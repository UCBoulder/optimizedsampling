import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchgeo.models import swin_v2_b, Swin_V2_B_Weights
from .base_model import BaseModel
from sklearn.model_selection import KFold  # CHANGE HERE: Use KFold instead of LeaveOneOut
from sklearn.metrics import r2_score
import numpy as np
import random  # CHANGE HERE: Added for setting random seed
from tqdm import tqdm  # CHANGE HERE: Added for progress bars
import os  # CHANGE HERE: Added for checkpoint saving


class SwinV2Model(BaseModel):
    """
    Swin Transformer V2 Base model with pretrained weights from TorchGeo.
    Finetuning with NAIP_RGB_MI_SATLAS pretrained weights.
    """
    
    def __init__(self, cfg):
        super().__init__(cfg)
        
        # CHANGE HERE: Set random seeds for reproducibility
        seed = getattr(self.cfg, 'SEED', 42)
        self._set_seed(seed)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._build_model().to(self.device)
        
        # Loss and optimizer
        self.criterion = self._get_criterion()
        self.optimizer = self._get_optimizer()
        self.scheduler = self._get_scheduler()
    
    def _set_seed(self, seed):
        """Set random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # CHANGE HERE: Make CUDA operations deterministic
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
    def _build_model(self):
        """Build Swin V2 model with pretrained weights, adapted for 4-channel input."""
        # Load pretrained weights
        weights = Swin_V2_B_Weights.NAIP_RGB_MI_SATLAS
        model = swin_v2_b(weights=weights)
        
        # Modify first conv layer to accept 4 channels
        # Original conv
        old_conv = model.features[0][0]

        new_conv = nn.Conv2d(
            in_channels=4,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None
        )

        with torch.no_grad():
            new_conv.weight[:, :3, :, :] = old_conv.weight  # copy pretrained RGB
            # CHANGE HERE: Initialize 4th channel as average of RGB channels instead of zeros
            # This provides better starting point for the additional channel
            # Citation: Common practice in transfer learning (bnsreenu/python_for_microscopists)
            new_conv.weight[:, 3:, :, :] = old_conv.weight.mean(dim=1, keepdim=True)

        model.features[0][0] = new_conv

        # CHANGE HERE: Freeze all layers except the final head
        for param in model.parameters():
            param.requires_grad = False

        # Modify the final layer for regression and unfreeze it
        if hasattr(model, 'head'):
            in_features = model.head.in_features
            model.head = nn.Linear(in_features, self.cfg.MODEL.OUTPUT_DIM)
            # CHANGE HERE: Ensure the new head is trainable and properly seeded
            for param in model.head.parameters():
                param.requires_grad = True
        elif hasattr(model, 'classifier'):
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, self.cfg.MODEL.OUTPUT_DIM)
            # CHANGE HERE: Ensure the new classifier is trainable and properly seeded
            for param in model.classifier.parameters():
                param.requires_grad = True
        
        return model

    
    def _get_criterion(self):
        """Get loss criterion based on config."""
        loss_type = getattr(self.cfg.MODEL, 'LOSS_TYPE', 'mse').lower()
        
        if loss_type == 'mse':
            return nn.MSELoss()
        elif loss_type == 'l1':
            return nn.L1Loss()
        elif loss_type == 'huber':
            return nn.HuberLoss()
        else:
            return nn.MSELoss()
    
    def _get_optimizer(self):
        """Get optimizer based on config."""
        optim_type = getattr(self.cfg.OPTIM, 'TYPE', 'adam').lower()
        # CHANGE HERE: Higher learning rate appropriate for training only the final layer
        # Default changed from 1e-4 to 1e-3
        lr = getattr(self.cfg.OPTIM, 'LR', 1e-3)
        # CHANGE HERE: Reduced weight decay since we're only training one layer
        weight_decay = getattr(self.cfg.OPTIM, 'WEIGHT_DECAY', 1e-4)
        
        # CHANGE HERE: Only pass trainable parameters to optimizer
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        
        if optim_type == 'adam':
            return optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay)
        elif optim_type == 'adamw':
            return optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
        elif optim_type == 'sgd':
            momentum = getattr(self.cfg.OPTIM, 'MOMENTUM', 0.9)
            return optim.SGD(trainable_params, lr=lr, momentum=momentum, weight_decay=weight_decay)
        else:
            return optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    
    def _get_scheduler(self):
        """Get learning rate scheduler based on config."""
        scheduler_type = getattr(self.cfg.OPTIM, 'SCHEDULER', None)
        
        if scheduler_type is None:
            return None
        elif scheduler_type.lower() == 'cosine':
            # CHANGE HERE: Reduced num_epochs default for faster convergence with frozen backbone
            num_epochs = getattr(self.cfg.OPTIM, 'NUM_EPOCHS', 50)
            return optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=num_epochs)
        elif scheduler_type.lower() == 'step':
            # CHANGE HERE: Reduced step_size for frozen backbone training
            step_size = getattr(self.cfg.OPTIM, 'STEP_SIZE', 15)
            gamma = getattr(self.cfg.OPTIM, 'GAMMA', 0.1)
            return optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        elif scheduler_type.lower() == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', patience=10)
        else:
            return None

    def train(self, dataset, logger=None):
        """
        Train the model using a single 80/20 train-validation split.
        Works for either image datasets or RCF feature datasets.

        Args:
            dataset: PyTorch Dataset object
            logger: optional logger
        """
        if logger:
            logger.info("Training model with 80/20 train-validation split...")

        batch_size = getattr(self.cfg.OPTIM, 'BATCH_SIZE', 32)
        num_epochs = getattr(self.cfg.OPTIM, 'NUM_EPOCHS', 10)
        seed = getattr(self.cfg, 'SEED', 42)

        # --- Create 80/20 split (same as one fold of 5-fold CV) ---
        n_total = len(dataset)
        n_val = n_total // 5  # 1/5 for validation
        n_train = n_total - n_val

        generator = torch.Generator().manual_seed(seed)
        train_subset, val_subset = random_split(dataset, [n_train, n_val], generator=generator)

        train_loader = DataLoader(
            train_subset, batch_size=batch_size, shuffle=True,
            num_workers=0, generator=generator
        )
        val_loader = DataLoader(
            val_subset, batch_size=batch_size, shuffle=False, num_workers=0
        )

        # --- Freeze backbone, train only head/classifier ---
        for name, param in self.model.named_parameters():
            if 'head' not in name and 'classifier' not in name:
                param.requires_grad = False

        # --- Reset optimizer and scheduler ---
        self.optimizer.zero_grad()
        self.scheduler = self._get_scheduler()

        # --- Training loop ---
        for epoch in tqdm(range(num_epochs), desc="Training", leave=False):
            self.model.train()
            epoch_loss = 0.0
            batch_count = 0

            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
                self.optimizer.zero_grad()

                if isinstance(batch, dict):
                    X = batch['image'].to(self.device)
                    y = batch['label'].to(self.device).squeeze()
                else:
                    X = batch[0].to(self.device)
                    y = batch[1].to(self.device).squeeze()

                outputs = self.model(X)
                if outputs.dim() > 1:
                    outputs = outputs.squeeze()

                loss = self.criterion(outputs, y)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                batch_count += 1
                tqdm.write(f"Epoch {epoch+1}, batch loss: {loss.item():.4f}")

            avg_loss = epoch_loss / batch_count if batch_count > 0 else 0
            tqdm.write(f"Epoch {epoch+1} complete - avg loss: {avg_loss:.4f}")

        # --- Validation ---
        self.model.eval()
        y_true, y_pred = [], []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation", leave=False):
                if isinstance(batch, dict):
                    X_val = batch['image'].to(self.device)
                    y_val = batch['label'].to(self.device).squeeze()
                else:
                    X_val = batch[0].to(self.device)
                    y_val = batch[1].to(self.device).squeeze()

                pred = self.model(X_val)
                if pred.dim() > 1:
                    pred = pred.squeeze()

                y_true.append(y_val.cpu().numpy())
                y_pred.append(pred.cpu().numpy())

        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        r2 = r2_score(y_true, y_pred)

        tqdm.write(f"\n{'='*50}")
        tqdm.write(f"Validation R²: {r2:.4f}")
        tqdm.write(f"{'='*50}\n")

        if logger:
            logger.info(f"Validation R²: {r2:.4f}")

        return r2


    def evaluate(self, dataset, logger=None):
        """Evaluate the model on a Dataset and return R² score."""
        self.model.eval()
        loader = DataLoader(dataset, batch_size=getattr(self.cfg.OPTIM, 'BATCH_SIZE', 32),
                            shuffle=False, num_workers=0)

        y_true_all = []
        y_pred_all = []

        with torch.no_grad():
            for batch in loader:
                if isinstance(batch, dict):
                    X = batch['image'].to(self.device)
                    y = batch['label'].to(self.device).squeeze()
                else:
                    X = batch[0].to(self.device)
                    y = batch[1].to(self.device).squeeze()

                pred = self.model(X)
                if pred.dim() > 1:
                    pred = pred.squeeze()

                y_true_all.append(y.cpu().numpy())
                y_pred_all.append(pred.cpu().numpy())

        y_true_all = np.concatenate(y_true_all)
        y_pred_all = np.concatenate(y_pred_all)

        r2 = r2_score(y_true_all, y_pred_all)

        if logger:
            logger.info(f"R² score: {r2:.4f}")

        return r2
    
    def predict(self, X):
        """Make predictions."""
        self.model.eval()
        
        with torch.no_grad():
            X_t = torch.FloatTensor(X).to(self.device)
            
            # Reshape for image input if needed
            if len(X_t.shape) == 2:
                n_samples = X_t.shape[0]
                total_pixels = X_t.shape[1]
                img_size = int((total_pixels // 3) ** 0.5)
                X_t = X_t.reshape(n_samples, 3, img_size, img_size)
            
            predictions = self.model(X_t)
            
            # Handle output shape
            if predictions.dim() > 1:
                predictions = predictions.squeeze()
            
            return predictions.cpu().numpy()
    
    def score(self, X, y):
        """Score the model (for sklearn compatibility)."""
        return self.evaluate(X, y)