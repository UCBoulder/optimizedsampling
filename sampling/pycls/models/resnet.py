import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchgeo.models import resnet18, ResNet18_Weights
from .base_model import BaseModel
from sklearn.model_selection import KFold  # CHANGE HERE: Use KFold instead of LeaveOneOut
from torch.utils.data import Subset, DataLoader
from sklearn.metrics import r2_score
import numpy as np
import random  # CHANGE HERE: Added for setting random seed
from tqdm import tqdm  # CHANGE HERE: Added for progress bars
import os  # CHANGE HERE: Added for checkpoint saving



class ResNet18Model(BaseModel):
    """
    ResNet18 regression model with TorchGeo Sentinel-2 MoCo pretrained weights.
    Finetuned for RGB + NIR (4 channels).
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        
        # CHANGE HERE: Set random seeds for reproducibility
        seed = getattr(self.cfg, 'SEED', 42)
        self._set_seed(seed)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._build_model().to(self.device)

        # Loss, optimizer, scheduler
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
        """Build ResNet18 with Sentinel-2 pretrained weights, adapted for 4-channel RGB+NIR input."""
        weights = ResNet18_Weights.SENTINEL2_ALL_MOCO
        model = resnet18(weights=weights)

        # Modify the first conv layer (originally 13 channels for Sentinel-2)
        old_conv = model.conv1
        in_channels = 4  # RGB + NIR
        model.conv1 = nn.Conv2d(
            in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )

        # CHANGE HERE: Initialize weights using average of RGB channels for NIR
        # This provides better starting point than using a single band
        # Citation: Common practice in transfer learning (bnsreenu/python_for_microscopists)
        with torch.no_grad():
            model.conv1.weight[:, :3] = old_conv.weight[:, :3]  # RGB
            # Average the first 3 channels for NIR initialization
            model.conv1.weight[:, 3] = old_conv.weight[:, :3].mean(dim=1)

        # CHANGE HERE: Freeze all layers except the final FC layer
        for param in model.layer4.parameters():
            param.requires_grad = True #also unfreezing second to last layer

        for param in model.parameters():
            param.requires_grad = False

        # Replace FC for regression and unfreeze it
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, self.cfg.MODEL.OUTPUT_DIM)
        
        # CHANGE HERE: Ensure the new FC layer is trainable
        for param in model.fc.parameters():
            param.requires_grad = True

        return model

    def _get_criterion(self):
        """Get loss criterion."""
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
        """Get optimizer."""
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
        """Get scheduler."""
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

    def train(self, dataset, logger=None, save_best_model=False, checkpoint_dir=None):
        """
        Train the model using 5-Fold Cross-Validation.
        Works for either image datasets or RCF feature datasets.
        
        Args:
            dataset: PyTorch Dataset object
            logger: optional logger
            save_best_model: If True, save the best performing fold model (CHANGE HERE)
            checkpoint_dir: Directory to save checkpoints (CHANGE HERE)
        """
        if logger:
            logger.info("Training model with 5-Fold Cross-Validation...")  # CHANGE HERE

        batch_size = getattr(self.cfg.OPTIM, 'BATCH_SIZE', 32)
        # CHANGE HERE: Added num_epochs parameter for training per fold
        num_epochs_per_fold = getattr(self.cfg.OPTIM, 'NUM_EPOCHS_PER_FOLD', 10)  # CHANGE HERE: Increased to 10 for k-fold
        # CHANGE HERE: Use 5-fold cross-validation instead of LOO
        n_splits = getattr(self.cfg.OPTIM, 'N_FOLDS', 5)
        seed = getattr(self.cfg, 'SEED', 42)  # CHANGE HERE: Get seed from config
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)  # CHANGE HERE: Use config seed
        all_r2 = []

        # CHANGE HERE: Only store weights of trainable parameters (final layer)
        init_state = {k: v.clone() for k, v in self.model.state_dict().items() 
                      if any(k.startswith(name) for name, p in self.model.named_parameters() if p.requires_grad)}
        
        # CHANGE HERE: Track best model for optional saving
        best_r2 = -float('inf')
        best_model_state = None
        best_fold = -1

        for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):  # CHANGE HERE
            if logger:
                logger.info(f"Fold {fold + 1}/{n_splits}")  # CHANGE HERE

            # CHANGE HERE: Reset only trainable layer weights for each fold
            current_state = self.model.state_dict()
            current_state.update(init_state)
            self.model.load_state_dict(current_state)
            self.model.train()
            
            # CHANGE HERE: Ensure backbone stays frozen during training
            for name, param in self.model.named_parameters():
                if 'fc' not in name:
                    param.requires_grad = False

            # Create subset datasets and loaders
            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)
            # CHANGE HERE: Added generator for reproducible shuffling
            generator = torch.Generator()
            generator.manual_seed(seed + fold)  # Different seed per fold for diversity
            train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, 
                                     num_workers=0, generator=generator)
            val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0)  # CHANGE HERE: batch_size instead of 1

            # Optimizer and scheduler reset
            self.optimizer.zero_grad()
            self.scheduler = self._get_scheduler()  # optional, recreate scheduler

            # CHANGE HERE: Train for multiple epochs per fold for better convergence with progress bar
            epoch_pbar = tqdm(range(num_epochs_per_fold), desc=f"Fold {fold+1}/{n_splits}", leave=False)
            for epoch in epoch_pbar:
                epoch_loss = 0.0
                batch_count = 0
                
                # CHANGE HERE: Add progress bar for batches
                train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs_per_fold}", leave=False)
                for batch in train_pbar:
                    self.optimizer.zero_grad()

                    if isinstance(batch, dict):
                        X = batch['image'].to(self.device)
                        y = batch['label'].to(self.device).squeeze()
                    else:  # tuple from RCF features
                        X = batch[0].to(self.device)
                        y = batch[1].to(self.device).squeeze()

                    outputs = self.model(X)
                    if outputs.dim() > 1:
                        outputs = outputs.squeeze()

                    loss = self.criterion(outputs, y)
                    loss.backward()
                    self.optimizer.step()
                    
                    # CHANGE HERE: Track loss for progress bar
                    epoch_loss += loss.item()
                    batch_count += 1
                    train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                
                # CHANGE HERE: Update epoch progress bar with average loss
                avg_loss = epoch_loss / batch_count if batch_count > 0 else 0
                epoch_pbar.set_postfix({'avg_loss': f'{avg_loss:.4f}'})

            # Validation on the held-out fold
            self.model.eval()
            y_true_fold = []
            y_pred_fold = []
            
            # CHANGE HERE: Add progress bar for validation
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Validation Fold {fold+1}", leave=False):
                    if isinstance(batch, dict):
                        X_val = batch['image'].to(self.device)
                        y_val = batch['label'].to(self.device).squeeze()
                    else:
                        X_val = batch[0].to(self.device)
                        y_val = batch[1].to(self.device).squeeze()

                    pred = self.model(X_val)
                    if pred.dim() > 1:
                        pred = pred.squeeze()

                    # CHANGE HERE: Collect all predictions and true values for R² calculation
                    y_true_fold.append(y_val.cpu().numpy())
                    y_pred_fold.append(pred.cpu().numpy())
            
            # CHANGE HERE: Compute R² for the entire fold
            y_true_fold = np.concatenate(y_true_fold) if len(y_true_fold) > 1 else y_true_fold[0]
            y_pred_fold = np.concatenate(y_pred_fold) if len(y_pred_fold) > 1 else y_pred_fold[0]
            r2 = r2_score(y_true_fold, y_pred_fold)
            all_r2.append(r2)
            
            # CHANGE HERE: Track best model
            if r2 > best_r2:
                best_r2 = r2
                best_fold = fold + 1
                best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            
            # CHANGE HERE: Print fold result with tqdm.write to avoid interfering with progress bars
            tqdm.write(f"Fold {fold + 1}/{n_splits} - R²: {r2:.4f}")
            if logger:
                logger.info(f"Fold {fold + 1} R²: {r2:.4f}")

        mean_r2 = sum(all_r2) / len(all_r2)
        std_r2 = np.std(all_r2)  # CHANGE HERE: Added standard deviation
        
        # CHANGE HERE: Use tqdm.write for final summary
        tqdm.write(f"\n{'='*50}")
        tqdm.write(f"5-Fold CV completed. Mean R²: {mean_r2:.4f} ± {std_r2:.4f}")
        tqdm.write(f"Best fold: {best_fold} with R²: {best_r2:.4f}")
        tqdm.write(f"{'='*50}\n")
        
        if logger:
            logger.info(f"5-Fold CV completed. Mean R²: {mean_r2:.4f} ± {std_r2:.4f}")
            logger.info(f"Best fold: {best_fold} with R²: {best_r2:.4f}")
        
        # CHANGE HERE: Save best model if requested
        if save_best_model and best_model_state is not None:
            if checkpoint_dir is None:
                checkpoint_dir = getattr(self.cfg, 'CHECKPOINT_DIR', './checkpoints')
            
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f'best_resnet18_fold{best_fold}_r2_{best_r2:.4f}.pth')
            
            checkpoint = {
                'model_state_dict': best_model_state,
                'fold': best_fold,
                'r2_score': best_r2,
                'mean_r2': mean_r2,
                'std_r2': std_r2,
                'all_fold_r2': all_r2,
                'config': self.cfg
            }
            
            torch.save(checkpoint, checkpoint_path)
            tqdm.write(f"Best model saved to: {checkpoint_path}")
            if logger:
                logger.info(f"Best model saved to: {checkpoint_path}")

        return mean_r2

    def evaluate(self, dataset, logger=None):
        """Evaluate the model on a Dataset and return R² score."""
        self.model.eval()
        batch_size = getattr(self.cfg.OPTIM, 'BATCH_SIZE', 32)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        y_true_all = []
        y_pred_all = []

        # CHANGE HERE: Added tqdm progress bar for evaluation
        with torch.no_grad():
            for batch in tqdm(loader, desc="Evaluating", leave=False):
                if isinstance(batch, dict):
                    X = batch['image'].to(self.device)
                    y = batch['label'].to(self.device).squeeze()
                else:  # tuple from RCF features
                    X = batch[0].to(self.device)
                    y = batch[1].to(self.device).squeeze()

                # Predict
                pred = self.model(X)
                if pred.dim() > 1:
                    pred = pred.squeeze()

                y_true_all.append(y.cpu().numpy())
                y_pred_all.append(pred.cpu().numpy())

        # Flatten arrays
        y_true_all = np.concatenate(y_true_all)
        y_pred_all = np.concatenate(y_pred_all)

        # Compute R²
        r2 = r2_score(y_true_all, y_pred_all)

        # CHANGE HERE: Use tqdm.write for output
        tqdm.write(f"Evaluation R² score: {r2:.4f}")
        if logger:
            logger.info(f"R² score: {r2:.4f}")

        return r2


    def predict(self, X):
        """Predict outputs."""
        self.model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X).to(self.device)
            if len(X_t.shape) == 2:
                n_samples = X_t.shape[0]
                total_pixels = X_t.shape[1]
                img_size = int((total_pixels // 4) ** 0.5)
                X_t = X_t.reshape(n_samples, 4, img_size, img_size)

            predictions = self.model(X_t)
            if predictions.dim() > 1:
                predictions = predictions.squeeze()
            return predictions.cpu().numpy()

    def score(self, X, y):
        """For sklearn compatibility."""
        return self.evaluate(X, y)
    
    def load_checkpoint(self, checkpoint_path, logger=None):
        """
        Load a saved model checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint file
            logger: optional logger
        
        Returns:
            Dictionary containing checkpoint metadata
        """
        # CHANGE HERE: Added method to load saved checkpoints
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if logger:
            logger.info(f"Loaded checkpoint from: {checkpoint_path}")
            logger.info(f"  Fold: {checkpoint['fold']}")
            logger.info(f"  R² Score: {checkpoint['r2_score']:.4f}")
            logger.info(f"  Mean R² across folds: {checkpoint['mean_r2']:.4f} ± {checkpoint['std_r2']:.4f}")
        
        tqdm.write(f"Checkpoint loaded from: {checkpoint_path}")
        tqdm.write(f"  Best fold: {checkpoint['fold']} with R²: {checkpoint['r2_score']:.4f}")
        
        return {
            'fold': checkpoint['fold'],
            'r2_score': checkpoint['r2_score'],
            'mean_r2': checkpoint['mean_r2'],
            'std_r2': checkpoint['std_r2'],
            'all_fold_r2': checkpoint['all_fold_r2']
        }