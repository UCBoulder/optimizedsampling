import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from torchgeo.models import resnet18, ResNet18_Weights
from tqdm import tqdm


class ResNet18Model:
    """ResNet18 regression model with Sentinel-2 MoCo pretrained weights, finetuned for RGB+NIR (4 channels)."""

    def __init__(self, cfg):
        self.cfg = cfg
        self._set_seed(getattr(cfg, 'SEED', 42))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._build_model().to(self.device)
        self.criterion = self._get_criterion()
        self.optimizer = self._get_optimizer()
        self.scheduler = self._get_scheduler()

    @staticmethod
    def _set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def _build_model(self):
        model = resnet18(weights=ResNet18_Weights.SENTINEL2_ALL_MOCO)

        old_conv = model.conv1
        model.conv1 = nn.Conv2d(4, old_conv.out_channels, kernel_size=old_conv.kernel_size,
                                 stride=old_conv.stride, padding=old_conv.padding,
                                 bias=old_conv.bias is not None)
        with torch.no_grad():
            model.conv1.weight[:, :3] = old_conv.weight[:, :3]
            model.conv1.weight[:, 3] = old_conv.weight[:, :3].mean(dim=1)

        for p in model.layer4.parameters():
            p.requires_grad = True
        for p in model.parameters():
            p.requires_grad = False

        model.fc = nn.Linear(model.fc.in_features, self.cfg.MODEL.OUTPUT_DIM)
        for p in model.fc.parameters():
            p.requires_grad = True
        return model

    def _get_criterion(self):
        return {'mse': nn.MSELoss, 'l1': nn.L1Loss, 'huber': nn.HuberLoss}.get(
            getattr(self.cfg.MODEL, 'LOSS_TYPE', 'mse').lower(), nn.MSELoss)()

    def _get_optimizer(self):
        optim_type = getattr(self.cfg.OPTIM, 'TYPE', 'adam').lower()
        lr = getattr(self.cfg.OPTIM, 'LR', 1e-3)
        weight_decay = getattr(self.cfg.OPTIM, 'WEIGHT_DECAY', 1e-4)
        params = filter(lambda p: p.requires_grad, self.model.parameters())
        if optim_type == 'sgd':
            momentum = getattr(self.cfg.OPTIM, 'MOMENTUM', 0.9)
            return optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
        if optim_type == 'adam':
            return optim.Adam(params, lr=lr, weight_decay=weight_decay)
        return optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    def _get_scheduler(self):
        scheduler_type = getattr(self.cfg.OPTIM, 'SCHEDULER', None)
        if scheduler_type is None:
            return None
        scheduler_type = scheduler_type.lower()
        if scheduler_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=getattr(self.cfg.OPTIM, 'NUM_EPOCHS', 50))
        if scheduler_type == 'step':
            return optim.lr_scheduler.StepLR(self.optimizer, step_size=getattr(self.cfg.OPTIM, 'STEP_SIZE', 15),
                                              gamma=getattr(self.cfg.OPTIM, 'GAMMA', 0.1))
        if scheduler_type == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', patience=10)
        return None

    @staticmethod
    def _unpack(batch):
        if isinstance(batch, dict):
            return batch['image'], batch['label'].squeeze()
        return batch[0], batch[1].squeeze()

    def train(self, dataset, logger=None, save_best_model=False, checkpoint_dir=None):
        batch_size = getattr(self.cfg.OPTIM, 'BATCH_SIZE', 32)
        num_epochs = getattr(self.cfg.OPTIM, 'NUM_EPOCHS_PER_FOLD', 10)
        n_splits = getattr(self.cfg.OPTIM, 'N_FOLDS', 5)
        seed = getattr(self.cfg, 'SEED', 42)
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

        trainable = [n for n, p in self.model.named_parameters() if p.requires_grad]
        init_state = {k: v.clone() for k, v in self.model.state_dict().items() if any(k.startswith(n) for n in trainable)}

        all_r2, best_r2, best_state, best_fold = [], -float('inf'), None, -1

        for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
            state = self.model.state_dict()
            state.update(init_state)
            self.model.load_state_dict(state)
            for name, p in self.model.named_parameters():
                p.requires_grad = 'fc' in name
            self.model.train()

            generator = torch.Generator().manual_seed(seed + fold)
            train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True, generator=generator)
            val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False)

            self.optimizer.zero_grad()
            self.scheduler = self._get_scheduler()

            for _ in tqdm(range(num_epochs), desc=f"Fold {fold + 1}/{n_splits}", leave=False):
                for batch in train_loader:
                    X, y = self._unpack(batch)
                    X, y = X.to(self.device), y.to(self.device)
                    self.optimizer.zero_grad()
                    out = self.model(X)
                    out = out.squeeze() if out.dim() > 1 else out
                    loss = self.criterion(out, y)
                    loss.backward()
                    self.optimizer.step()

            self.model.eval()
            y_true, y_pred = [], []
            with torch.no_grad():
                for batch in val_loader:
                    X, y = self._unpack(batch)
                    pred = self.model(X.to(self.device))
                    pred = pred.squeeze() if pred.dim() > 1 else pred
                    y_true.append(y.numpy())
                    y_pred.append(pred.cpu().numpy())

            r2 = r2_score(np.concatenate(y_true), np.concatenate(y_pred))
            all_r2.append(r2)
            if r2 > best_r2:
                best_r2, best_fold = r2, fold + 1
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            if logger:
                logger.info(f"Fold {fold + 1} R2: {r2:.4f}")

        mean_r2, std_r2 = float(np.mean(all_r2)), float(np.std(all_r2))
        if logger:
            logger.info(f"{n_splits}-fold CV mean R2: {mean_r2:.4f} +/- {std_r2:.4f} (best fold {best_fold}: {best_r2:.4f})")

        if save_best_model and best_state is not None:
            checkpoint_dir = checkpoint_dir or getattr(self.cfg, 'CHECKPOINT_DIR', './checkpoints')
            os.makedirs(checkpoint_dir, exist_ok=True)
            path = os.path.join(checkpoint_dir, f'best_resnet18_fold{best_fold}_r2_{best_r2:.4f}.pth')
            torch.save({'model_state_dict': best_state, 'fold': best_fold, 'r2_score': best_r2,
                        'mean_r2': mean_r2, 'std_r2': std_r2, 'all_fold_r2': all_r2, 'config': self.cfg}, path)
            if logger:
                logger.info(f"Best model saved to: {path}")

        return mean_r2

    def evaluate(self, dataset, logger=None):
        self.model.eval()
        loader = DataLoader(dataset, batch_size=getattr(self.cfg.OPTIM, 'BATCH_SIZE', 32), shuffle=False)
        y_true, y_pred = [], []
        with torch.no_grad():
            for batch in loader:
                X, y = self._unpack(batch)
                pred = self.model(X.to(self.device))
                pred = pred.squeeze() if pred.dim() > 1 else pred
                y_true.append(y.numpy())
                y_pred.append(pred.cpu().numpy())
        r2 = r2_score(np.concatenate(y_true), np.concatenate(y_pred))
        if logger:
            logger.info(f"R2 score: {r2:.4f}")
        return r2

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X).to(self.device)
            if X_t.dim() == 2:
                img_size = int((X_t.shape[1] // 4) ** 0.5)
                X_t = X_t.reshape(X_t.shape[0], 4, img_size, img_size)
            pred = self.model(X_t)
            return (pred.squeeze() if pred.dim() > 1 else pred).cpu().numpy()

    def score(self, X, y):
        return self.evaluate(X, y)

    def load_checkpoint(self, checkpoint_path, logger=None):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if logger:
            logger.info(f"Loaded checkpoint from: {checkpoint_path} (fold {checkpoint['fold']}, R2 {checkpoint['r2_score']:.4f})")
        return {k: checkpoint[k] for k in ('fold', 'r2_score', 'mean_r2', 'std_r2', 'all_fold_r2')}
