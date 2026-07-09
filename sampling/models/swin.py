import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import r2_score
from torchgeo.models import swin_v2_b, Swin_V2_B_Weights
from tqdm import tqdm


class SwinV2Model:
    """Swin Transformer V2 Base model, finetuned from NAIP_RGB_MI_SATLAS weights for 4-channel input."""

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
        model = swin_v2_b(weights=Swin_V2_B_Weights.NAIP_RGB_MI_SATLAS)

        old_conv = model.features[0][0]
        new_conv = nn.Conv2d(4, old_conv.out_channels, kernel_size=old_conv.kernel_size,
                              stride=old_conv.stride, padding=old_conv.padding,
                              bias=old_conv.bias is not None)
        with torch.no_grad():
            new_conv.weight[:, :3, :, :] = old_conv.weight
            new_conv.weight[:, 3:, :, :] = old_conv.weight.mean(dim=1, keepdim=True)
        model.features[0][0] = new_conv

        for p in model.parameters():
            p.requires_grad = False

        head_attr = 'head' if hasattr(model, 'head') else 'classifier'
        head = getattr(model, head_attr)
        new_head = nn.Linear(head.in_features, self.cfg.MODEL.OUTPUT_DIM)
        setattr(model, head_attr, new_head)
        for p in new_head.parameters():
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

    def train(self, dataset, logger=None):
        batch_size = getattr(self.cfg.OPTIM, 'BATCH_SIZE', 32)
        num_epochs = getattr(self.cfg.OPTIM, 'NUM_EPOCHS', 10)
        seed = getattr(self.cfg, 'SEED', 42)

        n_val = len(dataset) // 5
        generator = torch.Generator().manual_seed(seed)
        train_subset, val_subset = random_split(dataset, [len(dataset) - n_val, n_val], generator=generator)
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, generator=generator)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        for name, p in self.model.named_parameters():
            p.requires_grad = 'head' in name or 'classifier' in name

        self.optimizer.zero_grad()
        self.scheduler = self._get_scheduler()

        for _ in tqdm(range(num_epochs), desc="Training", leave=False):
            self.model.train()
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
        if logger:
            logger.info(f"Validation R2: {r2:.4f}")
        return r2

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
                img_size = int((X_t.shape[1] // 3) ** 0.5)
                X_t = X_t.reshape(X_t.shape[0], 3, img_size, img_size)
            pred = self.model(X_t)
            return (pred.squeeze() if pred.dim() > 1 else pred).cpu().numpy()

    def score(self, X, y):
        return self.evaluate(X, y)
