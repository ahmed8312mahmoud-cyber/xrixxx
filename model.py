"""
model.py
--------
إعادة تعريف معمارية SingleInputDenseNet المستخدمة في التدريب.
مستخرجة بشكل مطابق 100% من الـ notebook.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class LSEPooling(nn.Module):
    """
    Log-Sum-Exp Pooling بدلاً من Average/Max Pooling.
    مستخرجة من الـ notebook بدون أي تعديل.
    """
    def __init__(self, r: int = 10):
        super().__init__()
        self.r = r

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = x.view(B, C, -1)
        pooled = (
            torch.logsumexp(self.r * x, dim=2)
            - math.log(H * W)
        ) / self.r
        return pooled


class SingleInputDenseNet(nn.Module):
    """
    DenseNet121 Backbone + LSEPooling + Custom Classifier.
    Architecture مطابقة 100% للـ notebook:
    - features: DenseNet121 backbone
    - pool: LSEPooling(r=10)
    - classifier: Linear(1024→512) → BN → ReLU → Dropout(0.3) → Linear(512→14)
    """
    def __init__(self, n_classes: int = 14):
        super().__init__()

        base_model = models.densenet121(weights=None)
        self.features = base_model.features
        self.pool = LSEPooling(r=10)

        num_features = base_model.classifier.in_features  # 1024

        self.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, n_classes),
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        feats = self.features(img)
        feats = F.relu(feats, inplace=True)
        feats = self.pool(feats)
        x = self.classifier(feats)
        return x
