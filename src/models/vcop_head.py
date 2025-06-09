# 🔁 vcop_head.py
import torch
import torch.nn as nn
import math

class VCOPN(nn.Module):
    def __init__(self, base_network, feature_size=512, tuple_len=4):
        super().__init__()
        self.base = base_network
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.tuple_len = tuple_len
        self.num_classes = math.factorial(tuple_len)

        self.fc = nn.Sequential(
            nn.Linear(feature_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.num_classes)
        )

    def forward(self, x):  # x: (B, T, C, H, W)
        x = x.permute(0, 2, 1, 3, 4)       # → (B, C, T, H, W)
        feats = self.base(x)               # → (B, C, T, H, W)
        pooled = self.pool(feats)          # → (B, C, 1, 1, 1)
        flat = pooled.view(x.size(0), -1)  # → (B, C)
        return self.fc(flat)               # → (B, num_classes)