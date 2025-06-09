# models/age_regressor.py
#다운스트림 용도! SSL 이후
import torch
import torch.nn as nn

class AgeRegressor(nn.Module):
    def __init__(self, encoder, feature_dim=512, spatial_size=(7, 7)):
        super().__init__()
        self.encoder = encoder

        # Temporal 축만 평균 → (B, C, T, H, W) → (B, C, H, W)
        self.temporal_pool = nn.AdaptiveAvgPool3d((1, spatial_size[0], spatial_size[1]))

        # ResNet 스타일의 Head
        self.head = nn.Sequential(
            nn.Conv2d(feature_dim, 128, kernel_size=1),  # 채널만 줄이기 (spatial 유지)
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # (B, 128, 1, 1)
            nn.Flatten(),                  # (B, 128)
            nn.Linear(128, 1)              # 회귀 출력
        )

    def forward(self, x):  # x: (B, C, T, H, W)
        z = self.encoder(x)                # → (B, C, T, H, W)
        z = self.temporal_pool(z)         # → (B, C, H, W)
        z = z.squeeze(2)  
        out = self.head(z)                # → (B, 1)
        return out.squeeze(1)