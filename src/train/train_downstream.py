import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from torch.optim.lr_scheduler import CosineAnnealingLR

from models.r21d_mini import MiniR2Plus1D
from models.vcop_head import VCOPN
from models.age_regressor import AgeRegressor

class AugmentedAgeDataset(Dataset):
    def __init__(self, images, ages, transform=None, tuple_len=4):
        self.images = images  # shape: (B, 3, H, W)
        self.ages = ages      # shape: (B,)
        self.transform = transform
        self.tuple_len = tuple_len

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        img = self.images[idx]
        age = self.ages[idx]

        imgs = []
        for _ in range(self.tuple_len):
            augmented = self.transform(img)
            imgs.append(augmented)

        # Stack: (T, C, H, W) → (C, T, H, W)
        seq = torch.stack(imgs, dim=0).permute(1, 0, 2, 3)  # (C, T, H, W)
        return seq, age

def train_downstream(X_train, y_train, X_val, y_val, config):
    device = torch.device("cuda" if config.use_cuda and torch.cuda.is_available() else "cpu")

    transform = T.Compose([
        T.ToPILImage(),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.1, contrast=0.1),
        T.RandomAffine(degrees=10, translate=(0.05, 0.05)),
        T.ToTensor(),
    ])

    train_dataset = AugmentedAgeDataset(X_train, y_train, transform, config.tuple_len)
    val_dataset = AugmentedAgeDataset(X_val, y_val, transform, config.tuple_len)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    # Load pretrained VCOP encoder
    base = MiniR2Plus1D()
    vcop_model = VCOPN(base_network=base, feature_size=config.feature_size, tuple_len=config.tuple_len)
    vcop_model.load_state_dict(torch.load(config.pretrained_path))
    encoder = vcop_model.base

    # ✅ 모델 정의
    model = AgeRegressor(encoder, feature_dim=config.feature_size).to(device)
    criterion = nn.MSELoss()

    # ✅ 파라미터 그룹 설정 (인코더는 아주 작은 lr로 학습)
    encoder_params = []
    head_params = []

    for name, param in model.named_parameters():
        if 'encoder' in name:
            encoder_params.append(param)
        else:
            head_params.append(param)

    optimizer = optim.AdamW([
        {'params': encoder_params, 'lr': config.lr * 0.1},  # 인코더는 더 작은 lr
        {'params': head_params, 'lr': config.lr}             # 헤드는 원래 lr
    ])

    scheduler = CosineAnnealingLR(optimizer, T_max=config.num_epochs, eta_min=1e-7)
    
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0.0
        for imgs, targets in train_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)
        avg_train_loss = total_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs, targets = imgs.to(device), targets.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * imgs.size(0)
        avg_val_loss = val_loss / len(val_loader.dataset)

        # ✅ 스케줄러 step 호출
        scheduler.step()

        print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if epoch + 1 > config.warmup_epochs:
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                os.makedirs(config.save_dir, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(config.save_dir, config.model_name))
                print(f"✅ 모델 저장됨: {config.model_name}")
            else:
                patience_counter += 1
                if patience_counter >= config.patience:
                    print("🛑 Early Stopping")
                    break
        else:
            print(f"⏳ Warmup Epoch {epoch+1}/{config.warmup_epochs} - Early stopping 판단 보류")

    return model