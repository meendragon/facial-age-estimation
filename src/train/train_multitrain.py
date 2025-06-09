import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import mean_absolute_error

from models.r21d_mini import MiniR2Plus1D
from models.vcop_head import VCOPN
from models.age_regressor import AgeRegressor
from models.base_model import AgeEstimationModel


class AugmentedAgeDataset(Dataset):
    def __init__(self, images, labels, transform=None, tuple_len=4):
        self.images = images
        self.labels = labels
        self.transform = transform
        self.tuple_len = tuple_len

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        if isinstance(label, torch.Tensor):
            label = label.clone().detach().float()
        else:
            label = torch.tensor(label, dtype=torch.float32)
        imgs_aug = [self.transform(img) for _ in range(self.tuple_len)]
        seq = torch.stack(imgs_aug, dim=1)  # (C, T, H, W)
        return seq, label
        
        
class SimpleImageDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        if isinstance(label, torch.Tensor):
            label = label.clone().detach().float()
        else:
            label = torch.tensor(label, dtype=torch.float32)
        img = self.transform(img) if self.transform else img
        return img, label

class AdaptiveAgeLoss(nn.Module):
    def __init__(self, age_bins=[60], weights=[1.0, 3.0], device='cuda'):
        super().__init__()
        self.bins = torch.tensor(age_bins, device=device)
        self.weights = torch.tensor(weights, device=device)

    def forward(self, pred, target):
        target = target.squeeze().to(self.weights.device)
        bin_indices = torch.bucketize(target, self.bins)
        return (torch.abs(pred.squeeze() - target) * self.weights[bin_indices]).mean()






def train_multitrain(X_train, y_train, X_val, y_val,config, config_dict):
    device = torch.device("cuda" if config.use_cuda and torch.cuda.is_available() else "cpu")
    transform_vcop = T.Compose([
        T.ToPILImage(),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.1, contrast=0.1),
        T.RandomAffine(degrees=10, translate=(0.05, 0.05)),
        T.ToTensor(),
    ])
    transform_kor = T.Compose([
        T.ToPILImage(),
        T.ToTensor(),
    ])

    train_loader_vcop = DataLoader(AugmentedAgeDataset(X_train, y_train, transform_vcop, config.tuple_len), batch_size=config.batch_size, shuffle=False)
    train_loader_kor = DataLoader(SimpleImageDataset(X_train, y_train, transform_kor), batch_size=config_dict.batch_size, shuffle=False)
    val_loader_vcop = DataLoader(AugmentedAgeDataset(X_val, y_val, transform_vcop, config.tuple_len), batch_size=config.batch_size, shuffle=False)
    val_loader_kor = DataLoader(SimpleImageDataset(X_val, y_val, transform_kor), batch_size=config_dict.batch_size, shuffle=False)

    vcop_model = VCOPN(base_network=MiniR2Plus1D(), feature_size=config.feature_size, tuple_len=config.tuple_len)
    vcop_model.load_state_dict(torch.load(config.vcop_pretrained_path, map_location=device))
    regressor_vcop = AgeRegressor(vcop_model.base).to(device)

    korean_model = AgeEstimationModel(
        input_dim=3,
        output_nodes=1,
        model_name=config_dict.model_name,
        pretrain_weights='IMAGENET1K_V2'
    ).to(device)
    korean_model.load_state_dict(torch.load(config_dict.pretrained_weight_path, map_location=device))

    criterion = nn.MSELoss()

    optimizer = optim.AdamW([
        {'params': regressor_vcop.encoder.parameters(), 'lr': config.lr * 0.1},
        {'params': regressor_vcop.head.parameters(), 'lr': config.lr},
        {'params': korean_model.parameters(), 'lr': config_dict.lr}
    ], weight_decay=config_dict.wd)

    scheduler = CosineAnnealingLR(optimizer, T_max=config_dict.epochs, eta_min=1e-7)

    best_val_mae = float('inf')
    patience_counter = 0
    os.makedirs(config.save_dir, exist_ok=True)

    for epoch in range(config_dict.epochs):
        regressor_vcop.train()
        korean_model.train()
        total_loss = 0.0

        for (x1, y1), (x2, y2) in zip(train_loader_vcop, train_loader_kor):
            x1, y1 = x1.to(device), y1.to(device).view(-1)
            x2, y2 = x2.to(device), y2.to(device).view(-1)

            optimizer.zero_grad()
            pred1 = regressor_vcop(x1).view(-1)
            pred2 = korean_model(x2).view(-1)
    

            loss1 = criterion(pred1, y1)
            loss2 = criterion(pred2, y2)
            loss = config.lambda_task * loss1 + (1 - config.lambda_task) * loss2

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        regressor_vcop.eval()
        korean_model.eval()
        preds_all = []
        labels_all = []

        with torch.no_grad():
            for (x1, y1), (x2, y2) in zip(val_loader_vcop, val_loader_kor):
                x1, y1 = x1.to(device), y1.to(device).view(-1)
                x2, y2 = x2.to(device), y2.to(device).view(-1)

                pred1 = regressor_vcop(x1).view(-1)
                pred2 = korean_model(x2).view(-1)
                preds = (config.lambda_task * pred1 + (1 - config.lambda_task) * pred2).detach().cpu().numpy()
                preds_all.extend(preds)
                labels_all.extend(y1.cpu().numpy())

        val_mae = mean_absolute_error(labels_all, preds_all)
        scheduler.step()

        print(f"[Epoch {epoch+1}] Loss: {total_loss:.4f} | Val MAE: {val_mae:.4f}")

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_counter = 0
            torch.save({
                'regressor_vcop': regressor_vcop.state_dict(),
                'regressor_kor': korean_model.state_dict()
            }, os.path.join(config_dict.save_dir, config_dict.save_name))
            print(f"✅ Best model saved at epoch {epoch+1} (MAE={val_mae:.3f})")
        else:
            patience_counter += 1
            if patience_counter >= config_dict.patience:
                print("🛑 Early stopping triggered.")
                break

