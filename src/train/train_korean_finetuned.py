import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_absolute_error
import os

from config import BaseFinetuneConfig
from models.age_regressor_ResNet import AgeEstimationModel


def train_korean_finetuned(X_train, X_val, y_train, y_val, config):
    device = torch.device("cuda" if config.use_cuda and torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=config.batch_size, shuffle=False)

    model = AgeEstimationModel(pretrain_weights='IMAGENET1K_V2').to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    best_mae = float('inf')
    patience_counter = 0
    max_patience = config.patience if hasattr(config, 'patience') else 10

    for epoch in range(1, config.num_epochs + 1):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device).view(-1, 1)

            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()

        # 검증
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                preds = model(batch_x).squeeze().cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(batch_y.numpy())

        val_mae = mean_absolute_error(all_targets, all_preds)
        print(f"📍 Epoch {epoch:03d} | Validation MAE: {val_mae:.2f}")

        # Early stopping
        if val_mae < best_mae:
            best_mae = val_mae
            patience_counter = 0
            save_path = os.path.join(config.save_dir, config.model_name)
            os.makedirs(config.save_dir, exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"✅ Best model saved to: {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print("⏹️ Early stopping triggered.")
                break