import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os

from models.vcop_head import VCOPN
from models.r21d_mini import MiniR2Plus1D

def train_vcop(X_train, y_train, X_val, y_val, config):
    device = torch.device("cuda" if config.use_cuda and torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=config.batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val, y_val), batch_size=config.batch_size, shuffle=False)

    base = MiniR2Plus1D()
    model = VCOPN(base_network=base, feature_size=config.feature_size, tuple_len=config.tuple_len).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    best_val_loss = float('inf')
    patience_counter = 0
    patience = config.patience  # ⬅️ config에서 불러오기

    for epoch in range(config.num_epochs):
        model.train()
        train_loss, train_correct = 0.0, 0

        for imgs, targets in train_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            train_correct += (outputs.argmax(dim=1) == targets).sum().item()

        avg_train_loss = train_loss / len(train_loader.dataset)
        avg_train_acc = train_correct / len(train_loader.dataset)

        # 🔍 Validation
        model.eval()
        val_loss, val_correct = 0.0, 0
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs, targets = imgs.to(device), targets.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * imgs.size(0)
                val_correct += (outputs.argmax(dim=1) == targets).sum().item()

        avg_val_loss = val_loss / len(val_loader.dataset)
        avg_val_acc = val_correct / len(val_loader.dataset)

        print(f"[Epoch {epoch+1}] 🏋️‍♂️ Train Loss: {avg_train_loss:.4f}, Acc: {avg_train_acc:.4f} | 🔍 Val Loss: {avg_val_loss:.4f}, Acc: {avg_val_acc:.4f}")

        # 🛑 Early Stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            os.makedirs(config.save_dir, exist_ok=True)
            best_model_path = os.path.join(config.save_dir, config.model_name)
            torch.save(model.state_dict(), best_model_path)
            print(f"✅ 모델 저장됨: {best_model_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("🛑 Early stopping triggered!")
                break

    return model