import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os

# 🧮 Kendall tau distance 계산 함수
def kendall_tau_distance(p1, p2):
    assert len(p1) == len(p2)
    n = len(p1)
    inv_p2 = {v: i for i, v in enumerate(p2)}
    p2_indices = [inv_p2[v] for v in p1]
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if p2_indices[i] > p2_indices[j]:
                count += 1
    return count

# 🧠 Soft Permutation Loss (Kendall tau 기반 soft label)
def soft_permutation_loss(pred_logits, target_permutation, all_perms, temperature=1.0):
    B = pred_logits.size(0)
    soft_targets = []
    for i in range(B):
        gt_perm = all_perms[target_permutation[i]]
        distances = []
        for perm in all_perms:
            d = kendall_tau_distance(gt_perm, perm)
            distances.append(d)
        distances = torch.tensor(distances, dtype=torch.float32)
        soft_label = torch.softmax(-distances / temperature, dim=0)
        soft_targets.append(soft_label)
    soft_targets = torch.stack(soft_targets).to(pred_logits.device)
    log_probs = F.log_softmax(pred_logits, dim=1)
    return F.kl_div(log_probs, soft_targets, reduction='batchmean')

# 🏋️ VCOP 학습 함수
def train_vcop(X_train, y_train, X_val, y_val, config, all_perms):
    from models.vcop_head import VCOPN
    from models.r21d_mini import MiniR2Plus1D

    device = torch.device("cuda" if config.use_cuda and torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=config.batch_size, shuffle=False)

    base = MiniR2Plus1D()
    model = VCOPN(base_network=base, feature_size=config.feature_size, tuple_len=config.tuple_len).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=config.lr)
    best_val_loss = float('inf')
    patience_counter = 0
    patience = config.patience

    for epoch in range(config.num_epochs):
        model.train()
        train_loss, train_correct = 0.0, 0

        for imgs, targets in train_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = soft_permutation_loss(outputs, targets, all_perms, temperature=0.5)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
            train_correct += (outputs.argmax(dim=1) == targets).sum().item()

        avg_train_loss = train_loss / len(train_loader.dataset)
        avg_train_acc = train_correct / len(train_loader.dataset)

        model.eval()
        val_loss, val_correct = 0.0, 0
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs, targets = imgs.to(device), targets.to(device)
                outputs = model(imgs)
                loss = soft_permutation_loss(outputs, targets, all_perms, temperature=0.5)
                val_loss += loss.item() * imgs.size(0)
                val_correct += (outputs.argmax(dim=1) == targets).sum().item()

        avg_val_loss = val_loss / len(val_loader.dataset)
        avg_val_acc = val_correct / len(val_loader.dataset)

        print(f"[Epoch {epoch + 1}] 🏋️ Train Loss: {avg_train_loss:.4f}, Acc: {avg_train_acc:.4f} | 🔍 Val Loss: {avg_val_loss:.4f}, Acc: {avg_val_acc:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            os.makedirs(config.save_dir, exist_ok=True)
            model_name_with_suffix = config.model_name + 'w'
            best_model_path = os.path.join(config.save_dir, model_name_with_suffix)
            torch.save(model.state_dict(), best_model_path)
            print(f"✅ 모델 저장됨: {model_name_with_suffix}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("🛑 Early stopping triggered!")
                break

    return model
