import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import torchvision.transforms as T
import matplotlib.pyplot as plt

class EvalAugmentedDataset(Dataset):
    def __init__(self, images, ages, transform=None, tuple_len=4):
        self.images = images  # torch.Tensor (B, 3, H, W)
        self.ages = ages      # torch.Tensor (B,)
        self.transform = transform
        self.tuple_len = tuple_len

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        img = self.images[idx]  # (3, H, W)
        age = self.ages[idx]

        # 증강 적용 (텐서 기반)
        imgs = []
        for _ in range(self.tuple_len):
            augmented = img.clone()
            if self.transform:
                augmented = self.transform(augmented)
            imgs.append(augmented)

        seq = torch.stack(imgs, dim=0).permute(1, 0, 2, 3)  # (T, C, H, W) → (C, T, H, W)
        return seq, age


def evaluate_downstream(model, X_test, y_test, config):
    device = torch.device("cuda" if config.use_cuda and torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # ❗ Tensor에 직접 적용할 수 있는 transform만 사용
    transform = T.Compose([
        T.Resize((config.input_size, config.input_size)),  # 텐서에 적용 가능
        T.ConvertImageDtype(torch.float32)                 # float64 → float32 (필요시)
    ])

    test_dataset = EvalAugmentedDataset(X_test, y_test, transform=transform, tuple_len=config.tuple_len)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    all_preds, all_targets = [], []

    with torch.no_grad():
        for seqs, targets in test_loader:
            seqs = seqs.to(device)         # (B, C, T, H, W)
            targets = targets.to(device)   # (B,)

            outputs = model(seqs)          # (B, T)
            preds = outputs.mean(dim=1)    # (B,)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    mae = mean_absolute_error(all_targets, all_preds)
    mse = mean_squared_error(all_targets, all_preds)
    rmse = np.sqrt(mse)

    print("📊 [Age Regression Evaluation]")
    print(f"   MAE:  {mae:.2f}")
    print(f"   MSE:  {mse:.2f}")
    print(f"   RMSE: {rmse:.2f}")
    plot_age_group_mae(all_targets, all_preds)


def plot_age_group_mae(all_targets, all_preds, bin_size=10, title="MAE by Age Group"):
    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)

    max_age = int(np.max(all_targets)) + 1
    age_bins = list(range(0, max_age + bin_size, bin_size))
    bin_labels = [f"{b}~{b+bin_size-1}" for b in age_bins[:-1]]
    mae_per_bin = []

    for i in range(len(age_bins) - 1):
        low, high = age_bins[i], age_bins[i + 1]
        mask = (all_targets >= low) & (all_targets < high)
        if np.sum(mask) > 0:
            mae = mean_absolute_error(all_targets[mask], all_preds[mask])
        else:
            mae = np.nan
        mae_per_bin.append(mae)

    plt.figure(figsize=(10, 5))
    plt.bar(bin_labels, mae_per_bin)
    plt.xticks(rotation=45)
    plt.ylabel("MAE")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()