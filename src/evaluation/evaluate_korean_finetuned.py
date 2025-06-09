import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from config import FinetuneEvalConfig
from models.age_regressor_ResNet import AgeEstimationModel


def evaluate_korean_finetuned(model, X_test, y_test, config):
    device = torch.device("cuda" if config.use_cuda and torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    loader = DataLoader(TensorDataset(X_test, y_test), batch_size=config.batch_size, shuffle=False)

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for imgs, targets in loader:
            imgs = imgs.to(device)
            outputs = model(imgs).squeeze()
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(targets.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    mae = mean_absolute_error(all_targets, all_preds)
    rmse = root_mean_squared_error(all_targets, all_preds)

    print("\n📊 [Korean-Finetuned Model Evaluation]")
    print(f"   MAE:  {mae:.2f}")
    print(f"   RMSE: {rmse:.2f}")

    return all_preds, all_targets


def plot_age_group_mae(y_true, y_pred, bin_size=10):
    bins = range(0, 101, bin_size)
    mae_per_group = []

    for i in range(len(bins) - 1):
        idx = (y_true >= bins[i]) & (y_true < bins[i + 1])
        if np.any(idx):
            mae = np.mean(np.abs(y_pred[idx] - y_true[idx]))
            mae_per_group.append(mae)
        else:
            mae_per_group.append(0)

    plt.figure(figsize=(10, 4))
    plt.bar([f"{b}-{b+bin_size-1}" for b in bins[:-1]], mae_per_group, edgecolor="black")
    plt.title("MAE per Age Group (Korean Finetuned)")
    plt.xlabel("Age Group")
    plt.ylabel("MAE")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
