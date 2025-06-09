import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as T

from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 커스텀 모델 및 데이터셋 모듈
from models.age_regressor import AgeRegressor
from models.base_model import AgeEstimationModel
from models.r21d_mini import MiniR2Plus1D
from train.train_multitrain import AugmentedAgeDataset, SimpleImageDataset

def evaluate_multitrain(X_test, y_test, config, config_dict):
    device = torch.device("cuda" if config.use_cuda and torch.cuda.is_available() else "cpu")

    # ── 체크포인트 경로 구성 ──
    checkpoint_path = os.path.join(config_dict.save_dir, config_dict.save_name)

    # ── 데이터 변환 설정 ──
    transform_vcop = T.Compose([
        T.ToPILImage(),
        T.ToTensor()
    ])
    transform_kor = T.Compose([
        T.ToPILImage(),
        T.ToTensor()
    ])

    # ── 테스트 데이터 로더 ──
    test_loader_vcop = DataLoader(
        AugmentedAgeDataset(X_test, y_test, transform=transform_vcop, tuple_len=config.tuple_len),
        batch_size=config.batch_size,
        shuffle=False
    )

    test_loader_kor = DataLoader(
        SimpleImageDataset(X_test, y_test, transform=transform_kor),
        batch_size=config_dict.batch_size,
        shuffle=False
    )

    # ── 모델 초기화 및 로드 ──
    vcop_encoder = MiniR2Plus1D()
    regressor_vcop = AgeRegressor(vcop_encoder).to(device)

    korean_model = AgeEstimationModel(
        input_dim=3,
        output_nodes=1,
        model_name=config_dict.model_name,
        pretrain_weights=None
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    regressor_vcop.load_state_dict(checkpoint['regressor_vcop'])
    korean_model.load_state_dict(checkpoint['regressor_kor'])

    regressor_vcop.eval()
    korean_model.eval()

    preds_all = []
    labels_all = []

    with torch.no_grad():
        for (x1, y1), (x2, y2) in zip(test_loader_vcop, test_loader_kor):
            x1, y1 = x1.to(device), y1.to(device).view(-1)
            x2, y2 = x2.to(device), y2.to(device).view(-1)

            pred1 = regressor_vcop(x1).view(-1)
            pred2 = korean_model(x2).view(-1)
            final_pred = config.lambda_task * pred1 + (1 - config.lambda_task) * pred2

            preds_all.extend(final_pred.cpu().numpy())
            labels_all.extend(y1.cpu().numpy())

    # ── 지표 계산 ──
    preds_all = np.array(preds_all)
    labels_all = np.array(labels_all)

    mae = mean_absolute_error(labels_all, preds_all)
    mse = mean_squared_error(labels_all, preds_all)
    rmse = np.sqrt(mse)

    print(f"[Age Regression Evaluation]")
    print(f"MAE: {mae:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")

    # ── 연령대별 MAE 분석 ──
    bins = [0, 10, 20, 30, 40, 50, 60, 70]
    labels = ['0~9', '10~19', '20~29', '30~39', '40~49', '50~59', '60~69']
    age_groups = np.digitize(labels_all, bins) - 1  # index adjustment

    group_maes = []
    for i in range(len(labels)):
        mask = age_groups == i
        if np.sum(mask) > 0:
            group_mae = mean_absolute_error(labels_all[mask], preds_all[mask])
        else:
            group_mae = 0
        group_maes.append(group_mae)

    # ── 시각화 ──
    plt.figure(figsize=(10, 5))
    plt.bar(labels, group_maes)
    plt.title('MAE by Age Group')
    plt.ylabel('MAE')
    plt.xlabel('Age Group')
    plt.tight_layout()
    plt.show()

    return mae 