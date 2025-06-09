import os
import torch
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error

from src.models.base_model import AgeEstimationModel
from src.models.base_model import LegacyAgeEstimationModel
from src.korean_config import config


# ✅ 평가 함수
def evaluate_model(model, test_loader):
    model.eval()
    pred_ages = []
    true_ages = []

    with torch.no_grad():
        for images, ages in test_loader:
            images = images.to(config['device'])
            outputs = model(images)
            pred_ages.extend(outputs.squeeze().cpu().numpy())
            true_ages.extend(ages.squeeze().cpu().numpy())

    return mean_absolute_error(true_ages, pred_ages)

# ✅ 테스트셋 로더 함수
def get_data_loaders(batch_size=32):
    test_X = torch.load('./data/megaage_asian/test_X.pt')
    test_Y = torch.load('./data/megaage_asian/test_y.pt')
    test_set = TensorDataset(test_X, test_Y)
    return DataLoader(test_set, batch_size=batch_size, shuffle=True)

# ✅ 실행 진입점
if __name__ == "__main__":
    # 모델 로딩
    path = "./src/weights"
    korean_checkpoint = os.path.join(path, 'pretrained_weight_korean.pt')
    legacy_checkpoint = os.path.join(path, 'legacy_weight.pt')

    korean_model = AgeEstimationModel(
        input_dim=3, output_nodes=1,
        model_name=config['model_name'],
        pretrain_weights='IMAGENET1K_V2'
    ).to(config['device'])

    legacy_model = LegacyAgeEstimationModel(
        input_dim=3, output_nodes=1,
        model_name=config['model_name'],
        pretrain_weights='IMAGENET1K_V2'
    ).to(config['device'])

    korean_model.load_state_dict(torch.load(korean_checkpoint))
    legacy_model.load_state_dict(torch.load(legacy_checkpoint))

    # 테스트
    test_loader = get_data_loaders()
    legacy_mae = evaluate_model(legacy_model, test_loader)
    korean_mae = evaluate_model(korean_model, test_loader)

    print(f"동아시아인 이미지에 대한 Base 모델 MAE: {legacy_mae:.2f}")
    print(f"동아시아인 이미지에 대한 Finetuned 모델 MAE: {korean_mae:.2f}")
