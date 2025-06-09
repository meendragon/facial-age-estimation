import os
import sys
import torch
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
import torchmetrics as tm
from src.korean_config import config
from src.models.base_model import AgeEstimationModel
from src.train.functions import train_one_epoch, validation
from torch.optim.lr_scheduler import CosineAnnealingLR

# 시스템 경로 추가
sys.path.insert(0, os.path.abspath('./src'))

# 전역 변수 초기화
writer = SummaryWriter()
loss_train_hist = []
loss_valid_hist = []
metric_train_hist = []
metric_valid_hist = []

class AdaptiveAgeLoss(nn.Module):
    """가중치 기반 연령 추정 손실 함수"""
    def __init__(self, age_bins=[60], weights=[1.0, 3.0], device='cuda'):
        super().__init__()
        self.bins = torch.tensor(age_bins, device=device)
        self.weights = torch.tensor(weights, device=device)

    def forward(self, pred, target):
        target = target.squeeze().to(self.weights.device)
        bin_indices = torch.bucketize(target, self.bins)
        return (torch.abs(pred.squeeze() - target) * self.weights[bin_indices]).mean()

def initialize_model(checkpoint_path=None):
    """모델 초기화 및 체크포인트 로드"""
    model = AgeEstimationModel(
        input_dim=3,
        output_nodes=1,
        model_name='resnet',
        pretrain_weights='IMAGENET1K_V2'
    ).to(config['device'])

    if checkpoint_path and os.path.exists(checkpoint_path):
        load_checkpoint(model, checkpoint_path)
        
    return model

def load_checkpoint(model, checkpoint_path):
    """체크포인트 로딩 핸들러"""
    pretrained_dict = torch.load(checkpoint_path, map_location=config['device'])
    model_dict = model.state_dict()
    
    # 백본 가중치만 필터링
    pretrained_dict = {
        k: v for k, v in pretrained_dict.items()
        if k.startswith('model.') and not k.startswith('model.fc')
    }
    
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict=False)
    print("백본 가중치 성공적으로 로드됨")

def configure_optimizer(model):
    """계층별 학습률 설정 옵티마이저"""
    backbone_params = []
    fc_params = []

    for name, param in model.named_parameters():
        (fc_params if 'fc' in name else backbone_params).append(param)

    return optim.AdamW([
        {'params': backbone_params, 'lr': 1e-5},
        {'params': fc_params, 'lr': 1e-4}
    ], weight_decay=config['wd'])

def train_model(model, train_loader, valid_loader, epochs=config['epochs']):
    """메인 학습 루프"""
    optimizer = configure_optimizer(model)
    scheduler = CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-7)
    loss_fn = AdaptiveAgeLoss(age_bins=[60], weights=[1.0, 3.0], device=config['device'])
    metric = tm.MeanAbsoluteError().to(config['device'])
    
    best_loss = torch.inf
    checkpoint_dir = './src/train/checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(epochs):
        # Training phase
        model, train_loss, train_metric = train_one_epoch(
            model, train_loader, loss_fn, optimizer, metric, epoch
        )
        
        # Validation phase
        valid_loss, valid_metric = validation(
            model, valid_loader, loss_fn, metric
        )
        
        # 기록 업데이트
        update_training_history(train_loss, valid_loss, train_metric, valid_metric)
        update_checkpoint(model, valid_loss, epoch, checkpoint_dir, best_loss)
        
        scheduler.step()
        log_progress(epoch, train_loss, valid_loss, train_metric, valid_metric)

    writer.close()
    return model

def update_training_history(train_loss, valid_loss, train_metric, valid_metric):
    """학습 기록 업데이트"""
    loss_train_hist.append(train_loss)
    loss_valid_hist.append(valid_loss)
    metric_train_hist.append(train_metric)
    metric_valid_hist.append(valid_metric)

def update_checkpoint(model, valid_loss, epoch, checkpoint_dir, best_loss):
    """모델 체크포인트 저장"""
    if valid_loss < best_loss:
        checkpoint_path = os.path.join(
            checkpoint_dir, 
            f'finetune-epoch-{epoch}-loss_valid-{valid_loss:.3f}.pt'
        )
        torch.save(model.state_dict(), checkpoint_path)
        print(f'\nModel saved at epoch {epoch} with loss {valid_loss:.3f}')

def log_progress(epoch, train_loss, valid_loss, train_metric, valid_metric):
    """학습 진행 상황 로깅"""
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/valid', valid_loss, epoch)
    
    if epoch % 5 == 0:
        print(f"\nEpoch {epoch}:")
        print(f"  Train >> Loss: {train_loss:.3f}, MAE: {train_metric:.3f}")
        print(f"  Valid >> Loss: {valid_loss:.3f}, MAE: {valid_metric:.3f}")

def draw_plot():
    """학습 곡선 시각화"""
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(loss_train_hist, 'r-', label='Train Loss')
    plt.plot(loss_valid_hist, 'b-', label='Valid Loss')
    plt.title('Loss Curve')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(metric_train_hist, 'g-', label='Train MAE')
    plt.plot(metric_valid_hist, 'c-', label='Valid MAE')
    plt.title('Metric Curve')
    plt.legend()
    
    plt.savefig(os.path.join('./src', 'finetune_metric_plot.png'))
    plt.close()