from torch import nn
from torchvision.models import resnet, efficientnet_b0
import timm
from src.korean_config import config

class LegacyAgeEstimationModel(nn.Module): # 기존 모델, 성능 비교를 위해 남겨둠.
    def __init__(self, input_dim, output_nodes, model_name, pretrain_weights):
        super(LegacyAgeEstimationModel, self).__init__()

        self.input_dim = input_dim
        self.output_nodes = output_nodes
        self.pretrain_weights = pretrain_weights

        if model_name == 'resnet':
            self.model = resnet.resnet50(weights=pretrain_weights)
            self.model.fc = nn.Sequential(nn.Dropout(p=0.2, inplace=True),
                                          nn.Linear(in_features=2048, out_features=256, bias=True),
                                          nn.Linear(in_features=256, out_features=self.output_nodes, bias=True))

        elif model_name == 'efficientnet':
            self.model = efficientnet_b0()
            self.model.classifier = nn.Sequential(nn.Dropout(p=0.2, inplace=True),
                                                  nn.Linear(in_features=1280, out_features=256, bias=True),
                                                  nn.Linear(in_features=256, out_features=self.output_nodes, bias=True))

        elif model_name == 'vit':
            self.model = timm.create_model('vit_small_patch14_dinov2.lvd142m', img_size=config['img_size'], pretrained=pretrain_weights)
            
            # num_features = model.blocks[11].mlp.fc2.out_features
            num_features = 384
            self.model.head = nn.Sequential(nn.Dropout(p=0.2, inplace=True),
                                            nn.Linear(num_features, 256),
                                            nn.ReLU(),
                                            nn.Linear(256, self.output_nodes))

        else:
            raise ValueError(f"Unsupported model name: {model_name}")

    def forward(self, x):
        x = self.model(x)
        return x

# Custom Model
class AgeEstimationModel(nn.Module):

    def __init__(self, input_dim, output_nodes, model_name, pretrain_weights):
        super(AgeEstimationModel, self).__init__()

        self.input_dim = input_dim
        self.output_nodes = output_nodes
        self.pretrain_weights = pretrain_weights

        if model_name == 'resnet':
            self.model = resnet.resnet50(weights=pretrain_weights)
            # 헤드 레이어 강화
            self.model.fc = nn.Sequential(
                nn.Linear(2048, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(1024, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 1)
            )
            # BatchNorm 동적 적응
            for m in self.model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.track_running_stats = False

        elif model_name == 'efficientnet':
            self.model = efficientnet_b0()
            self.model.classifier = nn.Sequential(nn.Dropout(p=0.2, inplace=True),
                                                  nn.Linear(in_features=1280, out_features=256, bias=True),
                                                  nn.Linear(in_features=256, out_features=self.output_nodes, bias=True))

        elif model_name == 'vit':
            self.model = timm.create_model('vit_small_patch14_dinov2.lvd142m', img_size=config['img_size'], pretrained=pretrain_weights)
            
            # num_features = model.blocks[11].mlp.fc2.out_features
            num_features = 384
            self.model.head = nn.Sequential(nn.Dropout(p=0.2, inplace=True),
                                            nn.Linear(num_features, 256),
                                            nn.ReLU(),
                                            nn.Linear(256, self.output_nodes))

        else:
            raise ValueError(f"Unsupported model name: {model_name}")
        
        if model_name in ['resnet', 'efficientnet']:
            for m in self.model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.track_running_stats = False  # 한국인 데이터에 맞춘 동적 통계
                    m.reset_running_stats() 

    def forward(self, x):
        x = self.model(x)
        return x