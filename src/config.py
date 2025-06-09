# config.py
import torch.cuda

class BaseConfig:
    image_size = (224, 224)
    use_cuda = True
    seed = 42
    batch_size = 64


# 🔁 1. Self-Supervised Learning (VCOP)
class VCOPConfig(BaseConfig):
    tuple_len = 4
    k_permutations = 24
    feature_size = 512
    num_epochs = 50
    lr = 1e-4
    save_dir = "weights/"
    model_name = "vcop_mini.pth"
    topk = 3
    patience = 5 
    model_path = "weights/vcop_mini.pth"
      
    train_X_path = "./data/preTextData/train_X.pt"
    train_y_path = "./data/preTextData/train_y.pt"
    val_X_path = "./data/preTextData/val_X.pt"
    val_y_path = "./data/preTextData/val_y.pt"
    test_X_path  = "./data/preTextData/test_X.pt"
    test_y_path  = "./data/preTextData/test_y.pt"
    pretrained_path = None

# 📥 2. 다운스트림 태스크 (Age Regression)
class AgeRegConfig(BaseConfig):
    input_channels = 3
    num_outputs = 1
    lr = 5e-4                         # Head learning rate
    encoder_lr_ratio = 0.1           # 🔹 Encoder lr = lr * ratio
    num_epochs = 100
    patience = 5
    save_dir = "weights/"
    model_name = "age_regression.pth"
    pretrained_path = "weights/vcop_mini.pth"
    batch_size = 32
    feature_size = 512
    tuple_len = 4
    warmup_epochs = 5
    input_size = 112
    train_X_path = "./data/megaage_asian/train_X.pt"
    train_y_path = "./data/megaage_asian/train_y.pt"
    val_X_path   = "./data/megaage_asian/val_X.pt"
    val_y_path   = "./data/megaage_asian/val_y.pt"
    test_X_path  = "./data/megaage_asian/test_X.pt"
    test_y_path  = "./data/megaage_asian/test_y.pt"
    model_path = "./src/weights/age_regression.pth"
    
config = {
    'img_width': 128,
    'img_height': 128,
    'img_size': 128,
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225],
    'model_name': 'resnet',
    'root_dir': '',
    'csv_path': '',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'image_path_test_folder': './megaage_asian/test',
    'leaky_relu': False,
    'epochs': 20,
    'batch_size': 128,
    'eval_batch_size': 256,
    'seed': 42,
    'lr': 0.0001,
    'wd': 0.001,
    'save_interval': 1,
    'reload_checkpoint': None,
    'finetune': 'weights/FA_DOCS/crnn-fa-base.pt',
    # 'finetune': None,
    'weights_dir': 'weights',
    'log_dir': 'logs',
    'cpu_workers': 0,
}
