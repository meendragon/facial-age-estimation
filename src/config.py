# config.py

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
    train_X_path = "./data/megaage_asian/train_X.pt"
    train_y_path = "./data/megaage_asian/train_y.pt"
    val_X_path   = "./data/megaage_asian/val_X.pt"
    val_y_path   = "./data/megaage_asian/val_y.pt"
    test_X_path  = "./data/megaage_asian/test_X.pt"
    test_y_path  = "./data/megaage_asian/test_y.pt"

    
# 🔎 1. SSL 성능 평가
class VCOPEvalConfig(BaseConfig):
    model_path = "weights/vcop_mini.pth"
    tuple_len = 4
    feature_size = 512
    topk = 3

    test_X_path = "./data/preTextData/test_X.pt"
    test_y_path = "./data/preTextData/test_y.pt"

# 🔎 2. 다운스트림 성능 평가 (나이 회귀)
class AgeRegEvalConfig(BaseConfig):
    model_path = "weights/age_regression.pth"
    test_X_path = "./data/megaage_asian/test_X.pt"
    test_y_path = "./data/megaage_asian/test_y.pt"
    
    batch_size = 32             # 또는 64 등 적절한 값
    feature_size = 512          # VCOP에서 사용된 encoder output dim과 동일해야 함
    tuple_len = 4


