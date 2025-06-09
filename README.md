## 프로젝트 개요

본 프로젝트는 얼굴 이미지의 **시계열 데이터를 활용**하여, Self-Supervised Learning 기반 인코더를 사전 학습(pretraining)하고, 이를 통해 **얼굴 나이 예측(Age Regression)** 성능을 향상시키는 것을 목표로 합니다.

사전학습 단계에서는 Video Clip Order Prediction (VCOP) 아이디어를 바탕으로, 동일 인물의 다양한 연령대 사진을 올바른 순서로 배열하는 **순서 예측 태스크**를 수행하며, 시계열적인 노화 정보를 인코더가 학습하도록 합니다.  
이후 다운스트림에서는 학습된 인코더를 활용하여 단일 이미지 기반 나이 회귀 모델을 학습합니다.

---

## 비교 실험 조건

본 프로젝트에서는 다음 네 가지 실험 조건을 통해 모델 성능을 비교합니다:

1. **Baseline Age Regression Model**  
   사전학습 없이 일반 얼굴 이미지(UTKFace)로 학습한 기본 나이 예측 모델

2. **Korean Fine-tuned Model**  
   베이스라인 모델을 한국인 얼굴 이미지로 파인튜닝한 모델

3. **VCOP Pretrained + Downstream Model**  
   VCOP 순서 예측 태스크로 사전학습한 인코더를 활용하여 나이 회귀를 수행한 모델

4. **Multi-train Loss Sharing Model**  
   VCOP 인코더와 Age Regressor 인코더를 **각각 독립적으로 구성**하되, 두 태스크의 **손실 함수를 동시에 계산**하여 함께 학습시키는 방식의 모델

---

## 기대 효과

VCOP 기반 사전학습을 통해 인코더는 얼굴 이미지 간의 **시계열적 변화**와 **노화 흐름**을 학습할 수 있습니다.

특히 4번 방식은 서로 다른 인코더 구조를 사용하면서도 손실을 공유하여 학습하기 때문에,  
**노화 순서 인식 능력을 회귀 모델에 간접적으로 전달**할 수 있습니다.  

이를 통해 단일 이미지 기반 회귀에서도 시계열적인 노화 패턴을 포착할 수 있으며,  
**2D 이미지 입력만으로도 더 풍부한 노화 정보를 반영**할 수 있을 것으로 기대됩니다.

---

## 팀원

- 20233127 김민형  
- 20201463 박동민  
- 20211429 위명준  
- 20201502 정현우

---

## 프로젝트 구조

```bash
project/
├── data/                            # 데이터 디렉토리
│   ├── preTextData/                # VCOP 학습용 시계열 이미지 시퀀스
│   ├── megaage_asian/              # MegaAge-Asian 데이터셋
│   └── korean_image#1/             # 한국인 얼굴 이미지 (정제 및 전처리 완료본)
├── src/                             # 소스 코드 디렉토리
│   ├── train/                       # 학습 스크립트
│   │   ├── __init__.py
│   │   ├── functions.py
│   │   ├── train_base_to_kor.ipynb        # Resnet50 → Korean 데이터 파인튜닝
│   │   ├── train_downstream.py            # Pretrained 인코더 기반 Age Regression
│   │   ├── train_korean_finetuned.py      # 한국 이미지 전용 모델 파인튜닝
│   │   ├── train_multitrain.py            # VCOP + Age Regression 손실 동시 학습
│   │   └── train_vcop.py                  # VCOP 사전학습 (순서 예측)
│   ├── models/                     # 모델 정의
│   │   ├── __init__.py
│   │   ├── age_regressor.py         # SSL 이후 다운스트림 모델 구조 정의
│   │   ├── base_model.py            # Baseline 및 finetune된 모델 구조 정의
│   │   ├── r21d_mini.py             # 기존 R(2+1)D 구조를 간소화한 모델 구조 정의
│   │   └── vcop_head.py             # 실제 SSL에 사용되는 모델 구조 정의
│   ├── weights/                    # 학습된 모델 가중치 파일
│   │   ├── age_regression.pth
│   │   ├── legacy_weight.pt
│   │   ├── pretrained_weight_korean.pt
│   │   └── vcop_mini.pth
│   ├── loader/                     # 한국인 이미지셋 로딩 및 전처리
│   │   ├── custom_dataset_dataloader_korean.py     # 커스텀 한국 이미지 로더
│   │   └── custom_dataset_datasplitter_korean.py   # 학습/검증/테스트 분할 모듈
│   └── evaluation/                 # 성능 평가 스크립트
│       ├── __init__.py
│       ├── evaluate_base_kor.py           # Base model 및 Korean_finetuned Model 평가
│       ├── evaluate_downstream.py         # downstream된 SSL 모델 단독 평가
│       ├── evaluate_multitrain.py         # multitrain 모델 평가
│       └── evaluate_vcop.py               # SSL 모델의 순서 맞추기 정확도 평가
├── README.md                       # 이 문서
└── requirements.txt  

---
```
## 📦 데이터셋 다운로드 안내

`data/`에 업로드해둔 데이터를 사용하면 되며 원본 데이터 다운로드를 링크를 통해 하고싶다면

학습에 사용된 원본 데이터를 아래 링크에서 수동 다운로드 후,  
`data/` 디렉토리 아래에 배치해야 합니다:

| 구분 | 링크 | 설명 |
|------|------|------|
| AIHub 한국인 얼굴 시계열 데이터 | [https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=71415](https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=71415) | 동일 인물의 다양한 연령대 얼굴 이미지 시퀀스 |
| MegaAge-Asian | [https://www.dropbox.com/scl/fi/brq5o467fl2oz2u5oz0ng/megaage_asian.zip?rlkey=beyju63xv56jtyjuhn30367ae&dl=0](https://www.dropbox.com/scl/fi/brq5o467fl2oz2u5oz0ng/megaage_asian.zip?rlkey=beyju63xv56jtyjuhn30367ae&dl=0) | 아시아인 얼굴 이미지 + 나이 레이블|
---
📂 주요 코드 파일 설명 – 사전학습 (Finetuning Baseline with Korean imageset)

---

**models/base_model.py**

Baseline 모델 및, Baseline 모델을 한국인 이미지로 finetune한 모델의 구조를 정의하는 파일입니다.
Baseline 모델은 Resnet50을 기반으로 하며, 비교적 간단한 FC 레이어를 가집니다.
한국인 이미지로 finetune한 모델은 기존 Baseline 모델에 비해 더 깊은 FC 레이어를 가집니다.
이는 기존에 비해 복잡한 연령 패턴을 잘 파악할 수 있도록 도와줍니다.

---

**train/base_to_kor.py**

Baseline 모델을 한국인 이미지셋으로 finetune하기 위해 필요한 모듈들이 정의된 파일입니다.
전체 학습 과정은 아래와 같으며, 각 단계마다 필요한 모듈이 정의되어 있습니다.

- 모델 초기화
- 새로운 모델에 Base Model의 가중치로부터 백본 가중치만 로드 (FC 레이어는 재정의되며, 따라서 로드하지 않습니다)
- 백본과 FC 레이어에 서로 다른 LR을 적용 (이는 백본에는 사전 학습된 가중치가 적용되었지만, FC 레이어는 재정의되어 초기화되었으므로, FC 레이어에 더 높은 LR을 적용하기 위함입니다.)
- 학습 진행
- 에폭 별 체크포인트 저장
- 학습 결과 시각화

---
📂 주요 코드 파일 설명 – 사전학습 (VCOP)

---

**models/r21d_mini.py**  
VCOP 사전학습에서 사용하는 인코더 정의 파일입니다.  
영상 시계열 데이터를 처리하기 위해 **R(2+1)D 구조를 간소화한 MiniR2Plus1D 클래스**를 포함하고 있으며,  
입력 시퀀스로부터 **공간적·시간적 특징(Spatiotemporal Features)** 을 추출하는 역할을 합니다.

---

**models/vcop_head.py**  
VCOP 태스크를 위한 **분류 헤드(classification head)** 정의 파일입니다.  
인코더의 출력 특징을 기반으로, 입력된 시퀀스의 올바른 순서를 예측하는 데 사용됩니다.

대표적으로 `VCOPN` 클래스는 다음과 같은 구조를 가집니다:

- 인코더로부터 입력을 받아 `(B, C, T, H, W)` 형태의 피처맵을 출력  
- `AdaptiveAvgPool3d(1)`을 통해 전체 시공간 차원을 평균 풀링하여 `(B, C)` 벡터로 축소  
- 완전 연결층(Fully Connected Layers)을 거쳐 시퀀스 순열 수 `factorial(tuple_len)`에 해당하는 분류 결과 출력

👉 이 구조는 시퀀스 순서 예측 정확도를 극대화하도록 설계되어 있으며,  
**인코더가 시간적 순서감(time-awareness)을 학습하도록 돕는 핵심 구성 요소**입니다.

---

**train/train_vcop.py**  
VCOP 사전학습을 수행하는 메인 학습 루틴이 정의된 파일입니다.  
`train_vcop` 함수는 다음과 같은 과정을 통해 모델을 학습합니다:

- `MiniR2Plus1D` 인코더와 `VCOPN` 분류 헤드를 연결하여 모델을 구성합니다.  
- 입력 데이터는 `DataLoader`를 통해 배치 단위로 로딩되며, 손실 함수로는 `CrossEntropyLoss`를 사용합니다.  
- 학습은 `config` 설정값에 따라 진행되며, `optimizer`, 학습률(`lr`), 에폭 수(`num_epochs`), `patience` 등을 통해 조절됩니다.
- 검증(validation) 데이터에 대한 정확도와 손실을 매 epoch 마다 출력하며, **Early Stopping 및 모델 저장** 기능도 포함되어 있습니다.

이 함수는 사전학습된 가중치를 저장하고 최종적으로 학습된 `VCOPN` 모델을 반환합니다.

---

📂 주요 코드 파일 설명 – 다운스트림 (Age Regression)

---

**models/age_regressor.py**  
Self-Supervised Learning(VCOP) 이후 학습된 인코더를 기반으로 **얼굴 나이 회귀를 수행하는 모델 구조**가 정의된 파일입니다.  
입력은 시계열 형태의 이미지 텐서 `(B, C, T, H, W)`이며, 구조는 다음과 같습니다:

- Temporal 평균 풀링을 통해 시간 차원을 축소: `(B, C, T, H, W) → (B, C, H, W)`
- ResNet 스타일의 간단한 head 구성:
  - 1x1 Convolution → ReLU → AdaptiveAvgPool2d → Flatten → Linear(128 → 1)
- 최종 출력은 단일 실수 값 `(B,)` 형태의 나이 예측 결과입니다.

👉 이 구조는 **VCOP 인코더가 내재화한 시계열적 표현**을 효과적으로 활용하여 단일 이미지 기반 나이 예측을 수행할 수 있도록 설계되었습니다.

---

**train/train_downstream.py**  
사전학습된 VCOP 인코더를 기반으로 **나이 회귀 모델을 학습하는 메인 루틴**이 포함된 파일입니다.

- `train_downstream()` 함수는 다음의 과정을 수행합니다:
  1. `MiniR2Plus1D` 인코더 + `VCOPN` 헤드 → VCOP 모델 로드 및 base encoder 추출  
  2. `AgeRegressor`에 encoder를 연결하여 전체 회귀 모델 구성  
  3. 입력 이미지에 여러 augmentation을 적용하여 `(T, C, H, W)` 시퀀스로 구성  
  4. `MSELoss` 기준으로 회귀 손실 계산  
  5. 인코더는 작은 학습률로, 회귀 head는 일반 학습률로 따로 최적화  
  6. `CosineAnnealingLR`로 learning rate 조절  
  7. `warmup_epochs` 동안 early stopping 비활성화

- 매 epoch마다 훈련 손실, 검증 손실을 출력하며, **최적 모델은 `.pth`로 저장**됩니다.

👉 이 학습 루틴은 사전학습된 시계열 표현이 **나이 회귀에 실제로 효과가 있는지 평가하는 핵심 절차**입니다.

---

📂 주요 코드 파일 설명 – evaluation 관련

---
**evaluation/evaluation_vcop.py**

SSL 모델의 **순서 맞추기 성능을 평가하는 메인 루틴**이 포함된 파일입니다.
이 파일은 vcop 모델의 성능을 평가하며, 아래와 같은 평가 지표를 제시해줍니다.
vcop model의 평가 지표로는 
Top-1 Accuracy (단일 정확도) : 모델이 가장 확률이 높은 순열(permutation)을 정확히 예측한 비율
Top-k Accuracy (상위 k개 정확도) : 모델이 예측한 상위 k개 순열 중 정답이 포함된 비율
Kendall's Tau (켄달 타우) : 두 순열 간 순위 상관관계 측정 (값 범위: -1 ~ 1, -1이면 역순, 0이면 상관관계 존재하지 않음, 순서 일치 시 1) 
Spearman's Rho (스피어만 로) : 두 순열의 순위 간 피어슨 상관계수 (값 범위: -1 ~ 1, -1이면 역순, 0이면 상관관계 존재하지 않음, 순서 일치 시 1)
Exact Match Ratio (완전 일치율) : 예측 순열이 정답과 완전히 동일한 비율
을 사용합니다.
test 데이터로는 시계열 이미지 시퀀스를 미리 분할해둔 test 데이터를 활용합니다.

---
**evaluation/evaluation_base_kor.py**

Base 모델 및 korean_finetuned 모델의 **나이 예측 성능을 평가하는 메인 루틴**이 포함된 파일입니다.
이 파일은 두 모델 각각을 평가하면서, test 데이터셋에 대한 각각의 MAE(Mean Absolute Error)값을 제시합니다.
이를 통해 두 모델 간 성능 차이를 확인할 수 있도록 해줍니다.
test 데이터로는 미리 test로 분할해둔 megaage_asian의 데이터를 활용합니다.

---
**evaluation/evaluation_downstream.py**

"SSL 모델을 downstream task를 수행할 수 있도록 변환한 모델"의 **나이 예측 성능을 평가하는 메인 루틴**이 포함된 파일입니다.
이 파일은 위 모델의 성능을 평가하며, 아래와 같은 평가 지표를 제시해줍니다.
MAE : Mean Absolute Error, 평균 절대 오차. 예측값과 실제값의 차이의 절대값의 평균를 나타냅니다.
MSE : Mean Squared Error, 평균 제곱 오차. 예측값과 실제값의 차이의 제곱의 평균을 나타냅니다.
RMSE : Root MSE, 평균 제곱근 오차. MSE의 제곱근을 나타냅니다.
test 데이터로는 evaluation_base_kor.py에서와 동일하게, 미리 test로 분할해둔 megaage_asian의 데이터를 활용합니다.

---
**evaluation/evaluation_multitrain.py**

공란

---
## ⚙️ 설치 안내

본 프로젝트는 PyTorch 기반으로 구현되었으며, 필요한 라이브러리는 `requirements.txt` 파일을 통해 설치할 수 있습니다.

### 🔧 설치 방법

```bash
pip install -r requirements.txt
```
---
## ▶️ 실행 방법 (Step-by-Step)

본 프로젝트는 목적에 따라 분리된 `.ipynb` 노트북 파일들을 실행하는 방식으로 구성되어 있습니다.  
**학습 과정을 직접 실행해보고 싶다면**, 아래의 노트북 파일 중 해당 목적에 맞는 것을 열어 순서대로 실행하시면 됩니다.
**평가를 원한다면**, 아래의 `eval-all.ipynb`을 실행하면 됩니다.

### 🧪 실험별 실행 노트북 안내

| 노트북 파일명 | 설명 |
|---------------|------|
| `ssl-preTextTask.ipynb` | VCOP 기반 Self-Supervised 순서 예측 사전학습을 진행합니다. |
| `ssl-regression.ipynb` | 사전학습된 인코더를 활용해 얼굴 나이 회귀 모델을 학습합니다. |
| `finetune_base_to_kor.ipynb` | 베이스 회귀 모델을 한국인 얼굴 데이터셋에 파인튜닝합니다. |
| `basemodel-eval.ipynb` | 사전학습 없이 학습한 기본 베이스 모델의 회귀 성능을 평가합니다. |
| `eval-all.ipynb` | 학습된 모든 모델을 불러와 성능을 **한눈에 비교 평가**할 수 있습니다. |

> 📌 `.ipynb` 파일은 Google Colab 또는 Jupyter Notebook 환경에서 실행 가능합니다.  
> 사전 설정(`config.py`, `korean_config.py`)은 각 노트북 내에서 자동 불러오거나 수정 가능합니다.
