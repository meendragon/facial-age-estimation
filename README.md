## 프로젝트 개요

본 프로젝트는 얼굴 이미지의 **시계열 데이터를 활용**하여, Self-Supervised Learning 기반 인코더를 사전 학습(pretraining)하고, 이를 통해 **얼굴 나이 예측(Age Regression)** 성능을 향상시키는 것을 목표로 합니다.

사전학습 단계에서는 Video Clip Order Prediction (VCOP) 아이디어를 바탕으로, 동일 인물의 다양한 연령대 사진을 올바른 순서로 배열하는 **순서 예측 태스크**를 수행하며, 시계열적인 노화 정보를 인코더가 학습하도록 합니다.  
이후 다운스트림에서는 학습된 인코더를 활용하여 단일 이미지 기반 나이 회귀 모델을 학습합니다.

---

## 비교 실험 조건

본 프로젝트에서는 다음 네 가지 실험 조건을 통해 모델 성능을 비교합니다:

1. **Baseline Age Regression Model**  
   사전학습 없이 일반 얼굴 이미지로 학습한 기본 나이 예측 모델

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
│   ├── megaage_asian/             # MegaAge-Asian 데이터셋
│   └── korean_image#1/            # 한국인 얼굴 이미지 (정제 및 전처리 완료본)
│
├── src/                             # 소스 코드 디렉토리
│   ├── train/                      # 학습 스크립트
│   │   ├── __init__.py
│   │   ├── functions.py
│   │   ├── train_base_to_kor.py           # ImageNet → Korean 데이터 파인튜닝
│   │   ├── train_downstream.py            # Pretrained 인코더 기반 Age Regression
│   │   ├── train_korean_finetuned.py      # 한국 이미지 전용 모델 파인튜닝
│   │   ├── train_multitrain.py            # Multi-task 학습 (VCOP + Age Regression)
│   │   └── train_vcop.py                  # VCOP 사전학습 (순서 예측)
│
│   ├── models/                    # 모델 정의
│   │   ├── __init__.py
│   │   ├── age_regressor.py
│   │   ├── base_model.py
│   │   ├── r21d_mini.py
│   │   └── vcop_head.py
│
│   ├── weights/                   # 학습된 모델 가중치 파일
│   │   ├── age_regression.pth
│   │   ├── legacy_weight.pt
│   │   ├── pretrained_weight_korean.pt
│   │   └── vcop_mini.pth
│
│   ├── loader/                    # 데이터 로딩 및 전처리
│   │   ├── custom_dataset_dataloader_korean.py    # 커스텀 한국 이미지 로더
│   │   └── custom_dataset_datasplitter_korean.py  # 학습/검증/테스트 분할 모듈
│
│   └── evaluation/                # 성능 평가 스크립트
│       ├── __init__.py
│       ├── evaluate_base.py
│       ├── evaluate_downstream.py
│       ├── evaluate_korean_finetuned.py
│       ├── evaluate_multitrain.py
│       └── evaluate_vcop.py
│
└── README.md                       # 이 문서
