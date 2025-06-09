# 👤 얼굴 노화 예측을 위한 Self-Supervised Spatiotemporal Learning

본 프로젝트는 얼굴 이미지 시계열 데이터를 활용하여, Self-Supervised Learning을 통해 인코더를 사전 학습하고 이후 Age Regression 성능을 향상시키는 것을 목표로 합니다.  
사전학습에는 Video Clip Order Prediction(VCOP)을 활용하며, 다운스트림에서는 얼굴 나이 예측 회귀 모델을 학습합니다.

---

## 👥 팀원

- 20233127 김민형  
- 20201463 박동민  
- 20211429 위명준  
- 20201502 정현우  

---

## 📁 프로젝트 구조

```bash
project/
├── data/                        # 학습/검증/테스트용 시퀀스 .pt 파일들
│   ├── train_X.pt, train_y.pt
│   ├── val_X.pt, val_y.pt
│   └── test_X.pt, test_y.pt
├── models/                      # 모델 정의 (MiniR2Plus1D, VCOPN, AgeRegressor 등)
│   ├── r21d_mini.py
│   ├── vcop_head.py
│   ├── age_regressor.py
├── train_vcop.py                # VCOP 사전학습용 스크립트
├── train_age.py                 # 나이 회귀 학습 스크립트
├── evaluate_vcop.py            # VCOP 모델 평가 (Top-k, Kendall's Tau 등)
├── evaluate_age.py             # Age Regression 모델 평가 (MAE, RMSE)
├── README.md                   # 이 문서
├── report.pdf                  # 제출용 보고서
└── config.py                   # 실험 설정 정의
