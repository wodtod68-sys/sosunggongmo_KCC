# sosunggongmo_KCC

NBM을 이용한 조기탐지 모델 

대회 제출용 GMM 모델은 https://github.com/joonookwak/KAMP_ai 참고

# Press Machine Anomaly Detection with NBM-T²

프레스 설비의 진동·전류 신호로 **고장 징후를 조기 탐지**하는 모델입니다.
운전조건(전류 진폭)에 따라 달라지는 진동의 "정상 기대값"을 Normal Behavior Model(NBM)로 먼저 걸러낸 뒤,
남은 잔차에 Hotelling T²를 적용해 이상을 판정합니다.

> 대회 제출용 모델은 [joonookwak/KAMP_ai](https://github.com/joonookwak/KAMP_ai)를 참고하세요.
> 이 레포는 NBM 기반 조기탐지 모델과 그 검증 실험을 정리한 것입니다.

---

## 데이터

[KAMP](https://www.kamp-ai.kr/) 소성가공 예지보전 AI 데이터셋을 사용합니다.

| 파일 | 내용 |
|---|---|
| `press_data_normal.csv` | 정상 운전 데이터 |
| `outlier_data.csv` | 이상 발생 구간 데이터 |

각 샘플은 `TimeStamp`, 상부 진동 `AI0_Vibration`, 하부 진동 `AI1_Vibration`, 전류 `AI2_Current`로 구성됩니다.
데이터는 레포에 포함되어 있지 않으므로 `data/` 폴더에 직접 넣어야 합니다.

---

## 방법

### 1. Burst 단위 feature 추출

샘플 간격이 0.2초를 넘으면 새 burst로 분할하고, 5샘플 미만 burst는 통계량이 불안정하므로 제외합니다.
burst마다 다음 feature를 계산합니다.

| Feature | 정의 |
|---|---|
| `v0_rms`, `v1_rms` | 상·하부 진동 RMS |
| `v0_p2p`, `v1_p2p` | 상·하부 진동 peak-to-peak |
| `v0_kurt`, `v1_kurt` | 상·하부 진동 kurtosis (Fisher, 정규분포 = 0) |
| `c_std` | 전류 진폭 = burst 내 전류 표준편차 (운전조건 변수) |

양수·오른쪽 꼬리 분포인 RMS, P2P, `c_std`는 log 변환합니다.

### 2. Leakage audit

정상 파일과 이상 파일은 고장과 무관한 **수집 방식 차이**로도 구분될 수 있습니다. 모델이 고장이 아니라 "어느 파일에서 왔는지"를 학습하는 것을 막기 위해 세 가지를 수치로 점검합니다.

| 항목 | 정상 | 이상 | burst 단위 AUROC |
|---|---|---|---|
| (1) DC offset — 파일 전체 전류 평균 | 1.442 | 57.33 | 0.607 (`c_mean` 단독) |
| (2) 양자화 — 전류 고유값 수 | 19,934 | 340 | 1.000 (1.19209 배수 비율) |
| (3) 샘플링 타이밍 — 전류 alias 주파수 편차 | 중앙값 0.596 Hz | — | 0.917 |

이 차이들이 지름길 feature가 되지 않도록 전류의 **평균(`c_mean`)은 모델 입력에서 제외**하고, 진폭(`c_std`)만 운전조건 변수로 사용합니다.

### 3. NBM-T² (제안 모델)

1. **Normal Behavior Model:** 정상 train 데이터로 `log c_std`에서 진동 feature 6개의 기대값을 OLS 회귀합니다.
2. **이상 점수:** 회귀 잔차 $r$의 Hotelling T²

$$T^2 = (r - \bar r)^\top S^{-1} (r - \bar r)$$

   즉 "이 전류 진폭에서 기대되는 진동"과 얼마나 다른지를 측정합니다.
3. **임계값 (split-conformal):** 정상 val 점수 $n$개에서 $\lceil (n+1)(1-\alpha) \rceil$번째 값을 임계값으로 씁니다. 정상 데이터가 exchangeable하면 burst당 오경보 확률이 $\alpha$ 이하로 보장됩니다.
4. **경보 규칙 (2-of-3):** 최근 3개 burst 중 2개 이상이 임계값을 넘으면 경보를 냅니다.
5. **Triage:** 운전조건 항 $(\log c_{std} - \mu)^2 / \sigma^2$가 $\chi^2_1$ 99% 분위수(6.63)를 넘으면, 학습 때 보지 못한 운전조건이라 판단을 유보합니다.

| 판정 | 조건 |
|---|---|
| 정비 경보 | T² > 임계값 |
| 판단 보류 | T² ≤ 임계값이지만 운전조건 항 > 6.63 |
| 정상 | 그 외 |

경보가 나면 feature별 표준화 잔차로 **어느 진동 feature가 이상인지**도 함께 제시합니다.

### 4. 평가 프로토콜

- **정상 데이터만으로 학습**, 이상 데이터는 평가에만 사용
- 정상 burst를 시간순으로 train 60% / val 20% / test 20% 분할
- **모든 임계값은 정상 val(또는 forward-chaining의 보정 블록)에서만 결정**
- **FAR fwd:** 정상 데이터를 시간순 5블록으로 나눠 앞 블록 학습 → 다음 블록으로 임계값 → 그다음 블록에서 오경보 집계 (3 folds)
- **Detection delay:** 이상 파일 첫 샘플 시각부터 첫 경보(2-of-3) burst 종료 시각까지
- Recall·FAR에 Clopper-Pearson 95% 신뢰구간을 함께 보고
- AUROC는 부록 지표로만 사용 (실제 운영은 임계값 기반이므로)

---

## 결과

정상 burst 570개(train 342 / val 114 / test 114), 이상 burst 18개.

### 메인 결과 (α = 0.01)

| 모델 | Recall | FAR test | FAR fwd | FAR test (2-of-3) | Delay (s) |
|---|---|---|---|---|---|
| **NBM-T² (proposed)** | **16/18** | **1/114** | **2/342** | **0/114** | **20.3** |
| Joint T² (MSPC) | 16/18 | 1/114 | 2/342 | 0/114 | 20.3 |
| PCA-SPE (MSPC) | 6/18 | 1/114 | 1/342 | 0/114 | 95.1 |
| kNN | 17/18 | 6/114 | 6/342 | 4/114 | 20.3 |
| Isolation Forest | 17/18 | 5/114 | 10/342 | 2/114 | 20.3 |
| One-Class SVM | 17/18 | 5/114 | 5/342 | 2/114 | 20.3 |
| Feature-AE | 15/18 | 1/114 | 2/342 | 0/114 | 20.3 |
| RMS one-liner | 14/18 | 0/114 | 1/342 | 0/114 | 20.3 |
| Random | 0/18 | 0/114 | 1/342 | 0/114 | no alarm |

- 2-of-3 규칙 적용 후 test 오경보 **0건**, 첫 경보까지 **20.3초**
- kNN·IForest·OCSVM은 Recall이 1 burst 높지만 오경보가 5~10배 많습니다.
- Joint T²(운전조건 보정 없이 전류 진폭을 feature로 함께 넣은 T²)와 수치는 같지만, NBM-T²는 운전조건 항을 분리해 **"판단 보류" triage와 feature별 원인 제시**가 가능합니다.
- 부록 AUROC: NBM-T² 0.992, Joint T² 0.994, Random 0.471

### 설계 α vs 실측 오경보율

| 설계 α | FAR test | FAR fwd | Recall |
|---|---|---|---|
| 0.05 | 0.0351 | 0.0380 | 17/18 |
| 0.02 | 0.0088 | 0.0058 | 16/18 |
| 0.01 | 0.0088 | 0.0058 | 16/18 |

실측 오경보율이 모두 설계 α 이하로, conformal 보장이 실제로 지켜집니다.

### 미탐지 burst 분석

- **burst 19:** T²/임계값 = 0.28 → 진동 이상 신호가 실제로 약함 (정상)
- **burst 20:** 운전조건 항 12.5 > 6.63 → 미경험 운전조건으로 **판단 보류** 표시
- **test 오경보 burst 492:** 원인 feature `v1_kurt` (+7.3σ)

### Ablation

| 실험 | 변형 | Recall | FAR test | FAR fwd |
|---|---|---|---|---|
| A1 운전조건 보정 | **제안 (보정 O)** | 16/18 | **1/114** | **2/342** |
| | 보정 없음 (진동만) | 16/18 | 2/114 | 3/342 |
| A2 feature 구성 | **전체 6개** | **16/18** | 1/114 | 2/342 |
| | kurtosis 제외 | 14/18 | 0/114 | 1/342 |
| | P2P 제외 | 16/18 | 1/114 | 6/342 |
| | RMS만 | 12/18 | 0/114 | 4/342 |

| A3 경보 규칙 | test 경보 burst | 이상 경보 burst | Delay (s) |
|---|---|---|---|
| point (1-of-1) | 1/114 | 16/18 | 13.7 |
| **2-of-3 (proposed)** | **0/114** | **16/18** | **20.3** |
| 3-of-5 | 0/114 | 16/18 | 26.8 |
| MEWMA (λ = 0.2) | 64/114 | 17/18 | 20.3 |

**A4 학습 데이터 길이:** 3분 12/18 · 5분 12/18 · 10분 15/18 · 15분 이후 16/18 (FAR test 0~1/114). 약 15분 분량의 정상 데이터면 성능이 수렴합니다.

---

## 레포 구조

```
.
├── preprocess.py         # CSV 로드 → burst 분할 → feature 추출 + leakage audit
├── model.py              # NBM-T², split-conformal 임계값, k-of-n 경보, triage, 비교군 8개
├── experiments.py        # 메인 결과, 설계 α vs 실측 FAR, 미탐지·오경보 분석, AUROC
├── ablation.py           # A1 운전조건 보정 / A2 feature / A3 경보 규칙 / A4 학습 길이
├── requirements.txt
└── CLAUDE_CODE_PROMPT.md # 재현성 검증용 체크리스트 (기대값 포함)
```

---

## 실행 방법

### 환경

Python 3.12에서 검증했습니다.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 데이터 배치

```
data/
├── press_data_normal.csv
└── outlier_data.csv
```

### 실행 (순서대로)

```bash
python preprocess.py  --data-dir data --out-dir outputs
python experiments.py --data-dir data --out-dir outputs   # --alpha 로 설계 α 변경 가능 (기본 0.01)
python ablation.py    --data-dir data --out-dir outputs
```

### 산출물 (`outputs/`)

| 파일 | 내용 |
|---|---|
| `features_normal.csv`, `features_anomaly.csv` | burst feature (정상은 split 라벨 포함) |
| `main_results.csv` | 메인 결과표 |
| `alpha_calibration.csv` | 설계 α vs 실측 FAR |
| `error_analysis.csv` | 이상 burst 및 test 오경보 burst의 T², 운전조건 항, triage, 원인 feature |
| `ablation_A1~A4_*.csv` | ablation 결과 |

### 재현성

결정적(deterministic) 계산은 위 결과와 정확히 일치해야 합니다. 난수·최적화에 의존하는 IForest, OCSVM, Feature-AE는 라이브러리 버전에 따라 ±1 burst 차이가 날 수 있습니다. 재현 검증 체크리스트는 [`CLAUDE_CODE_PROMPT.md`](CLAUDE_CODE_PROMPT.md)를 참고하세요.

---

## 한계

- 이상 데이터가 단일 이상 구간의 18개 burst뿐이라 Recall 추정의 신뢰구간이 넓습니다.
- Detection delay는 이상 파일의 첫 샘플을 이상 시작 시점으로 가정한 값입니다.
- Leakage audit에서 확인한 수집 방식 차이 중 DC offset은 `c_mean` 제외로 막았지만, 운전조건 변수 `c_std`도 같은 전류 신호에서 계산되므로 양자화·샘플링 타이밍 차이의 영향을 완전히 배제하지는 못합니다.
