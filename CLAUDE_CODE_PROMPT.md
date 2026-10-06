# Claude Code 전달 프롬프트: KAMP 과제 ③ NBM-T² 재현성 검증

## 목적
이 폴더의 코드가 다른 환경에서도 **같은 숫자를 내는지** 확인한다.
코드를 개선하거나 결과를 맞추려고 로직을 바꾸는 것이 아니라, 재현 여부와 차이의 원인을 보고하는 것이 목표다.

## 폴더 구성
- `preprocess.py`: CSV 로드 → burst 분할(0.2초 초과 공백, 5샘플 미만 제외) → burst feature(진동 RMS·P2P·kurtosis ×2, 전류 진폭 = burst 내 전류 std) + 누수 감사
- `model.py`: 제안 모델 NBMT2(전류 진폭으로 진동 기대값 OLS → 잔차 Hotelling T²), split-conformal 임계값, 2-of-3 규칙, 경보 유형 분류, 비교군 8개
- `experiments.py`: 메인 결과표, 설계 α vs 실측 오경보율, 미탐지·오경보 조건 분석, 부록 AUROC
- `ablation.py`: A1 운전조건 보정 / A2 feature 구성 / A3 경보 규칙 / A4 학습 데이터 길이
- `requirements.txt`: 작성 환경 버전

## 준비
1. `data/` 폴더에 `press_data_normal.csv`, `outlier_data.csv`를 넣는다 (KAMP 소성가공 예지보전 AI 데이터셋).
2. 가상환경을 만들고 `pip install -r requirements.txt`. 설치 버전이 다르면 실제 설치된 버전을 기록한다.

## 실행 (순서대로)
```bash
python preprocess.py  --data-dir data --out-dir outputs
python experiments.py --data-dir data --out-dir outputs
python ablation.py    --data-dir data --out-dir outputs
```

## 기대값 (작성 환경: Python 3.12, 위 requirements 버전)
결정적(deterministic) 계산은 **정확히 일치**해야 한다. 아래 표시한 항목만 라이브러리 버전에 따라 미세하게 다를 수 있다.

### preprocess.py
- normal bursts 570 (train 342 / val 114 / test 114), anomaly bursts 18
- excluded short bursts: normal 29, anomaly 3
- 누수 감사: 파일 전체 전류 평균 normal 1.442 / anomaly 57.33, c_mean 단독 AUROC 0.6074,
  전류 고유값 수 normal 19934 / anomaly 340, 양자화 AUROC 1, 정상 alias 주파수 중앙값 0.5957 Hz, alias 편차 AUROC 0.9171

### experiments.py [1] 메인 결과 (α = 0.01)
| 모델 | Recall | FAR test | FAR fwd | FAR test (2-of-3) | Delay s | missed |
|---|---|---|---|---|---|---|
| NBM-T2 (proposed) | 16/18 | 1/114 | 2/342 | 0/114 | 20.3 | [19, 20] |
| Joint T2 (MSPC) | 16/18 | 1/114 | 2/342 | 0/114 | 20.3 | [19, 20] |
| PCA-SPE (MSPC) | 6/18 | 1/114 | 1/342 | 0/114 | 95.1 | 12개 |
| kNN | 17/18 | 6/114 | 6/342 | 4/114 | 20.3 | [19] |
| IForest * | 17/18 | 5/114 | 10/342 | 2/114 | 20.3 | [19] |
| OCSVM * | 17/18 | 5/114 | 5/342 | 2/114 | 20.3 | [19] |
| Feature-AE * | 15/18 | 1/114 | 2/342 | 0/114 | 20.3 | [7, 19, 20] |
| RMS one-liner | 14/18 | 0/114 | 1/342 | 0/114 | 20.3 | [3, 14, 19, 20] |
| Random | 0/18 | 0/114 | 1/342 | 0/114 | no alarm | 전부 |

\* 난수·최적화 의존 모델: 버전에 따라 ±1 burst 차이 허용, 차이가 나면 보고.

### experiments.py [2] 설계 α vs 실측 FAR (제안 모델)
| α | FAR test | FAR fwd | Recall |
|---|---|---|---|
| 0.05 | 0.0351 | 0.0380 | 17/18 |
| 0.02 | 0.0088 | 0.0058 | 16/18 |
| 0.01 | 0.0088 | 0.0058 | 16/18 |

### experiments.py [3] 조건 분석
- T2 threshold = 53.6
- burst 19: T2/threshold 0.28, triage "정상" / burst 20: 운전조건 항 12.5 (> 6.63), triage "판단 보류"
- test 오경보 burst 492: 원인 v1_kurt +7.3
- 부록 AUROC: proposed 0.992, Joint T2 0.994, Random 0.471

### ablation.py
- A1: proposed 16/18·1/114·2/342 / 보정 없음 16/18·2/114·3/342 / Joint T2 16/18·1/114·2/342
- A2: no kurtosis 14/18·0/114·1/342 / no p2p 16/18·1/114·6/342 / rms only 12/18·0/114·4/342
- A3 (test 경보 burst, 이상 경보 burst, delay s): point 1/114·16/18·13.7 / 2-of-3 0/114·16/18·20.3 / 3-of-5 0/114·16/18·26.8 / MEWMA 64/114·17/18·20.3
- A4 Recall·FAR test: 3분 12/18·1/114, 5분 12/18·0/114, 10분 15/18·0/114, 15분 이후 16/18·1/114

## 추가 확인 (각각 결과를 보고)
1. **결정성**: 세 스크립트를 두 번 실행해서 출력이 완전히 같은지 비교한다.
2. **CSV 일치**: `outputs/`의 CSV가 화면 출력과 같은 숫자인지 확인한다.
3. **누수 가드**: 모델 입력에 `c_mean`이 쓰이지 않는지 코드에서 확인한다 (누수 감사에만 쓰여야 함).
4. **임계값 규칙**: 모든 임계값이 정상 val(또는 forward-chaining의 보정 블록)에서만 계산되고, 이상 데이터가 임계값 결정에 쓰이지 않는지 확인한다.

## 보고 형식
- 실행 환경 (OS, Python, 실제 설치된 패키지 버전)
- 항목별 일치 / 불일치 표 (불일치는 기대값과 실제값을 나란히)
- 불일치가 있으면 원인 추정 (버전 차이, 난수, 정렬 순서 등). **로직을 바꿔서 숫자를 맞추지 말 것.**
- 버그로 보이는 부분은 수정하지 말고 위치와 이유만 보고
