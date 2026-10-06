"""전처리: 원시 CSV → burst 분할 → burst 단위 feature, 그리고 누수(leakage) 감사."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis
from sklearn.metrics import roc_auc_score

GAP_S = 0.2          # 샘플 간격이 이 값을 넘으면 새 burst
MIN_LEN = 5          # 샘플 수가 이보다 적은 burst는 통계량이 불안정하므로 제외
VIB = ['v0_rms', 'v0_p2p', 'v0_kurt', 'v1_rms', 'v1_p2p', 'v1_kurt']   # 0=상부(AI0), 1=하부(AI1)
LOG_COLS = {'v0_rms', 'v0_p2p', 'v1_rms', 'v1_p2p', 'c_std'}           # 양수·오른쪽 꼬리 → log 변환
NORMAL_FILE, ANOMALY_FILE = 'press_data_normal.csv', 'outlier_data.csv'


def load_raw(path):
    d = pd.read_csv(path, index_col=0).drop_duplicates()
    d['TimeStamp'] = pd.to_datetime(d['TimeStamp'])
    d['b'] = (d.TimeStamp.diff().dt.total_seconds() > GAP_S).cumsum()   # burst 번호
    return d


def burst_features(d):
    rows = []
    for b, g in d.groupby('b'):
        if len(g) < MIN_LEN:
            continue
        r = {'b': b, 't_start': g.TimeStamp.iloc[0], 't_end': g.TimeStamp.iloc[-1], 'n': len(g)}
        for col, k in [('AI0_Vibration', 'v0'), ('AI1_Vibration', 'v1')]:
            x = g[col].values
            r[k + '_rms'] = np.sqrt(np.mean(x ** 2))
            r[k + '_p2p'] = np.ptp(x)
            r[k + '_kurt'] = kurtosis(x)                  # Fisher 정의(정규분포 = 0)
        r['c_std'] = g.AI2_Current.std()                  # 전류 진폭 = burst 내 전류 표준편차
        r['c_mean'] = g.AI2_Current.mean()                # 누수 감사용으로만 계산, 모델 입력 아님
        rows.append(r)
    return pd.DataFrame(rows)


def design_matrix(D, cols):
    return np.column_stack([np.log(D[c]) if c in LOG_COLS else D[c] for c in cols])


def load_dataset(data_dir):
    data_dir = Path(data_dir)
    dn, da = load_raw(data_dir / NORMAL_FILE), load_raw(data_dir / ANOMALY_FILE)
    return dn, da, burst_features(dn), burst_features(da)


def time_split(N, ratios=(.6, .2, .2)):
    n = len(N)
    a, b = int(ratios[0] * n), int((ratios[0] + ratios[1]) * n)
    return N.iloc[:a], N.iloc[a:b], N.iloc[b:]


def _peak_freq(x, fs=10.0, pad=1024):
    x = x - x.mean()
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x)), pad))
    return np.fft.rfftfreq(pad, 1 / fs)[np.argmax(spec[1:]) + 1]


def leakage_audit(dn, da, N, A):
    """정상 파일과 이상 파일을 고장과 무관하게 가르는 수집 방식 차이 3가지를 수치로 확인.
    각 항목의 burst 단위 AUROC가 1에 가까울수록 '파일 구분자'로 쓰일 위험이 큼."""
    y = np.r_[np.zeros(len(N)), np.ones(len(A))]
    per_burst = lambda d, B, f: np.array([f(d.loc[d.b == b, 'AI2_Current'].values) for b in B.b])
    quant = lambda x: np.mean(np.abs(x / 1.19209 - np.round(x / 1.19209)) < 1e-3)
    f_ref = np.median(per_burst(dn, N, _peak_freq))
    alias_dev = lambda x: abs(_peak_freq(x) - f_ref)
    return {
        '(1) DC offset: 파일 전체 전류 평균 normal': dn.AI2_Current.mean(),
        '(1) DC offset: 파일 전체 전류 평균 anomaly': da.AI2_Current.mean(),
        '(1) DC offset: burst 평균(c_mean) 단독 AUROC': roc_auc_score(y, np.r_[N.c_mean, A.c_mean]),
        '(2) 양자화: 전류 고유값 수 normal': dn.AI2_Current.nunique(),
        '(2) 양자화: 전류 고유값 수 anomaly': da.AI2_Current.nunique(),
        '(2) 양자화: 1.19209 배수 비율(burst 단위) AUROC': roc_auc_score(y, np.r_[per_burst(dn, N, quant), per_burst(da, A, quant)]),
        '(3) 샘플링 타이밍: 정상 전류 alias 주파수 중앙값(Hz)': f_ref,
        '(3) 샘플링 타이밍: alias 주파수 편차 AUROC': roc_auc_score(y, np.r_[per_burst(dn, N, alias_dev), per_burst(da, A, alias_dev)]),
    }


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default='data')
    ap.add_argument('--out-dir', default='outputs')
    args = ap.parse_args()
    Path(args.out_dir).mkdir(exist_ok=True)
    dn, da, N, A = load_dataset(args.data_dir)
    tr, va, te = time_split(N)
    print(f'normal bursts {len(N)} (train {len(tr)} / val {len(va)} / test {len(te)}), anomaly bursts {len(A)}')
    print(f'excluded short bursts: normal {dn.b.nunique() - len(N)}, anomaly {da.b.nunique() - len(A)}')
    print('\n[leakage audit]')
    for k, v in leakage_audit(dn, da, N, A).items():
        print(f'  {k}: {v:.4g}' if isinstance(v, float) else f'  {k}: {v}')
    N.assign(split=['train'] * len(tr) + ['val'] * len(va) + ['test'] * len(te)).to_csv(Path(args.out_dir) / 'features_normal.csv', index=False)
    A.to_csv(Path(args.out_dir) / 'features_anomaly.csv', index=False)
