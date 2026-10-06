"""Ablation: 제안 모델 구성요소를 하나씩 바꿔 기여를 확인.
A1 운전조건 보정, A2 feature 구성, A3 경보 규칙, A4 학습 데이터 길이."""
import argparse
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from experiments import detection_delay, evaluate
from model import NBMT2, JointT2, conformal_threshold, k_of_n
from preprocess import load_dataset, time_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default='data')
    ap.add_argument('--out-dir', default='outputs')
    ap.add_argument('--alpha', type=float, default=0.01)
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(exist_ok=True)
    pd.set_option('display.width', 200)

    dn, da, N, A = load_dataset(args.data_dir)
    tr, va, te = time_split(N)
    t0 = da.TimeStamp.min()
    keep = ['Recall', 'FAR test', 'FAR fwd', 'FAR test (2-of-3)', 'Delay s (2-of-3)', 'missed bursts']

    def table(variants, title, fname):
        T = pd.DataFrame({k: evaluate(f, N, A, tr, va, te, t0, args.alpha)[2] for k, f in variants.items()}).T[keep]
        print(f'\n[{title}]'); print(T.to_string()); T.to_csv(out / fname)

    # A1 운전조건 보정의 역할
    table({'NBM-T2 (proposed)': NBMT2,
           'T2 without conditioning (vibration only)': partial(NBMT2, conditioned=False),
           'Joint T2 (vibration + current amplitude)': JointT2},
          'A1 운전조건 보정', 'ablation_A1_conditioning.csv')

    # A2 feature 구성
    table({'all 6 (rms, p2p, kurt x 2)': NBMT2,
           'no kurtosis': partial(NBMT2, mon=['v0_rms', 'v0_p2p', 'v1_rms', 'v1_p2p']),
           'no p2p': partial(NBMT2, mon=['v0_rms', 'v0_kurt', 'v1_rms', 'v1_kurt']),
           'rms only': partial(NBMT2, mon=['v0_rms', 'v1_rms'])},
          'A2 feature 구성', 'ablation_A2_features.csv')

    # A3 경보 규칙 (같은 점수, 같은 임계값)
    m = NBMT2().fit(tr)
    th = conformal_threshold(m.score(va), args.alpha)
    pt, pa = m.score(te) > th, m.score(A) > th
    R0 = m.resid(tr) - m.mu
    lam = 0.2
    Sinv_z = np.linalg.pinv(lam / (2 - lam) * np.cov(R0, rowvar=False))

    def mewma(D):
        z, s = np.zeros(R0.shape[1]), []
        for r in m.resid(D) - m.mu:
            z = lam * r + (1 - lam) * z
            s.append(z @ Sinv_z @ z)
        return np.array(s)

    th_m = conformal_threshold(mewma(va), args.alpha)
    rules = {'point (1-of-1)': (pt, pa), '2-of-3 (proposed)': (k_of_n(pt, 2, 3), k_of_n(pa, 2, 3)),
             '3-of-5': (k_of_n(pt, 3, 5), k_of_n(pa, 3, 5)), 'MEWMA (lambda=0.2)': (mewma(te) > th_m, mewma(A) > th_m)}
    T3 = pd.DataFrame({k: {'test bursts in alarm': f'{t.sum()}/{len(te)}', 'anomaly bursts in alarm': f'{a.sum()}/{len(A)}',
                           'Delay s': detection_delay(a, A, t0)[0]} for k, (t, a) in rules.items()}).T
    print('\n[A3 경보 규칙]'); print(T3.to_string()); T3.to_csv(out / 'ablation_A3_alarm_rule.csv')

    # A4 학습 데이터 길이 (val·test 고정)
    rows = {}
    for mins in [3, 5, 10, 15, 20, 30, 46]:
        sub = tr[(tr.t_start - tr.t_start.iloc[0]).dt.total_seconds() <= mins * 60]
        mm = NBMT2().fit(sub); t4 = conformal_threshold(mm.score(va), args.alpha)
        rows[f'{mins} min ({len(sub)} bursts)'] = {'Recall': f'{(mm.score(A) > t4).sum()}/{len(A)}',
                                                   'FAR test': f'{(mm.score(te) > t4).sum()}/{len(te)}'}
    T4 = pd.DataFrame(rows).T
    print('\n[A4 학습 데이터 길이]'); print(T4.to_string()); T4.to_csv(out / 'ablation_A4_train_length.csv')


if __name__ == '__main__':
    main()
