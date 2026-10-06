"""실험: 제안 모델과 비교군을 같은 조건에서 평가 (Recall / False alarm rate / Detection delay).
평가 규칙: 정상만으로 학습, 임계값은 정상 val로 정한 split-conformal 값, 이상 데이터는 평가에만 사용."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from model import BASELINES, NBMT2, clopper_pearson, conformal_threshold, k_of_n, triage
from preprocess import VIB, load_dataset, time_split

N_BLOCKS = 5          # forward-chaining: 정상 데이터를 시간순 5블록으로 나눔
ALARM_K, ALARM_N = 2, 3


def forward_chaining_far(make, N, alpha):
    """fold k: 앞 k+1블록 학습 → 다음 블록으로 임계값 → 그다음 블록에서 오경보 집계 (3 folds)."""
    blocks = np.array_split(np.arange(len(N)), N_BLOCKS)
    fa = total = 0
    for k in range(N_BLOCKS - 2):
        m = make().fit(N.iloc[np.concatenate(blocks[:k + 1])])
        th = conformal_threshold(m.score(N.iloc[blocks[k + 1]]), alpha)
        f = m.score(N.iloc[blocks[k + 2]]) > th
        fa += f.sum(); total += len(f)
    return fa, total


def detection_delay(alarm_flags, A, event_start):
    """이상 파일 첫 샘플 시각부터 첫 경보 burst가 끝난 시각까지(초)."""
    if not alarm_flags.any():
        return np.nan, None
    i = int(np.argmax(alarm_flags))
    return (A.t_end.iloc[i] - event_start).total_seconds(), i + 1


def evaluate(make, N, A, tr, va, te, event_start, alpha):
    m = make().fit(tr)
    th = conformal_threshold(m.score(va), alpha)
    st, sa = m.score(te), m.score(A)
    det, fa = sa > th, st > th
    fa_fwd, n_fwd = forward_chaining_far(make, N, alpha)
    delay, nth = detection_delay(k_of_n(det, ALARM_K, ALARM_N), A, event_start)
    r_lo, r_hi = clopper_pearson(det.sum(), len(A))
    f_lo, f_hi = clopper_pearson(fa.sum() + fa_fwd, len(te) + n_fwd)
    return m, th, {
        'Recall': f'{det.sum()}/{len(A)} [{r_lo:.2f}, {r_hi:.2f}]',
        'FAR test': f'{fa.sum()}/{len(te)}',
        'FAR fwd': f'{fa_fwd}/{n_fwd}',
        'FAR pooled [95% CI]': f'{(fa.sum() + fa_fwd) / (len(te) + n_fwd):.3f} [{f_lo:.3f}, {f_hi:.3f}]',
        'FAR test (2-of-3)': f'{k_of_n(fa, ALARM_K, ALARM_N).sum()}/{len(te)}',
        'Delay s (2-of-3)': round(delay, 1) if delay == delay else 'no alarm',
        'Delay burst #': nth,
        'missed bursts': list(A.b[~det]),
        '_auroc': roc_auc_score(np.r_[np.zeros(len(st)), np.ones(len(sa))], np.r_[st, sa]),
    }


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
    event_start = da.TimeStamp.min()

    # 1) 메인 결과표: 제안 모델 + 비교군
    rows = {}
    m0, th0, rows['NBM-T2 (proposed)'] = evaluate(NBMT2, N, A, tr, va, te, event_start, args.alpha)
    for name, cls in BASELINES.items():
        rows[name] = evaluate(cls, N, A, tr, va, te, event_start, args.alpha)[2]
    T = pd.DataFrame(rows).T
    print(f'\n[1] 메인 결과 (alpha = {args.alpha}, 임계값 = 정상 val split-conformal)')
    print(T.drop(columns='_auroc').to_string())
    T.drop(columns='_auroc').to_csv(out / 'main_results.csv')

    # 2) 설계 alpha vs 실측 FAR (제안 모델)
    a_rows = []
    for a in [0.05, 0.02, 0.01]:
        th = conformal_threshold(m0.score(va), a)
        fa_fwd, n_fwd = forward_chaining_far(NBMT2, N, a)
        a_rows.append({'design alpha': a, 'FAR test': round((m0.score(te) > th).mean(), 4),
                       'FAR fwd': round(fa_fwd / n_fwd, 4), 'Recall': f'{(m0.score(A) > th).sum()}/{len(A)}'})
    print('\n[2] 설계 alpha vs 실측 False alarm rate (제안 모델)')
    print(pd.DataFrame(a_rows).to_string(index=False))
    pd.DataFrame(a_rows).to_csv(out / 'alpha_calibration.csv', index=False)

    # 3) 미탐지·오경보 조건 분석: 이상 burst 전체 + test 오경보 burst
    def detail(D, kind):
        Z = m0.contributions(D)
        top = [', '.join(f'{VIB[j]} {z[j]:+.1f}' for j in np.argsort(-np.abs(z))[:2]) for z in Z]
        return pd.DataFrame({'kind': kind, 'burst': D.b.values, 'n': D.n.values,
                             'log_c_std': np.log(D.c_std.values).round(2),
                             'T2 / threshold': (m0.score(D) / th0).round(2),
                             'condition term': m0.condition_term(D).round(1),
                             'triage': triage(m0, D, th0), 'top features (z)': top})
    E = pd.concat([detail(A, 'anomaly'), detail(te[m0.score(te) > th0], 'test false alarm')])
    print(f'\n[3] 미탐지·오경보 조건 분석 (T2 threshold = {th0:.1f}, 운전조건 항 기준 = chi2_1 99% = 6.63)')
    print(E.to_string(index=False))
    E.to_csv(out / 'error_analysis.csv', index=False)

    # 부록: AUROC (본문 지표 아님)
    print('\n[부록] AUROC:', {k: round(v['_auroc'], 3) for k, v in rows.items()})


if __name__ == '__main__':
    main()
