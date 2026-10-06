"""모델: 제안 모델 NBM-T2(운전조건 보정 + Hotelling T2 + 2-of-3 경보)와 비교군."""
import math

import numpy as np
from scipy.stats import beta, chi2
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from preprocess import VIB, design_matrix


def t2(Z, Sinv):
    return np.einsum('ij,jk,ik->i', Z, Sinv, Z)


# ----------------------------------------------------------------------------- 제안 모델
class NBMT2:
    """NBM: 전류 진폭(log c_std)으로 진동 feature의 기대값을 OLS 회귀.
    점수: 잔차의 Hotelling T2(조건부 항). 운전조건 항은 경보가 아니라 '판단 보류' 표시에 사용."""

    def __init__(self, mon=VIB, conditioned=True):
        self.mon, self.conditioned = mon, conditioned

    def fit(self, TR):
        L = np.log(TR[['c_std']]).values
        X = design_matrix(TR, self.mon)
        if self.conditioned:
            self.reg = LinearRegression().fit(L, X)
        R = self.resid(TR)
        self.mu = R.mean(0)
        self.sd = R.std(0)
        self.Sinv = np.linalg.pinv(np.atleast_2d(np.cov(R, rowvar=False)))
        self.l_mu, self.l_var = L.mean(), L.var()
        return self

    def resid(self, D):
        X = design_matrix(D, self.mon)
        return X - self.reg.predict(np.log(D[['c_std']]).values) if self.conditioned else X

    def score(self, D):                       # 조건부 항 = 전류 진폭 대비 진동 이상 정도
        return t2(self.resid(D) - self.mu, self.Sinv)

    def condition_term(self, D):              # 운전조건 항 = 학습 때 본 전류 진폭에서 얼마나 벗어났나
        return (np.log(D.c_std.values) - self.l_mu) ** 2 / self.l_var

    def contributions(self, D):               # feature별 표준화 잔차(어느 진동 feature가 이상인지)
        return (self.resid(D) - self.mu) / self.sd


def triage(model, D, th, alpha_cond=0.01):
    """경보 유형 분류: 정비 경보 / 판단 보류(미경험 운전조건) / 정상."""
    alarm = model.score(D) > th
    unseen = model.condition_term(D) > chi2.ppf(1 - alpha_cond, df=1)
    return np.where(alarm, '정비 경보', np.where(unseen, '판단 보류', '정상'))


# ----------------------------------------------------------------------------- 공통 도구
def conformal_threshold(cal_scores, alpha):
    """split-conformal 임계값: 정상 데이터가 교환 가능하면 burst당 오경보 확률 <= alpha."""
    n = len(cal_scores)
    k = min(math.ceil((n + 1) * (1 - alpha)), n)
    return np.sort(cal_scores)[k - 1]


def k_of_n(flags, k=2, n=3):
    flags = np.asarray(flags, dtype=int)
    return np.array([flags[max(0, i - n + 1):i + 1].sum() >= k for i in range(len(flags))])


def clopper_pearson(k, n, conf=.95):
    a = (1 - conf) / 2
    return (beta.ppf(a, k, n - k + 1) if k > 0 else 0.0, beta.ppf(1 - a, k + 1, n - k) if k < n else 1.0)


# ----------------------------------------------------------------------------- 비교군 (입력: 진동 6 + log 전류 진폭)
FEATS = VIB + ['c_std']


class _Scaled:
    def _x(self, D, fit=False):
        X = design_matrix(D, FEATS)
        if fit:
            self.sc = StandardScaler().fit(X)
        return self.sc.transform(X)


class JointT2(_Scaled):               # MSPC: 운전조건 보정 없이 전체 feature에 T2
    def fit(self, TR):
        Z = self._x(TR, fit=True); self.mu = Z.mean(0); self.Sinv = np.linalg.pinv(np.cov(Z, rowvar=False)); return self
    def score(self, D):
        return t2(self._x(D) - self.mu, self.Sinv)


class PCASPE(_Scaled):                # MSPC: PCA 재구성 오차(SPE, Q 통계량)
    def fit(self, TR):
        self.pca = PCA(n_components=0.9, random_state=0).fit(self._x(TR, fit=True)); return self
    def score(self, D):
        Z = self._x(D); return ((Z - self.pca.inverse_transform(self.pca.transform(Z))) ** 2).sum(1)


class KNN(_Scaled):                   # 근접: train 정상 k개 이웃까지 평균 거리
    def fit(self, TR, k=5):
        self.nn = NearestNeighbors(n_neighbors=k).fit(self._x(TR, fit=True)); return self
    def score(self, D):
        return self.nn.kneighbors(self._x(D))[0].mean(1)


class IForest(_Scaled):               # 분리
    def fit(self, TR):
        self.m = IsolationForest(n_estimators=300, random_state=0).fit(self._x(TR, fit=True)); return self
    def score(self, D):
        return -self.m.score_samples(self._x(D))


class OCSVM(_Scaled):                 # 경계
    def fit(self, TR):
        self.m = OneClassSVM(nu=0.05, gamma='scale').fit(self._x(TR, fit=True)); return self
    def score(self, D):
        return -self.m.decision_function(self._x(D))


class FeatureAE(_Scaled):             # 재구성 (Serradilla et al. 계열)
    def fit(self, TR, seed=0):
        Z = self._x(TR, fit=True)
        self.m = MLPRegressor(hidden_layer_sizes=(16, 3, 16), max_iter=3000, random_state=seed).fit(Z, Z); return self
    def score(self, D):
        Z = self._x(D); return ((Z - self.m.predict(Z)) ** 2).mean(1)


class RMSOneLiner:                    # sanity check: 상·하부 진동 RMS 합(log)
    def fit(self, TR):
        return self
    def score(self, D):
        return np.log(D.v0_rms.values) + np.log(D.v1_rms.values)


class RandomScore:                    # sanity check: burst 번호로 시드를 고정한 난수
    def fit(self, TR):
        return self
    def score(self, D):
        return np.array([np.random.default_rng(10_000 + int(b)).random() for b in D.b])


BASELINES = {'Joint T2 (MSPC)': JointT2, 'PCA-SPE (MSPC)': PCASPE, 'kNN': KNN, 'IForest': IForest,
             'OCSVM': OCSVM, 'Feature-AE': FeatureAE, 'RMS one-liner': RMSOneLiner, 'Random': RandomScore}
