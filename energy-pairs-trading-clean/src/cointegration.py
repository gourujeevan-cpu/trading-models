"""
Cointegration toolkit.

The original ExxonMobil-vs-crude notebook this project upgrades regressed one
price level on another and read off R-squared. That is a spurious-regression
trap: two independent random walks will show a high R-squared and a significant
t-stat purely because both trend. The correct sequence is:

1. Confirm each series is non-stationary (I(1)) with an ADF test. If they are
   already stationary, ordinary regression is fine and cointegration is moot.
2. Test whether a linear combination of them is stationary. Two routes:
   - Engle-Granger: regress y on x, ADF-test the residual.
   - Johansen: a system test that also handles the cointegrating rank and does
     not depend on which variable is the regressand.
3. Only if they cointegrate does the spread mean-revert and a pairs trade make
   sense. The mean-reversion half-life then tells you the horizon to trade on.

Everything here works in logs, never raw levels.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen


@dataclass
class StationarityResult:
    statistic: float
    pvalue: float
    is_stationary: bool          # at 5%


def adf_test(series: pd.Series, alpha: float = 0.05) -> StationarityResult:
    """Augmented Dickey-Fuller. Null = unit root (non-stationary)."""
    s = series.dropna()
    stat, pval, *_ = adfuller(s, autolag="AIC")
    return StationarityResult(stat, pval, pval < alpha)


def estimate_hedge_ratio(log_y: pd.Series, log_x: pd.Series) -> tuple[float, float, pd.Series]:
    """OLS of log_y on log_x with intercept. Returns (alpha, beta, residual)."""
    s = pd.concat([log_y, log_x], axis=1).dropna()
    X = sm.add_constant(s.iloc[:, 1])
    res = sm.OLS(s.iloc[:, 0], X).fit()
    alpha, beta = res.params.iloc[0], res.params.iloc[1]
    resid = (s.iloc[:, 0] - (alpha + beta * s.iloc[:, 1])).rename("residual")
    return float(alpha), float(beta), resid


@dataclass
class EngleGrangerResult:
    beta: float
    alpha: float
    residual: pd.Series
    eg_pvalue: float             # p-value of the EG cointegration test
    resid_adf: StationarityResult
    cointegrated: bool


def engle_granger(log_y: pd.Series, log_x: pd.Series, alpha_level: float = 0.05) -> EngleGrangerResult:
    """Two-step Engle-Granger cointegration test."""
    s = pd.concat([log_y, log_x], axis=1).dropna()
    a, b, resid = estimate_hedge_ratio(s.iloc[:, 0], s.iloc[:, 1])
    # statsmodels.coint runs the EG test directly and returns its p-value.
    _, eg_p, _ = coint(s.iloc[:, 0], s.iloc[:, 1])
    resid_adf = adf_test(resid, alpha_level)
    return EngleGrangerResult(
        beta=b, alpha=a, residual=resid, eg_pvalue=float(eg_p),
        resid_adf=resid_adf, cointegrated=eg_p < alpha_level,
    )


@dataclass
class JohansenResult:
    rank: int                    # number of cointegrating relations at 5%
    trace_stats: np.ndarray
    crit_95: np.ndarray
    hedge_ratio: float           # normalised cointegrating vector (y on x)


def johansen_test(log_y: pd.Series, log_x: pd.Series, det_order: int = 0, k_ar_diff: int = 1) -> JohansenResult:
    """Johansen trace test on the 2-variable system [log_y, log_x]."""
    s = pd.concat([log_y, log_x], axis=1).dropna()
    jres = coint_johansen(s.values, det_order, k_ar_diff)
    trace = jres.lr1
    crit = jres.cvt[:, 1]        # 95% critical values
    rank = int(np.sum(trace > crit))
    # First cointegrating vector, normalised so the y-coefficient is 1.
    vec = jres.evec[:, 0]
    hedge = -vec[1] / vec[0]
    return JohansenResult(rank=rank, trace_stats=trace, crit_95=crit,
                          hedge_ratio=float(hedge))


def half_life(spread: pd.Series) -> float:
    """Half-life of mean reversion from an AR(1) / OU fit on the spread.

    Regress dS_t on S_{t-1}; with dS_t = lambda * S_{t-1} + e, the OU half-life
    is -ln(2) / ln(1 + lambda). A short half-life means fast mean reversion and
    a tradable horizon; a very long one means the "spread" barely reverts.
    """
    s = spread.dropna()
    lag = s.shift(1).dropna()
    ds = (s - s.shift(1)).dropna()
    common = ds.index.intersection(lag.index)
    X = sm.add_constant(lag.loc[common])
    lam = sm.OLS(ds.loc[common], X).fit().params.iloc[1]
    if lam >= 0:
        return np.inf
    return float(-np.log(2) / np.log(1 + lam))


if __name__ == "__main__":
    from data import get_pair_data

    df = get_pair_data()
    print("ADF log_y:", adf_test(df["log_y"]))
    print("ADF log_x:", adf_test(df["log_x"]))
    eg = engle_granger(df["log_y"], df["log_x"])
    print(f"\nEngle-Granger: beta={eg.beta:.3f}  EG p-value={eg.eg_pvalue:.4f}  "
          f"cointegrated={eg.cointegrated}")
    print(f"Residual ADF p-value: {eg.resid_adf.pvalue:.4f} (stationary={eg.resid_adf.is_stationary})")
    jo = johansen_test(df["log_y"], df["log_x"])
    print(f"Johansen rank: {jo.rank}  hedge ratio: {jo.hedge_ratio:.3f}")
    print(f"Spread half-life: {half_life(eg.residual):.1f} days")
