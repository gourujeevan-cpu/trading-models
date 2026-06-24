"""
Data layer for energy-pairs-trading.

Two interchangeable sources behind one function, ``get_pair_data``:

1. ``synthetic`` (default) builds two cointegrated log-price series. A common
   stochastic trend (the "crude" factor) drives both; the equity leg is a slowly
   drifting multiple of that trend plus a stationary, mean-reverting spread (an
   Ornstein-Uhlenbeck process). Because we control the data-generating process,
   we know the truth: the two series are cointegrated, the cointegrating
   residual is stationary with a known half-life, and the hedge ratio drifts
   over time. Every test and diagnostic downstream can be checked against that.

2. ``yahoo`` pulls two real tickers via yfinance. Defaults to ExxonMobil (XOM)
   and a crude proxy (CL=F), echoing the ExxonMobil-vs-crude example this
   project upgrades. CVX (Chevron) is the cleaner textbook cointegrated equity
   pair if you want a stronger relationship.

Switch one argument and everything downstream is identical:

    df = get_pair_data(source="synthetic")
    df = get_pair_data(source="yahoo", ticker_y="XOM", ticker_x="CVX")

Columns returned (DatetimeIndex, business-day frequency):
    y, x            the two price levels (y = equity leg, x = factor leg)
    log_y, log_x    natural logs
    true_spread     ground-truth OU spread (synthetic only; NaN otherwise)
    true_beta       ground-truth, drifting hedge ratio (synthetic only)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def simulate_cointegrated_pair(
    start: str = "2017-01-01",
    end: str = "2024-12-31",
    seed: int = 2,
    beta0: float = 0.85,
    beta_drift: float = 0.00001,
    half_life: float = 18.0,
    spread_innov_vol: float = 0.018,
    factor_vol: float = 0.019,
    factor_drift: float = 0.0,
    x0: float = 70.0,
    alpha: float = 0.9,
) -> pd.DataFrame:
    """Generate two log-cointegrated price series with a drifting hedge ratio.

    The equity leg is  log_y = alpha + beta_t * log_x + spread_t, where spread_t
    is a mean-reverting OU process. log_y and log_x are therefore cointegrated
    with a (slowly time-varying) cointegrating vector, and the spread is the
    stationary equilibrium error a pairs strategy trades.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, end=end)
    n = len(dates)

    # Factor leg ("crude"): a random walk in logs => I(1), non-stationary.
    log_x = np.log(x0) + np.cumsum(
        factor_drift + factor_vol * rng.standard_normal(n)
    )

    # Drifting hedge ratio: relationships move in real markets, which is exactly
    # why a full-sample OLS hedge ratio is a look-ahead trap.
    beta = beta0 + beta_drift * np.arange(n)

    # OU spread: phi from the target half-life, HL = -ln(2)/ln(phi).
    phi = np.exp(-np.log(2) / half_life)
    spread = np.zeros(n)
    for t in range(1, n):
        spread[t] = phi * spread[t - 1] + spread_innov_vol * rng.standard_normal()

    log_y = alpha + beta * log_x + spread
    x = np.exp(log_x)
    y = np.exp(log_y)

    df = pd.DataFrame(
        {
            "y": y,
            "x": x,
            "log_y": log_y,
            "log_x": log_x,
            "true_spread": spread,
            "true_beta": beta,
        },
        index=dates,
    )
    df.index.name = "date"
    return df


def _load_yahoo(ticker_y: str, ticker_x: str, start: str, end: str) -> pd.DataFrame:
    """Pull two real series via yfinance, aligned on common trading days."""
    import yfinance as yf  # lazy import so the project runs without it

    raw = yf.download([ticker_y, ticker_x], start=start, end=end,
                      progress=False, auto_adjust=True)["Close"]
    raw = raw.dropna()
    if raw.empty:
        raise RuntimeError(f"No overlapping data for {ticker_y}, {ticker_x}.")
    df = pd.DataFrame(index=raw.index)
    df["y"] = raw[ticker_y]
    df["x"] = raw[ticker_x]
    df["log_y"] = np.log(df["y"])
    df["log_x"] = np.log(df["x"])
    df["true_spread"] = np.nan
    df["true_beta"] = np.nan
    df.index.name = "date"
    return df


def get_pair_data(
    source: str = "synthetic",
    ticker_y: str = "XOM",
    ticker_x: str = "CL=F",
    start: str = "2017-01-01",
    end: str = "2024-12-31",
    **kwargs,
) -> pd.DataFrame:
    """Single entry point. ``source`` is 'synthetic' or 'yahoo'."""
    if source == "synthetic":
        return simulate_cointegrated_pair(start=start, end=end, **kwargs)
    if source == "yahoo":
        return _load_yahoo(ticker_y, ticker_x, start, end)
    raise ValueError(f"Unknown source '{source}'. Use 'synthetic' or 'yahoo'.")


if __name__ == "__main__":
    d = get_pair_data()
    print(d.head())
    print(f"\n{len(d)} rows | true half-life ~18d | "
          f"beta drift {d['true_beta'].iloc[0]:.3f} -> {d['true_beta'].iloc[-1]:.3f}")
