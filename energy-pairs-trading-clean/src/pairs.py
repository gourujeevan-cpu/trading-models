"""
Pairs trading strategy and backtest.

The trade: when the cointegrating spread is unusually high the equity leg is
rich relative to the factor leg, so we short the spread (short y, long beta*x)
and wait for it to revert; when it is unusually low we go long the spread. The
spread is measured in standard deviations (a z-score); we enter past an entry
threshold and exit near zero.

Two versions are implemented on purpose:

* ``rolling_signal`` re-estimates the hedge ratio and the spread's mean and
  standard deviation on a trailing window, using only past data, and acts on
  the next day. This is what is actually achievable in real time.

* ``naive_signal`` estimates the hedge ratio and the z-score normalisation once
  on the whole sample. It is the standard textbook mistake: the normalisation
  "knows" the spread's eventual range, which flatters the backtest. Comparing
  the two shows how much of a typical pairs result is look-ahead.

The backtest is dollar-neutral in spread units, charges a per-leg transaction
cost on every position change, and is reported against buy-and-hold of the
equity leg.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def _build_positions(z: pd.Series, entry: float, exit: float) -> pd.Series:
    """Stateful entry/exit: enter past +/-entry, exit inside +/-exit, else hold."""
    pos = np.zeros(len(z))
    cur = 0
    zv = z.to_numpy()
    for i in range(len(zv)):
        zi = zv[i]
        if np.isnan(zi):
            cur = 0
        elif cur == 0:
            if zi > entry:
                cur = -1          # spread rich -> short spread
            elif zi < -entry:
                cur = 1           # spread cheap -> long spread
        else:                     # currently in a position
            if abs(zi) < exit:
                cur = 0           # reverted -> close
        pos[i] = cur
    return pd.Series(pos, index=z.index, name="position")


def rolling_signal(df: pd.DataFrame, window: int = 90,
                   entry: float = 2.0, exit: float = 0.5):
    """Causal rolling hedge ratio + z-score. Returns (z, beta, positions).

    A trailing window estimates the hedge ratio and normalises the spread, both
    using only past data; the resulting position is acted on the next day. A
    short window keeps the spread locally tight and mean-reverting, which is
    what the z-score entry/exit logic needs.
    """
    ly, lx = df["log_y"], df["log_x"]
    beta = ly.rolling(window).cov(lx) / lx.rolling(window).var()
    spread = ly - beta * lx
    mu = spread.rolling(window).mean()
    sigma = spread.rolling(window).std()
    z = ((spread - mu) / sigma).rename("zscore")
    pos = _build_positions(z, entry, exit)
    return z, beta.rename("beta"), pos


def naive_signal(df: pd.DataFrame, entry: float = 2.0, exit: float = 0.5):
    """Full-sample hedge ratio + z-score (contains look-ahead). For contrast only."""
    ly, lx = df["log_y"], df["log_x"]
    beta_full = float(np.cov(ly, lx)[0, 1] / np.var(lx))
    spread = ly - beta_full * lx
    z = ((spread - spread.mean()) / spread.std()).rename("zscore")
    pos = _build_positions(z, entry, exit)
    beta = pd.Series(beta_full, index=df.index, name="beta")
    return z, beta, pos


@dataclass
class BacktestResult:
    daily_returns: pd.Series
    equity: pd.Series
    stats: dict
    positions: pd.Series


def backtest_pair(df: pd.DataFrame, positions: pd.Series, beta: pd.Series,
                  cost_bps: float = 5.0) -> BacktestResult:
    """Backtest a 0/+1/-1 spread signal. Costs charged on each leg per turn.

    The position decided on day t is applied to day t+1's return, and the hedge
    ratio used is also the one known at t. Nothing uses same-day information.
    """
    ret_y = df["log_y"].diff()
    ret_x = df["log_x"].diff()
    beta_lag = beta.shift(1)
    pos_lag = positions.shift(1).fillna(0)

    spread_ret = ret_y - beta_lag * ret_x          # return of one unit of spread
    gross = pos_lag * spread_ret

    # Transaction cost: a position change trades both legs (1 + beta units).
    turn = (positions - positions.shift(1)).abs().fillna(0)
    leg_units = (1 + beta.abs()).fillna(1)
    cost = turn * leg_units * (cost_bps / 1e4)

    net = (gross - cost).fillna(0.0)
    equity = net.cumsum().pipe(np.exp).rename("strategy")

    in_mkt = pos_lag != 0
    n_trades = int((turn > 0).sum())
    hit = (net[in_mkt] > 0).mean() if in_mkt.any() else np.nan
    ann_ret = net.mean() * 252
    ann_vol = net.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    eq = net.cumsum().pipe(np.exp)
    max_dd = (eq / eq.cummax() - 1).min()
    exposure = in_mkt.mean()

    stats = {
        "ann_return": round(float(ann_ret), 4),
        "ann_vol": round(float(ann_vol), 4),
        "sharpe": round(float(sharpe), 3),
        "max_drawdown": round(float(max_dd), 4),
        "hit_rate": round(float(hit), 4),
        "n_trades": n_trades,
        "time_in_market": round(float(exposure), 3),
    }
    return BacktestResult(net, equity, stats, positions)


def buy_and_hold(df: pd.DataFrame) -> pd.Series:
    """Equity-leg buy-and-hold equity curve for comparison."""
    return df["log_y"].diff().fillna(0).cumsum().pipe(np.exp).rename("buy_hold")


if __name__ == "__main__":
    from data import get_pair_data

    df = get_pair_data()
    z, beta, pos = rolling_signal(df)
    bt = backtest_pair(df, pos, beta)
    zn, betan, posn = naive_signal(df)
    btn = backtest_pair(df, posn, betan)
    print("Rolling (out-of-sample, achievable):")
    print("  ", bt.stats)
    print("Naive (full-sample, look-ahead):")
    print("  ", btn.stats)
