"""
End-to-end pipeline for energy-pairs-trading.

Run:  python run_analysis.py
      python run_analysis.py --source yahoo --ticker-y XOM --ticker-x CVX

Steps:
  1. Load the pair (synthetic by default).
  2. Test stationarity of each leg and cointegration of the pair
     (Engle-Granger + Johansen), and estimate the mean-reversion half-life.
  3. Generate the rolling out-of-sample signal and the naive look-ahead signal.
  4. Backtest both against buy-and-hold, with transaction costs.
  5. Save figures to outputs/ and print a summary.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data import get_pair_data
from src.cointegration import adf_test, engle_granger, johansen_test, half_life
from src.pairs import rolling_signal, naive_signal, backtest_pair, buy_and_hold

OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"figure.dpi": 120, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.25, "figure.autolayout": True})


def fig_prices(df, path):
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(df.index, df["y"] / df["y"].iloc[0] * 100, color="#1d4ed8", lw=1.0,
            label="Equity leg (y), rebased")
    ax.plot(df.index, df["x"] / df["x"].iloc[0] * 100, color="#b45309", lw=1.0,
            label="Factor leg (x), rebased")
    ax.set_ylabel("Index (start = 100)")
    ax.set_title("The two legs share a common stochastic trend")
    ax.legend(fontsize=8)
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def fig_spread(df, z, pos, entry, exit, path):
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(z.index, z, color="#374151", lw=0.8, label="Spread z-score (rolling)")
    for lvl, c in [(entry, "#dc2626"), (-entry, "#dc2626"),
                   (exit, "#9ca3af"), (-exit, "#9ca3af")]:
        ax.axhline(lvl, color=c, ls="--", lw=0.8)
    ax.fill_between(z.index, -entry, entry, where=(pos > 0), color="#16a34a",
                    alpha=0.10, label="Long spread")
    ax.fill_between(z.index, -entry, entry, where=(pos < 0), color="#dc2626",
                    alpha=0.10, label="Short spread")
    ax.set_ylabel("z-score")
    ax.set_title("Cointegrating spread, entry/exit bands and positions")
    ax.legend(fontsize=8, loc="upper right", ncol=3)
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def fig_drawdown(bt_roll, bt_naive, path):
    def dd(eq):
        return (eq / eq.cummax() - 1) * 100
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.fill_between(bt_roll.equity.index, dd(bt_roll.equity), 0, color="#1d4ed8",
                    alpha=0.25)
    ax.plot(bt_roll.equity.index, dd(bt_roll.equity), color="#1d4ed8", lw=1.0,
            label=f"Rolling OOS (achievable)  max DD {bt_roll.stats['max_drawdown']*100:.0f}%")
    ax.plot(bt_naive.equity.index, dd(bt_naive.equity), color="#dc2626", lw=1.0,
            label=f"Naive / look-ahead  max DD {bt_naive.stats['max_drawdown']*100:.0f}%")
    ax.set_ylabel("Drawdown (%)")
    ax.set_title("The look-ahead backtest hides risk: achievable drawdowns are far deeper")
    ax.legend(fontsize=8, loc="lower left")
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def fig_equity(bt_roll, bt_naive, bh, path):
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(bh.index, bh, color="#9ca3af", lw=1.0, label="Buy & hold (equity leg)")
    ax.plot(bt_naive.equity.index, bt_naive.equity, color="#dc2626", lw=1.1,
            label=f"Naive / look-ahead  (Sharpe {bt_naive.stats['sharpe']})")
    ax.plot(bt_roll.equity.index, bt_roll.equity, color="#1d4ed8", lw=1.5,
            label=f"Rolling OOS (achievable)  (Sharpe {bt_roll.stats['sharpe']})")
    ax.set_ylabel("Growth of 1 unit")
    ax.set_title("Pairs strategy: achievable vs look-ahead vs buy-and-hold (after costs)")
    ax.legend(fontsize=8, loc="upper left")
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="synthetic", choices=["synthetic", "yahoo"])
    ap.add_argument("--ticker-y", default="XOM")
    ap.add_argument("--ticker-x", default="CL=F")
    ap.add_argument("--window", type=int, default=90)
    ap.add_argument("--entry", type=float, default=2.0)
    ap.add_argument("--exit", type=float, default=0.5)
    args = ap.parse_args()

    print(f"[1/5] Loading pair (source={args.source}) ...")
    df = get_pair_data(source=args.source, ticker_y=args.ticker_y, ticker_x=args.ticker_x)
    print(f"      {len(df)} obs, {df.index[0].date()} to {df.index[-1].date()}")

    print("[2/5] Testing stationarity and cointegration ...")
    adf_y, adf_x = adf_test(df["log_y"]), adf_test(df["log_x"])
    eg = engle_granger(df["log_y"], df["log_x"])
    jo = johansen_test(df["log_y"], df["log_x"])
    hl = half_life(eg.residual)
    print(f"      ADF log_y p={adf_y.pvalue:.3f} (I(1)={not adf_y.is_stationary}) | "
          f"log_x p={adf_x.pvalue:.3f} (I(1)={not adf_x.is_stationary})")
    print(f"      Engle-Granger p={eg.eg_pvalue:.4f} cointegrated={eg.cointegrated} | "
          f"beta={eg.beta:.3f}")
    print(f"      Johansen rank={jo.rank} | half-life={hl:.1f} days")

    print("[3/5] Building rolling OOS and naive signals ...")
    z, beta_roll, pos = rolling_signal(df, window=args.window, entry=args.entry, exit=args.exit)
    zn, beta_n, posn = naive_signal(df, entry=args.entry, exit=args.exit)

    print("[4/5] Backtesting ...")
    bt = backtest_pair(df, pos, beta_roll)
    btn = backtest_pair(df, posn, beta_n)
    bh = buy_and_hold(df)
    print(f"      rolling OOS : {bt.stats}")
    print(f"      naive (LA)  : {btn.stats}")

    print("[5/5] Writing figures ...")
    fig_prices(df, OUT / "01_prices.png")
    fig_spread(df, z, pos, args.entry, args.exit, OUT / "02_spread_signal.png")
    fig_drawdown(bt, btn, OUT / "03_drawdown.png")
    fig_equity(bt, btn, bh, OUT / "04_backtest.png")

    print("\n=== SUMMARY ===")
    print(f"  cointegrated (EG)       : {eg.cointegrated} (p={eg.eg_pvalue:.4f})")
    print(f"  half-life (days)        : {hl:.1f}")
    print(f"  rolling OOS Sharpe      : {bt.stats['sharpe']}")
    print(f"  naive look-ahead Sharpe : {btn.stats['sharpe']}")
    print(f"  look-ahead inflation    : {btn.stats['sharpe'] / bt.stats['sharpe']:.2f}x")
    if df["true_beta"].notna().any():
        end_beta_err = abs(float(beta_roll.dropna().iloc[-1]) - float(df["true_beta"].iloc[-1]))
        print(f"  rolling-beta error vs truth (end): {end_beta_err:.3f}")
    print(f"\nFigures written to {OUT}/")


if __name__ == "__main__":
    main()
