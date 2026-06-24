"""
Tests for energy-pairs-trading.

Two properties worth guaranteeing:

1. The synthetic generator produces a genuinely cointegrated pair: each leg is
   non-stationary, but a linear combination is stationary, with a finite
   mean-reversion half-life. This validates that the cointegration machinery is
   measuring what we think it is.

2. The rolling signal is causal. The position on day t is built only from data
   up to t, so changing the series after a cut date must not move any position
   on or before that date. (The naive full-sample signal deliberately fails
   this; that is the whole point of comparing the two.)

Run with:  pytest -q
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import get_pair_data
from src.cointegration import adf_test, engle_granger, half_life
from src.pairs import rolling_signal, naive_signal


def test_synthetic_pair_is_cointegrated():
    df = get_pair_data(seed=2)
    # Each leg should be non-stationary (fail to reject the unit-root null).
    assert not adf_test(df["log_y"]).is_stationary
    assert not adf_test(df["log_x"]).is_stationary
    # The pair should cointegrate and the spread should be stationary.
    eg = engle_granger(df["log_y"], df["log_x"])
    assert eg.cointegrated, f"EG p-value {eg.eg_pvalue:.3f} should be < 0.05"
    assert eg.resid_adf.is_stationary
    # Half-life should be finite and in a tradable range.
    hl = half_life(eg.residual)
    assert 5 < hl < 60, f"half-life {hl:.1f} outside plausible range"


def test_rolling_signal_is_causal():
    df = get_pair_data(seed=11)
    cut = int(len(df) * 0.6)
    cut_date = df.index[cut]

    z1, b1, pos1 = rolling_signal(df)

    df2 = df.copy()
    rng = np.random.default_rng(0)
    for col in ["log_y", "log_x"]:
        v = df2[col].to_numpy(dtype=float).copy()
        v[cut + 1:] = v[cut + 1:] + rng.normal(0, 0.05, len(v) - cut - 1)
        df2[col] = v
    z2, b2, pos2 = rolling_signal(df2)

    before = z1.index[z1.index <= cut_date]
    # Positions up to the cut must be untouched by future changes.
    assert (pos1.loc[before].fillna(0) == pos2.loc[before].fillna(0)).all(), \
        "look-ahead detected in rolling signal"
    # Non-vacuous: positions after the cut should differ somewhere.
    after = z1.index[z1.index > cut_date]
    assert not (pos1.loc[after].fillna(0) == pos2.loc[after].fillna(0)).all()


def test_naive_signal_uses_future():
    """Sanity check that the naive signal is genuinely full-sample (not causal)."""
    df = get_pair_data(seed=11)
    cut = int(len(df) * 0.6)
    df2 = df.copy()
    v = df2["log_y"].to_numpy(dtype=float).copy()
    v[cut + 1:] = v[cut + 1:] + 0.2          # large future shift
    df2["log_y"] = v
    _, _, pos_a = naive_signal(df)
    _, _, pos_b = naive_signal(df2)
    before = df.index[df.index <= df.index[cut]]
    # Because normalisation is full-sample, past positions DO change. This is
    # the look-ahead the rolling version avoids.
    assert not (pos_a.loc[before].fillna(0) == pos_b.loc[before].fillna(0)).all()
