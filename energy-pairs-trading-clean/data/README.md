# Data

This project runs out of the box on a **synthetic** cointegrated pair, so it needs
no downloads, API keys or network access. The generator (`src/data.py`) builds two
log-price series that share a common stochastic trend plus a stationary,
mean-reverting spread, which lets every test and diagnostic be checked against a
known ground truth (the pair is cointegrated by construction, with a known
half-life).

## Using real data

Switch one argument to pull two real series via `yfinance`:

```python
from src.data import get_pair_data
df = get_pair_data(source="yahoo", ticker_y="XOM", ticker_x="CL=F")  # ExxonMobil vs crude
df = get_pair_data(source="yahoo", ticker_y="XOM", ticker_x="CVX")   # the cleaner textbook pair
```

Everything downstream is identical. Good candidate pairs in energy:
- XOM / CVX (two integrated majors — the classic cointegrated equity pair)
- XOM / crude front-month (CL=F) — closest to the ExxonMobil-vs-crude example this upgrades
- XLE / crude, or two refiners (VLO / MPC)

A caution that matters in practice: cointegration estimated in-sample can break
down out-of-sample (regime shifts, mergers, capital-structure changes). Re-test
the relationship on a rolling basis before trusting any pair, and treat a single
pair's backtest as fragile — real books trade baskets of pairs.
