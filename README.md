# Samsung Electronics Monte Carlo Risk Analysis

Samsung Electronics historical price analysis:

- log return calculation
- rolling volatility comparison for 30, 90, and 252 trading days
- Monte Carlo simulation with 100,000 one-year paths
- stopping-time probability that price falls below KRW 150,000 at least once within one year

The script also supports an optional KOSPI benchmark CSV in the same Investing.com-style format.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python src/samsung_mc.py ^
  --samsung-csv "C:\Users\a9405\Downloads\Samsung Electronics Co Stock Price History (2).csv" ^
  --kospi-csv "C:\Users\a9405\Downloads\KOSPI Historical Data.csv" ^
  --threshold 150000 ^
  --paths 100000 ^
  --horizon-days 252 ^
  --mc-start-date 2018-05-04 ^
  --output-dir outputs
```

For macOS/Linux, replace `^` with `\`.

## Outputs

The script writes:

- `samsung_returns_volatility.csv`: cleaned prices, log returns, and rolling volatilities
- `kospi_returns_volatility.csv`: optional benchmark returns and rolling volatilities
- `rolling_volatility_comparison.png`: 30/90/252-day volatility comparison
- `monte_carlo_summary.json`: simulation assumptions and stopping-time result
- `stopping_times.csv`: first day each path breached the threshold
- `sample_monte_carlo_paths.png`: sample simulated paths with the threshold line

## Method

Daily log return:

```text
r_t = ln(P_t / P_{t-1})
```

Annualized rolling volatility:

```text
rolling_std(log_returns, window) * sqrt(252)
```

Monte Carlo price paths use geometric Brownian motion:

```text
S_t = S_{t-1} * exp((mu - 0.5 * sigma^2) * dt + sigma * sqrt(dt) * Z_t)
```

where `mu` and `sigma` are annualized from historical daily log returns.

For Samsung Electronics, the sample command estimates Monte Carlo drift and volatility from `2018-05-04` onward to avoid treating the 50:1 stock split as an ordinary market return. The full historical series is still used for the returns and rolling-volatility output unless you filter the input file yourself.

The stopping time for each path is the first simulated trading day where:

```text
S_t < 150000
```

The reported probability is:

```text
number_of_paths_with_a_breach / total_paths
```

## Notes

This is a quantitative analysis example, not investment advice. Historical drift and volatility may not describe future Samsung Electronics behavior, especially around regime changes, stock splits, dividends, and market stress periods.
