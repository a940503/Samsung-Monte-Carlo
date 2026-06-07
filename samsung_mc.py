from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


TRADING_DAYS = 252
VOL_WINDOWS = (30, 90, 252)


def parse_price(value: object) -> float:
    if pd.isna(value):
        return np.nan
    return float(str(value).replace(",", "").strip())


def load_investing_csv(path: Path, label: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"Date", "Price"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df["Date"], format="%m/%d/%Y"),
            f"{label}_price": df["Price"].map(parse_price),
        }
    )
    out = out.dropna().sort_values("date").drop_duplicates("date")
    out[f"{label}_log_return"] = np.log(
        out[f"{label}_price"] / out[f"{label}_price"].shift(1)
    )

    for window in VOL_WINDOWS:
        out[f"{label}_rolling_vol_{window}d"] = (
            out[f"{label}_log_return"].rolling(window).std() * np.sqrt(TRADING_DAYS)
        )

    return out.reset_index(drop=True)


def estimate_gbm_params(log_returns: pd.Series) -> tuple[float, float]:
    clean = log_returns.dropna()
    if len(clean) < TRADING_DAYS:
        raise ValueError("At least 252 log-return observations are recommended.")

    mu = clean.mean() * TRADING_DAYS
    sigma = clean.std(ddof=1) * np.sqrt(TRADING_DAYS)
    return float(mu), float(sigma)


def filter_returns_for_mc(
    samsung: pd.DataFrame, start_date: str | None, end_date: str | None
) -> pd.Series:
    mask = pd.Series(True, index=samsung.index)
    if start_date:
        mask &= samsung["date"] >= pd.Timestamp(start_date)
    if end_date:
        mask &= samsung["date"] <= pd.Timestamp(end_date)

    returns = samsung.loc[mask, "samsung_log_return"].dropna()
    if returns.empty:
        raise ValueError("No Samsung log returns remain after applying MC date filters.")
    return returns


def simulate_gbm_paths(
    s0: float,
    mu: float,
    sigma: float,
    horizon_days: int,
    n_paths: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dt = 1.0 / TRADING_DAYS
    shocks = rng.standard_normal((horizon_days, n_paths))
    increments = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * shocks

    paths = np.empty((horizon_days + 1, n_paths), dtype=np.float64)
    paths[0] = s0
    paths[1:] = s0 * np.exp(np.cumsum(increments, axis=0))
    return paths


def scale_points(x_values: np.ndarray, y_values: np.ndarray, width: int, height: int, pad: int) -> str:
    x_min, x_max = float(np.min(x_values)), float(np.max(x_values))
    y_min, y_max = float(np.nanmin(y_values)), float(np.nanmax(y_values))
    if y_min == y_max:
        y_min -= 1.0
        y_max += 1.0

    points = []
    for x, y in zip(x_values, y_values):
        if np.isnan(y):
            continue
        sx = pad + (float(x) - x_min) / (x_max - x_min) * (width - 2 * pad)
        sy = height - pad - (float(y) - y_min) / (y_max - y_min) * (height - 2 * pad)
        points.append(f"{sx:.1f},{sy:.1f}")
    return " ".join(points)


def write_svg_line_chart(
    series: list[tuple[str, np.ndarray, np.ndarray, str, str]],
    output_path: Path,
    title: str,
    y_label: str,
    threshold: float | None = None,
) -> None:
    width, height, pad = 1100, 620, 70
    all_y = np.concatenate([y[np.isfinite(y)] for _, _, y, _, _ in series])
    y_min, y_max = float(np.min(all_y)), float(np.max(all_y))
    if threshold is not None:
        y_min = min(y_min, threshold)
        y_max = max(y_max, threshold)
    if y_min == y_max:
        y_min -= 1.0
        y_max += 1.0

    def xy(x: float, y: float, x_min: float, x_max: float) -> tuple[float, float]:
        sx = pad + (x - x_min) / (x_max - x_min) * (width - 2 * pad)
        sy = height - pad - (y - y_min) / (y_max - y_min) * (height - 2 * pad)
        return sx, sy

    x_min = min(float(np.min(x)) for _, x, _, _, _ in series)
    x_max = max(float(np.max(x)) for _, x, _, _, _ in series)

    body = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2}" y="34" text-anchor="middle" font-family="Arial" font-size="24" font-weight="700">{title}</text>',
        f'<text x="24" y="{height / 2}" transform="rotate(-90 24 {height / 2})" text-anchor="middle" font-family="Arial" font-size="14">{y_label}</text>',
        f'<line x1="{pad}" y1="{height - pad}" x2="{width - pad}" y2="{height - pad}" stroke="#222"/>',
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height - pad}" stroke="#222"/>',
    ]

    for i in range(6):
        y = y_min + (y_max - y_min) * i / 5
        _, sy = xy(x_min, y, x_min, x_max)
        body.append(f'<line x1="{pad}" y1="{sy:.1f}" x2="{width - pad}" y2="{sy:.1f}" stroke="#e5e7eb"/>')
        body.append(f'<text x="{pad - 10}" y="{sy + 4:.1f}" text-anchor="end" font-family="Arial" font-size="12">{y:,.2f}</text>')

    if threshold is not None:
        _, sy = xy(x_min, threshold, x_min, x_max)
        body.append(f'<line x1="{pad}" y1="{sy:.1f}" x2="{width - pad}" y2="{sy:.1f}" stroke="#d62828" stroke-width="2"/>')
        body.append(f'<text x="{width - pad - 6}" y="{sy - 8:.1f}" text-anchor="end" font-family="Arial" font-size="13" fill="#d62828">Threshold {threshold:,.0f}</text>')

    legend_x, legend_y = pad + 12, 60
    for idx, (name, x_values, y_values, color, dash) in enumerate(series):
        pts = []
        for x, y in zip(x_values, y_values):
            if np.isfinite(y):
                sx, sy = xy(float(x), float(y), x_min, x_max)
                pts.append(f"{sx:.1f},{sy:.1f}")
        body.append(
            f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color}" stroke-width="1.8" stroke-linejoin="round" stroke-linecap="round" {dash}/>'
        )
        ly = legend_y + idx * 20
        body.append(f'<line x1="{legend_x}" y1="{ly}" x2="{legend_x + 26}" y2="{ly}" stroke="{color}" stroke-width="2" {dash}/>')
        body.append(f'<text x="{legend_x + 34}" y="{ly + 4}" font-family="Arial" font-size="13">{name}</text>')

    body.append("</svg>")
    output_path.write_text("\n".join(body), encoding="utf-8")


def stopping_times(paths: np.ndarray, threshold: float) -> tuple[np.ndarray, float]:
    breached = paths[1:] < threshold
    ever_breached = breached.any(axis=0)

    first_breach = np.full(paths.shape[1], np.nan)
    first_breach[ever_breached] = breached[:, ever_breached].argmax(axis=0) + 1

    probability = float(ever_breached.mean())
    return first_breach, probability


def plot_rolling_volatility(
    samsung: pd.DataFrame, kospi: pd.DataFrame | None, output_path: Path
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        colors = ["#3772ff", "#2a9d8f", "#f4a261", "#6c757d", "#8d5a97", "#e76f51"]
        series = []
        x = samsung["date"].map(pd.Timestamp.toordinal).to_numpy()
        for idx, window in enumerate(VOL_WINDOWS):
            series.append(
                (
                    f"Samsung {window}D",
                    x,
                    samsung[f"samsung_rolling_vol_{window}d"].to_numpy(),
                    colors[idx],
                    "",
                )
            )
        if kospi is not None:
            kx = kospi["date"].map(pd.Timestamp.toordinal).to_numpy()
            for idx, window in enumerate(VOL_WINDOWS):
                series.append(
                    (
                        f"KOSPI {window}D",
                        kx,
                        kospi[f"kospi_rolling_vol_{window}d"].to_numpy(),
                        colors[idx + 3],
                        'stroke-dasharray="6 5"',
                    )
                )
        write_svg_line_chart(
            series,
            output_path.with_suffix(".svg"),
            "Annualized Rolling Volatility",
            "Volatility",
        )
        return

    plt.figure(figsize=(13, 7))
    for window in VOL_WINDOWS:
        plt.plot(
            samsung["date"],
            samsung[f"samsung_rolling_vol_{window}d"],
            label=f"Samsung {window}D",
            linewidth=1.4,
        )

    if kospi is not None:
        for window in VOL_WINDOWS:
            plt.plot(
                kospi["date"],
                kospi[f"kospi_rolling_vol_{window}d"],
                label=f"KOSPI {window}D",
                linewidth=1.0,
                linestyle="--",
                alpha=0.75,
            )

    plt.title("Annualized Rolling Volatility")
    plt.xlabel("Date")
    plt.ylabel("Volatility")
    plt.grid(True, alpha=0.25)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_sample_paths(paths: np.ndarray, threshold: float, output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        sample_count = min(75, paths.shape[1])
        x = np.arange(paths.shape[0])
        series = [
            (f"Path {i + 1}", x, paths[:, i], "#3772ff", 'opacity="0.18"')
            for i in range(sample_count)
        ]
        write_svg_line_chart(
            series,
            output_path.with_suffix(".svg"),
            f"Sample Monte Carlo Paths ({sample_count:,} of {paths.shape[1]:,})",
            "Simulated price",
            threshold=threshold,
        )
        return

    sample_count = min(250, paths.shape[1])
    x = np.arange(paths.shape[0])

    plt.figure(figsize=(13, 7))
    plt.plot(x, paths[:, :sample_count], color="#3772ff", alpha=0.08, linewidth=0.8)
    plt.axhline(threshold, color="#d62828", linewidth=1.8, label=f"Threshold {threshold:,.0f}")
    plt.title(f"Sample Monte Carlo Paths ({sample_count:,} of {paths.shape[1]:,})")
    plt.xlabel("Trading day")
    plt.ylabel("Simulated price")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Samsung Electronics log returns, rolling volatility, and Monte Carlo stopping-time analysis."
    )
    parser.add_argument("--samsung-csv", required=True, type=Path)
    parser.add_argument("--kospi-csv", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=150_000)
    parser.add_argument("--paths", type=int, default=100_000)
    parser.add_argument("--horizon-days", type=int, default=252)
    parser.add_argument(
        "--mc-start-date",
        default=None,
        help="Optional YYYY-MM-DD start date for returns used to estimate Monte Carlo mu/sigma.",
    )
    parser.add_argument(
        "--mc-end-date",
        default=None,
        help="Optional YYYY-MM-DD end date for returns used to estimate Monte Carlo mu/sigma.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    samsung = load_investing_csv(args.samsung_csv, "samsung")
    kospi = load_investing_csv(args.kospi_csv, "kospi") if args.kospi_csv else None

    samsung.to_csv(args.output_dir / "samsung_returns_volatility.csv", index=False)
    if kospi is not None:
        kospi.to_csv(args.output_dir / "kospi_returns_volatility.csv", index=False)

    plots_created = True
    try:
        plot_rolling_volatility(
            samsung=samsung,
            kospi=kospi,
            output_path=args.output_dir / "rolling_volatility_comparison.png",
        )
    except ModuleNotFoundError as exc:
        if exc.name != "matplotlib":
            raise
        plots_created = False

    s0 = float(samsung["samsung_price"].iloc[-1])
    mc_returns = filter_returns_for_mc(
        samsung=samsung,
        start_date=args.mc_start_date,
        end_date=args.mc_end_date,
    )
    mu, sigma = estimate_gbm_params(mc_returns)
    paths = simulate_gbm_paths(
        s0=s0,
        mu=mu,
        sigma=sigma,
        horizon_days=args.horizon_days,
        n_paths=args.paths,
        seed=args.seed,
    )
    first_breach, breach_probability = stopping_times(paths, args.threshold)

    pd.DataFrame(
        {
            "path_id": np.arange(args.paths),
            "first_breach_day": first_breach,
            "breached_within_horizon": ~np.isnan(first_breach),
        }
    ).to_csv(args.output_dir / "stopping_times.csv", index=False)

    try:
        plot_sample_paths(
            paths=paths,
            threshold=args.threshold,
            output_path=args.output_dir / "sample_monte_carlo_paths.png",
        )
    except ModuleNotFoundError as exc:
        if exc.name != "matplotlib":
            raise
        plots_created = False

    final_prices = paths[-1]
    summary = {
        "input_file": str(args.samsung_csv),
        "latest_date": samsung["date"].iloc[-1].strftime("%Y-%m-%d"),
        "latest_price": s0,
        "threshold": args.threshold,
        "paths": args.paths,
        "horizon_days": args.horizon_days,
        "seed": args.seed,
        "annualized_mu": mu,
        "annualized_sigma": sigma,
        "mc_return_start_date": (
            mc_returns.index.to_series().map(samsung["date"]).iloc[0].strftime("%Y-%m-%d")
        ),
        "mc_return_end_date": (
            mc_returns.index.to_series().map(samsung["date"]).iloc[-1].strftime("%Y-%m-%d")
        ),
        "mc_return_observations": int(mc_returns.shape[0]),
        "probability_price_below_threshold_at_least_once": breach_probability,
        "plots_created": plots_created,
        "final_price_mean": float(final_prices.mean()),
        "final_price_median": float(np.median(final_prices)),
        "final_price_p05": float(np.quantile(final_prices, 0.05)),
        "final_price_p95": float(np.quantile(final_prices, 0.95)),
        "breached_paths": int(np.sum(~np.isnan(first_breach))),
    }

    with (args.output_dir / "monte_carlo_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
