#!/usr/bin/env python3
"""
Estimate the generalized Renyi dimension spectrum D_q for the Henon map.

For a grid partition with box probabilities p_i(epsilon), this script fits

    D_q = slope of log(sum_i p_i^q) / (1 - q) versus log(1 / epsilon), q != 1,
    D_1 = slope of -sum_i p_i log(p_i) versus log(1 / epsilon).

Special cases:
- q = 0 gives the box-counting dimension of the occupied support.
- q = 1 gives the information dimension.
- q = 2 gives the box-probability Renyi dimension, closely related to but not
  identical to the pair-count correlation-dimension estimator.

All outputs are finite-scale numerical diagnostics. The bootstrap intervals
resample selected scale points, so they measure regression-window uncertainty,
not rigorous mathematical enclosures.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def henon_orbit(a: float, b: float, n: int, transient: int, x0: float, y0: float) -> np.ndarray:
    """Generate a deterministic Henon orbit after discarding an initial transient."""
    total = n + transient
    orbit = np.empty((n, 2), dtype=float)
    x, y = float(x0), float(y0)

    for i in range(total):
        x, y = 1.0 - a * x * x + y, b * x
        if not np.isfinite(x) or not np.isfinite(y):
            raise FloatingPointError(
                "Orbit diverged or produced non-finite values. Try different parameters or initial conditions."
            )
        if i >= transient:
            orbit[i - transient] = (x, y)

    return orbit


def geometric_scales(points: np.ndarray, count: int, min_fraction: float, max_fraction: float) -> np.ndarray:
    """Choose geometric scales from the attractor bounding-box size."""
    span = np.ptp(points, axis=0)
    extent = float(np.max(span))
    if extent <= 0.0:
        raise ValueError("Orbit has zero spatial extent; cannot estimate scale-dependent dimensions.")
    return np.geomspace(extent * min_fraction, extent * max_fraction, count)


def linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, int]:
    """Return slope, intercept, R^2, and number of finite points for y = slope*x + intercept."""
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if len(x) < 2:
        return np.nan, np.nan, np.nan, int(len(x))

    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    ss_res = float(np.sum((y - predicted) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else np.nan
    return float(slope), float(intercept), r_squared, int(len(x))


def percentile_interval(values: np.ndarray, confidence_level: float) -> tuple[float, float]:
    """Return a central percentile interval for finite values."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    alpha = 1.0 - confidence_level
    low, high = np.percentile(values, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
    return float(low), float(high)


def bootstrap_slope_interval(
    x: np.ndarray,
    y: np.ndarray,
    samples: int,
    confidence_level: float,
    seed: int,
) -> tuple[float, float]:
    """Bootstrap a regression slope by resampling scale points."""
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if len(x) < 2 or samples <= 0:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)
    slopes = []
    for _ in range(samples):
        indices = rng.choice(len(x), size=len(x), replace=True)
        xb = x[indices]
        yb = y[indices]
        if len(np.unique(xb)) < 2:
            continue
        try:
            slope = np.polyfit(xb, yb, 1)[0]
        except np.linalg.LinAlgError:
            continue
        if np.isfinite(slope):
            slopes.append(float(slope))

    return percentile_interval(np.asarray(slopes), confidence_level)


def sensitivity_slope_range(x: np.ndarray, y: np.ndarray, min_points: int) -> tuple[float, float, int]:
    """Return min/max slopes over contiguous scale windows with at least min_points."""
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if len(x) < min_points:
        return np.nan, np.nan, 0

    slopes = []
    for start in range(len(x)):
        for end in range(start + min_points, len(x) + 1):
            try:
                slope = np.polyfit(x[start:end], y[start:end], 1)[0]
            except np.linalg.LinAlgError:
                continue
            if np.isfinite(slope):
                slopes.append(float(slope))

    if not slopes:
        return np.nan, np.nan, 0
    return float(np.min(slopes)), float(np.max(slopes)), len(slopes)


def box_probabilities(points: np.ndarray, epsilon: float) -> np.ndarray:
    """Return nonzero empirical box probabilities for one grid scale."""
    mins = points.min(axis=0)
    boxes = np.floor((points - mins) / epsilon).astype(np.int64)
    _unique, counts = np.unique(boxes, axis=0, return_counts=True)
    return counts / counts.sum()


def renyi_scale_value(probabilities: np.ndarray, q: float, q_one_tol: float) -> tuple[float, float]:
    """
    Return the scale function whose slope is D_q, plus the raw moment sum.

    For q=1 the raw moment column is left as NaN because entropy replaces the
    singular q -> 1 expression.
    """
    if abs(q - 1.0) <= q_one_tol:
        entropy = -float(np.sum(probabilities * np.log(probabilities)))
        return entropy, np.nan

    moment_sum = float(np.sum(probabilities**q))
    scale_value = np.log(moment_sum) / (1.0 - q)
    return float(scale_value), moment_sum


def generalized_dimension_data(
    points: np.ndarray,
    epsilons: np.ndarray,
    q_values: np.ndarray,
    fit_start: int,
    fit_end: int,
    q_one_tol: float,
) -> pd.DataFrame:
    """Compute scale-dependent Renyi quantities for all q and epsilon values."""
    probability_by_scale = [box_probabilities(points, epsilon) for epsilon in epsilons]
    rows = []

    for q in q_values:
        for scale_index, (epsilon, probabilities) in enumerate(zip(epsilons, probability_by_scale, strict=True)):
            scale_value, moment_sum = renyi_scale_value(probabilities, float(q), q_one_tol)
            rows.append(
                {
                    "q": float(q),
                    "epsilon": float(epsilon),
                    "scale_index": scale_index,
                    "log_inverse_epsilon": float(np.log(1.0 / epsilon)),
                    "occupied_boxes": int(len(probabilities)),
                    "moment_sum": moment_sum,
                    "scale_value": scale_value,
                    "used_in_fit": fit_start <= scale_index < fit_end,
                }
            )

    return pd.DataFrame(rows)


def estimate_dimensions(
    data: pd.DataFrame,
    bootstrap_samples: int,
    confidence_level: float,
    sensitivity_min_points: int,
    seed: int,
) -> pd.DataFrame:
    """Fit D_q for each q and attach bootstrap and fit-window diagnostics."""
    rows = []
    for q_index, (q, group) in enumerate(data.groupby("q", sort=True)):
        fit_group = group[group["used_in_fit"]]
        slope, intercept, r_squared, points_used = linear_fit(
            fit_group["log_inverse_epsilon"].to_numpy(),
            fit_group["scale_value"].to_numpy(),
        )
        boot_low, boot_high = bootstrap_slope_interval(
            fit_group["log_inverse_epsilon"].to_numpy(),
            fit_group["scale_value"].to_numpy(),
            bootstrap_samples,
            confidence_level,
            seed + q_index,
        )
        sens_low, sens_high, sens_count = sensitivity_slope_range(
            group["log_inverse_epsilon"].to_numpy(),
            group["scale_value"].to_numpy(),
            sensitivity_min_points,
        )
        rows.append(
            {
                "q": float(q),
                "dimension_estimate": slope,
                "intercept": intercept,
                "r_squared": r_squared,
                "points_used": points_used,
                "confidence_level": confidence_level,
                "bootstrap_samples": bootstrap_samples,
                "bootstrap_dimension_low": boot_low,
                "bootstrap_dimension_high": boot_high,
                "sensitivity_min_points": sensitivity_min_points,
                "sensitivity_window_count": sens_count,
                "sensitivity_dimension_min": sens_low,
                "sensitivity_dimension_max": sens_high,
            }
        )

    return pd.DataFrame(rows)


def plot_spectrum(summary: pd.DataFrame, path: Path) -> None:
    """Plot q against D_q, with bootstrap intervals when available."""
    summary = summary.sort_values("q")
    q = summary["q"].to_numpy()
    d_q = summary["dimension_estimate"].to_numpy()
    low = summary["bootstrap_dimension_low"].to_numpy()
    high = summary["bootstrap_dimension_high"].to_numpy()

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    finite_interval = np.isfinite(low) & np.isfinite(high)
    if np.any(finite_interval):
        yerr = np.vstack([d_q - low, high - d_q])
        yerr = np.where(np.isfinite(yerr) & (yerr >= 0.0), yerr, 0.0)
        ax.errorbar(q, d_q, yerr=yerr, marker="o", linewidth=1.5, capsize=3, label="estimate")
    else:
        ax.plot(q, d_q, marker="o", linewidth=1.5, label="estimate")

    ax.set_title("Generalized dimension spectrum")
    ax.set_xlabel("q")
    ax.set_ylabel("D_q")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_scaling_curves(data: pd.DataFrame, summary: pd.DataFrame, path: Path, max_curves: int) -> None:
    """Plot scale functions used to estimate representative D_q values."""
    q_values = np.array(sorted(data["q"].unique()), dtype=float)
    if len(q_values) > max_curves:
        indices = np.unique(np.linspace(0, len(q_values) - 1, max_curves).round().astype(int))
        q_values = q_values[indices]

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    for q in q_values:
        group = data[np.isclose(data["q"], q)].sort_values("scale_index")
        estimate = summary[np.isclose(summary["q"], q)]["dimension_estimate"]
        label = f"q={q:g}"
        if not estimate.empty and np.isfinite(float(estimate.iloc[0])):
            label += f", D={float(estimate.iloc[0]):.3f}"
        ax.plot(group["log_inverse_epsilon"], group["scale_value"], marker="o", markersize=3, linewidth=1, label=label)

    ax.set_title("Renyi scaling functions")
    ax.set_xlabel("log(1/epsilon)")
    ax.set_ylabel("scale function")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def parse_q_values(raw_values: list[str]) -> np.ndarray:
    """Parse q values, allowing comma-separated groups or repeated CLI args."""
    parsed = []
    for raw in raw_values:
        for part in raw.split(","):
            part = part.strip()
            if part:
                parsed.append(float(part))
    if not parsed:
        raise ValueError("At least one q value is required.")
    return np.array(sorted(set(parsed)), dtype=float)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate the generalized Renyi dimension spectrum D_q.")
    parser.add_argument("--a", type=float, default=1.4, help="Henon parameter a.")
    parser.add_argument("--b", type=float, default=0.3, help="Henon parameter b.")
    parser.add_argument("--N", type=int, default=100_000, help="Number of orbit points to retain after transient.")
    parser.add_argument("--transient", type=int, default=10_000, help="Number of initial iterates to discard.")
    parser.add_argument("--x0", type=float, default=0.1, help="Initial x coordinate.")
    parser.add_argument("--y0", type=float, default=0.1, help="Initial y coordinate.")
    parser.add_argument(
        "--q-values",
        nargs="+",
        default=["0", "0.5", "1", "1.5", "2", "3", "4"],
        help="q values to estimate. Accepts spaces or commas, e.g. --q-values 0 0.5 1 2 or --q-values 0,0.5,1,2.",
    )
    parser.add_argument("--scale-count", type=int, default=20, help="Number of epsilon scales to evaluate.")
    parser.add_argument("--min-scale-fraction", type=float, default=1e-3, help="Smallest scale as a fraction of extent.")
    parser.add_argument("--max-scale-fraction", type=float, default=2e-1, help="Largest scale as a fraction of extent.")
    parser.add_argument("--fit-start", type=int, default=0, help="First scale index used in fits, inclusive.")
    parser.add_argument(
        "--fit-end",
        type=int,
        default=None,
        help="Last scale index used in fits, exclusive. Defaults to --scale-count.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=1000,
        help="Bootstrap resamples over selected scale points. Use 0 to disable.",
    )
    parser.add_argument("--confidence-level", type=float, default=0.95, help="Central bootstrap confidence level.")
    parser.add_argument(
        "--sensitivity-min-points",
        type=int,
        default=6,
        help="Minimum contiguous scale-window length for slope sensitivity ranges.",
    )
    parser.add_argument("--seed", type=int, default=12345, help="Random seed for bootstrap resampling.")
    parser.add_argument(
        "--q-one-tol",
        type=float,
        default=1e-10,
        help="Treat q values within this tolerance of 1 as the information-dimension case.",
    )
    parser.add_argument(
        "--max-scaling-curves",
        type=int,
        default=8,
        help="Maximum q curves shown in generalized_dimension_scaling_curves.png.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("henon_generalized_dimensions"),
        help="Directory for plots and CSVs.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace, q_values: np.ndarray, fit_end: int) -> None:
    if args.N <= 0 or args.transient < 0:
        raise ValueError("--N must be positive and --transient must be nonnegative.")
    if args.scale_count < 3:
        raise ValueError("--scale-count must be at least 3.")
    if args.min_scale_fraction <= 0.0 or args.max_scale_fraction <= args.min_scale_fraction:
        raise ValueError("Scale fractions must satisfy 0 < min < max.")
    if not (0 <= args.fit_start < fit_end <= args.scale_count):
        raise ValueError("--fit-start and --fit-end must satisfy 0 <= start < end <= scale-count.")
    if args.bootstrap_samples < 0:
        raise ValueError("--bootstrap-samples must be nonnegative.")
    if not (0.0 < args.confidence_level < 1.0):
        raise ValueError("--confidence-level must be between 0 and 1.")
    if args.sensitivity_min_points < 2:
        raise ValueError("--sensitivity-min-points must be at least 2.")
    if args.q_one_tol <= 0.0:
        raise ValueError("--q-one-tol must be positive.")
    if args.max_scaling_curves <= 0:
        raise ValueError("--max-scaling-curves must be positive.")
    if len(q_values) == 0 or not np.all(np.isfinite(q_values)):
        raise ValueError("All q values must be finite.")


def main() -> None:
    args = parse_args()
    q_values = parse_q_values(args.q_values)
    fit_end = args.scale_count if args.fit_end is None else args.fit_end
    validate_args(args, q_values, fit_end)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    points = henon_orbit(args.a, args.b, args.N, args.transient, args.x0, args.y0)
    pd.DataFrame(points, columns=["x", "y"]).to_csv(args.output_dir / "orbit.csv", index=False)

    epsilons = geometric_scales(points, args.scale_count, args.min_scale_fraction, args.max_scale_fraction)
    data = generalized_dimension_data(points, epsilons, q_values, args.fit_start, fit_end, args.q_one_tol)
    summary = estimate_dimensions(
        data,
        args.bootstrap_samples,
        args.confidence_level,
        args.sensitivity_min_points,
        args.seed,
    )

    data.to_csv(args.output_dir / "generalized_dimension_scale_data.csv", index=False)
    summary.to_csv(args.output_dir / "generalized_dimensions.csv", index=False)
    plot_spectrum(summary, args.output_dir / "generalized_dimension_spectrum.png")
    plot_scaling_curves(
        data,
        summary,
        args.output_dir / "generalized_dimension_scaling_curves.png",
        args.max_scaling_curves,
    )

    print(f"Saved generalized dimension outputs to {args.output_dir}")
    for row in summary.sort_values("q").itertuples(index=False):
        if np.isfinite(row.bootstrap_dimension_low) and np.isfinite(row.bootstrap_dimension_high):
            interval = f" [{row.bootstrap_dimension_low:.6f}, {row.bootstrap_dimension_high:.6f}]"
        else:
            interval = ""
        print(f"q={row.q:g}: D_q={row.dimension_estimate:.6f}{interval}")


if __name__ == "__main__":
    main()
