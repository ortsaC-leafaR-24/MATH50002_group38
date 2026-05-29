#!/usr/bin/env python3
"""
Study strange attractors of two-dimensional maps, with the Henon map included.

The script generates a long orbit, estimates Lyapunov exponents, and computes
several fractal-dimension diagnostics. All dimension estimates here are finite
sample regressions over user-chosen scale ranges; treat them as diagnostics, not
as rigorous mathematical values.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class FitResult:
    dimension: str
    slope: float
    intercept: float
    r_squared: float
    points_used: int
    x_label: str
    y_label: str


@dataclass(frozen=True)
class FitUncertainty:
    confidence_level: float
    bootstrap_samples: int
    bootstrap_slope_low: float
    bootstrap_slope_high: float
    sensitivity_min_points: int
    sensitivity_window_count: int
    sensitivity_slope_min: float
    sensitivity_slope_max: float


def henon_orbit(a: float, b: float, n: int, transient: int, x0: float, y0: float) -> np.ndarray:
    """Generate a Henon orbit after discarding an initial transient."""
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


def henon_jacobian(a: float, b: float, x: float) -> np.ndarray:
    """Jacobian of the Henon map at (x, y)."""
    return np.array([[-2.0 * a * x, 1.0], [b, 0.0]], dtype=float)


def lyapunov_exponents_qr(points: np.ndarray, a: float, b: float) -> np.ndarray:
    """
    Estimate both Lyapunov exponents using QR re-orthonormalization.

    The orbit is assumed to have already passed through its transient. QR is the
    numerically stable form of the usual Gram-Schmidt tangent-space iteration.
    """
    q = np.eye(2)
    log_diag_sum = np.zeros(2, dtype=float)

    for x, _y in points:
        z = henon_jacobian(a, b, x) @ q
        q, r = np.linalg.qr(z)
        diag = np.diag(r).copy()

        # Keep the QR convention stable so sign flips do not affect accumulated logs.
        signs = np.sign(diag)
        signs[signs == 0.0] = 1.0
        q *= signs
        diag *= signs
        log_diag_sum += np.log(np.abs(diag))

    return np.sort(log_diag_sum / len(points))[::-1]


def kaplan_yorke_dimension(exponents: np.ndarray) -> float:
    """Compute the Kaplan-Yorke dimension from sorted Lyapunov exponents."""
    exponents = np.asarray(exponents, dtype=float)
    cumulative = np.cumsum(exponents)

    nonnegative = np.where(cumulative >= 0.0)[0]
    if len(nonnegative) == 0:
        return 0.0

    j = int(nonnegative[-1])
    if j == len(exponents) - 1:
        return float(len(exponents))

    return float((j + 1) + cumulative[j] / abs(exponents[j + 1]))


def regression_fit(x: np.ndarray, y: np.ndarray, dimension: str, x_label: str, y_label: str) -> FitResult:
    """Fit y = slope*x + intercept and return slope plus a simple R^2."""
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if len(x) < 2:
        return FitResult(dimension, np.nan, np.nan, np.nan, int(len(x)), x_label, y_label)

    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    ss_res = float(np.sum((y - predicted) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else np.nan
    return FitResult(dimension, float(slope), float(intercept), r_squared, int(len(x)), x_label, y_label)


def finite_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return finite x/y pairs."""
    valid = np.isfinite(x) & np.isfinite(y)
    return x[valid], y[valid]


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
    """Bootstrap the regression slope over selected scale points."""
    x, y = finite_xy(x, y)
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
    x, y = finite_xy(x, y)
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


def fit_uncertainty(
    fit_x: np.ndarray,
    fit_y: np.ndarray,
    all_x: np.ndarray,
    all_y: np.ndarray,
    bootstrap_samples: int,
    confidence_level: float,
    seed: int,
    sensitivity_min_points: int,
) -> FitUncertainty:
    """Compute bootstrap and fit-window sensitivity intervals for one slope estimate."""
    boot_low, boot_high = bootstrap_slope_interval(
        fit_x,
        fit_y,
        bootstrap_samples,
        confidence_level,
        seed,
    )
    sens_low, sens_high, sens_count = sensitivity_slope_range(all_x, all_y, sensitivity_min_points)
    return FitUncertainty(
        confidence_level=confidence_level,
        bootstrap_samples=bootstrap_samples,
        bootstrap_slope_low=boot_low,
        bootstrap_slope_high=boot_high,
        sensitivity_min_points=sensitivity_min_points,
        sensitivity_window_count=sens_count,
        sensitivity_slope_min=sens_low,
        sensitivity_slope_max=sens_high,
    )


def box_counts(points: np.ndarray, epsilons: np.ndarray) -> pd.DataFrame:
    """Count occupied square boxes at each epsilon."""
    mins = points.min(axis=0)
    rows = []
    for eps in epsilons:
        boxes = np.floor((points - mins) / eps).astype(np.int64)
        occupied = np.unique(boxes, axis=0).shape[0]
        rows.append({"epsilon": eps, "occupied_boxes": occupied})
    return pd.DataFrame(rows)


def information_sums(points: np.ndarray, epsilons: np.ndarray) -> pd.DataFrame:
    """Estimate entropy of box probabilities at each epsilon."""
    mins = points.min(axis=0)
    rows = []
    for eps in epsilons:
        boxes = np.floor((points - mins) / eps).astype(np.int64)
        _unique, counts = np.unique(boxes, axis=0, return_counts=True)
        probabilities = counts / counts.sum()
        entropy = -float(np.sum(probabilities * np.log(probabilities)))
        rows.append({"epsilon": eps, "entropy": entropy, "occupied_boxes": len(counts)})
    return pd.DataFrame(rows)


def sampled_pairwise_distances(
    points: np.ndarray,
    max_points: int,
    seed: int,
    theiler_window: int,
) -> tuple[np.ndarray, int, int]:
    """Return pairwise distances, optionally excluding temporally close pairs."""
    rng = np.random.default_rng(seed)
    if len(points) > max_points:
        indices = rng.choice(len(points), size=max_points, replace=False)
        indices = np.sort(indices)
    else:
        indices = np.arange(len(points))

    sample = points[indices]
    if theiler_window <= 0:
        distances = pdist(sample)
        return distances, len(sample), len(distances)

    chunks = []
    pairs_used = 0
    for i in range(len(sample) - 1):
        valid = np.abs(indices[i + 1 :] - indices[i]) > theiler_window
        if not np.any(valid):
            continue
        differences = sample[i + 1 :][valid] - sample[i]
        distances = np.linalg.norm(differences, axis=1)
        chunks.append(distances)
        pairs_used += len(distances)

    if not chunks:
        return np.array([], dtype=float), len(sample), 0
    return np.concatenate(chunks), len(sample), pairs_used


def correlation_sums(distances: np.ndarray, radii: np.ndarray) -> pd.DataFrame:
    """Estimate C(r) = fraction of sampled pairs with distance below r."""
    distances = np.sort(distances[np.isfinite(distances)])
    total_pairs = len(distances)
    rows = []
    for radius in radii:
        count = int(np.searchsorted(distances, radius, side="right"))
        c_r = count / total_pairs if total_pairs else np.nan
        rows.append({"radius": radius, "correlation_sum": c_r, "pairs_within_radius": count})
    return pd.DataFrame(rows)


def geometric_scales(points: np.ndarray, count: int, min_fraction: float, max_fraction: float) -> np.ndarray:
    """Choose geometric scales from the attractor bounding-box size."""
    span = np.ptp(points, axis=0)
    extent = float(np.max(span))
    if extent <= 0.0:
        raise ValueError("Orbit has zero spatial extent; cannot estimate scale-dependent dimensions.")
    return np.geomspace(extent * min_fraction, extent * max_fraction, count)


def plot_attractor(points: np.ndarray, path: Path, title: str, point_size: float) -> None:
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    ax.scatter(points[:, 0], points[:, 1], s=point_size, alpha=0.45, linewidths=0)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_fit(
    x: np.ndarray,
    y: np.ndarray,
    fit: FitResult,
    path: Path,
    title: str,
    x_axis_label: str,
    y_axis_label: str,
    fit_mask: np.ndarray | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    if fit_mask is None:
        ax.plot(x, y, "o", label="scale data")
        fit_x = x
    else:
        ax.plot(x[~fit_mask], y[~fit_mask], "o", color="0.7", label="unused scales")
        ax.plot(x[fit_mask], y[fit_mask], "o", label="fit scales")
        fit_x = x[fit_mask]
    if np.isfinite(fit.slope):
        xx = np.linspace(float(np.min(fit_x)), float(np.max(fit_x)), 200)
        ax.plot(xx, fit.slope * xx + fit.intercept, "-", label=f"slope = {fit.slope:.4f}")
    ax.set_title(title)
    ax.set_xlabel(x_axis_label)
    ax.set_ylabel(y_axis_label)
    ax.legend()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_fit_plot_data(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and analyze the Henon strange attractor for configurable map parameters."
    )
    parser.add_argument("--a", type=float, default=1.4, help="Henon parameter a.")
    parser.add_argument("--b", type=float, default=0.3, help="Henon parameter b.")
    parser.add_argument("--N", type=int, default=100_000, help="Number of orbit points to retain after transient.")
    parser.add_argument("--transient", type=int, default=10_000, help="Number of initial iterates to discard.")
    parser.add_argument("--x0", type=float, default=0.1, help="Initial x coordinate.")
    parser.add_argument("--y0", type=float, default=0.1, help="Initial y coordinate.")
    parser.add_argument("--scale-count", type=int, default=20, help="Number of epsilon/radius scales to evaluate.")
    parser.add_argument("--min-scale-fraction", type=float, default=1e-3, help="Smallest scale as a fraction of extent.")
    parser.add_argument("--max-scale-fraction", type=float, default=2e-1, help="Largest scale as a fraction of extent.")
    parser.add_argument("--fit-start", type=int, default=0, help="First scale index used in dimension fits, inclusive.")
    parser.add_argument(
        "--fit-end",
        type=int,
        default=None,
        help="Last scale index used in dimension fits, exclusive. Defaults to --scale-count.",
    )
    parser.add_argument(
        "--max-correlation-points",
        type=int,
        default=8_000,
        help="Maximum points used for pairwise correlation-dimension distances.",
    )
    parser.add_argument(
        "--theiler-window",
        type=int,
        default=0,
        help="Exclude correlation pairs whose original orbit indices differ by at most this value.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=1000,
        help="Bootstrap resamples for regression-slope confidence intervals over selected scale points. Use 0 to disable.",
    )
    parser.add_argument(
        "--confidence-level",
        type=float,
        default=0.95,
        help="Central confidence level for bootstrap intervals.",
    )
    parser.add_argument(
        "--sensitivity-min-points",
        type=int,
        default=6,
        help="Minimum contiguous scale-window length for slope sensitivity ranges.",
    )
    parser.add_argument("--seed", type=int, default=12345, help="Random seed for correlation-dimension subsampling.")
    parser.add_argument("--point-size", type=float, default=0.08, help="Scatter point size for attractor plot.")
    parser.add_argument("--output-dir", type=Path, default=Path("henon_output"), help="Directory for plots and CSVs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.N <= 0 or args.transient < 0:
        raise ValueError("--N must be positive and --transient must be nonnegative.")
    if args.scale_count < 3:
        raise ValueError("--scale-count must be at least 3 for meaningful fits.")
    if args.min_scale_fraction <= 0.0 or args.max_scale_fraction <= args.min_scale_fraction:
        raise ValueError("Scale fractions must satisfy 0 < min < max.")
    if args.max_correlation_points < 2:
        raise ValueError("--max-correlation-points must be at least 2.")
    if args.theiler_window < 0:
        raise ValueError("--theiler-window must be nonnegative.")
    if args.bootstrap_samples < 0:
        raise ValueError("--bootstrap-samples must be nonnegative.")
    if not (0.0 < args.confidence_level < 1.0):
        raise ValueError("--confidence-level must be between 0 and 1.")
    if args.sensitivity_min_points < 2:
        raise ValueError("--sensitivity-min-points must be at least 2.")
    fit_end = args.scale_count if args.fit_end is None else args.fit_end
    if not (0 <= args.fit_start < fit_end <= args.scale_count):
        raise ValueError("--fit-start and --fit-end must satisfy 0 <= start < end <= scale-count.")
    fit_slice = slice(args.fit_start, fit_end)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    points = henon_orbit(args.a, args.b, args.N, args.transient, args.x0, args.y0)
    orbit_path = args.output_dir / "orbit.csv"
    pd.DataFrame(points, columns=["x", "y"]).to_csv(orbit_path, index=False)

    plot_attractor(
        points,
        args.output_dir / "attractor.png",
        f"Henon attractor: a={args.a:g}, b={args.b:g}, N={args.N}",
        args.point_size,
    )

    exponents = lyapunov_exponents_qr(points, args.a, args.b)
    ky_dimension = kaplan_yorke_dimension(exponents)

    scales = geometric_scales(points, args.scale_count, args.min_scale_fraction, args.max_scale_fraction)

    box_df = box_counts(points, scales)
    box_df["log_inverse_epsilon"] = np.log(1.0 / box_df["epsilon"])
    box_df["log_occupied_boxes"] = np.log(box_df["occupied_boxes"])
    box_df["used_in_fit"] = False
    box_df.loc[box_df.index[fit_slice], "used_in_fit"] = True
    box_fit = regression_fit(
        box_df.loc[box_df["used_in_fit"], "log_inverse_epsilon"].to_numpy(),
        box_df.loc[box_df["used_in_fit"], "log_occupied_boxes"].to_numpy(),
        "box_counting",
        "log(1/epsilon)",
        "log(N(epsilon))",
    )
    box_uncertainty = fit_uncertainty(
        box_df.loc[box_df["used_in_fit"], "log_inverse_epsilon"].to_numpy(),
        box_df.loc[box_df["used_in_fit"], "log_occupied_boxes"].to_numpy(),
        box_df["log_inverse_epsilon"].to_numpy(),
        box_df["log_occupied_boxes"].to_numpy(),
        args.bootstrap_samples,
        args.confidence_level,
        args.seed + 10,
        args.sensitivity_min_points,
    )
    save_fit_plot_data(box_df, args.output_dir / "box_counting_data.csv")
    plot_fit(
        box_df["log_inverse_epsilon"].to_numpy(),
        box_df["log_occupied_boxes"].to_numpy(),
        box_fit,
        args.output_dir / "box_counting_fit.png",
        "Box-counting dimension fit",
        "log(1/epsilon)",
        "log occupied boxes",
        box_df["used_in_fit"].to_numpy(),
    )

    distances, correlation_sample_size, correlation_pairs_used = sampled_pairwise_distances(
        points,
        args.max_correlation_points,
        args.seed,
        args.theiler_window,
    )
    corr_df = correlation_sums(distances, scales)
    corr_df["log_radius"] = np.log(corr_df["radius"])
    corr_df["log_correlation_sum"] = np.log(corr_df["correlation_sum"].replace(0.0, np.nan))
    corr_df["used_in_fit"] = False
    corr_df.loc[corr_df.index[fit_slice], "used_in_fit"] = True
    corr_fit = regression_fit(
        corr_df.loc[corr_df["used_in_fit"], "log_radius"].to_numpy(),
        corr_df.loc[corr_df["used_in_fit"], "log_correlation_sum"].to_numpy(),
        "correlation",
        "log(radius)",
        "log(C(radius))",
    )
    corr_uncertainty = fit_uncertainty(
        corr_df.loc[corr_df["used_in_fit"], "log_radius"].to_numpy(),
        corr_df.loc[corr_df["used_in_fit"], "log_correlation_sum"].to_numpy(),
        corr_df["log_radius"].to_numpy(),
        corr_df["log_correlation_sum"].to_numpy(),
        args.bootstrap_samples,
        args.confidence_level,
        args.seed + 20,
        args.sensitivity_min_points,
    )
    corr_df["sample_size"] = correlation_sample_size
    corr_df["pairs_used"] = correlation_pairs_used
    corr_df["theiler_window"] = args.theiler_window
    save_fit_plot_data(corr_df, args.output_dir / "correlation_dimension_data.csv")
    plot_fit(
        corr_df["log_radius"].to_numpy(),
        corr_df["log_correlation_sum"].to_numpy(),
        corr_fit,
        args.output_dir / "correlation_dimension_fit.png",
        "Correlation dimension fit",
        "log radius",
        "log correlation sum",
        corr_df["used_in_fit"].to_numpy(),
    )

    info_df = information_sums(points, scales)
    info_df["log_inverse_epsilon"] = np.log(1.0 / info_df["epsilon"])
    info_df["entropy"] = info_df["entropy"].replace(0.0, np.nan)
    info_df["used_in_fit"] = False
    info_df.loc[info_df.index[fit_slice], "used_in_fit"] = True
    info_fit = regression_fit(
        info_df.loc[info_df["used_in_fit"], "log_inverse_epsilon"].to_numpy(),
        info_df.loc[info_df["used_in_fit"], "entropy"].to_numpy(),
        "information",
        "log(1/epsilon)",
        "entropy",
    )
    info_uncertainty = fit_uncertainty(
        info_df.loc[info_df["used_in_fit"], "log_inverse_epsilon"].to_numpy(),
        info_df.loc[info_df["used_in_fit"], "entropy"].to_numpy(),
        info_df["log_inverse_epsilon"].to_numpy(),
        info_df["entropy"].to_numpy(),
        args.bootstrap_samples,
        args.confidence_level,
        args.seed + 30,
        args.sensitivity_min_points,
    )
    save_fit_plot_data(info_df, args.output_dir / "information_dimension_data.csv")
    plot_fit(
        info_df["log_inverse_epsilon"].to_numpy(),
        info_df["entropy"].to_numpy(),
        info_fit,
        args.output_dir / "information_dimension_fit.png",
        "Information dimension fit",
        "log(1/epsilon)",
        "box entropy",
        info_df["used_in_fit"].to_numpy(),
    )

    lyapunov_df = pd.DataFrame(
        {
            "lambda_1": [exponents[0]],
            "lambda_2": [exponents[1]],
            "kaplan_yorke_dimension": [ky_dimension],
        }
    )
    lyapunov_df.to_csv(args.output_dir / "lyapunov_and_ky.csv", index=False)

    def uncertainty_columns(uncertainty: FitUncertainty | None) -> dict[str, float | int]:
        if uncertainty is None:
            return {
                "confidence_level": np.nan,
                "bootstrap_samples": 0,
                "bootstrap_slope_low": np.nan,
                "bootstrap_slope_high": np.nan,
                "sensitivity_min_points": np.nan,
                "sensitivity_window_count": 0,
                "sensitivity_slope_min": np.nan,
                "sensitivity_slope_max": np.nan,
            }
        return {
            "confidence_level": uncertainty.confidence_level,
            "bootstrap_samples": uncertainty.bootstrap_samples,
            "bootstrap_slope_low": uncertainty.bootstrap_slope_low,
            "bootstrap_slope_high": uncertainty.bootstrap_slope_high,
            "sensitivity_min_points": uncertainty.sensitivity_min_points,
            "sensitivity_window_count": uncertainty.sensitivity_window_count,
            "sensitivity_slope_min": uncertainty.sensitivity_slope_min,
            "sensitivity_slope_max": uncertainty.sensitivity_slope_max,
        }

    summary = pd.DataFrame(
        [
            {
                "quantity": "lyapunov_lambda_1",
                "estimate": exponents[0],
                "slope": np.nan,
                "intercept": np.nan,
                "r_squared": np.nan,
                "points_used": len(points),
                "notes": "Largest Lyapunov exponent from QR tangent iteration.",
                **uncertainty_columns(None),
            },
            {
                "quantity": "lyapunov_lambda_2",
                "estimate": exponents[1],
                "slope": np.nan,
                "intercept": np.nan,
                "r_squared": np.nan,
                "points_used": len(points),
                "notes": "Smallest Lyapunov exponent from QR tangent iteration.",
                **uncertainty_columns(None),
            },
            {
                "quantity": "kaplan_yorke_dimension",
                "estimate": ky_dimension,
                "slope": np.nan,
                "intercept": np.nan,
                "r_squared": np.nan,
                "points_used": len(exponents),
                "notes": "Computed from sorted Lyapunov exponents.",
                **uncertainty_columns(None),
            },
            {
                "quantity": "box_counting_dimension",
                "estimate": box_fit.slope,
                "slope": box_fit.slope,
                "intercept": box_fit.intercept,
                "r_squared": box_fit.r_squared,
                "points_used": box_fit.points_used,
                "notes": f"Finite-size slope of log occupied boxes versus log inverse epsilon, scales [{args.fit_start}, {fit_end}).",
                **uncertainty_columns(box_uncertainty),
            },
            {
                "quantity": "correlation_dimension",
                "estimate": corr_fit.slope,
                "slope": corr_fit.slope,
                "intercept": corr_fit.intercept,
                "r_squared": corr_fit.r_squared,
                "points_used": corr_fit.points_used,
                "notes": (
                    f"Finite-size slope of log C(r) versus log r using {correlation_sample_size} sampled points, "
                    f"{correlation_pairs_used} pairs, Theiler window {args.theiler_window}, "
                    f"scales [{args.fit_start}, {fit_end})."
                ),
                **uncertainty_columns(corr_uncertainty),
            },
            {
                "quantity": "information_dimension",
                "estimate": info_fit.slope,
                "slope": info_fit.slope,
                "intercept": info_fit.intercept,
                "r_squared": info_fit.r_squared,
                "points_used": info_fit.points_used,
                "notes": f"Finite-size slope of Shannon box entropy versus log inverse epsilon, scales [{args.fit_start}, {fit_end}).",
                **uncertainty_columns(info_uncertainty),
            },
        ]
    )
    summary.to_csv(args.output_dir / "summary.csv", index=False)

    print("Henon attractor analysis complete.")
    print(f"Output directory: {args.output_dir.resolve()}")
    print(f"Lyapunov exponents: lambda_1={exponents[0]:.6f}, lambda_2={exponents[1]:.6f}")
    print(f"Kaplan-Yorke dimension: {ky_dimension:.6f}")
    print(f"Box-counting dimension slope: {box_fit.slope:.6f}")
    print(f"Correlation dimension slope: {corr_fit.slope:.6f}")
    print(f"Information dimension slope: {info_fit.slope:.6f}")
    print(
        f"{args.confidence_level:.0%} bootstrap slope intervals: "
        f"box=[{box_uncertainty.bootstrap_slope_low:.6f}, {box_uncertainty.bootstrap_slope_high:.6f}], "
        f"correlation=[{corr_uncertainty.bootstrap_slope_low:.6f}, {corr_uncertainty.bootstrap_slope_high:.6f}], "
        f"information=[{info_uncertainty.bootstrap_slope_low:.6f}, {info_uncertainty.bootstrap_slope_high:.6f}]"
    )
    print(
        "Warning: dimension estimates depend strongly on N, transient length, scale range, "
        "sampling, and regression window. Inspect the log-log plots before interpreting slopes."
    )


if __name__ == "__main__":
    main()
