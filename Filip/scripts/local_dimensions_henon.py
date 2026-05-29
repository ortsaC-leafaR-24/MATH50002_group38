#!/usr/bin/env python3
"""
Estimate local dimensions for the empirical SRB measure of the Henon map.

For centres z_i sampled from a long orbit, this script estimates

    mu_N(B(z_i, r)) = #{z_k : ||z_k - z_i|| < r} / N

and fits the local slope of log(mu_N(B(z_i, r))) versus log(r).

Two estimators are available:
- fixed-radius: use one physical radius grid for every N. This is useful for
  finite-scale diagnostics, but it does not test the exact-dimensional limit
  unless the radius window is also moved toward zero.
- k-nearest-neighbor: use centre-dependent radii given by neighbor counts
  k ~= N^alpha with 0 < alpha < 1. Then k -> infinity while k/N -> 0, so the
  fitted mass window shrinks toward the local dimension scale as N grows.

Warnings:
- Finite sample effects are severe for empirical local dimensions.
- Too-small radii often give zero or one counts and produce unstable slopes.
- Too-large radii see global attractor geometry rather than local scaling.
- High R^2 only says the selected finite scaling window is close to linear; it
  does not prove true exact dimensionality.
- Results depend strongly on the chosen scaling region, orbit length, transient,
  centre sample, and noise level.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def sample_unit_disk(rng: np.random.Generator) -> tuple[float, float]:
    """Sample uniformly from the unit disk B(0, 1)."""
    radius = np.sqrt(rng.random())
    angle = 2.0 * np.pi * rng.random()
    return float(radius * np.cos(angle)), float(radius * np.sin(angle))


def henon_orbit(
    a: float,
    b: float,
    n: int,
    transient: int,
    x0: float,
    y0: float,
    sigma: float,
    seed: int,
    divergence_bound: float,
) -> np.ndarray:
    """Generate a deterministic or bounded-noise Henon orbit after transient removal."""
    rng = np.random.default_rng(seed)
    total = n + transient
    orbit = np.empty((n, 2), dtype=float)
    x, y = float(x0), float(y0)

    for i in range(total):
        with np.errstate(over="ignore", invalid="ignore"):
            x_next = 1.0 - a * x * x + y
            y_next = b * x
            if sigma > 0.0:
                xi, eta = sample_unit_disk(rng)
                x_next += sigma * xi
                y_next += sigma * eta
        x, y = x_next, y_next
        if (
            not np.isfinite(x)
            or not np.isfinite(y)
            or abs(x) > divergence_bound
            or abs(y) > divergence_bound
        ):
            raise FloatingPointError(
                f"Orbit diverged at total iterate {i + 1} for sigma={sigma:g}. "
                "For the additively forced Henon map, sufficiently large bounded kicks "
                "can leave the attracting region and then the quadratic term escapes."
            )
        if i >= transient:
            orbit[i - transient] = (x, y)

    return orbit


def linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, int]:
    """Return slope, intercept, R^2, and number of valid points for y = slope*x + intercept."""
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


def bootstrap_statistic_interval(
    values: np.ndarray,
    statistic,
    samples: int,
    confidence_level: float,
    seed: int,
) -> tuple[float, float]:
    """Bootstrap a scalar statistic over sampled centre estimates."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0 or samples <= 0:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)
    bootstrap_values = np.empty(samples, dtype=float)
    for i in range(samples):
        resampled = rng.choice(values, size=len(values), replace=True)
        bootstrap_values[i] = statistic(resampled)
    return percentile_interval(bootstrap_values, confidence_level)


def sample_centres(n: int, centres: int, seed: int) -> np.ndarray:
    """Sample orbit indices used as ball centres."""
    if centres <= 0:
        raise ValueError("--centres must be positive.")
    rng = np.random.default_rng(seed)
    size = min(centres, n)
    return np.sort(rng.choice(n, size=size, replace=False))


def fixed_radius_dimension_estimates(
    points: np.ndarray,
    centre_indices: np.ndarray,
    radii: np.ndarray,
    fit_start: int,
    fit_end: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Estimate local dimensions for sampled centres with cKDTree radius queries.

    Centres are orbit points, so each count includes the centre itself. We subtract
    one at every positive radius to estimate counts from the rest of the empirical
    orbit. The denominator remains N, corresponding to mu_N with self-counting
    removed only to avoid the artificial one-point floor at tiny radii.
    """
    tree = cKDTree(points)
    n = len(points)
    log_r = np.log(radii)
    rows = []
    curve_rows = []

    fit_slice = slice(fit_start, fit_end)
    for centre_number, index in enumerate(centre_indices):
        centre = points[index]
        raw_counts = np.array(
            [tree.query_ball_point(centre, radius, return_length=True) for radius in radii],
            dtype=float,
        )
        counts = np.maximum(raw_counts - 1.0, 0.0)
        masses = counts / n
        log_mass = np.full_like(masses, np.nan, dtype=float)
        positive = masses > 0.0
        log_mass[positive] = np.log(masses[positive])

        slope, intercept, r_squared, fit_points = linear_fit(log_r[fit_slice], log_mass[fit_slice])
        rows.append(
            {
                "centre_number": centre_number,
                "orbit_index": int(index),
                "x": centre[0],
                "y": centre[1],
                "local_dimension": slope,
                "intercept": intercept,
                "r_squared": r_squared,
                "fit_points": fit_points,
                "min_fit_count": float(np.nanmin(counts[fit_slice])) if len(counts[fit_slice]) else np.nan,
                "max_fit_count": float(np.nanmax(counts[fit_slice])) if len(counts[fit_slice]) else np.nan,
                "estimator": "fixed_radius",
                "fit_min_radius": float(radii[fit_start]),
                "fit_max_radius": float(radii[fit_end - 1]),
                "fit_min_mass": float(np.nanmin(masses[fit_slice])) if len(masses[fit_slice]) else np.nan,
                "fit_max_mass": float(np.nanmax(masses[fit_slice])) if len(masses[fit_slice]) else np.nan,
            }
        )

        for radius, count, mass, lm in zip(radii, counts, masses, log_mass, strict=True):
            curve_rows.append(
                {
                    "centre_number": centre_number,
                    "orbit_index": int(index),
                    "radius": radius,
                    "count_excluding_self": int(count),
                    "mass": mass,
                    "log_radius": np.log(radius),
                    "log_mass": lm,
                    "estimator": "fixed_radius",
                }
            )

    return pd.DataFrame(rows), pd.DataFrame(curve_rows)


def neighbor_count_grid(n: int, min_exponent: float, max_exponent: float, count: int) -> np.ndarray:
    """Return logarithmically spaced k values with k -> infinity and k/N -> 0."""
    k_min = max(2, int(round(n**min_exponent)))
    k_max = max(k_min + 1, int(round(n**max_exponent)))
    k_max = min(k_max, n - 1)
    if k_max <= k_min:
        raise ValueError("The requested kNN count window is too small for this N.")
    values = np.unique(np.geomspace(k_min, k_max, count).astype(int))
    if len(values) < 2:
        values = np.array([k_min, k_max], dtype=int)
    return values


def knn_dimension_estimates(
    points: np.ndarray,
    centre_indices: np.ndarray,
    neighbor_counts: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Estimate local dimensions from k-nearest-neighbor radii.

    For each centre, the kth-neighbor distance r_k satisfies
    mu_N(B(x, r_k)) ~= k/N. Fitting log(k/N) against log(r_k) gives a local
    slope over a mass window that can shrink with N by choosing k = N^alpha,
    0 < alpha < 1.
    """
    tree = cKDTree(points)
    n = len(points)
    centres = points[centre_indices]
    query_counts = neighbor_counts + 1  # include the centre itself as nearest neighbor.
    distances, _ = tree.query(centres, k=query_counts, workers=-1)
    if distances.ndim == 1:
        distances = distances[:, np.newaxis]

    masses = neighbor_counts / n
    log_mass = np.log(masses)
    rows = []
    curve_rows = []

    for centre_number, (index, centre, centre_distances) in enumerate(
        zip(centre_indices, centres, distances, strict=True)
    ):
        valid = np.isfinite(centre_distances) & (centre_distances > 0.0)
        slope, intercept, r_squared, fit_points = linear_fit(np.log(centre_distances[valid]), log_mass[valid])

        valid_distances = centre_distances[valid]
        rows.append(
            {
                "centre_number": centre_number,
                "orbit_index": int(index),
                "x": centre[0],
                "y": centre[1],
                "local_dimension": slope,
                "intercept": intercept,
                "r_squared": r_squared,
                "fit_points": fit_points,
                "min_fit_count": int(neighbor_counts[valid][0]) if np.any(valid) else np.nan,
                "max_fit_count": int(neighbor_counts[valid][-1]) if np.any(valid) else np.nan,
                "estimator": "knn",
                "fit_min_radius": float(np.min(valid_distances)) if len(valid_distances) else np.nan,
                "fit_max_radius": float(np.max(valid_distances)) if len(valid_distances) else np.nan,
                "fit_min_mass": float(np.min(masses[valid])) if np.any(valid) else np.nan,
                "fit_max_mass": float(np.max(masses[valid])) if np.any(valid) else np.nan,
            }
        )

        for k, radius, mass in zip(neighbor_counts, centre_distances, masses, strict=True):
            positive = np.isfinite(radius) and radius > 0.0
            curve_rows.append(
                {
                    "centre_number": centre_number,
                    "orbit_index": int(index),
                    "radius": radius,
                    "count_excluding_self": int(k),
                    "mass": mass,
                    "log_radius": np.log(radius) if positive else np.nan,
                    "log_mass": np.log(mass),
                    "estimator": "knn",
                }
            )

    return pd.DataFrame(rows), pd.DataFrame(curve_rows)


def summary_statistics(
    estimates: pd.DataFrame,
    n: int,
    sigma: float,
    estimator: str,
    bootstrap_samples: int,
    confidence_level: float,
    seed: int,
) -> pd.DataFrame:
    """Summarize valid local dimension estimates, including robust spread diagnostics."""
    values = estimates["local_dimension"].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    if len(values) == 0:
        return pd.DataFrame(
            [
                {
                    "N": n,
                    "sigma": sigma,
                    "estimator": estimator,
                    "successful_centres": 0,
                    "mean": np.nan,
                    "median": np.nan,
                    "std": np.nan,
                    "q25": np.nan,
                    "q75": np.nan,
                    "iqr": np.nan,
                    "trimmed_min": np.nan,
                    "trimmed_max": np.nan,
                    "standard_error": np.nan,
                    "confidence_level": confidence_level,
                    "bootstrap_samples": bootstrap_samples,
                    "mean_ci_low": np.nan,
                    "mean_ci_high": np.nan,
                    "median_ci_low": np.nan,
                    "median_ci_high": np.nan,
                }
            ]
        )

    q25, q75 = np.percentile(values, [25, 75])
    iqr = q75 - q25
    lower = q25 - 1.5 * iqr
    upper = q75 + 1.5 * iqr
    trimmed = values[(values >= lower) & (values <= upper)]
    if len(trimmed) == 0:
        trimmed = values
    mean_ci_low, mean_ci_high = bootstrap_statistic_interval(
        values,
        np.mean,
        bootstrap_samples,
        confidence_level,
        seed,
    )
    median_ci_low, median_ci_high = bootstrap_statistic_interval(
        values,
        np.median,
        bootstrap_samples,
        confidence_level,
        seed + 1,
    )
    standard_error = float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0

    return pd.DataFrame(
        [
            {
                "N": n,
                "sigma": sigma,
                "estimator": estimator,
                "successful_centres": len(values),
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "q25": float(q25),
                "q75": float(q75),
                "iqr": float(iqr),
                "trimmed_min": float(np.min(trimmed)),
                "trimmed_max": float(np.max(trimmed)),
                "standard_error": standard_error,
                "confidence_level": confidence_level,
                "bootstrap_samples": bootstrap_samples,
                "mean_ci_low": mean_ci_low,
                "mean_ci_high": mean_ci_high,
                "median_ci_low": median_ci_low,
                "median_ci_high": median_ci_high,
            }
        ]
    )


def plot_attractor_with_centres(points: np.ndarray, centres: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    ax.scatter(points[:, 0], points[:, 1], s=0.08, alpha=0.35, linewidths=0, label="orbit")
    ax.scatter(centres[:, 0], centres[:, 1], s=9, c="crimson", alpha=0.8, linewidths=0, label="centres")
    ax.set_title("Henon attractor with sampled centres")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper right", markerscale=2)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_histogram(estimates: pd.DataFrame, path: Path) -> None:
    values = estimates["local_dimension"].replace([np.inf, -np.inf], np.nan).dropna()
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    ax.hist(values, bins=40, color="#3b6ea8", edgecolor="white", alpha=0.9)
    ax.set_title("Local dimension estimates")
    ax.set_xlabel("estimated local dimension")
    ax.set_ylabel("centre count")
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_boxplot(estimates: pd.DataFrame, path: Path) -> None:
    values = estimates["local_dimension"].replace([np.inf, -np.inf], np.nan).dropna()
    fig, ax = plt.subplots(figsize=(5, 6), constrained_layout=True)
    ax.boxplot(values, vert=True, showmeans=True)
    ax.set_title("Local dimension spread")
    ax.set_ylabel("estimated local dimension")
    ax.set_xticks([1], ["centres"])
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_representative_curves(curves: pd.DataFrame, estimates: pd.DataFrame, path: Path, count: int = 8) -> None:
    valid = estimates.replace([np.inf, -np.inf], np.nan).dropna(subset=["local_dimension"])
    if valid.empty:
        return

    quantiles = np.linspace(0.05, 0.95, min(count, len(valid)))
    targets = np.quantile(valid["local_dimension"], quantiles)
    chosen = []
    used = set()
    for target in targets:
        distances = np.abs(valid["local_dimension"] - target)
        for idx in distances.sort_values().index:
            centre_number = int(valid.loc[idx, "centre_number"])
            if centre_number not in used:
                chosen.append(centre_number)
                used.add(centre_number)
                break

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    for centre_number in chosen:
        centre_curve = curves[curves["centre_number"] == centre_number]
        dimension = float(valid.loc[valid["centre_number"] == centre_number, "local_dimension"].iloc[0])
        ax.plot(
            centre_curve["log_radius"],
            centre_curve["log_mass"],
            marker="o",
            markersize=3,
            linewidth=1,
            label=f"centre {centre_number}, d={dimension:.2f}",
        )
    ax.set_title("Representative local scaling curves")
    ax.set_xlabel("log radius")
    ax.set_ylabel("log mu_N(B(x,r))")
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_coloured_attractor(points: np.ndarray, estimates: pd.DataFrame, path: Path) -> None:
    valid = estimates.replace([np.inf, -np.inf], np.nan).dropna(subset=["local_dimension"])
    if valid.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    ax.scatter(points[:, 0], points[:, 1], s=0.06, color="0.75", alpha=0.35, linewidths=0)
    sc = ax.scatter(
        valid["x"],
        valid["y"],
        c=valid["local_dimension"],
        s=12,
        cmap="viridis",
        linewidths=0,
    )
    ax.set_title("Local dimension estimates on attractor")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(sc, ax=ax, label="estimated local dimension")
    fig.savefig(path, dpi=220)
    plt.close(fig)


def sigma_label(sigma: float) -> str:
    """Return a filesystem-friendly label for a noise level."""
    return f"sigma_{sigma:.12g}".replace("+", "").replace("-", "m").replace(".", "p")


def plot_dimension_summary(summary: pd.DataFrame, path: Path) -> None:
    if summary.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    group_keys = ["sigma"]
    if "estimator" in summary.columns:
        group_keys.append("estimator")
    for keys, group in summary.groupby(group_keys, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        sigma = keys[0]
        estimator = keys[1] if len(keys) > 1 else ""
        group = group.sort_values("N")
        label = f"sigma={sigma:g}" if not estimator else f"{estimator}, sigma={sigma:g}"
        axes[0].plot(group["N"], group["mean"], marker="o", label=f"mean, {label}")
        axes[0].plot(group["N"], group["median"], marker="s", linestyle="--", label=f"median, {label}")
        axes[1].plot(group["N"], group["std"], marker="o", label=f"std, {label}")
        axes[1].plot(group["N"], group["iqr"], marker="s", linestyle="--", label=f"IQR, {label}")

    axes[0].set_xscale("log")
    axes[0].set_title("Location versus orbit length")
    axes[0].set_xlabel("N")
    axes[0].set_ylabel("local dimension")
    axes[0].legend(fontsize=7)

    axes[1].set_xscale("log")
    axes[1].set_title("Spread versus orbit length")
    axes[1].set_xlabel("N")
    axes[1].set_ylabel("spread")
    axes[1].legend(fontsize=7)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_sigma_comparison(summary: pd.DataFrame, path: Path) -> None:
    if summary.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    group_keys = ["N"]
    if "estimator" in summary.columns:
        group_keys.append("estimator")
    for keys, group in summary.groupby(group_keys, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        n = keys[0]
        estimator = keys[1] if len(keys) > 1 else ""
        group = group.sort_values("sigma")
        label = f"N={n:g}" if not estimator else f"{estimator}, N={n:g}"
        axes[0].plot(group["sigma"], group["mean"], marker="o", label=f"mean, {label}")
        axes[0].plot(group["sigma"], group["median"], marker="s", linestyle="--", label=f"median, {label}")
        axes[1].plot(group["sigma"], group["std"], marker="o", label=f"std, {label}")
        axes[1].plot(group["sigma"], group["iqr"], marker="s", linestyle="--", label=f"IQR, {label}")

    for ax in axes:
        positive = summary["sigma"] > 0.0
        if positive.any():
            ax.set_xscale("symlog", linthresh=float(summary.loc[positive, "sigma"].min()))
        ax.set_xlabel("sigma")
        ax.legend(fontsize=7)

    axes[0].set_title("Local dimension versus noise")
    axes[0].set_ylabel("local dimension")
    axes[1].set_title("Spread versus noise")
    axes[1].set_ylabel("spread")
    fig.savefig(path, dpi=220)
    plt.close(fig)


def validate_args(args: argparse.Namespace) -> None:
    if any(n <= 0 for n in args.N) or args.transient < 0:
        raise ValueError("All --N values must be positive and --transient must be nonnegative.")
    if any(sigma < 0.0 for sigma in args.sigma):
        raise ValueError("All --sigma values must be nonnegative.")
    if args.estimator == "fixed-radius":
        if args.r_min <= 0.0 or args.r_max <= args.r_min:
            raise ValueError("Radii must satisfy 0 < --r-min < --r-max.")
        if args.num_radii < 3:
            raise ValueError("--num-radii must be at least 3.")
        if not (0 <= args.fit_start < args.fit_end <= args.num_radii):
            raise ValueError("--fit-start and --fit-end must satisfy 0 <= start < end <= num-radii.")
    if args.divergence_bound <= 0.0:
        raise ValueError("--divergence-bound must be positive.")
    if args.num_neighbors < 2:
        raise ValueError("--num-neighbors must be at least 2.")
    if not (0.0 < args.knn_min_exponent < args.knn_max_exponent < 1.0):
        raise ValueError("--knn-min-exponent and --knn-max-exponent must satisfy 0 < min < max < 1.")
    if args.bootstrap_samples < 0:
        raise ValueError("--bootstrap-samples must be nonnegative.")
    if not (0.0 < args.confidence_level < 1.0):
        raise ValueError("--confidence-level must be between 0 and 1.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate local dimensions of the Henon empirical SRB measure.")
    parser.add_argument("--a", type=float, default=1.4, help="Henon parameter a.")
    parser.add_argument("--b", type=float, default=0.3, help="Henon parameter b.")
    parser.add_argument(
        "--N",
        type=int,
        nargs="+",
        default=[100_000],
        help="One or more retained orbit lengths, e.g. --N 20000 50000 100000.",
    )
    parser.add_argument("--transient", type=int, default=10_000, help="Number of initial iterates to discard.")
    parser.add_argument("--centres", type=int, default=500, help="Number of orbit points used as local centres.")
    parser.add_argument("--r-min", type=float, default=1e-3, help="Smallest radius for ball counts.")
    parser.add_argument("--r-max", type=float, default=5e-1, help="Largest radius for ball counts.")
    parser.add_argument("--num-radii", type=int, default=24, help="Number of logarithmically spaced radii.")
    parser.add_argument("--fit-start", type=int, default=4, help="First radius index used in slope fit, inclusive.")
    parser.add_argument("--fit-end", type=int, default=18, help="Last radius index used in slope fit, exclusive.")
    parser.add_argument(
        "--estimator",
        choices=["fixed-radius", "knn"],
        default="fixed-radius",
        help=(
            "Local-dimension estimator. fixed-radius reuses the same physical radii for every N; "
            "knn fits kth-neighbor radii with k between N^alpha values so the mass window shrinks as N grows."
        ),
    )
    parser.add_argument(
        "--knn-min-exponent",
        type=float,
        default=0.25,
        help="Smallest kth-neighbor count is approximately N^this exponent for --estimator knn.",
    )
    parser.add_argument(
        "--knn-max-exponent",
        type=float,
        default=0.70,
        help="Largest kth-neighbor count is approximately N^this exponent for --estimator knn.",
    )
    parser.add_argument(
        "--num-neighbors",
        type=int,
        default=18,
        help="Number of logarithmically spaced kth-neighbor counts for --estimator knn.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=1000,
        help="Bootstrap resamples for confidence intervals over sampled centre estimates. Use 0 to disable.",
    )
    parser.add_argument(
        "--confidence-level",
        type=float,
        default=0.95,
        help="Central confidence level for bootstrap intervals.",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        nargs="+",
        default=[0.0],
        help=(
            "One or more bounded additive noise strengths. Each noise vector is sampled "
            "uniformly from sigma * B(0,1), e.g. --sigma 0 1e-4 5e-4 1e-3."
        ),
    )
    parser.add_argument("--seed", type=int, default=12345, help="Random seed for noise and centre sampling.")
    parser.add_argument("--outdir", type=Path, default=Path("henon_local_dimensions"), help="Output directory.")
    parser.add_argument("--x0", type=float, default=0.1, help="Initial x coordinate.")
    parser.add_argument("--y0", type=float, default=0.1, help="Initial y coordinate.")
    parser.add_argument(
        "--divergence-bound",
        type=float,
        default=1e12,
        help="Stop a noisy orbit if |x| or |y| exceeds this bound.",
    )
    return parser.parse_args()


def run_single_analysis(
    points: np.ndarray,
    centres: int,
    radii: np.ndarray | None,
    fit_start: int,
    fit_end: int,
    seed: int,
    sigma: float,
    estimator: str,
    knn_min_exponent: float,
    knn_max_exponent: float,
    num_neighbors: int,
    bootstrap_samples: int,
    confidence_level: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray]:
    centre_indices = sample_centres(len(points), centres, seed)
    summary_seed = seed + 50_000
    if estimator == "knn":
        k_values = neighbor_count_grid(len(points), knn_min_exponent, knn_max_exponent, num_neighbors)
        estimates, curves = knn_dimension_estimates(points, centre_indices, k_values)
        summary = summary_statistics(
            estimates,
            len(points),
            sigma,
            "knn",
            bootstrap_samples,
            confidence_level,
            summary_seed,
        )
        summary["knn_min_count"] = int(k_values[0])
        summary["knn_max_count"] = int(k_values[-1])
        summary["knn_min_mass"] = float(k_values[0] / len(points))
        summary["knn_max_mass"] = float(k_values[-1] / len(points))
        summary["knn_min_exponent"] = knn_min_exponent
        summary["knn_max_exponent"] = knn_max_exponent
    else:
        if radii is None:
            raise ValueError("Fixed-radius estimator requires a radius grid.")
        estimates, curves = fixed_radius_dimension_estimates(points, centre_indices, radii, fit_start, fit_end)
        summary = summary_statistics(
            estimates,
            len(points),
            sigma,
            "fixed_radius",
            bootstrap_samples,
            confidence_level,
            summary_seed,
        )
        summary["fit_min_radius"] = float(radii[fit_start])
        summary["fit_max_radius"] = float(radii[fit_end - 1])
    return estimates, curves, summary, centre_indices


def write_analysis_outputs(
    points: np.ndarray,
    estimates: pd.DataFrame,
    curves: pd.DataFrame,
    summary: pd.DataFrame,
    centre_indices: np.ndarray,
    outdir: Path,
) -> None:
    """Save all data and plots for one orbit length in a consistent directory."""
    outdir.mkdir(parents=True, exist_ok=True)
    estimates.to_csv(outdir / "local_dimension_estimates.csv", index=False)
    curves.to_csv(outdir / "local_dimension_curves.csv", index=False)
    summary.to_csv(outdir / "local_dimension_summary.csv", index=False)

    centre_points = points[centre_indices]
    plot_attractor_with_centres(points, centre_points, outdir / "attractor_with_centres.png")
    plot_histogram(estimates, outdir / "local_dimension_histogram.png")
    plot_boxplot(estimates, outdir / "local_dimension_boxplot.png")
    plot_representative_curves(curves, estimates, outdir / "representative_scaling_curves.png")
    plot_coloured_attractor(points, estimates, outdir / "local_dimensions_on_attractor.png")


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.outdir.mkdir(parents=True, exist_ok=True)

    orbit_lengths = sorted(set(args.N))
    sigmas = sorted(set(args.sigma))
    max_n = max(orbit_lengths)
    radii = None
    if args.estimator == "fixed-radius":
        radii = np.geomspace(args.r_min, args.r_max, args.num_radii)

    summary_rows = []
    failed_rows = []
    for sigma_index, sigma in enumerate(sigmas):
        # One long trajectory per sigma is generated and prefixes are reused across requested N.
        try:
            points_max = henon_orbit(
                args.a,
                args.b,
                max_n,
                args.transient,
                args.x0,
                args.y0,
                sigma,
                args.seed + 10_000 * sigma_index,
                args.divergence_bound,
            )
        except FloatingPointError as exc:
            message = str(exc)
            print(f"Skipping sigma={sigma:g}: {message}")
            for n in orbit_lengths:
                failed_rows.append(
                    {
                        "N": n,
                        "sigma": sigma,
                        "status": "failed_orbit_diverged",
                        "message": message,
                    }
                )
            continue

        sigma_dir = args.outdir / sigma_label(sigma)

        for n in orbit_lengths:
            estimates, curves, summary, centre_indices = run_single_analysis(
                points_max[:n],
                args.centres,
                radii,
                args.fit_start,
                args.fit_end,
                args.seed + 1000 + n + 10_000 * sigma_index,
                sigma,
                args.estimator,
                args.knn_min_exponent,
                args.knn_max_exponent,
                args.num_neighbors,
                args.bootstrap_samples,
                args.confidence_level,
            )
            summary["requested_centres"] = min(args.centres, n)
            summary_rows.append(summary)
            write_analysis_outputs(points_max[:n], estimates, curves, summary, centre_indices, sigma_dir / f"N_{n}")

    if failed_rows:
        pd.DataFrame(failed_rows).to_csv(args.outdir / "failed_runs.csv", index=False)

    if not summary_rows:
        raise RuntimeError(
            "All requested sigma values diverged before analysis. Try smaller sigma values, a shorter N, "
            "or a different random seed."
        )

    combined_summary = pd.concat(summary_rows, ignore_index=True)
    combined_summary.to_csv(args.outdir / "local_dimension_summary.csv", index=False)
    plot_dimension_summary(combined_summary, args.outdir / "local_dimension_summary_by_N.png")
    plot_sigma_comparison(combined_summary, args.outdir / "local_dimension_summary_by_sigma.png")

    row = combined_summary.iloc[-1]
    print("Henon local-dimension analysis complete.")
    print(f"Output directory: {args.outdir.resolve()}")
    print(
        "Final run in summary: "
        f"N={int(row['N'])}, centres={args.centres}, sigma={row['sigma']:g}, "
        f"estimator={row['estimator']}, successful_centres={int(row['successful_centres'])}"
    )
    print(
        "Local dimension summary: "
        f"mean={row['mean']:.6f}, median={row['median']:.6f}, std={row['std']:.6f}, "
        f"IQR={row['iqr']:.6f}, trimmed_min={row['trimmed_min']:.6f}, "
        f"trimmed_max={row['trimmed_max']:.6f}"
    )
    print(
        f"{row['confidence_level']:.0%} bootstrap intervals: "
        f"mean=[{row['mean_ci_low']:.6f}, {row['mean_ci_high']:.6f}], "
        f"median=[{row['median_ci_low']:.6f}, {row['median_ci_high']:.6f}]"
    )
    print("Per-run data and plots were saved under sigma_<value>/N_<value>/ directories.")
    print(
        "Combined comparisons saved to local_dimension_summary.csv, "
        "local_dimension_summary_by_N.png, and local_dimension_summary_by_sigma.png."
    )
    print(
        "Interpretation: fixed-radius runs diagnose finite-scale variation. For exact-dimensionality "
        "checks across N, use --estimator knn or otherwise shrink the fitted radius window as N grows."
    )


if __name__ == "__main__":
    main()
