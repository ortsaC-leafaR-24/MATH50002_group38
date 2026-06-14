from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# Step 3: finite-time stable--unstable angle diagnostics
# ============================================================
# Folder structure assumed:
#
# project/
# ├── Henon_orbit/
# │   └── henon_orbit.csv
# │
# └── angles/
#     └── angles.py
#
# This script reads the orbit from Henon_orbit/ and saves all Step 3
# outputs inside angles/.
#
# Important interpretation:
# The computed directions are finite-time approximations obtained from
# derivative propagation. They are not assumed to be true globally defined
# stable/unstable subspaces of a uniformly hyperbolic attractor.
# ============================================================


# ============================================================
# Path setup
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

ORBIT_PATH = PROJECT_DIR / "Henon_orbit" / "henon_orbit.csv"

ALL_VALUES_PATH = BASE_DIR / "angle_values_all_N.csv"
SUMMARY_CSV_PATH = BASE_DIR / "angle_summary_by_N.csv"
SUMMARY_TXT_PATH = BASE_DIR / "angle_summary.txt"

HIST_COMPARE_PATH = BASE_DIR / "angle_hist_compare_N.png"
LOWER_TAIL_PATH = BASE_DIR / "angle_lower_tail_cdf.png"
SUMMARY_PLOT_PATH = BASE_DIR / "angle_summary_vs_N.png"
ROBUSTNESS_PATH = BASE_DIR / "angle_robustness_check.csv"

SMALL_ANGLE_POINTS_PATH = BASE_DIR / "small_angle_points_N100.csv"
SMALL_ANGLE_GAPS_PATH = BASE_DIR / "small_angle_iteration_gaps_N100.csv"
SMALL_ANGLE_SPATIAL_PATH = BASE_DIR / "small_and_very_small_angle_points_N100.png"
# ============================================================
# Hénon map in Robinson notation
# F_{A,B}(x,y) = (A - B y - x^2, x)
# Classical parameters: A = 1.4, B = -0.3
# ============================================================

A_PARAM = 1.4
B_PARAM = -0.3


def DF(z: np.ndarray) -> np.ndarray:
    """
    Jacobian of F_{A,B}(x,y) = (A - B y - x^2, x).
    """
    x, y = z
    return np.array(
        [
            [-2.0 * x, -B_PARAM],
            [1.0, 0.0],
        ],
        dtype=float,
    )


def normalize(v: np.ndarray) -> np.ndarray:
    """
    Return v / ||v||.
    """
    norm_v = np.linalg.norm(v)
    if norm_v == 0:
        raise ValueError("Zero vector encountered during propagation.")
    return v / norm_v


def angle_between_lines(u: np.ndarray, s: np.ndarray) -> tuple[float, float]:
    """
    Angle between the unoriented lines spanned by u and s.

    Since the directions are lines rather than oriented vectors, we use
    |<u,s>|. The angle is therefore in [0, pi/2].

    Returns:
        theta_rad, theta_deg
    """
    u = normalize(u)
    s = normalize(s)

    dot = abs(float(np.dot(u, s)))
    dot = np.clip(dot, 0.0, 1.0)

    theta_rad = np.arccos(dot)
    theta_deg = np.degrees(theta_rad)

    return theta_rad, theta_deg


# ============================================================
# Numerical settings
# ============================================================

# Use small and moderate propagation lengths to show convergence of the
# finite-time direction approximation.
N_PROP_VALUES = [1, 2, 5, 10, 20, 50, 100]

SAMPLE_SIZE = 5000

# Representative value for lower-tail plot and robustness check.
REPRESENTATIVE_N = 100
ROBUSTNESS_N = 100

# Threshold used to detect finite-time near-tangencies.
SMALL_ANGLE_THRESHOLD_DEG = 5.0
VERY_SMALL_ANGLE_THRESHOLD_DEG = 1.0

# Initial tangent vectors for the main run.
# Bad choices would lie exactly in an exceptional direction. Generic vectors
# avoid this except for a zero-probability choice.
U_INIT_MAIN = normalize(np.array([1.0, 0.3], dtype=float))
S_INIT_MAIN = normalize(np.array([0.7, 1.0], dtype=float))

# Additional generic vector pairs for robustness check.
ROBUSTNESS_VECTOR_PAIRS = [
    (
        normalize(np.array([1.0, 0.3], dtype=float)),
        normalize(np.array([0.7, 1.0], dtype=float)),
    ),
    (
        normalize(np.array([0.4, 1.0], dtype=float)),
        normalize(np.array([1.0, -0.2], dtype=float)),
    ),
    (
        normalize(np.array([-0.8, 0.6], dtype=float)),
        normalize(np.array([0.3, -1.0], dtype=float)),
    ),
]


# ============================================================
# Load orbit
# ============================================================

if not ORBIT_PATH.exists():
    raise FileNotFoundError(
        f"Orbit file not found:\n{ORBIT_PATH}\n"
        "Check that Henon_orbit/henon_orbit.csv exists."
    )

df_orbit = pd.read_csv(ORBIT_PATH)
orbit = df_orbit[["x", "y"]].to_numpy(dtype=float)

N_ORBIT = len(orbit)

# Need enough space for all propagation lengths, including robustness check.
max_N = max(max(N_PROP_VALUES), REPRESENTATIVE_N, ROBUSTNESS_N)

if N_ORBIT <= 2 * max_N + 1:
    raise ValueError("Orbit is too short for the largest propagation length.")

# Need both z_{i-N} and z_{i+N}, so sample away from endpoints.
sample_indices = np.linspace(
    max_N,
    N_ORBIT - max_N - 1,
    SAMPLE_SIZE,
    dtype=int,
)


# ============================================================
# Direction propagation
# ============================================================

def unstable_direction_at_index(i: int, N: int, u_init: np.ndarray) -> np.ndarray:
    """
    Approximate the finite-time unstable direction at z_i.

    Method:
        propagate a generic vector forward from z_{i-N} to z_i.
    """
    u = u_init.copy()

    for j in range(i - N, i):
        u = DF(orbit[j]) @ u
        u = normalize(u)

    return u


def stable_direction_at_index(i: int, N: int, s_init: np.ndarray) -> np.ndarray:
    """
    Approximate the finite-time stable direction at z_i.

    Method:
        propagate a generic vector backward from z_{i+N} to z_i.

    Numerically, solve
        DF(z_{j-1}) s_{j-1} = s_j
    instead of explicitly forming the inverse matrix.
    """
    s = s_init.copy()

    for j in range(i + N, i, -1):
        s = np.linalg.solve(DF(orbit[j - 1]), s)
        s = normalize(s)

    return s


def finite_time_angle_at_index(
    i: int,
    N: int,
    u_init: np.ndarray,
    s_init: np.ndarray,
) -> tuple[float, float]:
    """
    Compute the finite-time angle at z_i.
    """
    u = unstable_direction_at_index(i, N, u_init)
    s = stable_direction_at_index(i, N, s_init)

    theta_rad, theta_deg = angle_between_lines(u, s)

    return theta_rad, theta_deg


# ============================================================
# Main angle computation
# ============================================================

all_rows = []
summary_rows = []

for N in N_PROP_VALUES:
    print(f"Computing finite-time angles for N = {N}")

    rows_N = []

    for i in sample_indices:
        theta_rad, theta_deg = finite_time_angle_at_index(
            i=i,
            N=N,
            u_init=U_INIT_MAIN,
            s_init=S_INIT_MAIN,
        )

        row = {
            "N": N,
            "i": int(i),
            "x": orbit[i, 0],
            "y": orbit[i, 1],
            "theta_rad": theta_rad,
            "theta_deg": theta_deg,
        }

        rows_N.append(row)
        all_rows.append(row)

    df_N = pd.DataFrame(rows_N)
    df_N.to_csv(BASE_DIR / f"angle_values_N{N}.csv", index=False)

    summary_rows.append(
        {
            "N": N,
            "mean_deg": df_N["theta_deg"].mean(),
            "std_deg": df_N["theta_deg"].std(),
            "min_deg": df_N["theta_deg"].min(),
            "q01_deg": df_N["theta_deg"].quantile(0.01),
            "q05_deg": df_N["theta_deg"].quantile(0.05),
            "median_deg": df_N["theta_deg"].median(),
            "q95_deg": df_N["theta_deg"].quantile(0.95),
            "max_deg": df_N["theta_deg"].max(),
            "fraction_below_1_deg": np.mean(df_N["theta_deg"] < 1.0),
            "fraction_below_2_deg": np.mean(df_N["theta_deg"] < 2.0),
            "fraction_below_5_deg": np.mean(df_N["theta_deg"] < 5.0),
        }
    )

df_all = pd.DataFrame(all_rows)
df_summary = pd.DataFrame(summary_rows)

df_all.to_csv(ALL_VALUES_PATH, index=False)
df_summary.to_csv(SUMMARY_CSV_PATH, index=False)

# ============================================================
# Small-angle test for representative N
# ============================================================

df_rep = df_all[df_all["N"] == REPRESENTATIVE_N].copy()

if df_rep.empty:
    raise ValueError(
        f"No data found for REPRESENTATIVE_N={REPRESENTATIVE_N}. "
        "Make sure REPRESENTATIVE_N is included in N_PROP_VALUES."
    )

df_small = df_rep[df_rep["theta_deg"] < SMALL_ANGLE_THRESHOLD_DEG].copy()
df_very_small = df_rep[df_rep["theta_deg"] < VERY_SMALL_ANGLE_THRESHOLD_DEG].copy()

df_small.to_csv(SMALL_ANGLE_POINTS_PATH, index=False)

# Iteration-gap statistics for small-angle events.
# This checks whether small-angle events occur in clusters along the sampled orbit.
small_indices = np.sort(df_small["i"].to_numpy(dtype=int))

if len(small_indices) >= 2:
    gaps = np.diff(small_indices)
    df_gaps = pd.DataFrame({"gap_between_small_angle_events": gaps})
else:
    gaps = np.array([])
    df_gaps = pd.DataFrame({"gap_between_small_angle_events": []})

df_gaps.to_csv(SMALL_ANGLE_GAPS_PATH, index=False)

small_angle_summary = {
    "representative_N": REPRESENTATIVE_N,
    "threshold_deg": SMALL_ANGLE_THRESHOLD_DEG,
    "very_small_threshold_deg": VERY_SMALL_ANGLE_THRESHOLD_DEG,
    "sample_size": len(df_rep),
    "small_angle_count": len(df_small),
    "small_angle_fraction": len(df_small) / len(df_rep),
    "very_small_angle_count": len(df_very_small),
    "very_small_angle_fraction": len(df_very_small) / len(df_rep),
    "min_angle_deg": df_rep["theta_deg"].min(),
    "q01_angle_deg": df_rep["theta_deg"].quantile(0.01),
    "q05_angle_deg": df_rep["theta_deg"].quantile(0.05),
    "median_angle_deg": df_rep["theta_deg"].median(),
    "mean_gap_between_small_angle_events": float(np.mean(gaps)) if len(gaps) > 0 else np.nan,
    "median_gap_between_small_angle_events": float(np.median(gaps)) if len(gaps) > 0 else np.nan,
    "min_gap_between_small_angle_events": int(np.min(gaps)) if len(gaps) > 0 else np.nan,
    "max_gap_between_small_angle_events": int(np.max(gaps)) if len(gaps) > 0 else np.nan,
}


# ============================================================
# Robustness check with different generic initial vectors
# ============================================================

robustness_rows = []

print(f"Running robustness check for N = {ROBUSTNESS_N}")

for pair_id, (u_init, s_init) in enumerate(ROBUSTNESS_VECTOR_PAIRS, start=1):
    theta_values = []

    for i in sample_indices:
        _, theta_deg = finite_time_angle_at_index(
            i=i,
            N=ROBUSTNESS_N,
            u_init=u_init,
            s_init=s_init,
        )
        theta_values.append(theta_deg)

    theta_values = np.array(theta_values)

    robustness_rows.append(
        {
            "pair_id": pair_id,
            "N": ROBUSTNESS_N,
            "mean_deg": theta_values.mean(),
            "std_deg": theta_values.std(ddof=1),
            "min_deg": theta_values.min(),
            "q01_deg": np.quantile(theta_values, 0.01),
            "q05_deg": np.quantile(theta_values, 0.05),
            "median_deg": np.median(theta_values),
            "q95_deg": np.quantile(theta_values, 0.95),
            "max_deg": theta_values.max(),
            "fraction_below_1_deg": np.mean(theta_values < 1.0),
            "fraction_below_2_deg": np.mean(theta_values < 2.0),
            "fraction_below_5_deg": np.mean(theta_values < 5.0),
        }
    )

df_robustness = pd.DataFrame(robustness_rows)
df_robustness.to_csv(ROBUSTNESS_PATH, index=False)


# ============================================================
# Figure 1: angle distributions for several N
# ============================================================

plt.figure(figsize=(7, 5))

for N in N_PROP_VALUES:
    vals = df_all.loc[df_all["N"] == N, "theta_deg"].to_numpy()

    plt.hist(
        vals,
        bins=60,
        density=True,
        histtype="step",
        linewidth=1.5,
        label=f"N={N}",
    )

plt.xlabel(r"$\theta^{(N)}(z)$ in degrees")
plt.ylabel("density")
plt.title("Finite-time stable--unstable angle distributions")
plt.legend()
plt.tight_layout()
plt.savefig(HIST_COMPARE_PATH, dpi=300)
plt.close()


# ============================================================
# Figure 2: lower-tail empirical CDF for representative N
# ============================================================

df_rep = df_all[df_all["N"] == REPRESENTATIVE_N]

if df_rep.empty:
    raise ValueError(
        f"No data found for REPRESENTATIVE_N={REPRESENTATIVE_N}. "
        "Make sure REPRESENTATIVE_N is included in N_PROP_VALUES."
    )

vals = np.sort(df_rep["theta_deg"].to_numpy())
cdf = np.arange(1, len(vals) + 1) / len(vals)

plt.figure(figsize=(7, 5))
plt.plot(vals, cdf, linewidth=1.5)

plt.xlim(0, 25)
plt.ylim(0, 0.25)
plt.xlabel(rf"$\theta^{{({REPRESENTATIVE_N})}}(z)$ in degrees")
plt.ylabel("empirical cumulative probability")
plt.title(rf"Lower tail of stable--unstable angle distribution, $N={REPRESENTATIVE_N}$")
plt.tight_layout()
plt.savefig(LOWER_TAIL_PATH, dpi=300)
plt.close()


# ============================================================
# Figure: very small-angle and small-angle points on the attractor
# ============================================================

df_rep = df_all[df_all["N"] == REPRESENTATIVE_N].copy()

df_very_small = df_rep[df_rep["theta_deg"] < VERY_SMALL_ANGLE_THRESHOLD_DEG].copy()
df_small_only = df_rep[
    (df_rep["theta_deg"] >= VERY_SMALL_ANGLE_THRESHOLD_DEG)
    & (df_rep["theta_deg"] < SMALL_ANGLE_THRESHOLD_DEG)
].copy()

plt.figure(figsize=(6, 5))

plt.scatter(
    df_rep["x"],
    df_rep["y"],
    s=1.0,
    c="lightgrey",
    label="sampled points",
)

plt.scatter(
    df_small_only["x"],
    df_small_only["y"],
    s=5.0,
    c="orange",
    label=rf"$1^\circ \leq \theta^{{({REPRESENTATIVE_N})}}(z)<5^\circ$",
)

plt.scatter(
    df_very_small["x"],
    df_very_small["y"],
    s=10.0,
    c="red",
    label=rf"$\theta^{{({REPRESENTATIVE_N})}}(z)<1^\circ$",
)

plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
plt.title(rf"Small-angle points on the Hénon attractor, $N={REPRESENTATIVE_N}$")
plt.legend(markerscale=2)
plt.tight_layout()
plt.savefig(SMALL_ANGLE_SPATIAL_PATH, dpi=300)
plt.close()



# ============================================================
# Figure 4: summary statistics versus N
# ============================================================

plt.figure(figsize=(7, 5))

plt.plot(
    df_summary["N"],
    df_summary["q01_deg"],
    marker="o",
    label="1% quantile",
)
plt.plot(
    df_summary["N"],
    df_summary["q05_deg"],
    marker="s",
    label="5% quantile",
)
plt.plot(
    df_summary["N"],
    df_summary["median_deg"],
    marker="^",
    label="median",
)

plt.xlabel(r"propagation length $N$")
plt.ylabel(r"angle in degrees")
plt.title(r"Angle statistics versus propagation length")
plt.legend()
plt.tight_layout()
plt.savefig(SUMMARY_PLOT_PATH, dpi=300)
plt.close()


# ============================================================
# Text summary
# ============================================================

lines = []

lines.append("Step 3: finite-time stable--unstable angle diagnostics")
lines.append("")
lines.append("Map: F_{A,B}(x,y) = (A - B y - x^2, x)")
lines.append(f"A = {A_PARAM}")
lines.append(f"B = {B_PARAM}")
lines.append("")
lines.append(f"Orbit file: {ORBIT_PATH}")
lines.append(f"Number of orbit points = {N_ORBIT}")
lines.append(f"Sample size = {SAMPLE_SIZE}")
lines.append(f"Propagation lengths N = {N_PROP_VALUES}")
lines.append(f"Representative N = {REPRESENTATIVE_N}")
lines.append(f"Robustness check N = {ROBUSTNESS_N}")
lines.append("")
lines.append("Interpretation warning:")
lines.append("The computed directions are finite-time approximations from derivative propagation.")
lines.append("They are not assumed to be true globally defined stable/unstable subspaces.")
lines.append("The angle distribution is used as a diagnostic for uniform transversality.")
lines.append("")
lines.append("Angle summary by N, in degrees:")
lines.append(df_summary.to_string(index=False))
lines.append("")
lines.append("Robustness check with different generic initial tangent vectors:")
lines.append(df_robustness.to_string(index=False))

lines.append("")
lines.append("Small-angle test:")
for key, value in small_angle_summary.items():
    lines.append(f"{key} = {value}")

lines.append("")
lines.append("Interpretation:")
lines.append("- The unstable direction is approximated by forward propagation from the past.")
lines.append("- The stable direction is approximated by backward propagation from the future.")
lines.append("- Small finite-time angles indicate near-tangencies in the finite-time direction diagnostic.")
lines.append("- For a uniformly hyperbolic invariant set, the true stable and unstable directions would be uniformly transverse.")
lines.append("- Similar robustness statistics indicate that the result is not dominated by the arbitrary initial tangent vectors.")
lines.append("")
lines.append("Saved outputs:")
lines.append(f"All values: {ALL_VALUES_PATH}")
lines.append(f"Summary CSV: {SUMMARY_CSV_PATH}")
lines.append(f"Summary TXT: {SUMMARY_TXT_PATH}")
lines.append(f"Histogram comparison: {HIST_COMPARE_PATH}")
lines.append(f"Lower-tail CDF: {LOWER_TAIL_PATH}")
lines.append(f"Summary plot versus N: {SUMMARY_PLOT_PATH}")
lines.append(f"Robustness check: {ROBUSTNESS_PATH}")
lines.append(f"Small-angle points CSV: {SMALL_ANGLE_POINTS_PATH}")
lines.append(f"Small-angle iteration gaps CSV: {SMALL_ANGLE_GAPS_PATH}")
lines.append(f"Small-angle spatial plot: {SMALL_ANGLE_SPATIAL_PATH}")

SUMMARY_TXT_PATH.write_text("\n".join(lines), encoding="utf-8")


print("Step 3 complete.")
print("")
print("Angle summary by N:")
print(df_summary)
print("")
print("Robustness check:")
print(df_robustness)
print("")
print(f"Saved outputs in: {BASE_DIR}")