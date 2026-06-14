from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# Step 2: finite-time Lyapunov exponents
# ============================================================
# Folder structure assumed:
#
# project/
# ├── Henon_orbit/
# │   └── henon_orbit.csv
# │
# └── ftle/
#     └── ftle.py
#
# This script reads the orbit from Henon_orbit/ and saves all Step 2
# outputs inside ftle/.
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

ORBIT_PATH = PROJECT_DIR / "Henon_orbit" / "henon_orbit.csv"

ALL_VALUES_PATH = BASE_DIR / "ftle_values_all_n.csv"
SUMMARY_CSV_PATH = BASE_DIR / "ftle_summary_by_n.csv"
SUMMARY_TXT_PATH = BASE_DIR / "ftle_summary.txt"

HIST_COMPARE_PATH = BASE_DIR / "hist_lambda1_compare_n.png"
MEAN_STD_PATH = BASE_DIR / "mean_std_lambda1_vs_n.png"
COLOURED_N50_PATH = BASE_DIR / "lambda1_coloured_attractor_n50.png"


# ============================================================
# Hénon map in Robinson notation
# F_{A,B}(x,y) = (A - B y - x^2, x)
#
# Classical parameters:
#     A = 1.4
#     B = -0.3
# ============================================================

A_PARAM = 1.4
B_PARAM = -0.3
LOG_ABS_B = np.log(abs(B_PARAM))


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


# ============================================================
# Numerical settings
# ============================================================

# Finite-time window lengths.
# n=50 still shows local finite-time variation.
# n=500 is used as a longer-window comparison with the global exponent.
N_VALUES = [50, 100, 200, 500]

# Number of sampled base points z_i.
SAMPLE_SIZE = 5000

# Representative window for the spatial coloured-attractor plot.
REPRESENTATIVE_N = 50


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
max_n = max(N_VALUES)

if N_ORBIT <= max_n + 1:
    raise ValueError("Orbit is too short for the largest finite-time window.")

sample_indices = np.linspace(
    0,
    N_ORBIT - max_n - 1,
    SAMPLE_SIZE,
    dtype=int,
)


# ============================================================
# Global Lyapunov estimate by repeated normalisation
# ============================================================

def estimate_global_lyapunov(orbit: np.ndarray) -> tuple[float, float]:
    """
    Estimate global Lyapunov exponents along the orbit.

    lambda_1 is computed by repeated normalisation of one tangent vector.
    lambda_2 is obtained from the exact determinant relation

        lambda_1 + lambda_2 = log |B|.
    """
    v = np.array([1.0, 0.0], dtype=float)
    log_sum = 0.0
    count = 0

    for z in orbit:
        w = DF(z) @ v
        norm_w = np.linalg.norm(w)

        if norm_w == 0:
            continue

        log_sum += np.log(norm_w)
        v = w / norm_w
        count += 1

    lambda1_global = log_sum / count
    lambda2_global = LOG_ABS_B - lambda1_global

    return lambda1_global, lambda2_global


lambda1_global, lambda2_global = estimate_global_lyapunov(orbit)


# ============================================================
# Finite-time Lyapunov exponents
# ============================================================

def finite_time_exponents(i: int, n: int) -> tuple[float, float, float]:
    """
    Compute finite-time Lyapunov exponents at z_i over window n.

    lambda_1^{(n)} is computed from the largest singular value of D(F^n)(z_i).

    lambda_2^{(n)} is computed using

        lambda_2^{(n)} = log |B| - lambda_1^{(n)},

    which avoids numerical underflow in the smaller singular value.
    """
    A_prod = np.eye(2)

    for j in range(n):
        A_prod = DF(orbit[i + j]) @ A_prod

    sigma1 = np.linalg.svd(A_prod, compute_uv=False)[0]

    lambda1 = np.log(sigma1) / n
    lambda2 = LOG_ABS_B - lambda1

    log_condition = n * (lambda1 - lambda2)

    return lambda1, lambda2, log_condition


all_rows = []
summary_rows = []

for n in N_VALUES:
    print(f"Computing finite-time Lyapunov exponents for n = {n}")

    rows_n = []

    for i in sample_indices:
        lam1, lam2, log_cond = finite_time_exponents(i, n)

        row = {
            "n": n,
            "i": int(i),
            "x": orbit[i, 0],
            "y": orbit[i, 1],
            "lambda1": lam1,
            "lambda2": lam2,
            "lambda_sum": lam1 + lam2,
            "log_condition": log_cond,
        }

        rows_n.append(row)
        all_rows.append(row)

    df_n = pd.DataFrame(rows_n)
    df_n.to_csv(BASE_DIR / f"finite_time_exponents_n{n}.csv", index=False)

    summary_rows.append(
        {
            "n": n,
            "mean_lambda1": df_n["lambda1"].mean(),
            "std_lambda1": df_n["lambda1"].std(),
            "min_lambda1": df_n["lambda1"].min(),
            "q01_lambda1": df_n["lambda1"].quantile(0.01),
            "q05_lambda1": df_n["lambda1"].quantile(0.05),
            "median_lambda1": df_n["lambda1"].median(),
            "q95_lambda1": df_n["lambda1"].quantile(0.95),
            "q99_lambda1": df_n["lambda1"].quantile(0.99),
            "max_lambda1": df_n["lambda1"].max(),
            "mean_lambda2": df_n["lambda2"].mean(),
            "std_lambda2": df_n["lambda2"].std(),
            "mean_lambda_sum": df_n["lambda_sum"].mean(),
            "std_lambda_sum": df_n["lambda_sum"].std(),
            "mean_log_condition": df_n["log_condition"].mean(),
        }
    )

df_all = pd.DataFrame(all_rows)
df_summary = pd.DataFrame(summary_rows)

df_all.to_csv(ALL_VALUES_PATH, index=False)
df_summary.to_csv(SUMMARY_CSV_PATH, index=False)


# ============================================================
# Figure 1: distributions of lambda_1^{(n)}
# ============================================================

plt.figure(figsize=(7, 5))

for n in N_VALUES:
    vals = df_all.loc[df_all["n"] == n, "lambda1"].to_numpy()

    plt.hist(
        vals,
        bins=55,
        density=True,
        histtype="step",
        linewidth=1.5,
        label=f"n={n}",
    )

plt.axvline(
    lambda1_global,
    linestyle="--",
    linewidth=1.5,
    label=rf"global $\lambda_1 \approx {lambda1_global:.3f}$",
)

plt.xlabel(r"$\lambda_1^{(n)}(z)$")
plt.ylabel("density")
plt.title(r"Distributions of finite-time Lyapunov exponents")
plt.legend()
plt.tight_layout()
plt.savefig(HIST_COMPARE_PATH, dpi=300)
plt.close()


# ============================================================
# Figure 2: mean and standard deviation versus n
# ============================================================

fig, ax1 = plt.subplots(figsize=(7, 5))

ax1.plot(
    df_summary["n"],
    df_summary["mean_lambda1"],
    marker="o",
    label=r"mean of $\lambda_1^{(n)}$",
)
ax1.axhline(
    lambda1_global,
    linestyle="--",
    linewidth=1.5,
    label=rf"global $\lambda_1 \approx {lambda1_global:.3f}$",
)

ax1.set_xlabel(r"window length $n$")
ax1.set_ylabel(r"mean of $\lambda_1^{(n)}$")
ax1.legend(loc="upper right")

ax2 = ax1.twinx()
ax2.plot(
    df_summary["n"],
    df_summary["std_lambda1"],
    marker="s",
    linestyle=":",
    label=r"std of $\lambda_1^{(n)}$",
)
ax2.set_ylabel(r"std of $\lambda_1^{(n)}$")

plt.title(r"Mean and spread of finite-time Lyapunov exponents")
fig.tight_layout()
plt.savefig(MEAN_STD_PATH, dpi=300)
plt.close()


# ============================================================
# Figure 3: representative coloured attractor, n = 50
# ============================================================

df_rep = df_all[df_all["n"] == REPRESENTATIVE_N]

plt.figure(figsize=(6, 5))
sc = plt.scatter(
    df_rep["x"],
    df_rep["y"],
    c=df_rep["lambda1"],
    s=1.0,
)
plt.colorbar(sc, label=rf"$\lambda_1^{{({REPRESENTATIVE_N})}}(z)$")
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
plt.title(rf"Hénon attractor coloured by $\lambda_1^{{({REPRESENTATIVE_N})}}$")
plt.tight_layout()
plt.savefig(COLOURED_N50_PATH, dpi=300)
plt.close()


# ============================================================
# Text summary
# ============================================================

lines = []

lines.append("Step 2: finite-time Lyapunov exponent diagnostics")
lines.append("")
lines.append("Map: F_{A,B}(x,y) = (A - B y - x^2, x)")
lines.append(f"A = {A_PARAM}")
lines.append(f"B = {B_PARAM}")
lines.append(f"log |B| = {LOG_ABS_B:.10f}")
lines.append("")
lines.append(f"Orbit file: {ORBIT_PATH}")
lines.append(f"Number of orbit points = {N_ORBIT}")
lines.append(f"Sample size = {SAMPLE_SIZE}")
lines.append(f"Window lengths n = {N_VALUES}")
lines.append("")
lines.append("Global Lyapunov estimate:")
lines.append(f"lambda1_global = {lambda1_global:.10f}")
lines.append(f"lambda2_global = {lambda2_global:.10f}")
lines.append(f"lambda1_global + lambda2_global = {lambda1_global + lambda2_global:.10f}")
lines.append("")
lines.append("Finite-time summary by n:")
lines.append(df_summary.to_string(index=False))
lines.append("")
lines.append("Interpretation:")
lines.append("- lambda1^{(n)} is computed from the largest singular value of D(F^n)(z).")
lines.append("- lambda2^{(n)} is computed using lambda2^{(n)} = log|B| - lambda1^{(n)}.")
lines.append("- The mean of lambda1^{(n)} is compared with the global Lyapunov estimate.")
lines.append("- The standard deviation of lambda1^{(n)} measures finite-time variability of stretching.")
lines.append("- A decreasing standard deviation as n increases indicates averaging of finite-time fluctuations.")
lines.append("")
lines.append("Saved outputs:")
lines.append(f"All values: {ALL_VALUES_PATH}")
lines.append(f"Summary CSV: {SUMMARY_CSV_PATH}")
lines.append(f"Histogram comparison: {HIST_COMPARE_PATH}")
lines.append(f"Mean/std plot: {MEAN_STD_PATH}")
lines.append(f"Representative coloured attractor: {COLOURED_N50_PATH}")

SUMMARY_TXT_PATH.write_text("\n".join(lines), encoding="utf-8")


print("Step 2 complete.")
print("")
print(f"Global lambda1 = {lambda1_global:.6f}")
print(f"Global lambda2 = {lambda2_global:.6f}")
print("")
print(df_summary)
print("")
print(f"Saved outputs in: {BASE_DIR}")