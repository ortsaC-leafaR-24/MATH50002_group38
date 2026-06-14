from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Path setup
# ============================================================
# All outputs are saved in the same folder as this script.
BASE_DIR = Path(__file__).resolve().parent

CSV_PATH = BASE_DIR / "henon_orbit.csv"
FIG_PATH = BASE_DIR / "henon_attractor.png"
SUMMARY_PATH = BASE_DIR / "orbit_summary.txt"


# ============================================================
# Hénon map in Robinson notation
# F_{A,B}(x,y) = (A - B y - x^2, x)
# Classical parameters: A = 1.4, B = -0.3
# ============================================================
A = 1.4
B = -0.3


def henon_F(z: np.ndarray) -> np.ndarray:
    """Hénon map F_{A,B}(x,y) = (A - B y - x^2, x)."""
    x, y = z
    return np.array([A - B * y - x**2, x], dtype=float)


# ============================================================
# Numerical parameters
# ============================================================
z0 = np.array([0.0, 0.0], dtype=float)

N_burn = 10_000
N_keep = 100_000
N_total = N_burn + N_keep


# ============================================================
# Generate orbit
# ============================================================
orbit_all = np.empty((N_total + 1, 2), dtype=float)
orbit_all[0] = z0

for j in range(N_total):
    orbit_all[j + 1] = henon_F(orbit_all[j])

# discard burn-in
orbit = orbit_all[N_burn:N_burn + N_keep]

# relabel kept orbit as z_0, ..., z_{N_keep-1}
df = pd.DataFrame({
    "i": np.arange(N_keep),
    "x": orbit[:, 0],
    "y": orbit[:, 1],
})

df.to_csv(CSV_PATH, index=False)


# ============================================================
# Plot attractor
# ============================================================
plt.figure(figsize=(6, 5))
plt.scatter(df["x"], df["y"], s=0.05, alpha=0.5)
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
plt.title(r"Hénon attractor, $F_{1.4,-0.3}$")
plt.tight_layout()
plt.savefig(FIG_PATH, dpi=300)
plt.close()


# ============================================================
# Summary statistics
# ============================================================
summary = []
summary.append("Step 1: orbit generation")
summary.append("")
summary.append("Map: F_{A,B}(x,y) = (A - B y - x^2, x)")
summary.append(f"A = {A}")
summary.append(f"B = {B}")
summary.append(f"Initial point z0 = {z0.tolist()}")
summary.append(f"Burn-in N_burn = {N_burn}")
summary.append(f"Kept points N_keep = {N_keep}")
summary.append("")
summary.append("Orbit coordinate summary:")
summary.append(df[["x", "y"]].describe().to_string())
summary.append("")
summary.append(f"CSV saved to: {CSV_PATH}")
summary.append(f"Figure saved to: {FIG_PATH}")

SUMMARY_PATH.write_text("\n".join(summary), encoding="utf-8")

print("Step 1 complete.")
print(f"CSV saved to: {CSV_PATH}")
print(f"Figure saved to: {FIG_PATH}")
print(f"Summary saved to: {SUMMARY_PATH}")