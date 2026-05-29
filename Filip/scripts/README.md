# M2R Scripts

Small Python scripts for generating and analysing the classical Henon map

```text
x_{n+1} = 1 - a*x_n^2 + y_n
y_{n+1} = b*x_n
```

The default parameters are `a=1.4`, `b=0.3`, `N=100000`, and
`transient=10000`. Plots are written to files using Matplotlib's non-interactive
`Agg` backend.

## Environment

Use the Nix development shell so the expected Python packages are available:

```bash
nix develop
```

Or run a command directly inside the shell:

```bash
nix develop --command python strange_attractors_2d.py
```

The Python dependencies are also listed in `requirements.txt` for non-Nix use.

## `strange_attractors_2d.py`

Generates a deterministic Henon orbit, saves the orbit and attractor plot, and
computes finite-sample diagnostics:

- QR Lyapunov exponents
- Kaplan-Yorke dimension
- box-counting slope
- correlation-dimension slope
- information-dimension slope

Default full run:

```bash
nix develop --command python strange_attractors_2d.py \
  --a 1.4 \
  --b 0.3 \
  --N 100000 \
  --transient 10000 \
  --fit-start 0 \
  --fit-end 20 \
  --output-dir henon_output
```

Fast smoke run:

```bash
nix develop --command python strange_attractors_2d.py \
  --N 5000 \
  --transient 1000 \
  --fit-start 0 \
  --fit-end 20 \
  --max-correlation-points 1000 \
  --output-dir /tmp/henon_attractor_smoke
```

Main outputs:

- `orbit.csv`
- `attractor.png`
- `lyapunov_and_ky.csv`
- `summary.csv`
- `box_counting_fit.png`
- `correlation_dimension_fit.png`
- `information_dimension_fit.png`

Use `--fit-start` and `--fit-end` to restrict the regression to a scaling
window. Use `--theiler-window` for correlation-dimension runs when temporal
neighbours along the orbit should be excluded from pair counts. `summary.csv`
also includes bootstrap slope intervals over the selected scale points and a
fit-window sensitivity range over contiguous scale windows.

For the classical deterministic parameters, the Lyapunov exponents should be
approximately `lambda_1 ~= 0.42` and `lambda_2 ~= -1.62`, with Kaplan-Yorke
dimension near `1.26`.

## `local_dimensions_henon.py`

Generates deterministic or bounded-noise Henon orbits, samples centres from the
post-transient orbit, and estimates local dimensions from KDTree ball counts:

```text
log(mu_N(B(x,r))) versus log(r)
```

Default deterministic run:

```bash
nix develop --command python local_dimensions_henon.py \
  --a 1.4 \
  --b 0.3 \
  --N 100000 \
  --transient 10000 \
  --sigma 0 \
  --centres 500 \
  --r-min 1e-3 \
  --r-max 5e-1 \
  --num-radii 24 \
  --fit-start 4 \
  --fit-end 18 \
  --outdir henon_local_dimensions
```

Exact-dimensionality N-sweep with a shrinking k-nearest-neighbor fitting
window:

```bash
nix develop --command python local_dimensions_henon.py \
  --N 50000 100000 300000 1000000 \
  --sigma 0 \
  --centres 500 \
  --estimator knn \
  --knn-min-exponent 0.25 \
  --knn-max-exponent 0.70 \
  --num-neighbors 18 \
  --bootstrap-samples 1000 \
  --confidence-level 0.95 \
  --outdir henon_local_dimensions_knn
```

Noise sweep example:

```bash
nix develop --command python local_dimensions_henon.py \
  --N 100000 \
  --sigma 0 1e-4 5e-4 1e-3 2e-3 \
  --centres 500 \
  --outdir henon_local_dimensions_sigma
```

Fast smoke run:

```bash
nix develop --command python local_dimensions_henon.py \
  --N 1000 \
  --centres 10 \
  --num-radii 8 \
  --fit-start 1 \
  --fit-end 6 \
  --outdir /tmp/henon_local_dimension_smoke
```

Main outputs:

- `local_dimension_summary.csv`
- `local_dimension_summary_by_N.png`
- `local_dimension_summary_by_sigma.png`
- per-run `local_dimension_estimates.csv`
- per-run `local_dimension_curves.csv`
- per-run attractor and local-dimension plots

`local_dimension_summary.csv` includes bootstrap confidence intervals for the
mean and median over the sampled centre estimates. These intervals quantify
centre-sampling uncertainty for the chosen orbit and fitting window; they are
not rigorous enclosures of the mathematical dimension.

Noise is added after the deterministic update as bounded disk noise:

```text
z_{n+1} = f(z_n) + sigma * xi_n
```

where `xi_n` is sampled uniformly from the unit disk.

## `generalized_dimensions_henon.py`

Estimates the generalized Renyi dimension spectrum `D_q` from box
probabilities:

```text
D_q = slope of log(sum_i p_i(epsilon)^q) / (1 - q) versus log(1/epsilon)
```

with the entropy expression used at `q=1`. The special cases line up with the
usual dimensions: `D_0` is the box-counting dimension, `D_1` is the information
dimension, and `D_2` is the box-probability Renyi dimension.

Default spectrum run:

```bash
nix develop --command python generalized_dimensions_henon.py \
  --a 1.4 \
  --b 0.3 \
  --N 100000 \
  --transient 10000 \
  --q-values 0 0.5 1 1.5 2 3 4 \
  --scale-count 20 \
  --fit-start 0 \
  --fit-end 20 \
  --bootstrap-samples 1000 \
  --confidence-level 0.95 \
  --output-dir henon_generalized_dimensions
```

Main outputs:

- `generalized_dimensions.csv`
- `generalized_dimension_scale_data.csv`
- `generalized_dimension_spectrum.png`
- `generalized_dimension_scaling_curves.png`

The `D_q` bootstrap intervals resample the selected scale points in the
finite-scale regression. They are useful diagnostics for the chosen scale
window, not rigorous mathematical confidence intervals.

## Dimension Results

For the classical deterministic Henon parameters `a=1.4`, `b=0.3`, the saved
global diagnostics in `henon_output/summary.csv` give:

| Quantity | Estimate | 95% bootstrap interval |
| --- | ---: | ---: |
| `lambda_1` | `0.420` | n/a |
| `lambda_2` | `-1.624` | n/a |
| Kaplan-Yorke dimension `D_KY` | `1.259` | n/a |
| Box-counting dimension `D_box` | `1.245` | `[1.235, 1.256]` |
| Correlation dimension `D_2` | `1.202` | `[1.196, 1.207]` |
| Information dimension `D_1` | `1.229` | `[1.222, 1.238]` |

The bootstrap intervals for `D_box`, `D_2`, and `D_1` resample the selected
scale points in the finite regression. The same `summary.csv` also reports
fit-window sensitivity ranges, which are usually the more conservative
diagnostic for systematic scale-window uncertainty.

The deterministic local-dimension runs used 500 sampled centres for each orbit
length. With the original fixed-radius fit window, the mean is stable but the
spread does not decrease much because the same physical radii are reused for
every `N`:

| `N` | Mean local dimension | Std. dev. |
| ---: | ---: | ---: |
| `50,000` | `1.259` | `0.250` |
| `100,000` | `1.239` | `0.228` |
| `300,000` | `1.240` | `0.223` |
| `1,000,000` | `1.244` | `0.224` |

Across `N=50k, 100k, 300k, 1M`, the mean local dimension is approximately
`1.24`, while the standard deviation remains around `0.22-0.25`. The spread is
therefore persistent for this finite-scale diagnostic.

The exact-dimensionality test should instead shrink the fitting scale as `N`
grows. The saved k-nearest-neighbor run in
`henon_local_dimensions_knn/local_dimension_summary.csv` uses
`k_min ~= N^0.25` and `k_max ~= N^0.70`, so `k -> infinity` while `k/N -> 0`.
The mean interval is a 95% bootstrap interval over the 500 sampled centres:

| `N` | Mean local dimension | 95% mean interval | Std. dev. | IQR |
| ---: | ---: | ---: | ---: | ---: |
| `50,000` | `1.266` | `[1.244, 1.288]` | `0.248` | `0.296` |
| `100,000` | `1.250` | `[1.232, 1.269]` | `0.214` | `0.258` |
| `300,000` | `1.244` | `[1.229, 1.261]` | `0.191` | `0.214` |
| `1,000,000` | `1.247` | `[1.231, 1.262]` | `0.171` | `0.172` |

This is the expected qualitative behaviour for an exact-dimensional measure:
the estimated location stays near the global dimension while the sampled-centre
spread decreases as the local scale window moves inward.

The saved generalized-dimension run in
`henon_generalized_dimensions/generalized_dimensions.csv` gives a decreasing
Renyi spectrum:

| `q` | `D_q` | 95% bootstrap interval |
| ---: | ---: | ---: |
| `0` | `1.245` | `[1.234, 1.257]` |
| `0.5` | `1.241` | `[1.234, 1.249]` |
| `1` | `1.229` | `[1.222, 1.238]` |
| `1.5` | `1.213` | `[1.205, 1.223]` |
| `2` | `1.191` | `[1.179, 1.202]` |
| `3` | `1.123` | `[1.102, 1.144]` |
| `4` | `1.048` | `[1.013, 1.079]` |

The monotone decrease reflects nonuniformity of the empirical invariant
measure: larger `q` gives more weight to denser regions of the attractor. The
box-probability `D_2` is close to, but not identical with, the pair-count
correlation-dimension estimate above because the finite-sample estimators use
different objects.

For bounded disk noise at `N=100,000`, the mean local dimension increases with
the forcing scale:

| `sigma` | Mean local dimension |
| ---: | ---: |
| `0` | `1.239` |
| `0.0001` | `1.245` |
| `0.0005` | `1.276` |
| `0.001` | `1.262` |
| `0.002` | `1.283` |
| `0.003` | `1.336` |
| `0.004` | `1.372` |
| `0.005` | `1.421` |

The deterministic estimates cluster consistently around the global dimension
estimates, but the local dimension distribution is broad. Small bounded noise
thickens the sampled measure in this finite-scale diagnostic, with the clearest
increase appearing for `sigma >= 0.003`. Larger attempted noise levels
(`sigma=0.01` and `0.05`) diverged from the attracting region in these runs.

## Notes

The dimension estimates are numerical diagnostics, not rigorous mathematical
values. They depend on orbit length, transient length, selected radii or
neighbor-count window, regression window, centre sample, and noise level.
Inspect the saved log-log plots before interpreting slopes.
