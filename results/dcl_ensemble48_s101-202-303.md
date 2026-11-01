# State-aware policy trained by simulation (DCL)

**Split:** fresh-instance generalisation. Every test instance is held out: training and validation use a different seed, so no test instance (or its supplier configuration) is seen.

Iterations 2, 230400 labelled states (9% with a non-base label), rollouts M = 16, H = 36, order offsets [-2, -1, 0, 1, 2]; device `cpu`, 11 min. Gaps are paired against the DP optimum on held-out test seeds (NumPy simulator), as % of excess cost. *Selected* = lowest validation cost among GMI, status table and DCL.

| n | GMI % | Status table % | DCL % | Selected % | DCL sig. better than table | DCL sig. worse than table | DCL sig. better than GMI | DCL selected |
|---|---|---|---|---|---|---|---|---|
| 16 | 8.18 | 3.96 | 2.83 | 2.86 | 0.81 | 0.00 | 0.94 | 0.88 |

| Instance | GMI % | Status table % | DCL % | ± | DCL − table (pp) | ± | Pathwise neural, per-instance % | Selected |
|---|---|---|---|---|---|---|---|---|
| F00_lcl | 16.23 | 7.49 | 2.41 | 0.81 | -5.08 | 0.47 | 4.12 | DCL |
| F00_lch | 13.77 | 9.08 | 4.65 | 1.02 | -4.43 | 0.70 | 2.96 | DCL |
| F01_lcl | 13.36 | 5.17 | 4.07 | 1.30 | -1.10 | 0.52 | – | DCL |
| F01_lch | 8.05 | 4.78 | 4.71 | 1.23 | -0.07 | 0.15 | – | Status-table |
| F02_lcl | 12.03 | 5.83 | 3.18 | 0.94 | -2.64 | 0.81 | 5.83 | DCL |
| F02_lch | 15.49 | 5.61 | 4.39 | 0.91 | -1.21 | 0.54 | – | DCL |
| F03_lcl | 9.67 | 1.68 | 1.59 | 0.98 | -0.09 | 0.07 | – | DCL |
| F03_lch | 4.42 | 1.25 | 1.14 | 0.68 | -0.11 | 0.10 | – | DCL |
| F04_lcl | 5.99 | 1.42 | 1.11 | 0.43 | -0.31 | 0.07 | – | DCL |
| F04_lch | 2.84 | 1.87 | 1.73 | 0.46 | -0.14 | 0.27 | – | DCL |
| F05_lcl | 7.05 | 6.86 | 5.84 | 0.58 | -1.02 | 0.27 | 3.08 | DCL |
| F05_lch | 0.87 | 0.51 | 0.50 | 0.46 | -0.01 | 0.02 | – | GMI |
| F06_lcl | 4.76 | 3.52 | 3.20 | 0.78 | -0.33 | 0.17 | – | DCL |
| F06_lch | 0.67 | 0.35 | 0.33 | 0.48 | -0.02 | 0.01 | – | DCL |
| F07_lcl | 5.17 | 3.33 | 2.96 | 0.84 | -0.36 | 0.26 | – | DCL |
| F07_lch | 10.50 | 4.55 | 3.45 | 0.61 | -1.10 | 0.09 | – | DCL |

**Ensemble vs. its members** (same test instances and seeds; each member is the network trained on one training seed, with its own validation-chosen iteration and threshold):

| Policy | Mean gap % | Sig. better than table | Sig. worse than table |
|---|---|---|---|
| Seed 101 | 2.87 | 0.62 | 0.06 |
| Seed 202 | 3.01 | 0.75 | 0.00 |
| Seed 303 | 3.81 | 0.88 | 0.06 |
| Ensemble (mean probability, threshold 0) | 2.83 | 0.81 | 0.00 |

| Instance | Status table % | Seed 101 % | Seed 202 % | Seed 303 % | Ensemble % |
|---|---|---|---|---|---|
| F00_lcl | 7.49 | 2.03 | 3.50 | 4.92 | 2.41 |
| F00_lch | 9.08 | 5.21 | 5.63 | 20.03 | 4.65 |
| F01_lcl | 5.17 | 4.21 | 3.43 | 4.05 | 4.07 |
| F01_lch | 4.78 | 5.02 | 4.74 | 4.38 | 4.71 |
| F02_lcl | 5.83 | 3.12 | 3.40 | 2.79 | 3.18 |
| F02_lch | 5.61 | 4.38 | 4.92 | 4.96 | 4.39 |
| F03_lcl | 1.68 | 1.68 | 1.59 | 1.61 | 1.59 |
| F03_lch | 1.25 | 1.14 | 1.18 | 1.05 | 1.14 |
| F04_lcl | 1.42 | 1.07 | 1.09 | 1.28 | 1.11 |
| F04_lch | 1.87 | 1.73 | 1.76 | 1.50 | 1.73 |
| F05_lcl | 6.86 | 5.20 | 5.98 | 3.93 | 5.84 |
| F05_lch | 0.51 | 0.35 | 0.49 | 0.46 | 0.50 |
| F06_lcl | 3.52 | 3.44 | 3.34 | 3.06 | 3.20 |
| F06_lch | 0.35 | 0.34 | 0.35 | 0.31 | 0.33 |
| F07_lcl | 3.33 | 3.67 | 3.08 | 2.86 | 2.96 |
| F07_lch | 4.55 | 3.41 | 3.65 | 3.70 | 3.45 |

