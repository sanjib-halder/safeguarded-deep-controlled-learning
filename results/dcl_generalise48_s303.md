# State-aware policy trained by simulation (DCL)

**Split:** fresh-instance generalisation. Every test instance is held out: training and validation use a different seed, so no test instance (or its supplier configuration) is seen.

Iterations 2, 76800 labelled states (9% with a non-base label), rollouts M = 16, H = 36, order offsets [-2, -1, 0, 1, 2]; device `cpu`, 110 min. Gaps are paired against the DP optimum on held-out test seeds (NumPy simulator), as % of excess cost. *Selected* = lowest validation cost among GMI, status table and DCL.

| n | GMI % | Status table % | DCL % | Selected % | DCL sig. better than table | DCL sig. worse than table | DCL sig. better than GMI | DCL selected |
|---|---|---|---|---|---|---|---|---|
| 16 | 8.18 | 3.96 | 3.81 | 3.15 | 0.88 | 0.06 | 0.88 | 0.88 |

| Instance | GMI % | Status table % | DCL % | ± | DCL − table (pp) | ± | Pathwise neural, per-instance % | Selected |
|---|---|---|---|---|---|---|---|---|
| F00_lcl | 16.23 | 7.49 | 4.92 | 0.74 | -2.58 | 0.44 | 4.12 | DCL |
| F00_lch | 13.77 | 9.08 | 20.03 | 1.63 | +10.95 | 1.29 | 2.96 | Status-table |
| F01_lcl | 13.36 | 5.17 | 4.05 | 1.22 | -1.12 | 0.59 | – | DCL |
| F01_lch | 8.05 | 4.78 | 4.38 | 1.22 | -0.40 | 0.17 | – | DCL |
| F02_lcl | 12.03 | 5.83 | 2.79 | 0.93 | -3.03 | 0.72 | 5.83 | DCL |
| F02_lch | 15.49 | 5.61 | 4.96 | 0.94 | -0.64 | 0.34 | – | DCL |
| F03_lcl | 9.67 | 1.68 | 1.61 | 0.98 | -0.07 | 0.07 | – | DCL |
| F03_lch | 4.42 | 1.25 | 1.05 | 0.68 | -0.20 | 0.14 | – | DCL |
| F04_lcl | 5.99 | 1.42 | 1.28 | 0.43 | -0.13 | 0.07 | – | DCL |
| F04_lch | 2.84 | 1.87 | 1.50 | 0.40 | -0.37 | 0.27 | – | DCL |
| F05_lcl | 7.05 | 6.86 | 3.93 | 0.61 | -2.92 | 0.45 | 3.08 | DCL |
| F05_lch | 0.87 | 0.51 | 0.46 | 0.46 | -0.05 | 0.07 | – | GMI |
| F06_lcl | 4.76 | 3.52 | 3.06 | 0.78 | -0.46 | 0.18 | – | DCL |
| F06_lch | 0.67 | 0.35 | 0.31 | 0.47 | -0.04 | 0.04 | – | DCL |
| F07_lcl | 5.17 | 3.33 | 2.86 | 0.79 | -0.47 | 0.28 | – | DCL |
| F07_lch | 10.50 | 4.55 | 3.70 | 0.64 | -0.85 | 0.22 | – | DCL |

