# State-aware policy trained by simulation (DCL)

**Split:** fresh-instance generalisation. Every test instance is held out: training and validation use a different seed, so no test instance (or its supplier configuration) is seen.

Iterations 2, 76800 labelled states (9% with a non-base label), rollouts M = 16, H = 36, order offsets [-2, -1, 0, 1, 2]; device `cpu`, 29 min. Gaps are paired against the DP optimum on held-out test seeds (NumPy simulator), as % of excess cost. *Selected* = lowest validation cost among GMI, status table and DCL.

| n | GMI % | Status table % | DCL % | Selected % | DCL sig. better than table | DCL sig. worse than table | DCL sig. better than GMI | DCL selected |
|---|---|---|---|---|---|---|---|---|
| 16 | 8.18 | 3.96 | 2.87 | 2.87 | 0.62 | 0.06 | 0.94 | 0.94 |

| Instance | GMI % | Status table % | DCL % | ± | DCL − table (pp) | ± | Pathwise neural, per-instance % | Selected |
|---|---|---|---|---|---|---|---|---|
| F00_lcl | 16.23 | 7.49 | 2.03 | 0.80 | -5.47 | 0.62 | 4.12 | DCL |
| F00_lch | 13.77 | 9.08 | 5.21 | 0.94 | -3.87 | 0.49 | 2.96 | DCL |
| F01_lcl | 13.36 | 5.17 | 4.21 | 1.31 | -0.97 | 0.54 | – | DCL |
| F01_lch | 8.05 | 4.78 | 5.02 | 1.31 | +0.24 | 0.34 | – | DCL |
| F02_lcl | 12.03 | 5.83 | 3.12 | 0.88 | -2.71 | 0.80 | 5.83 | DCL |
| F02_lch | 15.49 | 5.61 | 4.38 | 0.92 | -1.22 | 0.61 | – | DCL |
| F03_lcl | 9.67 | 1.68 | 1.68 | 1.00 | +0.00 | 0.00 | – | Status-table |
| F03_lch | 4.42 | 1.25 | 1.14 | 0.68 | -0.11 | 0.10 | – | DCL |
| F04_lcl | 5.99 | 1.42 | 1.07 | 0.49 | -0.35 | 0.38 | – | DCL |
| F04_lch | 2.84 | 1.87 | 1.73 | 0.51 | -0.14 | 0.33 | – | DCL |
| F05_lcl | 7.05 | 6.86 | 5.20 | 0.54 | -1.66 | 0.34 | 3.08 | DCL |
| F05_lch | 0.87 | 0.51 | 0.35 | 0.43 | -0.16 | 0.10 | – | DCL |
| F06_lcl | 4.76 | 3.52 | 3.44 | 0.77 | -0.08 | 0.13 | – | DCL |
| F06_lch | 0.67 | 0.35 | 0.34 | 0.48 | -0.01 | 0.01 | – | DCL |
| F07_lcl | 5.17 | 3.33 | 3.67 | 1.04 | +0.34 | 0.34 | – | DCL |
| F07_lch | 10.50 | 4.55 | 3.41 | 0.61 | -1.14 | 0.10 | – | DCL |

