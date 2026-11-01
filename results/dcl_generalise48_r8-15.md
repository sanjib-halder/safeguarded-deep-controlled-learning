# State-aware policy trained by simulation (DCL)

**Split:** fresh-instance generalisation. Every test instance is held out: training and validation use a different seed, so no test instance (or its supplier configuration) is seen.

Iterations 2, 76800 labelled states (9% with a non-base label), rollouts M = 16, H = 36, order offsets [-2, -1, 0, 1, 2]; device `cpu`, 38 min. Gaps are paired against the DP optimum on held-out test seeds (NumPy simulator), as % of excess cost. *Selected* = lowest validation cost among GMI, status table and DCL.

| n | GMI % | Status table % | DCL % | Selected % | DCL sig. better than table | DCL sig. worse than table | DCL sig. better than GMI | DCL selected |
|---|---|---|---|---|---|---|---|---|
| 16 | 7.67 | 3.84 | 2.87 | 2.88 | 0.62 | 0.00 | 1.00 | 0.75 |

| Instance | GMI % | Status table % | DCL % | ± | DCL − table (pp) | ± | Pathwise neural, per-instance % | Selected |
|---|---|---|---|---|---|---|---|---|
| F08_lcl | 5.73 | 2.75 | 2.76 | 1.55 | +0.01 | 0.02 | – | Status-table |
| F08_lch | 4.82 | 2.78 | 2.75 | 0.63 | -0.03 | 0.02 | – | DCL |
| F09_lcl | 15.96 | 5.85 | 4.41 | 1.01 | -1.44 | 0.32 | – | DCL |
| F09_lch | 6.64 | 2.76 | 2.71 | 0.83 | -0.05 | 0.02 | – | DCL |
| F10_lcl | 6.46 | 2.26 | 2.26 | 1.67 | -0.00 | 0.04 | – | Status-table |
| F10_lch | 5.30 | 3.57 | 2.84 | 0.73 | -0.73 | 0.22 | – | DCL |
| F11_lcl | 4.98 | 3.58 | 2.03 | 1.00 | -1.56 | 0.46 | – | DCL |
| F11_lch | 8.99 | 1.19 | 1.13 | 0.41 | -0.06 | 0.10 | – | DCL |
| F12_lcl | 12.44 | 10.32 | 4.79 | 1.04 | -5.53 | 0.52 | – | DCL |
| F12_lch | 4.88 | 3.76 | 2.25 | 0.61 | -1.51 | 0.29 | – | DCL |
| F13_lcl | 11.77 | 8.51 | 5.06 | 1.16 | -3.45 | 0.79 | – | DCL |
| F13_lch | 5.82 | 2.27 | 1.98 | 0.79 | -0.29 | 0.17 | – | DCL |
| F14_lcl | 4.30 | 1.24 | 1.21 | 0.40 | -0.03 | 0.05 | – | Status-table |
| F14_lch | 8.55 | 5.44 | 4.76 | 0.56 | -0.69 | 0.39 | – | DCL |
| F15_lcl | 11.59 | 3.13 | 2.99 | 1.04 | -0.14 | 0.43 | – | Status-table |
| F15_lch | 4.51 | 2.08 | 2.03 | 0.94 | -0.06 | 0.22 | – | DCL |

