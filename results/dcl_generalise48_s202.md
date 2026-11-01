# State-aware policy trained by simulation (DCL)

**Split:** fresh-instance generalisation. Every test instance is held out: training and validation use a different seed, so no test instance (or its supplier configuration) is seen.

Iterations 2, 76800 labelled states (9% with a non-base label), rollouts M = 16, H = 36, order offsets [-2, -1, 0, 1, 2]; device `cpu`, 103 min. Gaps are paired against the DP optimum on held-out test seeds (NumPy simulator), as % of excess cost. *Selected* = lowest validation cost among GMI, status table and DCL.

| n | GMI % | Status table % | DCL % | Selected % | DCL sig. better than table | DCL sig. worse than table | DCL sig. better than GMI | DCL selected |
|---|---|---|---|---|---|---|---|---|
| 16 | 8.18 | 3.96 | 3.01 | 3.04 | 0.75 | 0.00 | 0.88 | 0.81 |

| Instance | GMI % | Status table % | DCL % | ± | DCL − table (pp) | ± | Pathwise neural, per-instance % | Selected |
|---|---|---|---|---|---|---|---|---|
| F00_lcl | 16.23 | 7.49 | 3.50 | 0.90 | -4.00 | 0.46 | 4.12 | DCL |
| F00_lch | 13.77 | 9.08 | 5.63 | 0.98 | -3.45 | 0.73 | 2.96 | DCL |
| F01_lcl | 13.36 | 5.17 | 3.43 | 1.38 | -1.74 | 0.75 | – | DCL |
| F01_lch | 8.05 | 4.78 | 4.74 | 1.23 | -0.04 | 0.16 | – | Status-table |
| F02_lcl | 12.03 | 5.83 | 3.40 | 1.07 | -2.43 | 0.80 | 5.83 | DCL |
| F02_lch | 15.49 | 5.61 | 4.92 | 0.96 | -0.69 | 0.34 | – | DCL |
| F03_lcl | 9.67 | 1.68 | 1.59 | 0.98 | -0.09 | 0.07 | – | DCL |
| F03_lch | 4.42 | 1.25 | 1.18 | 0.68 | -0.07 | 0.05 | – | Status-table |
| F04_lcl | 5.99 | 1.42 | 1.09 | 0.43 | -0.32 | 0.16 | – | DCL |
| F04_lch | 2.84 | 1.87 | 1.76 | 0.51 | -0.11 | 0.15 | – | DCL |
| F05_lcl | 7.05 | 6.86 | 5.98 | 0.60 | -0.88 | 0.32 | 3.08 | DCL |
| F05_lch | 0.87 | 0.51 | 0.49 | 0.45 | -0.01 | 0.02 | – | GMI |
| F06_lcl | 4.76 | 3.52 | 3.34 | 0.78 | -0.19 | 0.12 | – | DCL |
| F06_lch | 0.67 | 0.35 | 0.35 | 0.48 | +0.00 | 0.04 | – | DCL |
| F07_lcl | 5.17 | 3.33 | 3.08 | 0.83 | -0.25 | 0.22 | – | DCL |
| F07_lch | 10.50 | 4.55 | 3.65 | 0.58 | -0.90 | 0.12 | – | DCL |

