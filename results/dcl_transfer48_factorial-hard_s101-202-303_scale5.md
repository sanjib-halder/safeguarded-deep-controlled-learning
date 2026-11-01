# State-aware policy trained by simulation (DCL)

**Split:** fresh-instance generalisation. Every test instance is held out: training and validation use a different seed, so no test instance (or its supplier configuration) is seen.

Iterations 2, 0 labelled states (0% with a non-base label), rollouts M = 16, H = 36, order offsets [-2, -1, 0, 1, 2], coord search; device `cpu`, 138 min. Gaps are paired against the best policy that uses only suppliers A, B, C (exact DP of the 3-supplier core; exact DP of the 5-supplier problem is intractable), on held-out test seeds, as % of its excess cost. A negative gap means better than any 3-supplier policy. *Selected* = lowest validation cost among GMI, status table and DCL.

| n | GMI % | Status table % | DCL % | Selected % | DCL sig. better than table | DCL sig. worse than table | DCL sig. better than GMI | DCL selected |
|---|---|---|---|---|---|---|---|---|
| 16 | -9.14 | -9.56 | -9.78 | -9.77 | 0.31 | 0.00 | 0.44 | 0.38 |

| GMI sig. better than 3-supplier optimum | Table sig. better than 3-supplier optimum | DCL sig. better than 3-supplier optimum |
|---|---|---|
| 0.81 | 0.81 | 0.81 |

| Instance | GMI % | Status table % | DCL % | ± | DCL − table (pp) | ± | Pathwise neural, per-instance % | Selected |
|---|---|---|---|---|---|---|---|---|
| N5_00_lcl | -6.80 | -6.37 | -6.37 | 2.41 | +0.00 | 0.00 | – | Status-table |
| N5_00_lch | -3.39 | -3.51 | -3.55 | 2.40 | -0.04 | 0.07 | – | GMI |
| N5_01_lcl | -5.55 | -5.49 | -5.57 | 1.80 | -0.08 | 0.07 | – | GMI |
| N5_01_lch | -11.11 | -10.60 | -10.86 | 1.37 | -0.26 | 0.17 | – | GMI |
| N5_02_lcl | -14.95 | -17.98 | -17.98 | 4.82 | +0.00 | 0.00 | – | Status-table |
| N5_02_lch | -16.15 | -15.13 | -15.13 | 4.08 | +0.00 | 0.00 | – | GMI |
| N5_03_lcl | -0.05 | 1.70 | -0.98 | 1.60 | -2.68 | 0.27 | – | DCL |
| N5_03_lch | -6.93 | -6.81 | -6.86 | 1.21 | -0.04 | 0.05 | – | DCL |
| N5_04_lcl | 0.74 | 1.27 | 1.10 | 1.34 | -0.18 | 0.09 | – | DCL |
| N5_04_lch | -4.31 | -4.83 | -4.83 | 1.19 | +0.00 | 0.00 | – | Status-table |
| N5_05_lcl | -14.60 | -17.08 | -17.12 | 4.30 | -0.04 | 0.08 | – | DCL |
| N5_05_lch | -17.58 | -17.54 | -17.58 | 3.51 | -0.04 | 0.07 | – | DCL |
| N5_06_lcl | 3.57 | 2.55 | 2.47 | 1.41 | -0.08 | 0.05 | – | GMI |
| N5_06_lch | -2.46 | -2.46 | -2.53 | 1.11 | -0.07 | 0.11 | – | Status-table |
| N5_07_lcl | -24.23 | -27.58 | -27.58 | 4.27 | -0.00 | 0.01 | – | Status-table |
| N5_07_lch | -22.42 | -23.04 | -23.04 | 3.07 | +0.00 | 0.00 | – | DCL |

