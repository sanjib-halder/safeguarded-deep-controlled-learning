# State-aware policy trained by simulation (DCL)

**Split:** per-instance (one network per instance). Nothing is held out; each network is trained and tested on its own instance (test uses separate random seeds).

Iterations 2, 12800 labelled states (12% with a non-base label), rollouts M = 16, H = 36, order offsets [-2, -1, 0, 1, 2]; device `cpu`, 9 min. Gaps are paired against the DP optimum on held-out test seeds (NumPy simulator), as % of excess cost. *Selected* = lowest validation cost among GMI, status table and DCL.

| n | GMI % | Status table % | DCL % | Selected % | DCL sig. better than table | DCL sig. worse than table | DCL sig. better than GMI | DCL selected |
|---|---|---|---|---|---|---|---|---|
| 4 | 12.27 | 7.31 | 5.44 | 5.44 | 0.75 | 0.00 | 1.00 | 1.00 |

| Instance | GMI % | Status table % | DCL % | ± | DCL − table (pp) | ± | Pathwise neural, per-instance % | Selected |
|---|---|---|---|---|---|---|---|---|
| F00_lcl | 16.23 | 7.49 | 7.46 | 0.87 | -0.03 | 0.27 | 4.12 | DCL |
| F00_lch | 13.77 | 9.08 | 5.42 | 1.00 | -3.66 | 0.80 | 2.96 | DCL |
| F02_lcl | 12.03 | 5.83 | 3.76 | 1.14 | -2.07 | 0.64 | 5.83 | DCL |
| F05_lcl | 7.05 | 6.86 | 5.14 | 0.58 | -1.72 | 0.36 | 3.08 | DCL |

