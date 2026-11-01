# State-aware policy trained by simulation (DCL)

**Split:** fresh-instance generalisation. Every test instance is held out: training and validation use a different seed, so no test instance (or its supplier configuration) is seen.

Iterations 2, 230400 labelled states (8% with a non-base label), rollouts M = 16, H = 36, order offsets [-2, -1, 0, 1, 2], coord search; device `cpu`, 524 min. Gaps are paired against the best policy that uses only suppliers A, B, C (exact DP of the 3-supplier core; exact DP of the 5-supplier problem is intractable), on held-out test seeds, as % of its excess cost. A negative gap means better than any 3-supplier policy. *Selected* = lowest validation cost among GMI, status table and DCL.

| n | GMI % | Status table % | DCL % | Selected % | DCL sig. better than table | DCL sig. worse than table | DCL sig. better than GMI | DCL selected |
|---|---|---|---|---|---|---|---|---|
| 16 | -9.14 | -9.56 | -10.12 | -10.06 | 0.56 | 0.00 | 0.50 | 0.69 |

| GMI sig. better than 3-supplier optimum | Table sig. better than 3-supplier optimum | DCL sig. better than 3-supplier optimum |
|---|---|---|
| 0.81 | 0.81 | 0.81 |

| Instance | GMI % | Status table % | DCL % | ± | DCL − table (pp) | ± | Pathwise neural, per-instance % | Selected |
|---|---|---|---|---|---|---|---|---|
| N5_00_lcl | -6.80 | -6.37 | -6.75 | 2.39 | -0.37 | 0.26 | – | DCL |
| N5_00_lch | -3.39 | -3.51 | -6.29 | 2.37 | -2.78 | 0.67 | – | DCL |
| N5_01_lcl | -5.55 | -5.49 | -5.77 | 1.80 | -0.28 | 0.16 | – | GMI |
| N5_01_lch | -11.11 | -10.60 | -11.45 | 1.36 | -0.85 | 0.34 | – | GMI |
| N5_02_lcl | -14.95 | -17.98 | -18.35 | 4.90 | -0.37 | 0.31 | – | DCL |
| N5_02_lch | -16.15 | -15.13 | -15.13 | 4.08 | +0.00 | 0.00 | – | GMI |
| N5_03_lcl | -0.05 | 1.70 | 0.24 | 1.62 | -1.45 | 0.32 | – | DCL |
| N5_03_lch | -6.93 | -6.81 | -6.94 | 1.25 | -0.12 | 0.12 | – | Status-table |
| N5_04_lcl | 0.74 | 1.27 | -0.77 | 1.27 | -2.05 | 0.69 | – | DCL |
| N5_04_lch | -4.31 | -4.83 | -4.89 | 1.20 | -0.06 | 0.06 | – | DCL |
| N5_05_lcl | -14.60 | -17.08 | -17.12 | 4.30 | -0.04 | 0.08 | – | DCL |
| N5_05_lch | -17.58 | -17.54 | -17.64 | 3.51 | -0.10 | 0.09 | – | DCL |
| N5_06_lcl | 3.57 | 2.55 | 2.28 | 1.39 | -0.27 | 0.14 | – | GMI |
| N5_06_lch | -2.46 | -2.46 | -2.51 | 1.11 | -0.05 | 0.09 | – | DCL |
| N5_07_lcl | -24.23 | -27.58 | -27.66 | 4.29 | -0.09 | 0.12 | – | DCL |
| N5_07_lch | -22.42 | -23.04 | -23.17 | 3.03 | -0.13 | 0.20 | – | DCL |

**Comparison on the same test instances and seeds.** *Seed s* = the network trained on one training seed (its own validation-chosen iteration and threshold); *zero-shot* = networks trained on a different number of suppliers, applied without retraining; ensemble rule: mean probability, threshold 0; zero-shot rule: mean probability, threshold 0.5 (chosen on this target's validation instances):

| Policy | Mean gap % | Sig. better than table | Sig. worse than table |
|---|---|---|---|
| Seed 101 | -9.80 | 0.38 | 0.00 |
| Seed 202 | -10.28 | 0.69 | 0.00 |
| Seed 303 | -10.38 | 0.94 | 0.00 |
| Zero-shot from N = 3 | -9.78 | 0.31 | 0.00 |
| Ensemble | -10.12 | 0.56 | 0.00 |

| Instance | Status table % | Seed 101 % | Seed 202 % | Seed 303 % | Zero-shot from N = 3 % | Ensemble % |
|---|---|---|---|---|---|---|
| N5_00_lcl | -6.37 | -6.37 | -6.97 | -6.82 | -6.37 | -6.75 |
| N5_00_lch | -3.51 | -3.51 | -6.32 | -7.04 | -3.55 | -6.29 |
| N5_01_lcl | -5.49 | -5.52 | -6.20 | -6.37 | -5.57 | -5.77 |
| N5_01_lch | -10.60 | -10.60 | -11.60 | -12.34 | -10.86 | -11.45 |
| N5_02_lcl | -17.98 | -18.12 | -18.36 | -18.19 | -17.98 | -18.35 |
| N5_02_lch | -15.13 | -15.13 | -15.13 | -15.26 | -15.13 | -15.13 |
| N5_03_lcl | 1.70 | -0.66 | -1.72 | 0.28 | -0.98 | 0.24 |
| N5_03_lch | -6.81 | -6.81 | -6.94 | -7.08 | -6.86 | -6.94 |
| N5_04_lcl | 1.27 | 0.34 | -0.11 | 0.84 | 1.10 | -0.77 |
| N5_04_lch | -4.83 | -4.83 | -5.02 | -4.93 | -4.83 | -4.89 |
| N5_05_lcl | -17.08 | -17.08 | -17.36 | -17.26 | -17.12 | -17.12 |
| N5_05_lch | -17.54 | -17.63 | -17.68 | -17.87 | -17.58 | -17.64 |
| N5_06_lcl | 2.55 | 2.41 | 2.24 | 0.73 | 2.47 | 2.28 |
| N5_06_lch | -2.46 | -2.46 | -2.51 | -2.82 | -2.53 | -2.51 |
| N5_07_lcl | -27.58 | -27.74 | -27.68 | -27.77 | -27.58 | -27.66 |
| N5_07_lch | -23.04 | -23.08 | -23.08 | -24.15 | -23.04 | -23.17 |

