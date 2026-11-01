# State-aware policy trained by simulation (DCL)

**Split:** fresh-instance generalisation. Every test instance is held out: training and validation use a different seed, so no test instance (or its supplier configuration) is seen.

Iterations 2, 230400 labelled states (8% with a non-base label), rollouts M = 16, H = 36, order offsets [-2, -1, 0, 1, 2], coord search; device `cpu`, 330 min. Gaps are paired against the best policy that uses only suppliers A, B, C (exact DP of the 3-supplier core; exact DP of the 4-supplier problem is intractable), on held-out test seeds, as % of its excess cost. A negative gap means better than any 3-supplier policy. *Selected* = lowest validation cost among GMI, status table and DCL.

| n | GMI % | Status table % | DCL % | Selected % | DCL sig. better than table | DCL sig. worse than table | DCL sig. better than GMI | DCL selected |
|---|---|---|---|---|---|---|---|---|
| 16 | -8.47 | -10.67 | -10.88 | -10.85 | 0.56 | 0.00 | 0.81 | 0.62 |

| GMI sig. better than 3-supplier optimum | Table sig. better than 3-supplier optimum | DCL sig. better than 3-supplier optimum |
|---|---|---|
| 0.88 | 0.94 | 0.94 |

| Instance | GMI % | Status table % | DCL % | ± | DCL − table (pp) | ± | Pathwise neural, per-instance % | Selected |
|---|---|---|---|---|---|---|---|---|
| N4_00_lcl | -16.90 | -21.36 | -21.70 | 4.66 | -0.35 | 0.30 | – | DCL |
| N4_00_lch | -16.74 | -18.69 | -18.87 | 3.91 | -0.19 | 0.17 | – | DCL |
| N4_01_lcl | -20.42 | -26.54 | -26.58 | 3.68 | -0.04 | 0.12 | – | DCL |
| N4_01_lch | -21.09 | -20.58 | -21.40 | 2.69 | -0.82 | 0.22 | – | DCL |
| N4_02_lcl | -10.93 | -15.37 | -15.50 | 4.87 | -0.13 | 0.12 | – | Status-table |
| N4_02_lch | -8.76 | -9.50 | -9.70 | 3.12 | -0.20 | 0.10 | – | DCL |
| N4_03_lcl | -9.28 | -14.09 | -14.14 | 2.52 | -0.05 | 0.06 | – | Status-table |
| N4_03_lch | -10.50 | -10.57 | -10.70 | 2.71 | -0.13 | 0.11 | – | GMI |
| N4_04_lcl | -5.89 | -11.54 | -11.85 | 1.61 | -0.31 | 0.16 | – | DCL |
| N4_04_lch | -1.27 | -5.11 | -5.55 | 1.25 | -0.44 | 0.14 | – | DCL |
| N4_05_lcl | 3.07 | 2.34 | 2.00 | 1.20 | -0.34 | 0.17 | – | DCL |
| N4_05_lch | -1.18 | -1.91 | -1.95 | 0.80 | -0.03 | 0.06 | – | Status-table |
| N4_06_lcl | -8.63 | -8.82 | -8.85 | 1.38 | -0.04 | 0.06 | – | Status-table |
| N4_06_lch | -4.52 | -4.55 | -4.59 | 0.98 | -0.04 | 0.09 | – | Status-table |
| N4_07_lcl | -0.45 | -1.47 | -1.53 | 1.34 | -0.05 | 0.08 | – | DCL |
| N4_07_lch | -2.06 | -2.90 | -3.14 | 0.95 | -0.24 | 0.27 | – | DCL |

**Comparison on the same test instances and seeds.** *Seed s* = the network trained on one training seed (its own validation-chosen iteration and threshold); *zero-shot* = networks trained on a different number of suppliers, applied without retraining; ensemble rule: mean probability, threshold 0; zero-shot rule: mean probability, threshold 0 (chosen on this target's validation instances):

| Policy | Mean gap % | Sig. better than table | Sig. worse than table |
|---|---|---|---|
| Seed 101 | -10.86 | 0.44 | 0.00 |
| Seed 202 | -11.17 | 0.62 | 0.00 |
| Seed 303 | -10.81 | 0.38 | 0.00 |
| Zero-shot from N = 3 | -11.12 | 0.56 | 0.00 |
| Ensemble | -10.88 | 0.56 | 0.00 |

| Instance | Status table % | Seed 101 % | Seed 202 % | Seed 303 % | Zero-shot from N = 3 % | Ensemble % |
|---|---|---|---|---|---|---|
| N4_00_lcl | -21.36 | -21.54 | -21.97 | -21.69 | -22.29 | -21.70 |
| N4_00_lch | -18.69 | -18.76 | -18.81 | -18.85 | -18.88 | -18.87 |
| N4_01_lcl | -26.54 | -26.60 | -26.51 | -26.56 | -26.54 | -26.58 |
| N4_01_lch | -20.58 | -20.70 | -21.83 | -21.05 | -21.12 | -21.40 |
| N4_02_lcl | -15.37 | -15.64 | -15.64 | -15.44 | -15.47 | -15.50 |
| N4_02_lch | -9.50 | -9.96 | -10.33 | -9.59 | -11.00 | -9.70 |
| N4_03_lcl | -14.09 | -14.15 | -14.20 | -14.11 | -14.24 | -14.14 |
| N4_03_lch | -10.57 | -10.54 | -10.75 | -10.46 | -10.71 | -10.70 |
| N4_04_lcl | -11.54 | -12.03 | -12.10 | -11.57 | -11.54 | -11.85 |
| N4_04_lch | -5.11 | -5.53 | -5.51 | -5.30 | -5.27 | -5.55 |
| N4_05_lcl | 2.34 | 1.73 | 1.16 | 1.59 | -0.58 | 2.00 |
| N4_05_lch | -1.91 | -1.88 | -1.95 | -1.94 | -1.95 | -1.95 |
| N4_06_lcl | -8.82 | -8.85 | -8.85 | -8.82 | -8.86 | -8.85 |
| N4_06_lch | -4.55 | -4.58 | -4.59 | -4.56 | -4.59 | -4.59 |
| N4_07_lcl | -1.47 | -1.60 | -3.63 | -1.47 | -1.75 | -1.53 |
| N4_07_lch | -2.90 | -3.15 | -3.18 | -3.06 | -3.10 | -3.14 |

