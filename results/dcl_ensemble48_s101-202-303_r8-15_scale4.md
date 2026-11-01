# State-aware policy trained by simulation (DCL)

**Split:** fresh-instance generalisation. Every test instance is held out: training and validation use a different seed, so no test instance (or its supplier configuration) is seen.

Iterations 2, 230400 labelled states (8% with a non-base label), rollouts M = 16, H = 36, order offsets [-2, -1, 0, 1, 2], coord search; device `cpu`, 42 min. Gaps are paired against the best policy that uses only suppliers A, B, C (exact DP of the 3-supplier core; exact DP of the 4-supplier problem is intractable), on held-out test seeds, as % of its excess cost. A negative gap means better than any 3-supplier policy. *Selected* = lowest validation cost among GMI, status table and DCL.

| n | GMI % | Status table % | DCL % | Selected % | DCL sig. better than table | DCL sig. worse than table | DCL sig. better than GMI | DCL selected |
|---|---|---|---|---|---|---|---|---|
| 16 | -6.94 | -8.79 | -8.94 | -8.92 | 0.44 | 0.00 | 0.56 | 0.56 |

| GMI sig. better than 3-supplier optimum | Table sig. better than 3-supplier optimum | DCL sig. better than 3-supplier optimum |
|---|---|---|
| 0.75 | 0.88 | 0.88 |

| Instance | GMI % | Status table % | DCL % | ± | DCL − table (pp) | ± | Pathwise neural, per-instance % | Selected |
|---|---|---|---|---|---|---|---|---|
| N4_08_lcl | -1.17 | -7.13 | -7.23 | 1.71 | -0.09 | 0.06 | – | DCL |
| N4_08_lch | -6.69 | -8.90 | -9.05 | 1.18 | -0.16 | 0.10 | – | DCL |
| N4_09_lcl | 4.94 | 2.19 | 2.04 | 0.87 | -0.15 | 0.14 | – | DCL |
| N4_09_lch | 2.01 | 1.68 | 1.63 | 0.60 | -0.05 | 0.05 | – | DCL |
| N4_10_lcl | -12.02 | -20.11 | -20.41 | 3.46 | -0.29 | 0.15 | – | DCL |
| N4_10_lch | -21.43 | -22.69 | -22.76 | 2.95 | -0.07 | 0.11 | – | DCL |
| N4_11_lcl | -2.35 | -3.69 | -3.82 | 0.67 | -0.13 | 0.10 | – | Status-table |
| N4_11_lch | -2.86 | -2.64 | -2.70 | 0.47 | -0.06 | 0.06 | – | Status-table |
| N4_12_lcl | -6.13 | -6.23 | -6.25 | 1.41 | -0.01 | 0.03 | – | Status-table |
| N4_12_lch | -10.05 | -9.98 | -10.00 | 1.60 | -0.02 | 0.03 | – | GMI |
| N4_13_lcl | -15.61 | -15.56 | -15.58 | 1.34 | -0.02 | 0.04 | – | GMI |
| N4_13_lch | -11.58 | -12.94 | -13.00 | 0.74 | -0.06 | 0.07 | – | DCL |
| N4_14_lcl | -13.85 | -13.87 | -13.92 | 2.89 | -0.05 | 0.05 | – | Status-table |
| N4_14_lch | -9.68 | -9.65 | -9.65 | 2.20 | +0.00 | 0.00 | – | Status-table |
| N4_15_lcl | -2.38 | -5.93 | -6.46 | 2.33 | -0.53 | 0.23 | – | DCL |
| N4_15_lch | -2.24 | -5.13 | -5.81 | 1.74 | -0.68 | 0.16 | – | DCL |

**Comparison on the same test instances and seeds.** *Seed s* = the network trained on one training seed (its own validation-chosen iteration and threshold); *zero-shot* = networks trained on a different number of suppliers, applied without retraining; ensemble rule: mean probability, threshold 0; zero-shot rule: mean probability, threshold 0 (chosen on this target's validation instances):

| Policy | Mean gap % | Sig. better than table | Sig. worse than table | Difference to Ensemble (pp) | Sig. better than Ensemble | Sig. worse than Ensemble |
|---|---|---|---|---|---|---|
| Seed 101 | -8.74 | 0.25 | 0.06 | +0.19 | 0.00 | 0.12 |
| Seed 202 | -8.95 | 0.50 | 0.06 | -0.01 | 0.00 | 0.19 |
| Seed 303 | -8.90 | 0.25 | 0.00 | +0.04 | 0.06 | 0.25 |
| Zero-shot from N = 3 | -8.86 | 0.31 | 0.06 | +0.07 | 0.00 | 0.25 |
| Ensemble | -8.94 | 0.44 | 0.00 | +0.00 | – | – |

| Instance | Status table % | Seed 101 % | Seed 202 % | Seed 303 % | Zero-shot from N = 3 % | Ensemble % |
|---|---|---|---|---|---|---|
| N4_08_lcl | -7.13 | -7.24 | -7.18 | -7.14 | -7.18 | -7.23 |
| N4_08_lch | -8.90 | -8.98 | -9.13 | -8.97 | -8.98 | -9.05 |
| N4_09_lcl | 2.19 | 2.07 | 2.09 | 1.89 | 2.09 | 2.04 |
| N4_09_lch | 1.68 | 1.62 | 1.60 | 1.69 | 1.67 | 1.63 |
| N4_10_lcl | -20.11 | -18.54 | -20.59 | -20.24 | -20.46 | -20.41 |
| N4_10_lch | -22.69 | -22.11 | -22.78 | -22.70 | -22.79 | -22.76 |
| N4_11_lcl | -3.69 | -3.95 | -3.84 | -3.69 | -3.78 | -3.82 |
| N4_11_lch | -2.64 | -2.69 | -2.71 | -2.67 | -2.70 | -2.70 |
| N4_12_lcl | -6.23 | -6.23 | -6.27 | -6.22 | -6.24 | -6.25 |
| N4_12_lch | -9.98 | -10.00 | -10.02 | -10.00 | -10.02 | -10.00 |
| N4_13_lcl | -15.56 | -15.58 | -15.59 | -15.56 | -15.59 | -15.58 |
| N4_13_lch | -12.94 | -12.97 | -13.00 | -12.97 | -12.99 | -13.00 |
| N4_14_lcl | -13.87 | -13.92 | -13.92 | -13.88 | -13.90 | -13.92 |
| N4_14_lch | -9.65 | -9.65 | -9.65 | -9.65 | -9.68 | -9.65 |
| N4_15_lcl | -5.93 | -6.55 | -6.55 | -6.12 | -5.74 | -6.46 |
| N4_15_lch | -5.13 | -5.17 | -5.64 | -6.16 | -5.55 | -5.81 |

