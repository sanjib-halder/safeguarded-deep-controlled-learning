# State-aware policy trained by simulation (DCL)

**Split:** fresh-instance generalisation. Every test instance is held out: training and validation use a different seed, so no test instance (or its supplier configuration) is seen.

Iterations 2, 230400 labelled states (8% with a non-base label), rollouts M = 16, H = 36, order offsets [-2, -1, 0, 1, 2], coord search; device `cpu`, 64 min. Gaps are paired against the best policy that uses only suppliers A, B, C (exact DP of the 3-supplier core; exact DP of the 5-supplier problem is intractable), on held-out test seeds, as % of its excess cost. A negative gap means better than any 3-supplier policy. *Selected* = lowest validation cost among GMI, status table and DCL.

| n | GMI % | Status table % | DCL % | Selected % | DCL sig. better than table | DCL sig. worse than table | DCL sig. better than GMI | DCL selected |
|---|---|---|---|---|---|---|---|---|
| 16 | -5.55 | -6.64 | -6.82 | -6.80 | 0.38 | 0.00 | 0.69 | 0.69 |

| GMI sig. better than 3-supplier optimum | Table sig. better than 3-supplier optimum | DCL sig. better than 3-supplier optimum |
|---|---|---|
| 0.56 | 0.81 | 0.81 |

| Instance | GMI % | Status table % | DCL % | ± | DCL − table (pp) | ± | Pathwise neural, per-instance % | Selected |
|---|---|---|---|---|---|---|---|---|
| N5_08_lcl | -30.47 | -30.74 | -30.74 | 3.94 | -0.00 | 0.00 | – | Status-table |
| N5_08_lch | -15.37 | -17.58 | -18.52 | 3.36 | -0.93 | 0.46 | – | DCL |
| N5_09_lcl | 0.82 | -2.61 | -2.62 | 1.15 | -0.02 | 0.03 | – | Status-table |
| N5_09_lch | -5.07 | -5.12 | -5.31 | 0.99 | -0.20 | 0.12 | – | DCL |
| N5_10_lcl | 0.13 | -1.46 | -1.46 | 1.36 | +0.00 | 0.00 | – | Status-table |
| N5_10_lch | 0.46 | -1.13 | -1.22 | 0.94 | -0.10 | 0.17 | – | DCL |
| N5_11_lcl | -10.18 | -10.18 | -10.18 | 0.75 | +0.00 | 0.00 | – | DCL |
| N5_11_lch | -5.22 | -5.32 | -5.55 | 0.51 | -0.23 | 0.23 | – | DCL |
| N5_12_lcl | 1.63 | -0.97 | -0.99 | 0.87 | -0.03 | 0.05 | – | DCL |
| N5_12_lch | 2.75 | 1.88 | 1.86 | 0.78 | -0.01 | 0.02 | – | DCL |
| N5_13_lcl | -10.65 | -11.32 | -11.61 | 1.60 | -0.29 | 0.17 | – | DCL |
| N5_13_lch | -7.45 | -8.27 | -8.31 | 1.41 | -0.04 | 0.09 | – | DCL |
| N5_14_lcl | 2.77 | 1.43 | 1.23 | 1.21 | -0.21 | 0.16 | – | Status-table |
| N5_14_lch | 0.47 | -0.59 | -0.67 | 0.95 | -0.08 | 0.09 | – | Status-table |
| N5_15_lcl | -5.65 | -6.56 | -7.15 | 1.05 | -0.58 | 0.23 | – | DCL |
| N5_15_lch | -7.82 | -7.75 | -7.82 | 0.88 | -0.07 | 0.07 | – | DCL |

**Comparison on the same test instances and seeds.** *Seed s* = the network trained on one training seed (its own validation-chosen iteration and threshold); *zero-shot* = networks trained on a different number of suppliers, applied without retraining; ensemble rule: mean probability, threshold 0; zero-shot rule: mean probability, threshold 0.5 (chosen on this target's validation instances):

| Policy | Mean gap % | Sig. better than table | Sig. worse than table | Difference to Ensemble (pp) | Sig. better than Ensemble | Sig. worse than Ensemble |
|---|---|---|---|---|---|---|
| Seed 101 | -6.70 | 0.19 | 0.00 | +0.12 | 0.06 | 0.31 |
| Seed 202 | -6.88 | 0.50 | 0.00 | -0.06 | 0.31 | 0.06 |
| Seed 303 | -7.29 | 0.94 | 0.00 | -0.47 | 0.81 | 0.00 |
| Zero-shot from N = 3 | -6.68 | 0.06 | 0.00 | +0.13 | 0.06 | 0.38 |
| Ensemble | -6.82 | 0.38 | 0.00 | +0.00 | – | – |

| Instance | Status table % | Seed 101 % | Seed 202 % | Seed 303 % | Zero-shot from N = 3 % | Ensemble % |
|---|---|---|---|---|---|---|
| N5_08_lcl | -30.74 | -30.92 | -30.86 | -30.98 | -30.74 | -30.74 |
| N5_08_lch | -17.58 | -17.70 | -18.68 | -21.39 | -17.59 | -18.52 |
| N5_09_lcl | -2.61 | -2.61 | -2.69 | -2.77 | -2.61 | -2.62 |
| N5_09_lch | -5.12 | -5.12 | -5.41 | -5.69 | -5.21 | -5.31 |
| N5_10_lcl | -1.46 | -1.49 | -1.45 | -1.64 | -1.46 | -1.46 |
| N5_10_lch | -1.13 | -1.13 | -1.28 | -1.46 | -1.13 | -1.22 |
| N5_11_lcl | -10.18 | -10.18 | -10.19 | -10.18 | -10.18 | -10.18 |
| N5_11_lch | -5.32 | -5.32 | -5.61 | -5.56 | -5.32 | -5.55 |
| N5_12_lcl | -0.97 | -0.97 | -1.00 | -1.20 | -0.97 | -0.99 |
| N5_12_lch | 1.88 | 1.91 | 1.61 | 0.17 | 1.40 | 1.86 |
| N5_13_lcl | -11.32 | -11.32 | -11.80 | -11.85 | -11.35 | -11.61 |
| N5_13_lch | -8.27 | -8.27 | -8.31 | -8.63 | -8.30 | -8.31 |
| N5_14_lcl | 1.43 | 1.44 | 0.90 | 0.70 | 1.43 | 1.23 |
| N5_14_lch | -0.59 | -0.59 | -0.73 | -0.91 | -0.59 | -0.67 |
| N5_15_lcl | -6.56 | -7.12 | -6.77 | -7.18 | -6.56 | -7.15 |
| N5_15_lch | -7.75 | -7.75 | -7.83 | -8.00 | -7.75 | -7.82 |

**Rolling-horizon SAA** (two-stage MILP re-solved every period; 2 episodes x 250 periods per instance, paired with the table and DCL on the same seeds; 0.05 s per decision for SAA). Difference to the status table, as % of the reference excess cost (negative = better):

| Policy | Mean difference to table (pp) | Sig. better than table | Sig. worse than table |
|---|---|---|---|
| SAA | +63.74 | 0.06 | 0.06 |
| DCL | +0.02 | 0.06 | 0.00 |

