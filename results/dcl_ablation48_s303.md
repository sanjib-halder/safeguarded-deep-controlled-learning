# State-aware policy trained by simulation (DCL)

**Split:** fresh-instance generalisation. Every test instance is held out: training and validation use a different seed, so no test instance (or its supplier configuration) is seen.

Iterations 2, 76800 labelled states (9% with a non-base label), rollouts M = 16, H = 36, order offsets [-2, -1, 0, 1, 2], joint search; device `cpu`, 5 min. Gaps are paired against the DP optimum on held-out test seeds (NumPy simulator), as % of excess cost. *Selected* = lowest validation cost among GMI, status table and DCL.

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

**Comparison on the same test instances and seeds.** *Seed s* = the network trained on one training seed (its own validation-chosen iteration and threshold); *zero-shot* = networks trained on a different number of suppliers, applied without retraining; *Ablation: v* = the full method's network retrained on the same instances with one component changed (see experiments/run_ablations.py); differences to the full method are paired on the same test seeds:

| Policy | Mean gap % | Sig. better than table | Sig. worse than table | Difference to Full method (pp) | Sig. better than Full method | Sig. worse than Full method |
|---|---|---|---|---|---|---|
| Ablation: no_confirm | 3.45 | 0.56 | 0.06 | -0.36 | 0.19 | 0.44 |
| Ablation: coord | 3.05 | 0.81 | 0.00 | -0.75 | 0.31 | 0.31 |
| Ablation: one_iter | 3.28 | 0.50 | 0.00 | -0.53 | 0.12 | 0.69 |
| Ablation: no_prior | 3.29 | 0.62 | 0.06 | -0.51 | 0.19 | 0.31 |
| Ablation: short_H | 2.97 | 0.62 | 0.00 | -0.84 | 0.19 | 0.38 |
| Full method | 3.81 | 0.88 | 0.06 | +0.00 | – | – |

| Instance | Status table % | Ablation: no_confirm % | Ablation: coord % | Ablation: one_iter % | Ablation: no_prior % | Ablation: short_H % | Full method % |
|---|---|---|---|---|---|---|---|
| F00_lcl | 7.49 | 10.15 | 7.73 | 6.40 | 9.74 | 3.41 | 4.92 |
| F00_lch | 9.08 | 6.46 | 7.38 | 7.24 | 6.38 | 6.31 | 20.03 |
| F01_lcl | 5.17 | 5.12 | 3.85 | 5.17 | 4.73 | 4.76 | 4.05 |
| F01_lch | 4.78 | 4.76 | 4.61 | 4.78 | 4.67 | 4.92 | 4.38 |
| F02_lcl | 5.83 | 3.46 | 3.15 | 3.43 | 3.17 | 3.67 | 2.79 |
| F02_lch | 5.61 | 5.12 | 4.14 | 5.27 | 5.19 | 4.81 | 4.96 |
| F03_lcl | 1.68 | 1.90 | 1.68 | 1.68 | 1.57 | 1.68 | 1.61 |
| F03_lch | 1.25 | 1.75 | 1.09 | 1.25 | 1.14 | 1.19 | 1.05 |
| F04_lcl | 1.42 | 0.80 | 1.11 | 1.33 | 1.16 | 1.27 | 1.28 |
| F04_lch | 1.87 | 1.83 | 1.54 | 1.87 | 1.72 | 1.87 | 1.50 |
| F05_lcl | 6.86 | 3.06 | 2.59 | 2.75 | 2.76 | 2.63 | 3.93 |
| F05_lch | 0.51 | 0.52 | 0.49 | 0.51 | 0.48 | 0.51 | 0.46 |
| F06_lcl | 3.52 | 2.96 | 3.24 | 3.14 | 3.10 | 3.08 | 3.06 |
| F06_lch | 0.35 | 0.32 | 0.32 | 0.35 | 0.31 | 0.33 | 0.31 |
| F07_lcl | 3.33 | 3.09 | 3.06 | 3.33 | 2.87 | 3.07 | 2.86 |
| F07_lch | 4.55 | 3.94 | 2.85 | 3.97 | 3.70 | 3.97 | 3.70 |

