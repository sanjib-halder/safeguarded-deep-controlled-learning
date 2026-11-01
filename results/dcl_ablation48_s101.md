# State-aware policy trained by simulation (DCL)

**Split:** fresh-instance generalisation. Every test instance is held out: training and validation use a different seed, so no test instance (or its supplier configuration) is seen.

Iterations 2, 76800 labelled states (9% with a non-base label), rollouts M = 16, H = 36, order offsets [-2, -1, 0, 1, 2], joint search; device `cpu`, 5 min. Gaps are paired against the DP optimum on held-out test seeds (NumPy simulator), as % of excess cost. *Selected* = lowest validation cost among GMI, status table and DCL.

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

**Comparison on the same test instances and seeds.** *Seed s* = the network trained on one training seed (its own validation-chosen iteration and threshold); *zero-shot* = networks trained on a different number of suppliers, applied without retraining; *Ablation: v* = the full method's network retrained on the same instances with one component changed (see experiments/run_ablations.py); differences to the full method are paired on the same test seeds:

| Policy | Mean gap % | Sig. better than table | Sig. worse than table | Difference to Full method (pp) | Sig. better than Full method | Sig. worse than Full method |
|---|---|---|---|---|---|---|
| Ablation: no_confirm | 3.65 | 0.62 | 0.00 | +0.77 | 0.06 | 0.56 |
| Ablation: coord | 3.02 | 0.69 | 0.00 | +0.15 | 0.06 | 0.31 |
| Ablation: one_iter | 3.32 | 0.69 | 0.00 | +0.45 | 0.12 | 0.38 |
| Ablation: no_prior | 3.38 | 0.56 | 0.00 | +0.51 | 0.19 | 0.38 |
| Ablation: short_H | 2.80 | 0.69 | 0.00 | -0.07 | 0.25 | 0.25 |
| Full method | 2.87 | 0.62 | 0.06 | +0.00 | – | – |

| Instance | Status table % | Ablation: no_confirm % | Ablation: coord % | Ablation: one_iter % | Ablation: no_prior % | Ablation: short_H % | Full method % |
|---|---|---|---|---|---|---|---|
| F00_lcl | 7.49 | 7.12 | 2.65 | 5.19 | 7.16 | 1.94 | 2.03 |
| F00_lch | 9.08 | 7.81 | 5.48 | 6.24 | 7.20 | 5.51 | 5.21 |
| F01_lcl | 5.17 | 5.07 | 4.13 | 4.92 | 3.86 | 3.82 | 4.21 |
| F01_lch | 4.78 | 4.69 | 4.70 | 4.73 | 4.77 | 4.74 | 5.02 |
| F02_lcl | 5.83 | 4.49 | 3.55 | 5.27 | 3.12 | 4.02 | 3.12 |
| F02_lch | 5.61 | 5.37 | 4.91 | 4.06 | 5.23 | 4.66 | 4.38 |
| F03_lcl | 1.68 | 1.74 | 1.68 | 1.67 | 1.68 | 1.67 | 1.68 |
| F03_lch | 1.25 | 1.24 | 1.19 | 1.14 | 1.25 | 1.23 | 1.14 |
| F04_lcl | 1.42 | 1.19 | 1.15 | 0.86 | 1.29 | 0.96 | 1.07 |
| F04_lch | 1.87 | 1.87 | 1.86 | 1.56 | 1.87 | 1.68 | 1.73 |
| F05_lcl | 6.86 | 6.02 | 6.05 | 6.27 | 5.95 | 3.23 | 5.20 |
| F05_lch | 0.51 | 0.51 | 0.43 | 0.49 | 0.51 | 0.51 | 0.35 |
| F06_lcl | 3.52 | 3.32 | 3.55 | 3.52 | 3.28 | 3.35 | 3.44 |
| F06_lch | 0.35 | 0.35 | 0.34 | 0.33 | 0.33 | 0.33 | 0.34 |
| F07_lcl | 3.33 | 3.33 | 3.30 | 2.88 | 3.26 | 2.98 | 3.67 |
| F07_lch | 4.55 | 4.24 | 3.41 | 3.98 | 3.43 | 4.19 | 3.41 |

