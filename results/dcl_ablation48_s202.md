# State-aware policy trained by simulation (DCL)

**Split:** fresh-instance generalisation. Every test instance is held out: training and validation use a different seed, so no test instance (or its supplier configuration) is seen.

Iterations 2, 76800 labelled states (9% with a non-base label), rollouts M = 16, H = 36, order offsets [-2, -1, 0, 1, 2], joint search; device `cpu`, 5 min. Gaps are paired against the DP optimum on held-out test seeds (NumPy simulator), as % of excess cost. *Selected* = lowest validation cost among GMI, status table and DCL.

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

**Comparison on the same test instances and seeds.** *Seed s* = the network trained on one training seed (its own validation-chosen iteration and threshold); *zero-shot* = networks trained on a different number of suppliers, applied without retraining; *Ablation: v* = the full method's network retrained on the same instances with one component changed (see experiments/run_ablations.py); differences to the full method are paired on the same test seeds:

| Policy | Mean gap % | Sig. better than table | Sig. worse than table | Difference to Full method (pp) | Sig. better than Full method | Sig. worse than Full method |
|---|---|---|---|---|---|---|
| Ablation: no_confirm | 3.15 | 0.38 | 0.00 | +0.14 | 0.12 | 0.25 |
| Ablation: coord | 3.14 | 0.75 | 0.06 | +0.13 | 0.12 | 0.25 |
| Ablation: one_iter | 3.18 | 0.62 | 0.00 | +0.17 | 0.06 | 0.44 |
| Ablation: no_prior | 3.49 | 0.75 | 0.06 | +0.48 | 0.12 | 0.19 |
| Ablation: short_H | 2.92 | 0.88 | 0.00 | -0.09 | 0.44 | 0.06 |
| Full method | 3.01 | 0.75 | 0.00 | +0.00 | – | – |

| Instance | Status table % | Ablation: no_confirm % | Ablation: coord % | Ablation: one_iter % | Ablation: no_prior % | Ablation: short_H % | Full method % |
|---|---|---|---|---|---|---|---|
| F00_lcl | 7.49 | 6.00 | 2.72 | 2.63 | 10.69 | 3.61 | 3.50 |
| F00_lch | 9.08 | 5.70 | 6.84 | 7.03 | 6.67 | 5.66 | 5.63 |
| F01_lcl | 5.17 | 4.52 | 4.17 | 4.23 | 4.21 | 4.29 | 3.43 |
| F01_lch | 4.78 | 4.67 | 4.67 | 4.72 | 4.48 | 4.51 | 4.74 |
| F02_lcl | 5.83 | 3.51 | 3.44 | 3.41 | 3.31 | 3.12 | 3.40 |
| F02_lch | 5.61 | 5.25 | 5.12 | 5.33 | 4.45 | 4.33 | 4.92 |
| F03_lcl | 1.68 | 1.58 | 1.59 | 1.68 | 1.50 | 1.65 | 1.59 |
| F03_lch | 1.25 | 1.77 | 1.06 | 1.25 | 1.04 | 1.04 | 1.18 |
| F04_lcl | 1.42 | 1.28 | 1.10 | 1.07 | 1.20 | 1.08 | 1.09 |
| F04_lch | 1.87 | 1.67 | 1.84 | 1.85 | 1.75 | 1.41 | 1.76 |
| F05_lcl | 6.86 | 2.80 | 6.14 | 6.15 | 5.66 | 5.45 | 5.98 |
| F05_lch | 0.51 | 0.51 | 0.77 | 0.51 | 0.69 | 0.48 | 0.49 |
| F06_lcl | 3.52 | 3.00 | 3.29 | 3.50 | 3.43 | 3.02 | 3.34 |
| F06_lch | 0.35 | 0.37 | 0.32 | 0.32 | 0.32 | 0.24 | 0.35 |
| F07_lcl | 3.33 | 3.30 | 3.23 | 3.27 | 2.97 | 2.83 | 3.08 |
| F07_lch | 4.55 | 4.46 | 3.94 | 3.95 | 3.42 | 3.93 | 3.65 |

