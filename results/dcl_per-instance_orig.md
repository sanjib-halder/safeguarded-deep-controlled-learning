# State-aware policy trained by simulation (DCL)

**Split:** per-instance (one network per instance). Nothing is held out; each network is trained and tested on its own instance (test uses separate random seeds).

Iterations 3, 76800 labelled states (37% with a non-base label), rollouts M = 16, H = 36, order offsets ['abs', 8], sh search; device `cpu`, 48 min. Gaps are paired against the DP optimum on held-out test seeds (NumPy simulator), as % of excess cost. *Selected* = lowest validation cost among GMI, status table and DCL.

| n | GMI % | Status table % | DCL % | Selected % | DCL sig. better than table | DCL sig. worse than table | DCL sig. better than GMI | DCL selected |
|---|---|---|---|---|---|---|---|---|
| 16 | 8.18 | 3.96 | 2.82 | 2.93 | 0.44 | 0.00 | 0.81 | 0.69 |

| Instance | GMI % | Status table % | DCL % | ± | DCL − table (pp) | ± | Pathwise neural, per-instance % | Selected |
|---|---|---|---|---|---|---|---|---|
| F00_lcl | 16.23 | 7.49 | 3.01 | 0.83 | -4.48 | 0.82 | 4.12 | DCL |
| F00_lch | 13.77 | 9.08 | 5.61 | 1.08 | -3.47 | 0.92 | 2.96 | DCL |
| F01_lcl | 13.36 | 5.17 | 3.97 | 1.31 | -1.20 | 0.67 | – | DCL |
| F01_lch | 8.05 | 4.78 | 4.80 | 1.36 | +0.02 | 0.72 | – | DCL |
| F02_lcl | 12.03 | 5.83 | 6.52 | 1.12 | +0.69 | 0.92 | 5.83 | DCL |
| F02_lch | 15.49 | 5.61 | 3.73 | 0.78 | -1.88 | 0.96 | – | Status-table |
| F03_lcl | 9.67 | 1.68 | 1.72 | 0.97 | +0.05 | 0.51 | – | Status-table |
| F03_lch | 4.42 | 1.25 | 0.90 | 0.68 | -0.35 | 0.50 | – | DCL |
| F04_lcl | 5.99 | 1.42 | 1.27 | 0.46 | -0.15 | 0.32 | – | DCL |
| F04_lch | 2.84 | 1.87 | 1.84 | 0.60 | -0.03 | 0.26 | – | DCL |
| F05_lcl | 7.05 | 6.86 | 2.41 | 0.49 | -4.45 | 0.54 | 3.08 | DCL |
| F05_lch | 0.87 | 0.51 | 0.87 | 0.47 | +0.37 | 0.41 | – | GMI |
| F06_lcl | 4.76 | 3.52 | 1.57 | 0.71 | -1.95 | 0.69 | – | DCL |
| F06_lch | 0.67 | 0.35 | 0.61 | 0.39 | +0.26 | 0.42 | – | Status-table |
| F07_lcl | 5.17 | 3.33 | 3.12 | 0.81 | -0.21 | 0.81 | – | Status-table |
| F07_lch | 10.50 | 4.55 | 3.21 | 0.67 | -1.34 | 0.41 | – | DCL |

**Comparison on the same test instances and seeds.** *Seed s* = the network trained on one training seed (its own validation-chosen iteration and threshold); *zero-shot* = networks trained on a different number of suppliers, applied without retraining:

| Policy | Mean gap % | Sig. better than table | Sig. worse than table | Difference to Original DCL (per instance) (pp) | Sig. better than Original DCL (per instance) | Sig. worse than Original DCL (per instance) |
|---|---|---|---|---|---|---|
| S-DCL ensemble (trained across instances) | 2.83 | 0.81 | 0.00 | +0.01 | 0.12 | 0.12 |
| Original DCL (per instance) | 2.82 | 0.44 | 0.00 | +0.00 | – | – |

| Instance | Status table % | S-DCL ensemble (trained across instances) % | Original DCL (per instance) % |
|---|---|---|---|
| F00_lcl | 7.49 | 2.41 | 3.01 |
| F00_lch | 9.08 | 4.65 | 5.61 |
| F01_lcl | 5.17 | 4.07 | 3.97 |
| F01_lch | 4.78 | 4.71 | 4.80 |
| F02_lcl | 5.83 | 3.18 | 6.52 |
| F02_lch | 5.61 | 4.39 | 3.73 |
| F03_lcl | 1.68 | 1.59 | 1.72 |
| F03_lch | 1.25 | 1.14 | 0.90 |
| F04_lcl | 1.42 | 1.11 | 1.27 |
| F04_lch | 1.87 | 1.73 | 1.84 |
| F05_lcl | 6.86 | 5.84 | 2.41 |
| F05_lch | 0.51 | 0.50 | 0.87 |
| F06_lcl | 3.52 | 3.20 | 1.57 |
| F06_lch | 0.35 | 0.33 | 0.61 |
| F07_lcl | 3.33 | 2.96 | 3.12 |
| F07_lch | 4.55 | 3.45 | 3.21 |

