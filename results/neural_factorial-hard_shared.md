# State-aware neural policy

Target `factorial-hard`, mode `shared`, 600 iterations on `cpu` (433 s training). Gaps are paired against the DP optimum on held-out test seeds (NumPy simulator), as % of excess cost. *Selected* = lowest validation cost among GMI, status-table and neural (heuristic as fallback).

## Summary (mean gap %, share of instances)

| Group | n | GMI | Status-table | Neural | Selected | Neural sig. better than table | Neural sig. worse than table | Neural sig. better than GMI | Neural selected |
|---|---|---|---|---|---|---|---|---|---|
| training instances | 12 | 9.15 | 4.29 | 5.92 | 3.86 | 0.17 | 0.83 | 0.67 | 0.17 |
| held-out instances | 4 | 5.27 | 2.94 | 27.72 | 2.94 | 0.00 | 1.00 | 0.00 | 0.00 |
| all | 16 | 8.18 | 3.96 | 11.37 | 3.63 | 0.12 | 0.88 | 0.50 | 0.12 |

## Per instance

| Instance | GMI % | Table % | Neural % | ± | Selected % | Chosen | Neural − table (pp) | ± |
|---|---|---|---|---|---|---|---|---|
| F00_lcl | 16.23 | 7.49 | 12.20 | 0.67 | 7.49 | Status-table | +4.70 | 0.97 |
| F00_lch | 13.77 | 9.08 | 5.01 | 0.59 | 5.01 | Neural | -4.07 | 0.98 |
| F01_lcl | 13.36 | 5.17 | 8.08 | 1.18 | 5.17 | Status-table | +2.90 | 1.65 |
| F01_lch | 8.05 | 4.78 | 8.13 | 1.07 | 4.78 | Status-table | +3.35 | 1.67 |
| F02_lcl | 12.03 | 5.83 | 7.46 | 1.00 | 5.83 | Status-table | +1.63 | 1.34 |
| F02_lch | 15.49 | 5.61 | 6.89 | 0.72 | 5.61 | Status-table | +1.28 | 1.14 |
| F03_lcl | 9.67 | 1.68 | 3.11 | 1.10 | 1.68 | Status-table | +1.43 | 0.92 |
| F03_lch | 4.42 | 1.25 | 2.11 | 0.78 | 1.25 | Status-table | +0.86 | 0.83 |
| F04_lcl | 5.99 | 1.42 | 5.88 | 0.81 | 1.42 | Status-table | +4.47 | 0.83 |
| F04_lch | 2.84 | 1.87 | 3.20 | 0.44 | 1.87 | Status-table | +1.33 | 0.77 |
| F05_lcl | 7.05 | 6.86 | 5.40 | 0.67 | 5.40 | Neural | -1.46 | 0.74 |
| F05_lch | 0.87 | 0.51 | 3.57 | 0.40 | 0.87 | GMI | +3.06 | 0.59 |
| F06_lcl (held out) | 4.76 | 3.52 | 26.34 | 1.11 | 3.52 | Status-table | +22.82 | 1.13 |
| F06_lch (held out) | 0.67 | 0.35 | 21.24 | 0.48 | 0.35 | Status-table | +20.89 | 0.67 |
| F07_lcl (held out) | 5.17 | 3.33 | 17.86 | 0.86 | 3.33 | Status-table | +14.53 | 1.18 |
| F07_lch (held out) | 10.50 | 4.55 | 45.45 | 0.80 | 4.55 | Status-table | +40.90 | 0.90 |

