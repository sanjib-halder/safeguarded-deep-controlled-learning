# State-aware neural policy

Target `factorial-hard`, mode `per-instance`, base `table`, bound ±2, 1000 iterations on `cpu` (1281 s training). Gaps are paired against the DP optimum on held-out test seeds (NumPy simulator), as % of excess cost. *Selected* = lowest validation cost among GMI, status-table and neural (heuristic as fallback).

## Summary (mean gap %, share of instances)

| Group | n | GMI | Status-table | Neural | Selected | Neural sig. better than table | Neural sig. worse than table | Neural sig. better than GMI | Neural selected |
|---|---|---|---|---|---|---|---|---|---|
| training instances | 4 | 12.27 | 7.31 | 4.00 | 4.00 | 0.75 | 0.00 | 1.00 | 0.75 |
| all | 4 | 12.27 | 7.31 | 4.00 | 4.00 | 0.75 | 0.00 | 1.00 | 0.75 |

## Per instance

| Instance | GMI % | Table % | Neural % | ± | Selected % | Chosen | Neural − table (pp) | ± |
|---|---|---|---|---|---|---|---|---|
| F00_lcl | 16.23 | 7.49 | 4.12 | 0.88 | 4.12 | Neural | -3.37 | 0.61 |
| F00_lch | 13.77 | 9.08 | 2.96 | 0.69 | 2.96 | Neural | -6.12 | 0.95 |
| F02_lcl | 12.03 | 5.83 | 5.83 | 1.04 | 5.83 | Status-table | +0.00 | 0.00 |
| F05_lcl | 7.05 | 6.86 | 3.08 | 0.73 | 3.08 | Neural | -3.77 | 0.60 |

