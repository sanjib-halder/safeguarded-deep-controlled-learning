# State-aware neural policy

**Split:** fresh-instance generalisation. Every test instance is held out: training and validation use a different seed, so no test instance (or its supplier configuration) is seen.

Target `factorial-hard`, mode `generalise`, base `table`, bound ±2, 1500 iterations on `cpu` (1459 s training). Gaps are paired against the DP optimum on held-out test seeds (NumPy simulator), as % of excess cost. *Selected* = lowest validation cost among GMI, status-table and neural (heuristic as fallback).

## Summary (mean gap %, share of instances)

| Group | n | GMI | Status-table | Neural | Selected | Neural sig. better than table | Neural sig. worse than table | Neural sig. better than GMI | Neural selected |
|---|---|---|---|---|---|---|---|---|---|
| held-out instances | 16 | 8.18 | 3.96 | 3.96 | 3.98 | 0.00 | 0.00 | 0.81 | 0.00 |
| all | 16 | 8.18 | 3.96 | 3.96 | 3.98 | 0.00 | 0.00 | 0.81 | 0.00 |

## Per instance

| Instance | GMI % | Table % | Neural % | ± | Selected % | Chosen | Neural − table (pp) | ± |
|---|---|---|---|---|---|---|---|---|
| F00_lcl (held out) | 16.23 | 7.49 | 7.49 | 0.88 | 7.49 | Status-table | +0.00 | 0.00 |
| F00_lch (held out) | 13.77 | 9.08 | 9.08 | 1.10 | 9.08 | Status-table | +0.00 | 0.00 |
| F01_lcl (held out) | 13.36 | 5.17 | 5.17 | 1.36 | 5.17 | Status-table | +0.00 | 0.00 |
| F01_lch (held out) | 8.05 | 4.78 | 4.78 | 1.27 | 4.78 | Status-table | +0.00 | 0.00 |
| F02_lcl (held out) | 12.03 | 5.83 | 5.83 | 1.04 | 5.83 | Status-table | +0.00 | 0.00 |
| F02_lch (held out) | 15.49 | 5.61 | 5.61 | 0.98 | 5.61 | Status-table | +0.00 | 0.00 |
| F03_lcl (held out) | 9.67 | 1.68 | 1.68 | 1.00 | 1.68 | Status-table | +0.00 | 0.00 |
| F03_lch (held out) | 4.42 | 1.25 | 1.25 | 0.68 | 1.25 | Status-table | +0.00 | 0.00 |
| F04_lcl (held out) | 5.99 | 1.42 | 1.42 | 0.45 | 1.42 | Status-table | +0.00 | 0.00 |
| F04_lch (held out) | 2.84 | 1.87 | 1.87 | 0.55 | 1.87 | Status-table | +0.00 | 0.00 |
| F05_lcl (held out) | 7.05 | 6.86 | 6.86 | 0.59 | 6.86 | Status-table | +0.00 | 0.00 |
| F05_lch (held out) | 0.87 | 0.51 | 0.51 | 0.46 | 0.87 | GMI | +0.00 | 0.00 |
| F06_lcl (held out) | 4.76 | 3.52 | 3.52 | 0.77 | 3.52 | Status-table | +0.00 | 0.00 |
| F06_lch (held out) | 0.67 | 0.35 | 0.35 | 0.48 | 0.35 | Status-table | +0.00 | 0.00 |
| F07_lcl (held out) | 5.17 | 3.33 | 3.33 | 0.96 | 3.33 | Status-table | +0.00 | 0.00 |
| F07_lch (held out) | 10.50 | 4.55 | 4.55 | 0.60 | 4.55 | Status-table | +0.00 | 0.00 |

