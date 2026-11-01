# Blocked factorial study (N = 3)

Design: 2 × 2 × 2 factors × 16 replicates = 128 instances, budget `quick`. Within a replicate all eight cells share lead times, MOQ and disruption probabilities, so they differ only in the factors. Gaps are paired against the DP optimum on held-out test seeds, as % of excess cost. The *selected* policy is chosen on validation seeds among the heuristic (GMI) and the learner runs, so the heuristic is the fallback.

Factor levels: **fixed_cost** low = (0.0, 10.0), high = (25.0, 45.0); **shock** none = (0.0, 0.0), correlated = (0.015, 0.03); **premium** low = (0.05, 0.1), high = (0.18, 0.25).

## Cell means (± 95% CI across replicates)

| Fixed cost | Shock | Premium | n | GMI gap % | ± | Selected gap % | ± | Gap closed % | ± | Share sig. better | Share fallback | Single-source median gap % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| low | none | low | 16 | 7.98 | 3.19 | 4.62 | 2.30 | 38 | 16 | 0.81 | 0.00 | 186 |
| low | none | high | 16 | 3.74 | 1.29 | 2.08 | 1.01 | 44 | 13 | 0.81 | 0.06 | 175 |
| low | correlated | low | 16 | 7.88 | 2.52 | 4.23 | 1.58 | 41 | 13 | 0.81 | 0.19 | 203 |
| low | correlated | high | 16 | 5.11 | 1.73 | 3.04 | 1.26 | 37 | 13 | 0.75 | 0.12 | 193 |
| high | none | low | 16 | 5.04 | 1.23 | 3.82 | 1.20 | 25 | 14 | 0.69 | 0.19 | 95 |
| high | none | high | 16 | 3.62 | 1.05 | 2.19 | 0.86 | 40 | 15 | 0.81 | 0.12 | 109 |
| high | correlated | low | 16 | 6.41 | 1.68 | 3.88 | 1.00 | 35 | 12 | 0.81 | 0.00 | 125 |
| high | correlated | high | 16 | 4.63 | 1.33 | 3.21 | 0.98 | 30 | 14 | 0.62 | 0.19 | 127 |

## Heuristic gap: effects

Linear model with replicate fixed effects, effect-coded factors and all interactions (n = 128, residual dof = 105, R² = 0.51). Main-effect coefficients are average high-minus-low differences in percentage points; two-way terms are the change in one factor's effect when the other moves from low to high.

| Term | Effect (high − low) | SE | t | p |
|---|---|---|---|---|
| fixed_cost | -1.25 | 0.52 | -2.42 | 0.0173 |
| shock | 0.91 | 0.52 | 1.76 | 0.0809 |
| premium | -2.55 | 0.52 | -4.93 | 0.0000 |
| fixed_cost × shock | 0.28 | 0.52 | 0.53 | 0.5947 |
| fixed_cost × premium | 0.95 | 0.52 | 1.84 | 0.0688 |
| shock × premium | 0.28 | 0.52 | 0.53 | 0.5954 |
| fixed_cost × shock × premium | -0.46 | 0.52 | -0.88 | 0.3797 |

Nonparametric check (within-replicate differences, Wilcoxon signed-rank):

| Factor | Median within-replicate diff | Mean diff | Replicates with increase | Replicates | Wilcoxon p |
|---|---|---|---|---|---|
| fixed_cost | -0.50 | -1.25 | 6 | 16 | 0.1167 |
| shock | 0.83 | 0.91 | 11 | 16 | 0.0507 |
| premium | -2.61 | -2.55 | 4 | 16 | 0.0063 |

## Share of the heuristic gap closed by the selected learner: effects

| Term | Effect (high − low) | SE | t | p |
|---|---|---|---|---|
| fixed_cost | -7.81 | 4.30 | -1.82 | 0.0724 |
| shock | -1.29 | 4.29 | -0.30 | 0.7634 |
| premium | 3.15 | 4.30 | 0.73 | 0.4661 |
| fixed_cost × shock | 1.04 | 4.29 | 0.24 | 0.8081 |
| fixed_cost × premium | 1.92 | 4.30 | 0.45 | 0.6561 |
| shock × premium | -7.78 | 4.29 | -1.81 | 0.0728 |
| fixed_cost × shock × premium | -2.88 | 4.29 | -0.67 | 0.5032 |

| Factor | Median within-replicate diff | Mean diff | Replicates with increase | Replicates | Wilcoxon p |
|---|---|---|---|---|---|
| fixed_cost | -12.41 | -8.32 | 5 | 16 | 0.1439 |
| shock | -4.74 | -1.53 | 6 | 16 | 0.3225 |
| premium | 7.66 | 3.89 | 11 | 16 | 0.1928 |

## Reading the results

- Treat main effects with Wilcoxon p < 0.05 *and* a consistent sign across replicates as findings; with 3 main effects, a Holm correction is advisable for the paper.
- Interactions are estimated with less precision; report them as exploratory unless p is small and the interaction plot shows a clear pattern.
- These are quick-budget results; rerun the significant cells at `--budget full` before publishing.

