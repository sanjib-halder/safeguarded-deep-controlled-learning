# Rolling-horizon stochastic MILP: two-stage vs. multi-stage

Setting: P3 (long outages) and P4 (correlated disruptions). Each instance runs 4 test
episodes × 190 periods (after 30 warm-up periods) with common random numbers. The gap is
paired against the simulated DP-optimal policy, as % of excess cost. The horizon is
H = 8 periods; HiGHS (via SciPy) has a 5 s limit per solve. Demand scenarios use
stratified sampling.

| Instance | Variant | Gap % excess | ±95% | Fill rate | Mean solve (s) | p95 solve (s) |
|---|---|---|---|---|---|---|
| P3 | two-stage, M = 20 | 34.9 | 14.1 | 0.9933 | 1.21 | 3.14 |
| P3 | multi-stage, tree 5×3×2 (30 scenarios) | 17.4 | 16.3 | 0.9966 | 3.55 | 5.03 |
| P4 | two-stage, M = 20 | 33.2 | 19.2 | 0.9906 | 1.06 | 3.81 |
| P4 | multi-stage, tree 5×3×2 (30 scenarios) | 21.1 | 27.9 | 0.9946 | 3.17 | 5.02 |

For reference, the GMI heuristic's gap at the full budget is 3.3% (P3) and 5.7% (P4).

Observations:

- The multi-stage tree roughly halves the two-stage gap and raises the fill rate. This is
  what removing the two-stage optimism should do, but the confidence intervals overlap at
  4 episodes. Use the full budget (`--saa both`) for the paper.
- Without stratified sampling, an earlier run of the tree variant (4×3×2) was *worse*
  than two-stage (63–99% gap, fill rate about 0.95–0.97). A first branch of only 4 random
  draws often missed the high-demand outcome, so the first decision under-protected.
  Stratification fixed this; it is now the default for both variants.
- Both MILP variants stay far behind the tuned GMI heuristic and need 1–3.5 s per decision.
  The p95 solve time of the tree variant sits at the 5 s limit, so some solves end
  time-limited. A longer limit or a larger tree is the next lever for a stronger baseline.
