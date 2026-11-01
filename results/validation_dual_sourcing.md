# Dual-sourcing validation (N = 2)

Simulator vs DP consistency (simulated cost of the DP-optimal policy vs g*):

| Instance | g* (DP) | lower | upper | simulated | rel. diff | boundary hits | states | DP s |
|---|---|---|---|---|---|---|---|---|
| dual_ce105_lr2 | 216.7698 | 216.7698 | 216.7698 | 216.8786 | 5.02e-04 | 0.0000 | 333 | 0.0 |
| dual_ce105_lr3 | 216.8767 | 216.8767 | 216.8767 | 216.9952 | 5.46e-04 | 0.0000 | 2997 | 0.1 |
| dual_ce105_lr4 | 216.8931 | 216.8931 | 216.8931 | 217.0161 | 5.67e-04 | 0.0000 | 26973 | 0.4 |
| dual_ce110_lr2 | 219.7333 | 219.7333 | 219.7333 | 219.8490 | 5.27e-04 | 0.0000 | 333 | 0.0 |
| dual_ce110_lr3 | 220.3411 | 220.3411 | 220.3411 | 220.4699 | 5.85e-04 | 0.0000 | 2997 | 0.0 |
| dual_ce110_lr4 | 220.6122 | 220.6122 | 220.6122 | 220.7562 | 6.52e-04 | 0.0000 | 26973 | 0.3 |

Policies on the held-out test seeds. Gaps are paired (common random numbers) against the simulated DP-optimal policy; excess cost = cost minus mean demand x cheapest price.

| Instance | Policy | Avg cost | ±95% | Gap total % | Gap excess % | ±95% | Fill rate |
|---|---|---|---|---|---|---|---|
| dual_ce105_lr2 | DP-optimal | 216.825 | 0.194 | 0.00 | 0.00 | 0.00 | 1.0000 |
| dual_ce105_lr2 | single[expedited](s=4,S=4) | 220.051 | 0.205 | 1.49 | 19.24 | 0.08 | 1.0000 |
| dual_ce105_lr2 | single[regular](s=11,S=11) | 229.083 | 0.211 | 5.65 | 73.10 | 0.43 | 0.9960 |
| dual_ce105_lr2 | DI(Se=4,Sr=7) | 217.073 | 0.195 | 0.11 | 1.48 | 0.03 | 1.0000 |
| dual_ce105_lr2 | CDI(Se=4,Sr=7,cap=2) | 216.834 | 0.196 | 0.00 | 0.06 | 0.03 | 1.0000 |
| dual_ce105_lr3 | DP-optimal | 216.931 | 0.198 | 0.00 | 0.00 | 0.00 | 1.0000 |
| dual_ce105_lr3 | single[expedited](s=4,S=4) | 220.051 | 0.205 | 1.44 | 18.49 | 0.06 | 1.0000 |
| dual_ce105_lr3 | single[regular](s=14,S=14) | 234.950 | 0.221 | 8.31 | 106.77 | 0.61 | 0.9951 |
| dual_ce105_lr3 | DI(Se=4,Sr=8) | 217.511 | 0.196 | 0.27 | 3.44 | 0.04 | 1.0000 |
| dual_ce105_lr3 | CDI(Se=4,Sr=9,cap=1) | 216.931 | 0.198 | 0.00 | 0.00 | 0.00 | 1.0000 |
| dual_ce105_lr4 | DP-optimal | 216.946 | 0.197 | 0.00 | 0.00 | 0.00 | 1.0000 |
| dual_ce105_lr4 | single[expedited](s=4,S=4) | 220.051 | 0.205 | 1.43 | 18.38 | 0.06 | 1.0000 |
| dual_ce105_lr4 | single[regular](s=17,S=17) | 239.652 | 0.219 | 10.47 | 134.41 | 0.74 | 0.9954 |
| dual_ce105_lr4 | DI(Se=4,Sr=8) | 217.720 | 0.198 | 0.36 | 4.58 | 0.04 | 1.0000 |
| dual_ce105_lr4 | CDI(Se=4,Sr=11,cap=1) | 216.946 | 0.198 | -0.00 | -0.00 | 0.00 | 1.0000 |
| dual_ce110_lr2 | DP-optimal | 219.789 | 0.194 | 0.00 | 0.00 | 0.00 | 1.0000 |
| dual_ce110_lr2 | single[expedited](s=4,S=4) | 230.053 | 0.215 | 4.67 | 52.01 | 0.12 | 1.0000 |
| dual_ce110_lr2 | single[regular](s=11,S=11) | 229.083 | 0.211 | 4.23 | 47.10 | 0.35 | 0.9960 |
| dual_ce110_lr2 | DI(Se=4,Sr=8) | 220.177 | 0.196 | 0.18 | 1.96 | 0.04 | 1.0000 |
| dual_ce110_lr2 | CDI(Se=4,Sr=9,cap=2) | 219.856 | 0.191 | 0.03 | 0.34 | 0.03 | 1.0000 |
| dual_ce110_lr3 | DP-optimal | 220.398 | 0.196 | 0.00 | 0.00 | 0.00 | 1.0000 |
| dual_ce110_lr3 | single[expedited](s=4,S=4) | 230.053 | 0.215 | 4.38 | 47.47 | 0.12 | 1.0000 |
| dual_ce110_lr3 | single[regular](s=14,S=14) | 234.950 | 0.221 | 6.60 | 71.54 | 0.49 | 0.9951 |
| dual_ce110_lr3 | DI(Se=4,Sr=9) | 221.193 | 0.199 | 0.36 | 3.91 | 0.04 | 1.0000 |
| dual_ce110_lr3 | CDI(Se=4,Sr=10,cap=2) | 220.519 | 0.194 | 0.05 | 0.59 | 0.03 | 1.0000 |
| dual_ce110_lr4 | DP-optimal | 220.665 | 0.196 | 0.00 | 0.00 | 0.00 | 1.0000 |
| dual_ce110_lr4 | single[expedited](s=4,S=4) | 230.053 | 0.215 | 4.26 | 45.55 | 0.12 | 1.0000 |
| dual_ce110_lr4 | single[regular](s=17,S=17) | 239.652 | 0.219 | 8.61 | 92.12 | 0.59 | 0.9954 |
| dual_ce110_lr4 | DI(Se=4,Sr=11) | 221.730 | 0.196 | 0.48 | 5.17 | 0.04 | 1.0000 |
| dual_ce110_lr4 | CDI(Se=4,Sr=11,cap=2) | 220.910 | 0.196 | 0.11 | 1.19 | 0.03 | 1.0000 |
