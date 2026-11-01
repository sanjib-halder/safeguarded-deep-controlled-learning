# Random-instance study (N = 3)

40 random DP-tractable instances, budget `quick`. Gaps are paired against the DP optimum on held-out test seeds, as % of excess cost (cost minus mean demand × cheapest price).

## Heuristic (GMI) gap to the optimum

mean 4.13%, median 3.52%, 90th percentile 8.12%, max 13.09%. Best single-source policy: median gap 105.8%.

## Status-table learner

mean gap 2.71% (median 2.04%); all training runs beat GMI significantly on 48% of instances; at least one run significantly worse on 5%; share of the GMI gap closed: median 25%, mean 27% (instances with GMI gap > 0.25%).

## What predicts a large heuristic gap?

Spearman rank correlations with the GMI gap:

| Feature | rho | p |
|---|---|---|
| shock_rate | 0.32 | 0.046 |
| K_max | 0.29 | 0.068 |
| premium | -0.26 | 0.111 |
| has_shock | 0.25 | 0.118 |
| lead_gap | 0.21 | 0.188 |
| down_B | 0.18 | 0.267 |
| lead_C | 0.17 | 0.303 |
| down_C | -0.16 | 0.314 |
| down_A | -0.13 | 0.429 |
| outage_C | 0.09 | 0.595 |
| moq_c | -0.09 | 0.600 |

OLS on standardised features (R² = 0.38, n = 40; descriptive only):

| Feature | β per SD | SE | t |
|---|---|---|---|
| premium | -1.30 | 0.54 | -2.39 |
| K_max | 1.11 | 0.51 | 2.19 |
| moq_c | 0.05 | 0.52 | 0.09 |
| shock_rate | 1.74 | 1.22 | 1.43 |
| has_shock | -1.29 | 1.27 | -1.01 |
| down_A | -0.04 | 0.52 | -0.09 |
| down_B | 0.68 | 0.53 | 1.27 |
| down_C | -0.81 | 0.55 | -1.47 |
| outage_C | 0.83 | 0.57 | 1.45 |
| lead_C | 0.61 | 0.86 | 0.71 |
| lead_gap | -0.29 | 0.81 | -0.36 |

Feature key: premium = price premium of the fastest supplier; K_max = largest fixed cost; down_X = long-run share of time supplier X is down; outage_C = mean outage length of the slow supplier (periods); shock_rate / has_shock = regional shock hitting B and C together; lead_gap = lead-time difference fastest vs slowest.

## Per-instance results

| Instance | Shock rate | Down share C | GMI gap % | Table gap % | Single gap % |
|---|---|---|---|---|---|
| R000 | 0.000 | 0.085 | 0.40 | 0.55 | 64.1 |
| R001 | 0.000 | 0.081 | 3.47 | 3.24 | 80.2 |
| R002 | 0.024 | 0.075 | 3.18 | 1.70 | 64.2 |
| R003 | 0.000 | 0.086 | 3.62 | 0.92 | 65.6 |
| R004 | 0.000 | 0.067 | 11.12 | 5.04 | 33.9 |
| R005 | 0.000 | 0.113 | 4.33 | 1.75 | 117.7 |
| R006 | 0.026 | 0.151 | 5.00 | 3.62 | 74.6 |
| R007 | 0.000 | 0.047 | 13.09 | 1.76 | 102.2 |
| R008 | 0.023 | 0.011 | 9.08 | 7.29 | 68.7 |
| R009 | 0.018 | 0.053 | 2.06 | 2.10 | 62.6 |
| R010 | 0.007 | 0.027 | 7.80 | 5.13 | 124.4 |
| R011 | 0.019 | 0.245 | 6.53 | 5.90 | 175.6 |
| R012 | 0.024 | 0.076 | 4.85 | 2.04 | 108.7 |
| R013 | 0.000 | 0.118 | 5.00 | 2.04 | 140.0 |
| R014 | 0.007 | 0.047 | 1.98 | 1.81 | 191.4 |
| R015 | 0.017 | 0.015 | 4.73 | 4.59 | 77.3 |
| R016 | 0.000 | 0.037 | 4.24 | 3.54 | 62.3 |
| R017 | 0.019 | 0.069 | 8.01 | 5.59 | 193.9 |
| R018 | 0.018 | 0.067 | 1.20 | 1.34 | 143.3 |
| R019 | 0.000 | 0.117 | 0.98 | 0.54 | 196.6 |
| R020 | 0.015 | 0.251 | 1.21 | 0.97 | 152.6 |
| R021 | 0.022 | 0.076 | 7.04 | 2.34 | 244.0 |
| R022 | 0.016 | 0.120 | 5.42 | 4.67 | 96.1 |
| R023 | 0.000 | 0.062 | 0.18 | 0.26 | 41.4 |
| R024 | 0.000 | 0.252 | 10.33 | 8.47 | 185.6 |
| R025 | 0.027 | 0.104 | 5.73 | 4.19 | 76.1 |
| R026 | 0.000 | 0.085 | 2.01 | 1.41 | 326.9 |
| R027 | 0.000 | 0.188 | 0.98 | 0.60 | 78.5 |
| R028 | 0.016 | 0.040 | 3.58 | 2.66 | 142.4 |
| R029 | 0.017 | 0.062 | 6.13 | 4.92 | 204.4 |
| R030 | 0.000 | 0.198 | 2.70 | 1.89 | 145.1 |
| R031 | 0.000 | 0.131 | 0.77 | 1.01 | 102.8 |
| R032 | 0.000 | 0.070 | 1.34 | 1.24 | 99.4 |
| R033 | 0.016 | 0.030 | 2.59 | 2.06 | 54.0 |
| R034 | 0.000 | 0.061 | 1.27 | 1.05 | 13.0 |
| R035 | 0.000 | 0.069 | 4.98 | 4.34 | 143.0 |
| R036 | 0.000 | 0.150 | 2.64 | 1.69 | 144.2 |
| R037 | 0.010 | 0.140 | 1.82 | 1.37 | 97.0 |
| R038 | 0.023 | 0.077 | 2.57 | 2.48 | 120.9 |
| R039 | 0.000 | 0.120 | 1.40 | 0.11 | 147.6 |
