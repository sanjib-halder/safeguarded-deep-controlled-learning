# Gradient check

**Verdict:** BIASED/UNINFORMATIVE: gradients do not reliably point downhill; use a gradient-free learner.

Mean sign agreement (autograd vs exact finite differences, at the step size with most non-zero differences): 0.48. Descent along the negative gradient lowered the exact cost in 25% of cases (two smallest steps).

| Instance | Point | Horizon | Sign agreement | at e | dL, step 1 | dL, step 2 | dL random dir, step 1 | Descent works |
|---|---|---|---|---|---|---|---|---|
| F00_lch | untrained (= status table) | 48 | – | 0.03 | +0.0000 | +0.0000 | +0.0000 | False |
| F00_lch | perturbed | 48 | – | 0.03 | +0.0000 | +0.0000 | +0.0000 | False |
| F00_lch | untrained (= status table) | 240 | – | 0.03 | +0.0000 | +0.0000 | +0.0000 | False |
| F00_lch | perturbed | 240 | – | 0.03 | +0.0000 | +0.0000 | +0.0000 | False |
| F05_lcl | untrained (= status table) | 48 | – | 0.03 | +0.0000 | +0.0000 | +0.0000 | False |
| F05_lcl | perturbed | 48 | 0.64 | 0.1 | -0.4104 | -4.2202 | +0.0000 | True |
| F05_lcl | untrained (= status table) | 240 | – | 0.03 | +0.0000 | +0.0000 | +0.0000 | False |
| F05_lcl | perturbed | 240 | 0.33 | 0.1 | -1.0951 | -7.9180 | +0.0000 | True |

Negative dL = the exact cost went down. Full details in `gradient_check.json`.
