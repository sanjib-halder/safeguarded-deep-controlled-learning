"""Validation on the classic two-supplier (dual-sourcing) problem.

Checks, before any learning:
  1. the simulator and the DP agree (simulated cost of the DP policy = g*);
  2. tuned DI and CDI land close to the DP optimum, as the literature reports
     (DI within ~1-2%, CDI closer still).
Setting in the style of Gijsbrechts et al. (2022): c_r = 100, c_e in {105,110},
l_e = 0, l_r in {2,3} (4 with --full), demand U{0..4}, h = 5, b = 495.

Costs include purchase cost. Gaps are reported both on total cost and on
"excess cost" = cost - mean demand x cheapest price (purchase cost that no
policy can avoid), which is the more informative scale.

Usage:  python experiments/validate_dual_sourcing.py [--full]
"""
from __future__ import annotations

import argparse
import time

from common import RESULTS, Budget, md_table, save_json  # noqa: E402

from supplier_pilot import dp
from supplier_pilot.instance import dual_sourcing_instance
from supplier_pilot.metrics import summarize
from supplier_pilot.simulator import make_streams, simulate
from supplier_pilot.tuning import SeedSplit, tune_dual_index, tune_single_source


def run(full: bool):
    B = Budget.preset("full" if full else "quick")
    seeds = SeedSplit.default(B.tune_E, B.val_E, B.test_E)
    lrs = [2, 3, 4] if full else [2, 3]
    rows, checks = [], []
    for ce in (105, 110):
        for lr in lrs:
            inst = dual_sourcing_instance(ce, 100, 0, lr, capacity=8)
            t0 = time.time()
            sol = dp.solve_instance(inst, -12, 24)
            chk = dp.check_boundaries(sol, inst, T=20000)
            chk.update(sol.summary())
            checks.append(chk)
            print(f"[{inst.name}] DP g*={sol.g:.4f} [{sol.g_lower:.4f},{sol.g_upper:.4f}] "
                  f"sim={chk['sim_avg_cost']:.4f} boundary={chk['boundary_hit_share']:.4f} "
                  f"({time.time() - t0:.1f}s)")
            tr = make_streams(inst, seeds.train, B.tune_T)
            pols = [dp.DPPolicy(sol)]
            for i in range(2):
                p, _, _ = tune_single_source(inst, i, tr, B.tune_warm)
                pols.append(p)
            for capped in (False, True):
                p, _, ev = tune_dual_index(inst, tr, B.tune_warm, capped)
                p.name += f"(Se={int(p.S[0])},Sr={int(p.S[1])}" + (
                    f",cap={int(p.cap[1])})" if capped else ")")
                pols.append(p)
            test = make_streams(inst, seeds.test, B.test_T)
            ref = None
            for p in pols:
                res = simulate(inst, p, streams=test, warmup=B.test_warm)
                ref = ref or res
                r = summarize(res, inst, ref=ref, ref_cost=sol.g)
                r["dp_g"] = sol.g
                rows.append(r)
                print(f"   {r['policy']:<40s} cost={r['avg_cost']:.3f}±{r['ci95']:.3f} "
                      f"paired gap_excess={r['paired_gap_excess_pct']:6.2f}% ±{r['paired_gap_excess_ci95']:.2f}")
    save_json({"rows": rows, "dp_checks": checks}, "validation_dual_sourcing.json")
    md = ["# Dual-sourcing validation (N = 2)", "",
          "Simulator vs DP consistency (simulated cost of the DP-optimal policy vs g*):", "",
          md_table(checks, [("instance", "Instance", ""), ("g_star", "g* (DP)", ".4f"),
                            ("g_lower", "lower", ".4f"), ("g_upper", "upper", ".4f"),
                            ("sim_avg_cost", "simulated", ".4f"), ("rel_diff", "rel. diff", ".2e"),
                            ("boundary_hit_share", "boundary hits", ".4f"),
                            ("states", "states", ""), ("seconds", "DP s", "")]),
          "", "Policies on the held-out test seeds. Gaps are paired (common random numbers) against the simulated DP-optimal policy; excess cost = cost minus mean demand x cheapest price.", "",
          md_table(rows, [("instance", "Instance", ""), ("policy", "Policy", ""),
                          ("avg_cost", "Avg cost", ".3f"), ("ci95", "±95%", ".3f"),
                          ("paired_gap_total_pct", "Gap total %", ".2f"),
                          ("paired_gap_excess_pct", "Gap excess %", ".2f"),
                          ("paired_gap_excess_ci95", "±95%", ".2f"),
                          ("fill_rate", "Fill rate", ".4f")])]
    path = f"{RESULTS}/validation_dual_sourcing.md"
    with open(path, "w") as f:
        f.write("\n".join(md) + "\n")
    print("wrote", path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    run(ap.parse_args().full)
