"""Truncation check of the exact DP (Supplementary material, Section S1, Table S2).

For 16 test instances (replicates 0-3 of the test seed in cells lcl, lch, hnh
and hch) it solves the DP with net inventory in [-10, 26] and in [-20, 40], and
reports (i) how often the truncated DP policy reaches a bound in an untruncated
simulation, (ii) the change in g* when the range is widened, and (iii) the
paired cost difference of the two DP policies in the untruncated simulator.
On two instances it also widens the range to [-30, 60].

Usage (from the project folder):
    python experiments/check_truncation.py          # about 5-10 minutes on a laptop

Writes results/truncation_check.json and results/truncation_check.md.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from supplier_pilot import dp  # noqa: E402
from supplier_pilot.instance import factorial_instance  # noqa: E402
from supplier_pilot.simulator import make_streams, simulate  # noqa: E402

RESULTS = os.path.join(ROOT, "results")
CELLS = [("low", "correlated", "low"), ("low", "correlated", "high"),
         ("high", "none", "high"), ("high", "correlated", "high")]
TEST_SEED = 7


def main():
    rows = []
    for rep in range(4):
        for cell in CELLS:
            inst = factorial_instance(TEST_SEED, rep, *cell)
            a = dp.solve_instance(inst, -10, 26)
            b = dp.solve_instance(inst, -20, 40)
            chk = dp.check_boundaries(a, inst, T=20000, seeds=range(4))
            streams = make_streams(inst, range(9000, 9032), 10000)
            ra = simulate(inst, dp.DPPolicy(a), streams=streams, warmup=500)
            rb = simulate(inst, dp.DPPolicy(b), streams=streams, warmup=500)
            excess = b.g - inst.unavoidable_cost()
            d = 100 * (ra.episode_cost - rb.episode_cost) / excess
            half = 2.04 * d.std(ddof=1) / np.sqrt(len(d))
            r = {"instance": inst.name, "boundary_share_pct": 100 * chk["boundary_hit_share"],
                 "dg_star_pct_excess": 100 * (b.g - a.g) / excess,
                 "policy_diff_pct_excess": float(d.mean()), "policy_diff_ci": float(half)}
            rows.append(r)
            print(json.dumps(r), flush=True)
    wider = []
    for rep, cell in [(0, CELLS[0]), (2, CELLS[1])]:
        inst = factorial_instance(TEST_SEED, rep, *cell)
        b = dp.solve_instance(inst, -20, 40)
        w = dp.solve_instance(inst, -30, 60)
        wider.append({"instance": inst.name, "dg_star_pct_excess": 100 * (w.g - b.g) / (w.g - inst.unavoidable_cost())})
        print(json.dumps(wider[-1]), flush=True)
    os.makedirs(RESULTS, exist_ok=True)
    json.dump({"rows": rows, "wider": wider}, open(os.path.join(RESULTS, "truncation_check.json"), "w"), indent=1)
    lines = ["# Truncation check of the exact DP", "",
             "| Instance | Boundary (%) | Δg* (% of excess) | Policy difference (%) |", "|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['instance']} | {r['boundary_share_pct']:.3f} | {r['dg_star_pct_excess']:.2f} | "
                     f"{r['policy_diff_pct_excess']:+.2f} ± {r['policy_diff_ci']:.2f} |")
    lines += ["", "Range widened to [-30, 60]:", ""] + [f"- {w['instance']}: Δg* = {w['dg_star_pct_excess']:.3f}% of excess cost"
                                                        for w in wider]
    open(os.path.join(RESULTS, "truncation_check.md"), "w").write("\n".join(lines) + "\n")
    print("wrote results/truncation_check.md")


if __name__ == "__main__":
    main()
