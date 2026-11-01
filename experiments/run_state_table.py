"""Inventory-aware status table (non-neural): does inventory/pipeline
awareness explain the neural policy's per-instance gains?

For each instance it starts from the tuned status table (same cached
baselines and test seeds as train_neural.py), splits every status row into
inventory-position bands (default 3: low / normal / high, cut at the terciles
seen under the status table) and tunes the rows by direct coordinate search
with common random numbers. Evaluation is identical to train_neural.py:
paired against the DP optimum on the same held-out test seeds, with a
validation-selected policy (heuristic and status table as fallbacks).

Usage
  python experiments/run_state_table.py                      # the 4 diagnostic instances
  python experiments/run_state_table.py --only F00_lch --bands 5
  python experiments/run_state_table.py --all-hard --jobs 8  # all 16 factorial-hard instances
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from common import RESULTS, Budget, md_table, save_json  # noqa: E402
from train_neural import build_targets, gmi_from, prepare  # noqa: E402

from supplier_pilot import dp
from supplier_pilot.learner import StatusTablePolicy, tune_state_table
from supplier_pilot.metrics import ci95, summarize
from supplier_pilot.simulator import make_streams, mean_cost, simulate

DIAG = ["F00_lch", "F00_lcl", "F05_lcl", "F02_lcl"]


def run_one(args):
    inst, key, budget, bands, table_runs = args
    t0 = time.time()
    prep = prepare((inst, key, budget, table_runs))
    B = Budget.preset(budget)
    gmi = gmi_from(prep)
    tab = StatusTablePolicy(gmi, np.array(prep["table"], float), "Status-table")
    tr = make_streams(inst, [key * 10 + 7_000_000 + j for j in range(B.tune_E)], B.tune_T)
    stt, evals, info = tune_state_table(inst, tab, tr, B.tune_warm, bands=bands)
    stt.name = "Status x inventory"
    va = make_streams(inst, prep["seeds"]["val"], B.tune_T)
    val = {"GMI": prep["gmi_val"], "Status-table": prep["table_val"],
           "Status x inventory": mean_cost(inst, stt, va, B.tune_warm)}
    chosen = min(val, key=val.get)
    sol = dp.solve_instance(inst, -10, 26)
    test = make_streams(inst, prep["seeds"]["test"], B.test_T)
    ref = simulate(inst, dp.DPPolicy(sol), streams=test, warmup=B.test_warm)
    res, row = {}, {"instance": inst.name, "val": val, "selected": chosen, "bands": info, "evals": evals}
    for name, pol in (("GMI", gmi), ("Status-table", tab), ("Status x inventory", stt)):
        res[name] = simulate(inst, pol, streams=test, warmup=B.test_warm)
        s = summarize(res[name], inst, ref=ref, ref_cost=sol.g)
        row[name] = {"gap": s["paired_gap_excess_pct"], "gap_ci": s["paired_gap_excess_ci95"],
                     "gap_total": s["paired_gap_total_pct"], "fill": s["fill_rate"], "volume_share": s["volume_share"]}
    row["Selected"] = dict(row[chosen], chosen=chosen)
    exc = sol.g - inst.unavoidable_cost()
    d, h = ci95(res["Status x inventory"].episode_cost - res["Status-table"].episode_cost)
    row["inv_vs_table"] = {"diff_pct_excess": 100 * d / exc, "ci": 100 * h / exc,
                           "better": bool(d + h < 0), "worse": bool(d - h > 0)}
    row["table_params"] = stt.table.tolist()
    row["seconds"] = round(time.time() - t0, 1)
    print(f"[{inst.name}] GMI {row['GMI']['gap']:.2f}%  table {row['Status-table']['gap']:.2f}%  "
          f"status x inventory {row['Status x inventory']['gap']:.2f}%  "
          f"(diff {row['inv_vs_table']['diff_pct_excess']:+.2f} ± {row['inv_vs_table']['ci']:.2f} pp; "
          f"{evals} evals, {row['seconds']:.0f}s)", flush=True)
    return row


def neural_reference():
    """Per-instance neural results, if that run's JSON is in results/."""
    path = os.path.join(RESULTS, "neural_factorial-hard_per-instance_table.json")
    if not os.path.exists(path):
        return {}
    d = json.load(open(path))
    return {r["instance"]: r["Neural"]["gap"] for r in d["rows"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=DIAG)
    ap.add_argument("--all-hard", action="store_true", help="all 16 factorial-hard instances (seed 7)")
    ap.add_argument("--bands", type=int, default=3)
    ap.add_argument("--budget", default="quick", choices=["smoke", "quick", "full"])
    ap.add_argument("--table-runs", type=int, default=2)
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    import types
    targets = build_targets(types.SimpleNamespace(target="factorial-hard", seed=7, reps=list(range(8))))
    if not a.all_hard:
        targets = [t for t in targets if t[0].name in set(a.only)]
    tasks = [(inst, key, a.budget, a.bands, a.table_runs) for inst, key, _ in targets]
    if a.jobs > 1:
        with ProcessPoolExecutor(a.jobs) as ex:
            rows = list(ex.map(run_one, tasks))
    else:
        rows = [run_one(t) for t in tasks]
    neu = neural_reference()
    for r in rows:
        r["neural_per_instance_gap"] = neu.get(r["instance"], float("nan"))
    m = lambda k: float(np.mean([r[k]["gap"] for r in rows]))
    summary = {"n": len(rows), "GMI": m("GMI"), "Status-table": m("Status-table"),
               "Status x inventory": m("Status x inventory"), "Selected": m("Selected"),
               "inv_beats_table": float(np.mean([r["inv_vs_table"]["better"] for r in rows])),
               "inv_worse_table": float(np.mean([r["inv_vs_table"]["worse"] for r in rows])),
               "bands": a.bands, "budget": a.budget}
    tag = f"state_table_b{a.bands}{'_all' if a.all_hard else ''}"
    save_json({"summary": summary, "rows": rows}, f"{tag}.json")
    L = ["# Inventory-aware status table (non-neural)", "",
         f"{len(rows)} instances, {a.bands} inventory-position bands, budget `{a.budget}`. Same baselines and "
         "held-out test seeds as `train_neural.py`; gaps are paired against the DP optimum, as % of excess cost. "
         "*Neural (per-instance)* is taken from `neural_factorial-hard_per-instance_table.json` when present.", "",
         md_table([summary], [("n", "n", ""), ("GMI", "GMI %", ".2f"), ("Status-table", "Status table %", ".2f"),
                              ("Status x inventory", "Status × inventory %", ".2f"), ("Selected", "Selected %", ".2f"),
                              ("inv_beats_table", "Sig. better than table", ".2f"),
                              ("inv_worse_table", "Sig. worse than table", ".2f")]), "",
         md_table([{"instance": r["instance"], "gmi": r["GMI"]["gap"], "tab": r["Status-table"]["gap"],
                    "inv": r["Status x inventory"]["gap"], "diff": r["inv_vs_table"]["diff_pct_excess"],
                    "ci": r["inv_vs_table"]["ci"], "neu": r["neural_per_instance_gap"], "chosen": r["selected"],
                    "thr": ", ".join(f"{x:g}" for x in r["bands"]["thresholds"])} for r in rows],
                  [("instance", "Instance", ""), ("gmi", "GMI %", ".2f"), ("tab", "Status table %", ".2f"),
                   ("inv", "Status × inventory %", ".2f"), ("diff", "Diff vs table (pp)", "+.2f"), ("ci", "±", ".2f"),
                   ("neu", "Neural per-instance %", ".2f"), ("chosen", "Selected", ""),
                   ("thr", "Band thresholds (IP)", "")]), ""]
    path = os.path.join(RESULTS, f"{tag}.md")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
    print(json.dumps(summary, indent=1))
    print("wrote", path)


if __name__ == "__main__":
    main()
