"""Pilot study (E0): N = 3 suppliers with fixed costs, MOQs and disruptions.

Compares, on held-out test seeds with common random numbers:
  1. best tuned single-source (s,S) policy
  2. generalised multi-index heuristic (GMI), tuned by simulation
  3. rolling-horizon stochastic MILP (SAA), on a smaller test subset
  4. a simple structured learning policy (MLP -> GMI parameters, ES-trained),
     several independent training seeds, heuristic kept as fallback
  5. the exact DP optimum

and answers the four pilot questions:
  Q1 does the best supplier combination change with the state?
  Q2 is rolling-horizon optimisation too slow at the needed scale?
  Q3 does the tuned heuristic already get close to the optimum?
  Q4 does learning improve decisions consistently across seeds?

Usage:
  python experiments/run_pilot.py                 # quick budget, all 4 instances
  python experiments/run_pilot.py --budget full --jobs 4
  python experiments/run_pilot.py --instances P2_fixed_moq P3_long_outages
  python experiments/run_pilot.py --no-saa        # skip the slow MILP baseline
  python experiments/run_pilot.py --saa both      # two-stage and multi-stage MILP
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from common import RESULTS, Budget, md_table, save_json  # noqa: E402

from supplier_pilot import dp
from supplier_pilot.instance import pilot_instances
from supplier_pilot.learner import train_es, tune_status_table
from supplier_pilot.metrics import activation_structure, ci95, summarize
from supplier_pilot.saa import RollingSAAPolicy

SAA_TREE = (5, 3, 2)   # multi-stage scenario tree: 30 scenarios, non-anticipative for 3 periods
from supplier_pilot.simulator import make_streams, mean_cost, simulate
from supplier_pilot.tuning import SeedSplit, tune_gmi, tune_single_source


def log(name, msg):
    print(f"[{name} {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run_instance(args):
    inst, budget_name, saa_mode = args
    B = Budget.preset(budget_name)
    seeds = SeedSplit.default(B.tune_E, B.val_E, B.test_E)
    out = {"instance": inst.name, "desc": inst.meta.get("desc", ""), "budget": budget_name}
    t_start = time.time()

    # ---------------------------------------------------------- DP optimum
    sol = dp.solve_instance(inst, -10, 26)
    chk = dp.check_boundaries(sol, inst, T=20000)
    out["dp"] = {**sol.summary(), **chk}
    log(inst.name, f"DP g*={sol.g:.3f} ({sol.seconds:.1f}s, {sol.model.Z * sol.model.R} states, "
                   f"boundary hits {chk['boundary_hit_share']:.4f})")
    dp_pol = dp.DPPolicy(sol)

    tr = make_streams(inst, seeds.train, B.tune_T)
    va = make_streams(inst, seeds.val, B.tune_T)

    # ------------------------------------------------- single-source (s,S)
    singles = []
    for i in range(inst.n):
        p, _, _ = tune_single_source(inst, i, tr, B.tune_warm)
        singles.append((mean_cost(inst, p, va, B.tune_warm), p))
    best_single = min(singles, key=lambda x: x[0])[1]
    best_single.name = "Single-source best " + best_single.name.split("(")[0].replace("single", "")
    log(inst.name, f"single-source tuned: {best_single.name}")

    # ------------------------------------------------------ GMI heuristic
    t0 = time.time()
    gmi, gmi_train, gmi_evals = tune_gmi(inst, tr, B.tune_warm)
    gmi.name = "GMI heuristic"
    out["gmi_params"] = gmi.describe(inst)
    out["gmi_tuning"] = {"evals": gmi_evals, "seconds": round(time.time() - t0, 1)}
    log(inst.name, f"GMI tuned ({gmi_evals} evals, {time.time() - t0:.0f}s): {gmi.describe(inst)}")

    # ------------------------------------------ structured learner (ES)
    es_runs = []
    for k in range(B.es_seeds):
        r = train_es(inst, gmi, seeds.train, seeds.val, T=B.tune_T, warmup=B.tune_warm,
                     iters=B.es_iters, pairs=B.es_pairs, seed=100 + k)
        r.policy.name = f"Structured-ES seed{k}"
        es_runs.append(r)
        log(inst.name, f"ES seed {k}: val={r.val_cost:.3f} vs GMI val={r.base_val_cost:.3f} "
                       f"({r.seconds:.0f}s)")
    # ------------------- status-table learner (direct coordinate search)
    tab_runs = []
    for k in range(B.es_seeds):
        tr_k = make_streams(inst, [5000 + 100 * k + j for j in range(B.tune_E)], B.tune_T)
        t0 = time.time()
        tab, ev, _ = tune_status_table(inst, gmi, tr_k, B.tune_warm)
        tab.name = f"Status-table seed{k}"
        tab_runs.append((mean_cost(inst, tab, va, B.tune_warm), tab))
        log(inst.name, f"status-table seed {k}: val={tab_runs[-1][0]:.3f} ({ev} evals, {time.time() - t0:.0f}s)")

    # ------------- validation-based selection, heuristic kept as fallback
    gmi_val = mean_cost(inst, gmi, va, B.tune_warm)
    cands = [(gmi_val, gmi)] + [(r.val_cost, r.policy) for r in es_runs] + tab_runs
    sel_val, selected = min(cands, key=lambda c: c[0])
    out["selection"] = {"selected": selected.name, "selected_val": sel_val, "gmi_val": gmi_val,
                        "candidates": {p.name: v for v, p in cands}}

    # ------------------------------------------------------------- testing
    test = make_streams(inst, seeds.test, B.test_T)
    dp_res = simulate(inst, dp_pol, streams=test, warmup=B.test_warm)
    rows = [summarize(dp_res, inst, ref=dp_res, ref_cost=sol.g)]
    results = {"DP": dp_res}
    for p in [best_single, gmi] + [r.policy for r in es_runs] + [t for _, t in tab_runs]:
        res = simulate(inst, p, streams=test, warmup=B.test_warm)
        results[p.name] = res
        rows.append(summarize(res, inst, ref=dp_res, ref_cost=sol.g))
    sel_row = dict(next(r for r in rows if r["policy"] == selected.name))
    sel_row["policy"] = f"Validation-selected: {selected.name}"
    rows.append(sel_row)

    # paired comparison of each ES seed against the GMI heuristic
    gmi_res = results[gmi.name]
    def vs_gmi(names):
        lst = []
        for nm in names:
            d = results[nm].episode_cost - gmi_res.episode_cost
            m, h = ci95(d)
            lst.append({"seed": nm, "diff_vs_gmi": m, "ci95": h,
                        "significant_improvement": bool(m + h < 0),
                        "significantly_worse": bool(m - h > 0)})
        return lst
    out["es_vs_gmi"] = vs_gmi([r.policy.name for r in es_runs])
    out["table_vs_gmi"] = vs_gmi([t.name for _, t in tab_runs])

    # ----------------------------------------------------- SAA (subset)
    if saa_mode != "none":
        sub_seeds = seeds.test[:B.saa_E]
        sub = make_streams(inst, sub_seeds, B.saa_T)
        dp_sub = simulate(inst, dp_pol, streams=sub, warmup=B.saa_warm)
        sub_rows = []
        for name, pol in [("DP-optimal", dp_pol), ("GMI heuristic", gmi), (f"Selected: {selected.name}", selected)]:
            res = dp_sub if name == "DP-optimal" else simulate(inst, pol, streams=sub, warmup=B.saa_warm)
            row = summarize(res, inst, ref=dp_sub, ref_cost=sol.g); row["policy"] = name
            sub_rows.append(row)
        variants = {"two": [RollingSAAPolicy(M=B.saa_M)],
                    "multi": [RollingSAAPolicy(tree=SAA_TREE)],
                    "both": [RollingSAAPolicy(M=B.saa_M), RollingSAAPolicy(tree=SAA_TREE)]}[saa_mode]
        out["saa"] = []
        for saa in variants:
            t0 = time.time()
            saa_res = simulate(inst, saa, streams=sub, warmup=B.saa_warm)
            st = np.array(saa.solve_times)
            row = summarize(saa_res, inst, ref=dp_sub, ref_cost=sol.g); row["policy"] = saa.name
            sub_rows.append(row)
            out["saa"].append({"name": saa.name, "decisions": len(st), "mean_solve_s": float(st.mean()),
                               "p95_solve_s": float(np.quantile(st, 0.95)), "max_solve_s": float(st.max()),
                               "failures": saa.failures, "wall_s": round(time.time() - t0, 1),
                               "scenarios": saa.M, "H": int(inst.lead_times.max() + 5),
                               "gap_excess": row["paired_gap_excess_pct"]})
            log(inst.name, f"{saa.name}: {len(st)} solves, mean {st.mean():.2f}s, p95 "
                           f"{np.quantile(st, .95):.2f}s; paired gap excess {row['paired_gap_excess_pct']:.1f}%")
        out["saa_subset_rows"] = sub_rows

    out["rows"] = rows
    out["activation_dp"] = activation_structure(dp_res, inst)
    out["activation_gmi"] = activation_structure(gmi_res, inst)
    out["wall_s"] = round(time.time() - t_start, 1)
    out["answers"] = answer_questions(out)
    log(inst.name, f"done in {out['wall_s']:.0f}s")
    return out


def answer_questions(o) -> dict:
    """Mechanical answers with explicit thresholds (state them in the paper)."""
    rows = {r["policy"]: r for r in o["rows"]}
    gmi = rows["GMI heuristic"]
    act = o["activation_dp"]
    modal = {r["modal_pattern"] for r in act["by_status"] if r["share_of_time"] > 0.01}
    q1 = (f"DP activation pattern differs from its status-modal pattern in "
          f"{100 * act['overall_deviation']:.1f}% of periods; {len(modal)} distinct modal patterns "
          f"across disruption states. -> "
          + ("state-dependent (YES)" if act["overall_deviation"] > 0.10 or len(modal) > 1 else "mostly static (NO)"))
    if o.get("saa"):
        sub = {r["policy"]: r for r in o["saa_subset_rows"]}
        lst = o["saa"] if isinstance(o["saa"], list) else [dict(o["saa"], name=next(
            k for k in sub if k.startswith("RH-SAA")), scenarios=o["saa"].get("M"))]
        parts = []
        for v in lst:
            g = sub[v["name"]]["paired_gap_excess_pct"]
            parts.append(f"{v['name']}: mean {v['mean_solve_s']:.2f}s / p95 {v['p95_solve_s']:.2f}s per "
                         f"decision, gap to DP {g:.1f}% of excess cost")
        q2 = "; ".join(parts) + (f" (GMI on the same subset: "
                                 f"{sub['GMI heuristic']['paired_gap_excess_pct']:.1f}%).")
    else:
        q2 = "SAA not run (--no-saa)."
    ge = gmi["paired_gap_excess_pct"]
    q3 = (f"GMI gap to DP = {gmi['paired_gap_total_pct']:.2f}% of total cost, {ge:.1f}% of excess cost. -> "
          + ("heuristic near-optimal (<=1% total)" if gmi["paired_gap_total_pct"] <= 1.0 else
             "room for improvement (>1% total)"))
    def consist(lst, label):
        imp = [e for e in lst if e["significant_improvement"]]
        worse = [e for e in lst if e["significantly_worse"]]
        return (f"{label}: {len(imp)}/{len(lst)} runs significantly beat GMI, {len(worse)} significantly "
                f"worse (diffs " + ", ".join(f"{e['diff_vs_gmi']:+.3f}±{e['ci95']:.3f}" for e in lst) + ")"), \
            len(imp) == len(lst)
    q4a, es_ok = consist(o["es_vs_gmi"], "ES (generic neural learner)")
    q4b, tab_ok = consist(o["table_vs_gmi"], "status-table (direct search)")
    q4 = q4a + "; " + q4b
    tab_gaps = [rows[e["seed"]]["paired_gap_excess_pct"] for e in o["table_vs_gmi"]]
    closed = 1 - np.mean(tab_gaps) / ge if ge > 0 else float("nan")
    q4 += f". Status-table closes {100 * closed:.0f}% of the GMI-to-DP gap on average."
    gap_total = gmi["paired_gap_total_pct"]
    if es_ok or tab_ok:
        verdict = ("GO: state-dependent adaptation beats the tuned heuristic consistently across "
                   "training runs" + ("" if es_ok else "; the generic ES learner does not find it, so use a "
                                      "model-exploiting learner") + ".")
    elif ge > 3.0:
        verdict = ("OPEN: the heuristic leaves a material gap to the optimum, but status-only adaptation "
                   "does not close it. The missing gain depends on inventory/pipeline state (see Q1): "
                   "the case for a richer state-dependent policy.")
    else:
        verdict = ("NO-GO for learning on this instance: the heuristic is near-optimal and learning adds "
                   "no consistent gain.")
    return {"Q1_state_dependence": q1, "Q2_rolling_horizon_cost": q2,
            "Q3_heuristic_gap": q3, "Q4_learning_consistency": q4, "verdict": verdict}


def write_report(all_out, budget):
    lines = ["# Pilot study results (N = 3)", "",
             f"Budget preset: `{budget}`. All gaps are paired (common random numbers) against the "
             "simulated DP-optimal policy on held-out test seeds. *Excess cost* = average cost "
             "minus mean demand x cheapest unit price. All policies are tuned/trained on expected "
             "average cost; CVaR and gaps are evaluation metrics only.", ""]
    cols = [("policy", "Policy", ""), ("avg_cost", "Avg cost", ".3f"), ("ci95", "±95%", ".3f"),
            ("paired_gap_total_pct", "Gap total %", ".2f"),
            ("paired_gap_excess_pct", "Gap excess %", ".1f"),
            ("paired_gap_excess_ci95", "±95%", ".1f"),
            ("fill_rate", "Fill rate", ".4f"), ("cvar95_block50", "CVaR95 (50-period blocks)", ".2f"),
            ("active_suppliers_per_period", "Active suppl./period", ".2f"), ("volume_share", "Volume share A/B/C", "")]
    lines += ["Status strings list suppliers in the order A, B, C (U = up, D = down). "
              "Patterns list which of A, B, C received an order (1 = ordered).", ""]
    for o in all_out:
        lines += [f"## {o['instance']} — {o['desc']}", "",
                  f"DP: g* = {o['dp']['g']:.4f} (bounds {o['dp']['g_lower']:.4f}–{o['dp']['g_upper']:.4f}), "
                  f"{o['dp']['states']} states, {o['dp']['seconds']} s; simulated DP cost "
                  f"{o['dp']['sim_avg_cost']:.4f}; truncation boundary hits {o['dp']['boundary_hit_share']:.4f}.",
                  "", md_table(o["rows"], cols), ""]
        if "saa_subset_rows" in o:
            lines += ["Rolling-horizon stochastic MILP on a smaller test subset (same seeds for all rows):", "",
                      md_table(o["saa_subset_rows"], cols[:6]), ""]
        lines += ["GMI parameters: " + "; ".join(
            f"{p['supplier']}: {'on' if p['active'] else 'off'}, S={p['S']}, cap={p['cap']}, r={p['trigger']}"
            for p in o["gmi_params"]), "",
            "DP activation by disruption status (U = up, D = down; pattern = suppliers ordered from, A/B/C):", "",
            md_table(o["activation_dp"]["by_status"],
                     [("status", "Status A/B/C", ""), ("share_of_time", "Share of time", ".4f"),
                      ("modal_pattern", "Modal pattern", ""), ("distinct_patterns", "Distinct patterns", ""),
                      ("deviation_from_modal", "Deviation from modal", ".3f")]), "",
            "**Pilot questions**", ""]
        for k, v in o["answers"].items():
            lines.append(f"- **{k}**: {v}")
        lines.append("")
    path = f"{RESULTS}/pilot_results.md"
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", default="quick", choices=["smoke", "quick", "full"])
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--instances", nargs="*")
    ap.add_argument("--no-saa", action="store_true", help="same as --saa none")
    ap.add_argument("--saa", default="multi", choices=["none", "two", "multi", "both"],
                    help="rolling-horizon MILP variant(s): two-stage, multi-stage (scenario tree), both")
    ap.add_argument("--report-only", action="store_true",
                    help="rebuild the markdown report from the saved JSON")
    a = ap.parse_args()
    if a.report_only:
        import json
        outs = json.load(open(f"{RESULTS}/pilot_results_{a.budget}.json"))
        for o in outs:
            o["answers"] = answer_questions(o)
        save_json(outs, f"pilot_results_{a.budget}.json")
        print("wrote", write_report(outs, a.budget))
        return
    insts = [i for i in pilot_instances() if not a.instances or i.name in a.instances]
    saa_mode = "none" if a.no_saa else a.saa
    tasks = [(i, a.budget, saa_mode) for i in insts]
    if a.jobs > 1:
        with ProcessPoolExecutor(a.jobs) as ex:
            outs = list(ex.map(run_instance, tasks))
    else:
        outs = [run_instance(t) for t in tasks]
    save_json(outs, f"pilot_results_{a.budget}.json")
    print("wrote", write_report(outs, a.budget))
    for o in outs:
        print(f"\n== {o['instance']}")
        for k, v in o["answers"].items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
