"""Random-instance study (N = 3): does the pilot pattern generalise?

For each randomly generated, DP-tractable instance (see
``random_study_instance``) this script computes the exact DP optimum, tunes
the GMI heuristic and the best single-source policy, trains the status-table
learner (and optionally the ES learner and the SAA baselines), and evaluates
everything on held-out test seeds with common random numbers.

It then summarises, across instances:
  * the distribution of the heuristic's gap to the optimum;
  * how often and by how much the learners beat the heuristic;
  * which instance features (disruption severity, correlation, fixed costs,
    price premium, MOQ) predict a large heuristic gap (Spearman + OLS).

Usage:
  python experiments/run_random_study.py --n 40 --jobs 4                  # ~30 min on an M4
  python experiments/run_random_study.py --n 60 --jobs 8 --learners both  # adds the ES learner
  python experiments/run_random_study.py --n 20 --saa multi --jobs 4      # adds a SAA baseline (slow)
  python experiments/run_random_study.py --report-only                    # rebuild report from JSON
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from common import RESULTS, Budget, md_table, save_json  # noqa: E402

from supplier_pilot import dp
from supplier_pilot.instance import random_study_instance
from supplier_pilot.learner import train_es, tune_status_table
from supplier_pilot.metrics import ci95, summarize
from supplier_pilot.saa import RollingSAAPolicy
from supplier_pilot.simulator import make_streams, mean_cost, simulate
from supplier_pilot.tuning import SeedSplit, tune_gmi, tune_single_source

FEATURES = ["premium", "K_max", "moq_c", "shock_rate", "has_shock", "down_A", "down_B", "down_C",
            "outage_C", "lead_C", "lead_gap"]


def features(inst) -> dict:
    m = inst.meta
    L = m["lead_times"]
    return {"premium": m["premium"], "K_max": m["K_max"], "moq_c": m["moq_c"],
            "shock_rate": m["shock"], "has_shock": float(m["shock"] > 0),
            "down_A": m["down_share"][0], "down_B": m["down_share"][1], "down_C": m["down_share"][2],
            "outage_C": m["mean_outage"][2], "lead_C": L[2], "lead_gap": L[2] - L[0]}


def run_one(args):
    k, seed, budget, learners, table_runs, saa_mode = args
    rng = np.random.default_rng((seed, k))
    inst = random_study_instance(rng, f"R{k:03d}")
    return evaluate_instance(inst, k, budget, learners, table_runs, saa_mode)


def evaluate_instance(inst, k, budget, learners="table", table_runs=2, saa_mode="none",
                      extra: dict | None = None):
    """DP, tuned heuristics, learners, validation-based selection with the
    heuristic as fallback, and paired test evaluation for one instance."""
    B = Budget.preset(budget)
    t0 = time.time()
    seeds = SeedSplit.default(B.tune_E, B.val_E, B.test_E, base=k * 10_000)
    sol = dp.solve_instance(inst, -10, 26)
    chk = dp.check_boundaries(sol, inst, T=8000, seeds=range(2))
    dp_pol = dp.DPPolicy(sol)
    tr = make_streams(inst, seeds.train, B.tune_T)
    va = make_streams(inst, seeds.val, B.tune_T)

    gmi, _, _ = tune_gmi(inst, tr, B.tune_warm, rng=np.random.default_rng(k))
    gmi.name = "GMI"
    singles = [tune_single_source(inst, i, tr, B.tune_warm)[0] for i in range(inst.n)]
    single = min(singles, key=lambda p: mean_cost(inst, p, va, B.tune_warm))
    single.name = "single"
    pols = [single, gmi]
    groups = {"table": [], "es": []}
    if learners in ("table", "both"):
        for r in range(table_runs):
            trr = make_streams(inst, [k * 10_000 + 5000 + 100 * r + j for j in range(B.tune_E)], B.tune_T)
            tab, _, _ = tune_status_table(inst, gmi, trr, B.tune_warm)
            tab.name = f"table{r}"
            pols.append(tab); groups["table"].append(tab.name)
    if learners in ("es", "both"):
        for r in range(table_runs):
            res = train_es(inst, gmi, seeds.train, seeds.val, T=B.tune_T, warmup=B.tune_warm,
                           iters=B.es_iters, pairs=B.es_pairs, seed=100 + r)
            res.policy.name = f"es{r}"
            pols.append(res.policy); groups["es"].append(res.policy.name)

    # validation-based selection: the heuristic is a candidate (fallback), so the
    # selected policy is the one with the lowest *validation* cost; test numbers
    # below show whether that choice held up out of sample
    learner_pols = [p for p in pols if p.name not in ("GMI", "single")]
    val_cost = {p.name: mean_cost(inst, p, va, B.tune_warm) for p in [gmi] + learner_pols}
    selected_name = min(val_cost, key=val_cost.get)

    test = make_streams(inst, seeds.test, B.test_T)
    ref = simulate(inst, dp_pol, streams=test, warmup=B.test_warm)
    gmi_res = None
    out_pol = {}
    for p in pols:
        res = simulate(inst, p, streams=test, warmup=B.test_warm)
        s = summarize(res, inst, ref=ref, ref_cost=sol.g)
        entry = {"gap_excess": s["paired_gap_excess_pct"], "gap_excess_ci": s["paired_gap_excess_ci95"],
                 "gap_total": s["paired_gap_total_pct"], "fill": s["fill_rate"],
                 "volume_share": s["volume_share"], "cost": s["avg_cost"]}
        if p.name == "GMI":
            gmi_res = res
        elif gmi_res is not None:
            d, h = ci95(res.episode_cost - gmi_res.episode_cost)
            entry.update({"diff_vs_gmi": d, "diff_ci": h, "beats_gmi": bool(d + h < 0),
                          "worse_than_gmi": bool(d - h > 0)})
        out_pol[p.name] = entry

    sel = dict(out_pol[selected_name])
    sel.setdefault("diff_vs_gmi", 0.0); sel.setdefault("diff_ci", 0.0)
    sel.setdefault("beats_gmi", False); sel.setdefault("worse_than_gmi", False)
    sel["chosen"] = selected_name
    out_pol["selected"] = sel

    saa = {}
    if saa_mode != "none":
        sub = make_streams(inst, seeds.test[:B.saa_E], B.saa_T)
        dp_sub = simulate(inst, dp_pol, streams=sub, warmup=B.saa_warm)
        modes = ["two", "multi"] if saa_mode == "both" else [saa_mode]
        for m in modes:
            pol = RollingSAAPolicy(M=B.saa_M) if m == "two" else RollingSAAPolicy(tree=(5, 3, 2))
            res = simulate(inst, pol, streams=sub, warmup=B.saa_warm)
            s = summarize(res, inst, ref=dp_sub, ref_cost=sol.g)
            g = summarize(simulate(inst, gmi, streams=sub, warmup=B.saa_warm), inst, ref=dp_sub, ref_cost=sol.g)
            saa[m] = {"gap_excess": s["paired_gap_excess_pct"], "gmi_gap_same_subset": g["paired_gap_excess_pct"],
                      "mean_solve_s": float(np.mean(pol.solve_times))}

    row = {"instance": inst.name, **features(inst), **(extra or {}), "dp_g": sol.g,
           "val_cost": val_cost, "selected": selected_name,
           "dp_excess": sol.g - inst.unavoidable_cost(), "dp_boundary_hits": chk["boundary_hit_share"],
           "dp_volume_share": summarize(ref, inst)["volume_share"],
           "gmi_params": gmi.describe(inst), "policies": out_pol, "groups": groups, "saa": saa,
           "suppliers": [s.__dict__ for s in inst.suppliers], "seconds": round(time.time() - t0, 1)}
    gl = out_pol["GMI"]["gap_excess"]
    tg = [out_pol[n]["gap_excess"] for n in groups["table"]]
    print(f"[{inst.name}] GMI gap {gl:5.2f}%  table {np.mean(tg) if tg else float('nan'):5.2f}%  "
          f"selected({selected_name}) {sel['gap_excess']:5.2f}%  single {out_pol['single']['gap_excess']:6.1f}%  "
          f"({row['seconds']:.0f}s)", flush=True)
    return row


# ------------------------------------------------------------------ analysis
def analyse(rows: list[dict]) -> dict:
    from scipy.stats import spearmanr
    gmi = np.array([r["policies"]["GMI"]["gap_excess"] for r in rows])
    out = {"n": len(rows),
           "gmi_gap": {"mean": float(gmi.mean()), "median": float(np.median(gmi)),
                       "p90": float(np.quantile(gmi, 0.9)), "max": float(gmi.max())},
           "single_gap_median": float(np.median([r["policies"]["single"]["gap_excess"] for r in rows]))}
    for grp in ("table", "es"):
        names = [r["groups"][grp] for r in rows]
        if not any(names):
            continue
        g = np.array([np.mean([r["policies"][n]["gap_excess"] for n in r["groups"][grp]]) for r in rows])
        all_beat = np.array([all(r["policies"][n]["beats_gmi"] for n in r["groups"][grp]) for r in rows])
        any_worse = np.array([any(r["policies"][n]["worse_than_gmi"] for n in r["groups"][grp]) for r in rows])
        closed = np.where(gmi > 0.25, 1 - g / np.maximum(gmi, 1e-9), np.nan)
        out[grp] = {"gap_mean": float(g.mean()), "gap_median": float(np.median(g)),
                    "share_all_runs_beat_gmi": float(all_beat.mean()),
                    "share_any_run_worse": float(any_worse.mean()),
                    "gap_closed_median_pct": float(100 * np.nanmedian(closed)),
                    "gap_closed_mean_pct": float(100 * np.nanmean(closed))}
    if all("selected" in r["policies"] for r in rows):
        sg = np.array([r["policies"]["selected"]["gap_excess"] for r in rows])
        out["selected"] = {
            "gap_mean": float(sg.mean()), "gap_median": float(np.median(sg)),
            "share_beats_gmi": float(np.mean([r["policies"]["selected"]["beats_gmi"] for r in rows])),
            "share_worse_than_gmi": float(np.mean([r["policies"]["selected"]["worse_than_gmi"] for r in rows])),
            "share_fallback_to_gmi": float(np.mean([r["selected"] == "GMI" for r in rows])),
            "gap_closed_median_pct": float(100 * np.nanmedian(np.where(gmi > 0.25, 1 - sg / np.maximum(gmi, 1e-9), np.nan)))}
    X = np.array([[r[f] for f in FEATURES] for r in rows], dtype=float)
    corr = []
    for j, f in enumerate(FEATURES):
        if np.std(X[:, j]) == 0:
            continue
        rho, p = spearmanr(X[:, j], gmi)
        corr.append({"feature": f, "spearman_rho": float(rho), "p_value": float(p)})
    out["spearman_gmi_gap"] = sorted(corr, key=lambda c: -abs(c["spearman_rho"]))
    # OLS on standardised features (descriptive; needs enough instances)
    keep = [j for j in range(X.shape[1]) if np.std(X[:, j]) > 0]
    if len(rows) < len(keep) + 10:
        out["ols_gmi_gap"] = {"r2": float("nan"), "n": len(rows), "coef": [],
                              "note": f"skipped: need at least {len(keep) + 10} instances"}
        return _saa_summary(rows, out)
    Xs = (X[:, keep] - X[:, keep].mean(0)) / X[:, keep].std(0)
    A = np.column_stack([np.ones(len(rows)), Xs])
    beta, *_ = np.linalg.lstsq(A, gmi, rcond=None)
    resid = gmi - A @ beta
    dof = max(len(rows) - A.shape[1], 1)
    s2 = resid @ resid / dof
    try:
        se = np.sqrt(np.diag(s2 * np.linalg.inv(A.T @ A)))
    except np.linalg.LinAlgError:
        se = np.full(len(beta), np.nan)
    r2 = 1 - resid @ resid / np.sum((gmi - gmi.mean()) ** 2)
    out["ols_gmi_gap"] = {"r2": float(r2), "n": len(rows),
                          "coef": [{"feature": FEATURES[j], "beta_per_sd": float(beta[i + 1]),
                                    "se": float(se[i + 1]), "t": float(beta[i + 1] / se[i + 1])}
                                   for i, j in enumerate(keep)]}
    return _saa_summary(rows, out)


def _saa_summary(rows, out):
    saa_rows = [r for r in rows if r["saa"]]
    if saa_rows:
        out["saa"] = {m: {"gap_median": float(np.median([r["saa"][m]["gap_excess"] for r in saa_rows if m in r["saa"]])),
                          "gmi_same_subset_median": float(np.median([r["saa"][m]["gmi_gap_same_subset"] for r in saa_rows if m in r["saa"]])),
                          "solve_s_mean": float(np.mean([r["saa"][m]["mean_solve_s"] for r in saa_rows if m in r["saa"]]))}
                      for m in ("two", "multi") if any(m in r["saa"] for r in saa_rows)}
    return out


def plot(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    gmi = np.array([r["policies"]["GMI"]["gap_excess"] for r in rows])
    tab = np.array([np.mean([r["policies"][n]["gap_excess"] for n in r["groups"]["table"]])
                    if r["groups"]["table"] else np.nan for r in rows])
    sev = np.array([r["K_max"] for r in rows])                       # largest fixed ordering cost
    shock = np.array([r["has_shock"] for r in rows]) > 0
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), dpi=160)
    ax = axes[0]
    ax.scatter(sev[~shock], gmi[~shock], s=28, color="#2a78d6", label="independent disruptions",
               edgecolor="white", linewidth=0.8, zorder=3)
    ax.scatter(sev[shock], gmi[shock], s=28, color="#eb6834", marker="D",
               label="with regional shock", edgecolor="white", linewidth=0.8, zorder=3)
    ax.set_xlabel("Largest fixed ordering cost K_max", fontsize=8)
    ax.set_ylabel("Heuristic gap to optimum (% of excess cost)", fontsize=8)
    ax.set_title("Where does the heuristic fall short?", fontsize=9, loc="left")
    ax.legend(frameon=False, fontsize=7)
    ax2 = axes[1]
    lim = max(np.nanmax(gmi), np.nanmax(tab)) * 1.05 + 0.1
    ax2.plot([0, lim], [0, lim], color="#999", lw=1, ls="--", zorder=1)
    ax2.scatter(gmi, tab, s=28, color="#1baf7a", edgecolor="white", linewidth=0.8, zorder=3)
    ax2.set_xlim(0, lim); ax2.set_ylim(0, lim)
    ax2.set_xlabel("Heuristic (GMI) gap, % of excess cost", fontsize=8)
    ax2.set_ylabel("Status-table learner gap, % of excess cost", fontsize=8)
    ax2.set_title("Learner vs heuristic (below the line = learner better)", fontsize=9, loc="left")
    for a in axes:
        a.grid(color="#e5e5e5", zorder=0)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        a.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(path)


def report(rows, summary, budget):
    L = ["# Random-instance study (N = 3)", "",
         f"{summary['n']} random DP-tractable instances, budget `{budget}`. Gaps are paired against the "
         "DP optimum on held-out test seeds, as % of excess cost (cost minus mean demand × cheapest price).", "",
         "## Heuristic (GMI) gap to the optimum", "",
         f"mean {summary['gmi_gap']['mean']:.2f}%, median {summary['gmi_gap']['median']:.2f}%, "
         f"90th percentile {summary['gmi_gap']['p90']:.2f}%, max {summary['gmi_gap']['max']:.2f}%. "
         f"Best single-source policy: median gap {summary['single_gap_median']:.1f}%.", ""]
    if "selected" in summary:
        s = summary["selected"]
        L += ["## Validation-selected policy (heuristic kept as fallback)", "",
              f"mean gap {s['gap_mean']:.2f}% (median {s['gap_median']:.2f}%); significantly better than GMI on "
              f"{100 * s['share_beats_gmi']:.0f}% of instances, significantly worse on "
              f"{100 * s['share_worse_than_gmi']:.0f}%; fell back to GMI on {100 * s['share_fallback_to_gmi']:.0f}%; "
              f"median share of the GMI gap closed {s['gap_closed_median_pct']:.0f}%.", ""]
    for grp, label in (("table", "Status-table learner"), ("es", "ES neural learner")):
        if grp in summary:
            s = summary[grp]
            L += [f"## {label}", "",
                  f"mean gap {s['gap_mean']:.2f}% (median {s['gap_median']:.2f}%); all training runs beat GMI "
                  f"significantly on {100 * s['share_all_runs_beat_gmi']:.0f}% of instances; at least one run "
                  f"significantly worse on {100 * s['share_any_run_worse']:.0f}%; share of the GMI gap closed: "
                  f"median {s['gap_closed_median_pct']:.0f}%, mean {s['gap_closed_mean_pct']:.0f}% "
                  "(instances with GMI gap > 0.25%).", ""]
    L += ["## What predicts a large heuristic gap?", "",
          "Spearman rank correlations with the GMI gap:", "",
          md_table(summary["spearman_gmi_gap"], [("feature", "Feature", ""), ("spearman_rho", "rho", ".2f"),
                                                 ("p_value", "p", ".3f")]), "",
          (f"OLS on standardised features (R² = {summary['ols_gmi_gap']['r2']:.2f}, n = {summary['ols_gmi_gap']['n']}; "
           "descriptive only):") if summary["ols_gmi_gap"]["coef"] else f"OLS {summary['ols_gmi_gap'].get('note', '')}.", "",
          md_table(summary["ols_gmi_gap"]["coef"], [("feature", "Feature", ""), ("beta_per_sd", "β per SD", ".2f"),
                                                    ("se", "SE", ".2f"), ("t", "t", ".2f")]), "",
          "Feature key: premium = price premium of the fastest supplier; K_max = largest fixed cost; "
          "down_X = long-run share of time supplier X is down; outage_C = mean outage length of the slow "
          "supplier (periods); shock_rate / has_shock = regional shock hitting B and C together; "
          "lead_gap = lead-time difference fastest vs slowest.", ""]
    if "saa" in summary:
        L += ["## Rolling-horizon stochastic MILP (subset of periods)", "",
              md_table([{"variant": k, **v} for k, v in summary["saa"].items()],
                       [("variant", "Variant", ""), ("gap_median", "Median gap %", ".1f"),
                        ("gmi_same_subset_median", "GMI same subset %", ".1f"),
                        ("solve_s_mean", "Solve s/decision", ".2f")]), ""]
    L += ["## Per-instance results", "",
          md_table([{"instance": r["instance"], "shock": r["shock_rate"], "down_C": r["down_C"], "gmi": r["policies"]["GMI"]["gap_excess"],
                     "table": float(np.mean([r["policies"][n]["gap_excess"] for n in r["groups"]["table"]]))
                     if r["groups"]["table"] else float("nan"),
                     "single": r["policies"]["single"]["gap_excess"]} for r in rows],
                   [("instance", "Instance", ""), ("shock", "Shock rate", ".3f"), ("down_C", "Down share C", ".3f"),
                    ("gmi", "GMI gap %", ".2f"), ("table", "Table gap %", ".2f"), ("single", "Single gap %", ".1f")])]
    path = f"{RESULTS}/random_study.md"
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--budget", default="quick", choices=["smoke", "quick", "full"])
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--learners", default="table", choices=["none", "table", "es", "both"])
    ap.add_argument("--runs", type=int, default=2, help="training runs per learner")
    ap.add_argument("--saa", default="none", choices=["none", "two", "multi", "both"])
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    jpath = f"{RESULTS}/random_study_{a.budget}.json"
    if a.report_only:
        rows = json.load(open(jpath))["rows"]
    else:
        tasks = [(k, a.seed, a.budget, a.learners, a.runs, a.saa) for k in range(a.n)]
        if a.jobs > 1:
            with ProcessPoolExecutor(a.jobs) as ex:
                rows = list(ex.map(run_one, tasks))
        else:
            rows = [run_one(t) for t in tasks]
    summary = analyse(rows)
    save_json({"summary": summary, "rows": rows}, f"random_study_{a.budget}.json")
    plot(rows, f"{RESULTS}/random_study_{a.budget}.png")
    print("wrote", report(rows, summary, a.budget))
    print(json.dumps({k: summary[k] for k in summary if k in ("gmi_gap", "table", "es", "selected")}, indent=1))


if __name__ == "__main__":
    main()
