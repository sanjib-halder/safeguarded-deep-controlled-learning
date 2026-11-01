"""Blocked 2x2x2 factorial study: what makes simple sourcing rules fail?

Factors (levels in ``FACTOR_LEVELS``):
  * fixed_cost  low / high        fixed ordering costs K_i
  * shock       none / correlated regional shock hitting B and C together
  * premium     low / high        price premium of the fastest supplier

Each *replicate* draws one set of the other parameters (lead times, MOQ,
disruption probabilities) and evaluates all eight cells with them, so the
cells of a replicate differ only in the factors (a randomised block design).
The analysis therefore compares cells *within* replicates:

  1. cell means of the heuristic's gap to the DP optimum (and of the learner's
     share of the gap closed), with 95% CIs across replicates;
  2. a linear model with replicate fixed effects and effect-coded factors
     (+/- 1/2), so each main-effect coefficient is the average high-minus-low
     difference, plus all interactions;
  3. a nonparametric check per main effect: within each replicate, mean of
     the four high-level cells minus mean of the four low-level cells,
     tested with the Wilcoxon signed-rank test across replicates.

Usage:
  python experiments/run_factorial.py --reps 8 --jobs 8         # 64 instances, ~1 h on an M4
  python experiments/run_factorial.py --reps 12 --jobs 8        # more power
  python experiments/run_factorial.py --report-only
"""
from __future__ import annotations

import argparse
import itertools
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from common import RESULTS, md_table, save_json  # noqa: E402
from run_random_study import evaluate_instance  # noqa: E402

from supplier_pilot.instance import FACTOR_LEVELS, factorial_instance

FACTORS = ["fixed_cost", "shock", "premium"]
HIGH = {"fixed_cost": "high", "shock": "correlated", "premium": "high"}
CELLS = list(itertools.product(*[list(FACTOR_LEVELS[f]) for f in FACTORS]))


def run_cell(args):
    rep, c, seed, budget, learners, runs = args
    levels = dict(zip(FACTORS, CELLS[c]))
    inst = factorial_instance(seed, rep, **levels)
    k = 100_000 + rep * 8 + c                     # unique seed block per instance
    extra = {"replicate": rep, "cell": inst.meta["cell"],
             **{f: int(levels[f] == HIGH[f]) for f in FACTORS}}
    return evaluate_instance(inst, k, budget, learners, runs, "none", extra=extra)


# ---------------------------------------------------------------- outcomes
def outcomes(r) -> dict:
    gmi = r["policies"]["GMI"]["gap_excess"]
    sel = r["policies"]["selected"]["gap_excess"]
    tab = [r["policies"][n]["gap_excess"] for n in r["groups"]["table"]]
    return {"gmi_gap": gmi, "selected_gap": sel,
            "learner_gap": float(np.mean(tab)) if tab else float("nan"),
            "closed_pct": 100 * (1 - sel / gmi) if gmi > 0.25 else float("nan"),
            "single_gap": r["policies"]["single"]["gap_excess"]}


def ci(x):
    from scipy import stats
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return float(np.mean(x)) if len(x) else float("nan"), float("nan")
    return float(x.mean()), float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))


def effects_model(rows, y_key):
    """OLS: y ~ replicate FE + effect-coded factors + all interactions."""
    from scipy import stats
    data = [(r, outcomes(r)[y_key]) for r in rows]
    data = [(r, y) for r, y in data if np.isfinite(y)]
    reps = sorted({r["replicate"] for r, _ in data})
    names, cols = [], []
    xs = {f: np.array([r[f] - 0.5 for r, _ in data]) for f in FACTORS}
    for kk in range(1, 4):
        for combo in itertools.combinations(FACTORS, kk):
            v = np.ones(len(data))
            for f in combo:
                v = v * xs[f]
            # scale interactions so coefficients stay "difference" units
            names.append(" × ".join(combo)); cols.append(v * (2 ** (kk - 1)))
    fe = [np.array([1.0 if r["replicate"] == rp else 0.0 for r, _ in data]) for rp in reps]
    X = np.column_stack(fe + cols)
    y = np.array([y for _, y in data])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(y) - np.linalg.matrix_rank(X)
    s2 = resid @ resid / max(dof, 1)
    cov = s2 * np.linalg.pinv(X.T @ X)
    out = []
    for j, nm in enumerate(names):
        b = beta[len(fe) + j]
        se = float(np.sqrt(cov[len(fe) + j, len(fe) + j]))
        t = b / se if se > 0 else float("nan")
        out.append({"term": nm, "effect": float(b), "se": se, "t": float(t),
                    "p": float(2 * stats.t.sf(abs(t), max(dof, 1)))})
    ss_tot = np.sum((y - y.mean()) ** 2)
    return {"terms": out, "dof": int(dof), "n": len(y), "r2": float(1 - resid @ resid / ss_tot)}


def paired_main_effects(rows, y_key):
    from scipy.stats import wilcoxon
    reps = sorted({r["replicate"] for r in rows})
    res = []
    for f in FACTORS:
        diffs = []
        for rp in reps:
            hi = [outcomes(r)[y_key] for r in rows if r["replicate"] == rp and r[f] == 1]
            lo = [outcomes(r)[y_key] for r in rows if r["replicate"] == rp and r[f] == 0]
            hi, lo = np.array(hi), np.array(lo)
            if np.isfinite(hi).any() and np.isfinite(lo).any():
                diffs.append(np.nanmean(hi) - np.nanmean(lo))
        diffs = np.array(diffs)
        p = float(wilcoxon(diffs).pvalue) if len(diffs) >= 5 and np.any(diffs != 0) else float("nan")
        res.append({"factor": f, "median_diff": float(np.median(diffs)), "mean_diff": float(diffs.mean()),
                    "replicates_up": int((diffs > 0).sum()), "replicates": len(diffs), "wilcoxon_p": p})
    return res


def cell_table(rows):
    out = []
    for cell in CELLS:
        lv = dict(zip(FACTORS, cell))
        sub = [r for r in rows if all(r[f] == int(lv[f] == HIGH[f]) for f in FACTORS)]
        o = [outcomes(r) for r in sub]
        g, gh = ci([x["gmi_gap"] for x in o])
        s, sh = ci([x["selected_gap"] for x in o])
        c, chh = ci([x["closed_pct"] for x in o])
        out.append({"fixed_cost": lv["fixed_cost"], "shock": lv["shock"], "premium": lv["premium"],
                    "n": len(sub), "gmi_gap": g, "gmi_ci": gh, "selected_gap": s, "selected_ci": sh,
                    "closed_pct": c, "closed_ci": chh,
                    "beats_gmi": float(np.mean([r["policies"]["selected"]["beats_gmi"] for r in sub])),
                    "fallback": float(np.mean([r["selected"] == "GMI" for r in sub])),
                    "single_gap": float(np.median([x["single_gap"] for x in o]))})
    return out


def plot(cells, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"none": "#2a78d6", "correlated": "#eb6834"}   # validated categorical slots 1-2
    markers = {"none": "o", "correlated": "D"}
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7), dpi=160, sharey="row")
    rows_spec = [("gmi_gap", "gmi_ci", "Heuristic gap to optimum\n(% of excess cost)"),
                 ("closed_pct", "closed_ci", "Share of heuristic gap closed\nby selected learner (%)")]
    for i, (yk, ek, ylab) in enumerate(rows_spec):
        for j, prem in enumerate(["low", "high"]):
            ax = axes[i, j]
            for sh in ["none", "correlated"]:
                pts = [c for c in cells if c["premium"] == prem and c["shock"] == sh]
                pts = sorted(pts, key=lambda c: c["fixed_cost"] == "high")
                x = np.array([0, 1]) + (0.1 if sh == "correlated" else -0.1)
                y = [c[yk] for c in pts]; e = [c[ek] for c in pts]
                ax.errorbar(x, y, yerr=e, color=colors[sh], marker=markers[sh], ms=6, lw=2, capsize=3,
                            label=f"shock: {sh}", zorder=3)
                right = sh == "correlated"
                for xi, yi in zip(x, y):
                    ax.annotate(f"{yi:.1f}", (xi, yi), textcoords="offset points",
                                xytext=(0, 9 if right else -9), ha="center",
                                va="bottom" if right else "top", fontsize=7, color="#333",
                                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85),
                                zorder=5)
            ax.set_xticks([0, 1], ["low fixed cost", "high fixed cost"], fontsize=8)
            ax.set_xlim(-0.55, 1.55)
            if i == 0:
                ax.set_title(f"Price premium of fast supplier: {prem}", fontsize=9, loc="left")
            if j == 0:
                ax.set_ylabel(ylab, fontsize=8)
            ax.grid(axis="y", color="#e5e5e5", zorder=0)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            ax.tick_params(labelsize=7)
    axes[0, 0].legend(frameon=False, fontsize=7, loc="upper left")
    fig.suptitle("Blocked factorial study (N = 3): cell means ± 95% CI across replicates", fontsize=10, x=0.02,
                 ha="left")
    fig.tight_layout()
    fig.savefig(path)


def report(rows, summ, budget):
    reps = len({r["replicate"] for r in rows})
    eff_cols = [("term", "Term", ""), ("effect", "Effect (high − low)", ".2f"), ("se", "SE", ".2f"),
                ("t", "t", ".2f"), ("p", "p", ".4f")]
    pair_cols = [("factor", "Factor", ""), ("median_diff", "Median within-replicate diff", ".2f"),
                 ("mean_diff", "Mean diff", ".2f"), ("replicates_up", "Replicates with increase", ""),
                 ("replicates", "Replicates", ""), ("wilcoxon_p", "Wilcoxon p", ".4f")]
    L = ["# Blocked factorial study (N = 3)", "",
         f"Design: 2 × 2 × 2 factors × {reps} replicates = {len(rows)} instances, budget `{budget}`. "
         "Within a replicate all eight cells share lead times, MOQ and disruption probabilities, so they "
         "differ only in the factors. Gaps are paired against the DP optimum on held-out test seeds, as % "
         "of excess cost. The *selected* policy is chosen on validation seeds among the heuristic (GMI) "
         "and the learner runs, so the heuristic is the fallback.", "",
         "Factor levels: " + "; ".join(f"**{f}** " + ", ".join(f"{k} = {v}" for k, v in FACTOR_LEVELS[f].items())
                                      for f in FACTORS) + ".", "",
         "## Cell means (± 95% CI across replicates)", "",
         md_table(summ["cells"], [("fixed_cost", "Fixed cost", ""), ("shock", "Shock", ""), ("premium", "Premium", ""),
                                  ("n", "n", ""), ("gmi_gap", "GMI gap %", ".2f"), ("gmi_ci", "±", ".2f"),
                                  ("selected_gap", "Selected gap %", ".2f"), ("selected_ci", "±", ".2f"),
                                  ("closed_pct", "Gap closed %", ".0f"), ("closed_ci", "±", ".0f"),
                                  ("beats_gmi", "Share sig. better", ".2f"), ("fallback", "Share fallback", ".2f"),
                                  ("single_gap", "Single-source median gap %", ".0f")]), "",
         "## Heuristic gap: effects", "",
         f"Linear model with replicate fixed effects, effect-coded factors and all interactions "
         f"(n = {summ['model_gmi']['n']}, residual dof = {summ['model_gmi']['dof']}, R² = {summ['model_gmi']['r2']:.2f}). "
         "Main-effect coefficients are average high-minus-low differences in percentage points; two-way "
         "terms are the change in one factor's effect when the other moves from low to high.", "",
         md_table(summ["model_gmi"]["terms"], eff_cols), "",
         "Nonparametric check (within-replicate differences, Wilcoxon signed-rank):", "",
         md_table(summ["paired_gmi"], pair_cols), "",
         "## Share of the heuristic gap closed by the selected learner: effects", "",
         md_table(summ["model_closed"]["terms"], eff_cols), "",
         md_table(summ["paired_closed"], pair_cols), "",
         "## Reading the results", "",
         "- Treat main effects with Wilcoxon p < 0.05 *and* a consistent sign across replicates as findings; "
         "with 3 main effects, a Holm correction is advisable for the paper.",
         "- Interactions are estimated with less precision; report them as exploratory unless p is small "
         "and the interaction plot shows a clear pattern.",
         "- These are quick-budget results; rerun the significant cells at `--budget full` before publishing.", ""]
    path = f"{RESULTS}/factorial_study.md"
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
    return path


def summarise(rows):
    return {"cells": cell_table(rows),
            "model_gmi": effects_model(rows, "gmi_gap"), "paired_gmi": paired_main_effects(rows, "gmi_gap"),
            "model_closed": effects_model(rows, "closed_pct"),
            "paired_closed": paired_main_effects(rows, "closed_pct")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--budget", default="quick", choices=["smoke", "quick", "full"])
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--learners", default="table", choices=["none", "table", "es", "both"])
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    jname = f"factorial_study_{a.budget}.json"
    if a.report_only:
        rows = json.load(open(f"{RESULTS}/{jname}"))["rows"]
    else:
        tasks = [(rep, c, a.seed, a.budget, a.learners, a.runs) for rep in range(a.reps) for c in range(len(CELLS))]
        if a.jobs > 1:
            with ProcessPoolExecutor(a.jobs) as ex:
                rows = list(ex.map(run_cell, tasks))
        else:
            rows = [run_cell(t) for t in tasks]
    summ = summarise(rows)
    save_json({"summary": summ, "rows": rows}, jname)
    plot(summ["cells"], f"{RESULTS}/factorial_study_{a.budget}.png")
    print("wrote", report(rows, summ, a.budget))
    for p in summ["paired_gmi"]:
        print(f"  heuristic gap, {p['factor']:<10s}: median within-replicate change {p['median_diff']:+.2f} pp, "
              f"{p['replicates_up']}/{p['replicates']} replicates up, Wilcoxon p = {p['wilcoxon_p']:.4f}")


if __name__ == "__main__":
    main()
