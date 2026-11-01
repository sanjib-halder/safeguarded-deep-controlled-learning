"""Deep controlled learning (DCL): train the state-aware policy by simulation.

Iteration 1: sample states under the status-table policy, label each state
with the best nearby order vector found by simulation (dcl.label_states),
train the classifier to imitate the labels.
Iteration 2+: sample states under the *trained* policy (DAgger-style), label
them the same way, retrain on all labels collected so far.

Modes (same instances, caches and test seeds as train_neural.py):
  per-instance (default):  one classifier per instance, --only names
  generalise:              --fresh-train N: one classifier trained on N new
                           replicates (seed --train-seed), iteration chosen on
                           --fresh-val further new replicates, tested on the
                           16 seed-7 instances (never seen)
  ensemble:                --fresh-train N --ensemble-seeds 101 202 303: one
                           network per training seed (saved as
                           results/dcl_member{N}_s{seed}.pt/.json and reused
                           by later runs with the same settings), combined by
                           mean probability or unanimous agreement, the rule
                           chosen on the members' validation instances

  more suppliers:          --target scale4 / scale5: N = 4 / 5 suppliers (A, B, C
                           as in the factorial + D, E in a third region). Exact
                           DP is intractable; labels use coordinate search
                           (linear in N) instead of the 6^N joint enumeration.
  zero-shot transfer:      --transfer-from factorial-hard: the saved N = 3
                           networks applied to N = 4 / 5 without retraining
                           (the network is permutation-equivariant and has no
                           fixed number of suppliers); --transfer-only skips
                           training on the new target.

Evaluation is in the NumPy simulator on held-out test seeds, with a
validation-selected policy (GMI and status table as fallbacks). Gaps are
paired against the DP optimum (N = 3) or, for N >= 4, against the best policy
that uses only suppliers A, B, C (exact DP of that 3-supplier core); a
negative gap then means the policy profits from suppliers D/E. --saa adds
rolling-horizon SAA on a shorter paired subset of the test seeds.

Usage
  python experiments/train_dcl.py --jobs 8                          # 4 diagnostic instances
  python experiments/train_dcl.py --fresh-train 48 --jobs 8         # generalisation test
  python experiments/train_dcl.py --fresh-train 48 --train-seed 202 --jobs 8      # another training seed
  python experiments/train_dcl.py --fresh-train 48 --ensemble-seeds 101 202 303 --jobs 8
  python experiments/train_dcl.py --iterations 1 --only F00_lch     # quick check
  python experiments/train_dcl.py --target scale5 --fresh-train 48 --ensemble-seeds 101 202 303 \
         --transfer-from factorial-hard --jobs 8                     # N = 5, with zero-shot N = 3 networks

Outputs: results/dcl_generalise{N}_s{seed}[_r{a}-{b}].md/.json for one seed,
results/dcl_ensemble{N}_s{seeds}[_r{a}-{b}].md/.json for an ensemble.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import types
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from common import RESULTS, Budget, md_table, save_json  # noqa: E402
from train_neural import build_fresh, build_targets, gmi_from, prepare, split_description, target_n  # noqa: E402

from supplier_pilot import dcl, dp
from supplier_pilot.instance import core_instance
from supplier_pilot.learner import StatusTablePolicy
from supplier_pilot.policies import SubsetPolicy
from supplier_pilot.metrics import ci95, summarize
from supplier_pilot.simulator import make_streams, mean_cost, simulate

DIAG = ["F00_lch", "F00_lcl", "F05_lcl", "F02_lcl"]


def active_np(gmi):
    return np.ones(len(gmi.S)) if gmi.active is None else np.asarray(gmi.active, float)


# ----------------------------------------------------- labelling (workers)
def label_job(args):
    """Label one instance's states (NumPy only, runs in a worker process)."""
    inst, table, active, states, deltas, M, H, z_conf, seed, confirm_M, search, rounds = args
    bases = dcl.Bases.build([inst], [table], [active])
    S = len(states["I"])
    lab = dcl.label_states(bases, np.zeros(S, dtype=int), states["I"], states["O"], states["z"], deltas, M, H,
                           z_conf, seed=seed, confirm_M=confirm_M, search=search, rounds=rounds)
    return lab


def collect_job(args):
    inst, table, active, episodes, periods, every, seed = args
    bases = dcl.Bases.build([inst], [table], [active])
    return dcl.collect_states(bases, [0], lambda spec, idx, I, O, z: bases.orders(spec, idx, I, O, z),
                              episodes=episodes, periods=periods, every=every, seed=seed)


def fit_width(O, L):
    """Pad or cut pipeline arrays (S, N, L') to width L (extra slots are always 0)."""
    if O.shape[2] == L:
        return O
    if O.shape[2] > L:
        assert np.all(O[:, :, L:] == 0)
        return O[:, :, :L]
    return np.concatenate([O, np.zeros(O.shape[:2] + (L - O.shape[2],))], axis=2)


def concat_labels(labs):
    return dcl.Labels(*[np.concatenate([getattr(l, f) for l in labs]) for f in
                        ("classes", "base_classes", "advantage", "changed", "q_label", "q_base")])


# ------------------------------------------------------------- evaluation
def reference_policy(inst):
    """N = 3: the exact DP optimum. N >= 4: exact DP is intractable, so the
    reference is the optimal policy using only suppliers A, B, C (exact DP of
    the core), an upper bound on the full optimum. Returns (policy, cost)."""
    if inst.n <= 3:
        sol = dp.solve_instance(inst, -10, 26)
        return dp.DPPolicy(sol), sol.g
    core = core_instance(inst)
    sol = dp.solve_instance(core, -10, 26)
    return SubsetPolicy(dp.DPPolicy(sol), core, [0, 1, 2], "DP (A, B, C only)"), sol.g


def evaluate(inst, prep, pol, budget, extra=None, saa=False):
    """Test-seed evaluation. ``extra`` = {name: policy} (e.g. ensemble members),
    each compared with the status table on the same test seeds. Gaps are paired
    against the reference policy (see reference_policy), as % of its excess
    cost; for N >= 4 a negative gap means better than any 3-supplier policy."""
    B = Budget.preset(budget)
    ref_pol, g_ref = reference_policy(inst)
    gmi = gmi_from(prep)
    tab = StatusTablePolicy(gmi, np.array(prep["table"], float), "Status-table")
    va = make_streams(inst, prep["seeds"]["val"], B.tune_T)
    val = {"GMI": prep["gmi_val"], "Status-table": prep["table_val"], "DCL": mean_cost(inst, pol, va, B.tune_warm)}
    chosen = min(val, key=val.get)
    test = make_streams(inst, prep["seeds"]["test"], B.test_T)
    ref = simulate(inst, ref_pol, streams=test, warmup=B.test_warm)
    res, row = {}, {"instance": inst.name, "val": val, "selected": chosen}
    for name, p in (("GMI", gmi), ("Status-table", tab), ("DCL", pol)):
        res[name] = simulate(inst, p, streams=test, warmup=B.test_warm)
        s = summarize(res[name], inst, ref=ref, ref_cost=g_ref)
        row[name] = {"gap": s["paired_gap_excess_pct"], "gap_ci": s["paired_gap_excess_ci95"],
                     "gap_total": s["paired_gap_total_pct"], "fill": s["fill_rate"], "volume_share": s["volume_share"]}
    row["Selected"] = dict(row[chosen], chosen=chosen)
    exc = g_ref - inst.unavoidable_cost()
    for other in ("GMI", "Status-table"):
        d, h = ci95(res["DCL"].episode_cost - res[other].episode_cost)
        row[f"dcl_vs_{other}"] = {"diff_pct_excess": 100 * d / exc, "ci": 100 * h / exc,
                                  "better": bool(d + h < 0), "worse": bool(d - h > 0)}
    if extra:
        row["members"] = {}
        for name, p in extra.items():
            r = simulate(inst, p, streams=test, warmup=B.test_warm)
            s = summarize(r, inst, ref=ref, ref_cost=g_ref)
            d, h = ci95(r.episode_cost - res["Status-table"].episode_cost)
            d2, h2 = ci95(r.episode_cost - res["DCL"].episode_cost)
            row["members"][name] = {"gap": s["paired_gap_excess_pct"], "diff_vs_table": 100 * d / exc,
                                    "ci": 100 * h / exc, "better": bool(d + h < 0), "worse": bool(d - h > 0),
                                    "diff_vs_main": 100 * d2 / exc, "ci_main": 100 * h2 / exc,
                                    "better_than_main": bool(d2 + h2 < 0), "worse_than_main": bool(d2 - h2 > 0)}
    if saa:
        # rolling-horizon SAA is slow (one MILP per period), so it runs on a shorter
        # paired subset of the test seeds, together with the table and DCL
        from supplier_pilot.saa import RollingSAAPolicy
        sub = make_streams(inst, prep["seeds"]["test"][:B.saa_E], B.saa_T)
        r_tab = simulate(inst, tab, streams=sub, warmup=B.saa_warm)
        r_dcl = simulate(inst, pol, streams=sub, warmup=B.saa_warm)
        sp = RollingSAAPolicy(M=B.saa_M)
        t0 = time.time()
        r_saa = simulate(inst, sp, streams=sub, warmup=B.saa_warm)
        secs = time.time() - t0
        row["saa"] = {"episodes": B.saa_E, "periods": B.saa_T,
                      "seconds_per_decision": secs / max(1, B.saa_E * B.saa_T)}
        for name, r in (("SAA", r_saa), ("DCL", r_dcl)):
            d, h = ci95(r.episode_cost - r_tab.episode_cost)
            row["saa"][name] = {"diff_vs_table": 100 * d / exc, "ci": 100 * h / exc,
                                "better": bool(d + h < 0), "worse": bool(d - h > 0)}
    return row


def neural_reference():
    path = os.path.join(RESULTS, "neural_factorial-hard_per-instance_table.json")
    if not os.path.exists(path):
        return {}
    return {r["instance"]: r["Neural"]["gap"] for r in json.load(open(path))["rows"]}


def write_report(rows, meta, path):
    neu = neural_reference()
    m = lambda k: float(np.mean([r[k]["gap"] for r in rows]))
    summ = {"n": len(rows), "GMI": m("GMI"), "Status-table": m("Status-table"), "DCL": m("DCL"),
            "Selected": m("Selected"),
            "dcl_beats_table": float(np.mean([r["dcl_vs_Status-table"]["better"] for r in rows])),
            "dcl_worse_table": float(np.mean([r["dcl_vs_Status-table"]["worse"] for r in rows])),
            "dcl_beats_gmi": float(np.mean([r["dcl_vs_GMI"]["better"] for r in rows])),
            "dcl_selected": float(np.mean([r["selected"] == "DCL" for r in rows]))}
    big = meta.get("n_suppliers", 3) > 3
    if big:
        for k in ("GMI", "Status-table", "DCL"):
            summ[f"{k}_beats_core"] = float(np.mean([r[k]["gap"] + r[k]["gap_ci"] < 0 for r in rows]))
    ref_txt = ("against the DP optimum on held-out test seeds (NumPy simulator), as % of excess cost" if not big else
               f"against the best policy that uses only suppliers A, B, C (exact DP of the 3-supplier core; exact DP "
               f"of the {meta['n_suppliers']}-supplier problem is intractable), on held-out test seeds, as % of its "
               "excess cost. A negative gap means better than any 3-supplier policy")
    rule = meta["split"]["held_out_rule"]
    L = ["# State-aware policy trained by simulation (DCL)", "",
         f"**Split:** {meta['split']['scheme']}. {rule[:1].upper() + rule[1:]}.", "",
         f"Iterations {meta['iterations']}, {meta['labelled_states']} labelled states "
         f"({100 * meta['share_changed']:.0f}% with a non-base label), rollouts M = {meta['M']}, H = {meta['H']}, "
         f"order offsets {meta['deltas']}, {meta.get('search', 'joint')} search; device `{meta['device']}`, "
         f"{meta['minutes']:.0f} min. Gaps are paired {ref_txt}. *Selected* = "
         "lowest validation cost among GMI, status table and DCL.", "",
         md_table([summ], [("n", "n", ""), ("GMI", "GMI %", ".2f"), ("Status-table", "Status table %", ".2f"),
                           ("DCL", "DCL %", ".2f"), ("Selected", "Selected %", ".2f"),
                           ("dcl_beats_table", "DCL sig. better than table", ".2f"),
                           ("dcl_worse_table", "DCL sig. worse than table", ".2f"),
                           ("dcl_beats_gmi", "DCL sig. better than GMI", ".2f"),
                           ("dcl_selected", "DCL selected", ".2f")]), "",
         *([md_table([summ], [("GMI_beats_core", "GMI sig. better than 3-supplier optimum", ".2f"),
                              ("Status-table_beats_core", "Table sig. better than 3-supplier optimum", ".2f"),
                              ("DCL_beats_core", "DCL sig. better than 3-supplier optimum", ".2f")]), ""]
           if big else []),
         md_table([{"instance": r["instance"], "gmi": r["GMI"]["gap"], "tab": r["Status-table"]["gap"],
                    "dcl": r["DCL"]["gap"], "ci": r["DCL"]["gap_ci"], "diff": r["dcl_vs_Status-table"]["diff_pct_excess"],
                    "dci": r["dcl_vs_Status-table"]["ci"], "neu": neu.get(r["instance"], float("nan")),
                    "sel": r["selected"], "it": r.get("iteration", ""), "th": r.get("threshold", 0.0)} for r in rows],
                  [("instance", "Instance", ""), ("gmi", "GMI %", ".2f"), ("tab", "Status table %", ".2f"),
                   ("dcl", "DCL %", ".2f"), ("ci", "±", ".2f"), ("diff", "DCL − table (pp)", "+.2f"), ("dci", "±", ".2f"),
                   ("neu", "Pathwise neural, per-instance %", ".2f"), ("sel", "Selected", "")]), ""]
    if rows and "members" in rows[0]:
        names = list(rows[0]["members"])
        mem = [{"name": nm, "gap": float(np.mean([r["members"][nm]["gap"] for r in rows])),
                "better": float(np.mean([r["members"][nm]["better"] for r in rows])),
                "worse": float(np.mean([r["members"][nm]["worse"] for r in rows])),
                "dmain": float(np.mean([r["members"][nm].get("diff_vs_main", np.nan) for r in rows])),
                "bmain": float(np.mean([r["members"][nm].get("better_than_main", np.nan) for r in rows])),
                "wmain": float(np.mean([r["members"][nm].get("worse_than_main", np.nan) for r in rows]))}
               for nm in names]
        mem.append({"name": meta.get("main_label", "DCL"), "gap": summ["DCL"],
                    "better": summ["dcl_beats_table"], "worse": summ["dcl_worse_table"],
                    "dmain": 0.0, "bmain": float("nan"), "wmain": float("nan")})
        summ["members"] = mem
        rules = []
        if "ensemble" in meta:
            rules.append(f"ensemble rule: {meta['ensemble']['rule']}")
        if "transfer" in meta:
            rules.append(f"zero-shot rule: {meta['transfer']['rule']} (chosen on this target's validation instances)")
        if "ablations" in meta:
            rules.append("*Ablation: v* = the full method's network retrained on the same instances with one "
                         "component changed (see experiments/run_ablations.py); differences to the full method "
                         "are paired on the same test seeds")
        L += ["**Comparison on the same test instances and seeds.** *Seed s* = the network trained on one "
              "training seed (its own validation-chosen iteration and threshold); *zero-shot* = networks trained "
              "on a different number of suppliers, applied without retraining"
              + (f"; {'; '.join(rules)}" if rules else "") + ":", "",
              md_table(mem, [("name", "Policy", ""), ("gap", "Mean gap %", ".2f"),
                             ("better", "Sig. better than table", ".2f"),
                             ("worse", "Sig. worse than table", ".2f"),
                             ("dmain", f"Difference to {meta.get('main_label', 'DCL')} (pp)", "+.2f"),
                             ("bmain", f"Sig. better than {meta.get('main_label', 'DCL')}", ".2f"),
                             ("wmain", f"Sig. worse than {meta.get('main_label', 'DCL')}", ".2f")]), "",
              md_table([dict({"instance": r["instance"], "tab": r["Status-table"]["gap"], "ens": r["DCL"]["gap"]},
                             **{f"m{k}": r["members"][nm]["gap"] for k, nm in enumerate(names)}) for r in rows],
                       [("instance", "Instance", ""), ("tab", "Status table %", ".2f")]
                       + [(f"m{k}", f"{nm} %", ".2f") for k, nm in enumerate(names)]
                       + [("ens", f"{meta.get('main_label', 'DCL')} %", ".2f")]), ""]
    if rows and "saa" in rows[0]:
        sa = [{"name": nm, "diff": float(np.mean([r["saa"][nm]["diff_vs_table"] for r in rows])),
               "better": float(np.mean([r["saa"][nm]["better"] for r in rows])),
               "worse": float(np.mean([r["saa"][nm]["worse"] for r in rows]))} for nm in ("SAA", "DCL")]
        sec = float(np.mean([r["saa"]["seconds_per_decision"] for r in rows]))
        summ["saa"] = sa
        L += [f"**Rolling-horizon SAA** (two-stage MILP re-solved every period; {rows[0]['saa']['episodes']} "
              f"episodes x {rows[0]['saa']['periods']} periods per instance, paired with the table and DCL on the "
              f"same seeds; {sec:.2f} s per decision for SAA). Difference to the status table, as % of the "
              "reference excess cost (negative = better):", "",
              md_table(sa, [("name", "Policy", ""), ("diff", "Mean difference to table (pp)", "+.2f"),
                            ("better", "Sig. better than table", ".2f"),
                            ("worse", "Sig. worse than table", ".2f")]), ""]
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
    return summ


# --------------------------------------------------------------------- main
MEMBER_KEYS = ("iterations", "episodes", "periods", "every", "M", "H", "confirm_M", "z", "deltas", "epochs",
               "thresholds", "fresh_train", "fresh_val", "budget", "table_runs")
LEGACY_TARGET = "factorial-hard"


def target_suffix(target):
    return "" if target == LEGACY_TARGET else f"_{target}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default=LEGACY_TARGET, choices=[LEGACY_TARGET, "scale4", "scale5"],
                    help="instance family: N = 3 factorial hard cells (exact DP), or N = 4 / 5 suppliers")
    ap.add_argument("--only", nargs="*", default=DIAG)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--reps", type=int, nargs="*", default=list(range(8)))
    ap.add_argument("--fresh-train", type=int, default=0)
    ap.add_argument("--fresh-val", type=int, default=8)
    ap.add_argument("--train-seed", type=int, default=101)
    ap.add_argument("--ensemble-seeds", type=int, nargs="*", default=None,
                    help="train (or load) one network per training seed and combine them; needs --fresh-train")
    ap.add_argument("--retrain", action="store_true", help="ignore saved ensemble members and train again")
    ap.add_argument("--transfer-from", default=None, choices=[LEGACY_TARGET, "scale4", "scale5"],
                    help="also evaluate, zero-shot, the saved networks trained on another target (same seeds)")
    ap.add_argument("--transfer-only", action="store_true",
                    help="with --transfer-from: do not train on this target, only test the transferred networks")
    ap.add_argument("--search", default="auto", choices=["auto", "joint", "coord"],
                    help="label search: joint enumeration (N = 3 default) or coordinate search (N >= 4 default)")
    ap.add_argument("--rounds", type=int, default=2, help="coordinate-search rounds")
    ap.add_argument("--saa", action="store_true", help="also evaluate rolling-horizon SAA (slow; short paired runs)")
    ap.add_argument("--prior", default="3", help="initial base-action prior of the classifier; 'none' removes it")
    ap.add_argument("--variant", default="", help="ablation name: saved networks and reports get this suffix")
    ap.add_argument("--train-only", action="store_true", help="train/save the networks, skip the test evaluation")
    ap.add_argument("--compare-variants", nargs="*", default=None,
                    help="also evaluate saved ablation networks (same seed) paired against the main policy")
    ap.add_argument("--iterations", type=int, default=2)
    ap.add_argument("--episodes", type=int, default=None, help="state-collection episodes per instance "
                                                               "(default 32; 8 with --fresh-train)")
    ap.add_argument("--periods", type=int, default=400)
    ap.add_argument("--every", type=int, default=8)
    ap.add_argument("--M", type=int, default=16, help="simulated futures per candidate")
    ap.add_argument("--H", type=int, default=36, help="look-ahead periods per candidate")
    ap.add_argument("--confirm-M", type=int, default=64,
                    help="fresh futures for confirming each label (0 = no confirmation stage)")
    ap.add_argument("--z", type=float, default=1.0, help="label only improvements > z standard errors")
    ap.add_argument("--deltas", type=int, nargs="*", default=list(dcl.DELTAS))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--thresholds", type=float, nargs="*", default=[0.0, 0.5, 0.65, 0.8],
                    help="confidence thresholds tried on validation (deviate from base only if P > threshold)")
    ap.add_argument("--budget", default="quick")
    ap.add_argument("--table-runs", type=int, default=2)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    if a.episodes is None:
        a.episodes = 8 if a.fresh_train else 32
    if (a.ensemble_seeds or a.transfer_from) and not a.fresh_train:
        ap.error("--ensemble-seeds / --transfer-from need --fresh-train N")
    if a.transfer_only and not a.transfer_from:
        ap.error("--transfer-only needs --transfer-from")
    prior = None if str(a.prior).lower() == "none" else float(a.prior)
    n_sup = target_n(a.target)
    if a.search == "auto":
        a.search = "joint" if n_sup <= 3 else "coord"

    import torch
    from supplier_pilot import dcl_torch as dt

    t_start = time.time()
    device = dt.pick_device(a.device)
    deltas = tuple(a.deltas)
    K = len(deltas) + 1
    B = Budget.preset(a.budget)
    targets = build_targets(types.SimpleNamespace(target=a.target, seed=a.seed, reps=a.reps))
    if not a.fresh_train:
        only = set(a.only) if a.target == LEGACY_TARGET or a.only != DIAG else {t[0].name for t in targets[:4]}
        targets = [t for t in targets if t[0].name in only]
    prep_of = {}

    def prepare_all(items):
        todo = [t for t in items if t[0].name not in prep_of]
        if not todo:
            return
        print(f"Preparing baselines for {len(todo)} instances ...", flush=True)
        with ProcessPoolExecutor(a.jobs) as ex:
            for t, p in zip(todo, ex.map(prepare, [(i, k, a.budget, a.table_runs) for i, k, _ in todo])):
                prep_of[t[0].name] = p

    def base_of(inst):
        p = prep_of[inst.name]
        g = gmi_from(p)
        return np.array(p["table"], float), active_np(g)

    def bases1(inst):
        tab, act = base_of(inst)
        return dcl.Bases.build([inst], [tab], [act])

    def val_score(make_pol, val_insts):
        """Mean validation (cost - table) / excess over the validation instances."""
        rel = []
        for inst in val_insts:
            p = prep_of[inst.name]
            va = make_streams(inst, p["seeds"]["val"], B.tune_T)
            c = mean_cost(inst, make_pol(bases1(inst)), va, B.tune_warm)
            rel.append((c - p["table_val"]) / max(p["table_val"] - p["unavoidable"], 1e-6))
        return float(np.mean(rel))

    def train_on(train_insts, tag=""):
        """DCL iterations on ``train_insts``; returns one net per iteration."""
        tabs = [base_of(i) for i in train_insts]
        bases = dcl.Bases.build(train_insts, [t[0] for t in tabs], [t[1] for t in tabs])
        all_states, all_labels, nets = [], [], []
        net = None
        for it in range(1, a.iterations + 1):
            t0 = time.time()
            if it == 1:
                with ProcessPoolExecutor(a.jobs) as ex:
                    st_list = list(ex.map(collect_job, [(inst, tabs[k][0], tabs[k][1], a.episodes, a.periods, a.every,
                                                         1000 * it + k) for k, inst in enumerate(train_insts)]))
            else:
                fn = dt.policy_fn_for({None: net.eval()}, bases, deltas, device, threshold=0.5)
                st_list = []
                for k, inst in enumerate(train_insts):
                    s = dcl.collect_states(bases, [k], fn, episodes=a.episodes, periods=a.periods, every=a.every,
                                           seed=1000 * it + k)
                    st_list.append(s)
            own_L = [dcl.Bases.build([i], [tabs[k][0]], [tabs[k][1]]).spec1.Lmax for k, i in enumerate(train_insts)]
            for k, s_ in enumerate(st_list):
                s_["O"] = fit_width(s_["O"], own_L[k])
            jobs = [(inst, tabs[k][0], tabs[k][1], st_list[k], deltas, a.M, a.H, a.z, 77 * it + k, a.confirm_M,
                     a.search, a.rounds) for k, inst in enumerate(train_insts)]
            with ProcessPoolExecutor(a.jobs) as ex:
                labs = list(ex.map(label_job, jobs))
            for k, s_ in enumerate(st_list):
                s_["inst"] = np.full(len(s_["I"]), k)
                s_["O"] = fit_width(s_["O"], bases.spec1.Lmax)
            all_states += st_list; all_labels += labs
            S = {f: np.concatenate([s[f] for s in all_states]) for f in ("I", "O", "z", "inst")}
            Lb = concat_labels(all_labels)
            n = len(S["I"])
            print(f"{tag}iteration {it}: {n} labelled states, {100 * Lb.changed.mean():.0f}% non-base labels, "
                  f"mean advantage {Lb.advantage.mean():.3f} ({time.time() - t0:.0f}s labelling)", flush=True)
            rng = np.random.default_rng(it)
            perm = rng.permutation(n); cut = int(0.85 * n)
            sub = lambda d, ix: {f: d[f][ix] for f in d}
            subl = lambda l, ix: dcl.Labels(*[getattr(l, f)[ix] for f in ("classes", "base_classes", "advantage",
                                                                          "changed", "q_label", "q_base")])
            tr = dt.tensorize(bases, sub(S, perm[:cut]), subl(Lb, perm[:cut]), device)
            va = dt.tensorize(bases, sub(S, perm[cut:]), subl(Lb, perm[cut:]), device)
            res = dt.fit(tr, va, K, net=net, epochs=a.epochs, seed=it, prior=prior)
            net = res.net
            nets.append(dt.copy.deepcopy(net).cpu())
            print(f"{tag}  classifier: validation joint accuracy {res.val_acc:.3f} ({res.seconds:.0f}s)", flush=True)
        return nets, int(len(S["I"])), float(Lb.changed.mean())

    # ------------------------------------------------ generalisation members
    def fresh_sets(seed, target=a.target):
        ns = types.SimpleNamespace(target=target, train_seed=seed)
        return (build_fresh(ns, range(a.fresh_train), "T", 400_000),
                build_fresh(ns, range(a.fresh_train, a.fresh_train + a.fresh_val), "V", 400_000))

    def member_paths(seed, target=a.target, variant=a.variant):
        stem = os.path.join(RESULTS, f"dcl_member{a.fresh_train}_s{seed}{target_suffix(target)}"
                                     + (f"_{variant}" if variant else ""))
        return stem + ".pt", stem + ".json"

    def config():
        cfg = {k: (list(v) if isinstance(v, (list, tuple)) else v) for k, v in vars(a).items() if k in MEMBER_KEYS}
        if a.target != LEGACY_TARGET or a.search != "joint":       # N = 3 joint members keep their old config
            cfg.update(target=a.target, search=a.search, rounds=a.rounds)
        if prior != 3.0:                                            # default-prior members keep their old config
            cfg["prior"] = prior
        return cfg

    def prior_of(info):
        return info.get("config", {}).get("prior", 3.0)

    def get_member(seed):
        """Train the network for one training seed, or load it if a matching one is saved."""
        tr, va = fresh_sets(seed)
        val_insts = [t[0] for t in va]
        pt, js = member_paths(seed)
        if not a.retrain and os.path.exists(pt) and os.path.exists(js):
            info = json.load(open(js))
            if info.get("config") == config():
                prepare_all(va)
                print(f"[seed {seed}] loaded saved network {os.path.basename(pt)} (iteration {info['iteration']}, "
                      f"threshold {info['threshold']:g})", flush=True)
                return dict(info, net=dt.load_net(pt, K, prior_of(info)), val_insts=val_insts, fresh_tr=tr,
                            fresh_va=va)
            print(f"[seed {seed}] saved network was trained with different settings; retraining", flush=True)
        prepare_all(tr + va)
        t_train = time.time()
        nets, n_lab, ch = train_on([t[0] for t in tr], tag=f"[seed {seed}] ")
        scores, combos = [], []
        for j, net in enumerate(nets):
            for th in a.thresholds:
                scores.append(val_score(lambda b, n=net, th=th: dt.DCLPolicy(n, b, 0, deltas, threshold=th),
                                        val_insts))
                combos.append((j, th))
                print(f"[seed {seed}]   iteration {j + 1}, threshold {th:g}: validation (cost - table)/excess = "
                      f"{scores[-1]:+.4f}", flush=True)
        k = int(np.argmin(scores))
        best, th = combos[k]
        info = {"train_seed": seed, "iteration": best + 1, "threshold": th, "val_score": scores[k],
                "labelled_states": n_lab, "share_changed": ch, "config": config(),
                "train_minutes": (time.time() - t_train) / 60}
        torch.save(nets[best].state_dict(), pt)
        json.dump(info, open(js, "w"), indent=1)
        return dict(info, net=nets[best], val_insts=val_insts, fresh_tr=tr, fresh_va=va)

    def choose_combination(nets, ths, val_insts, tag):
        """Pick how to combine several networks (mean probability with a threshold,
        or unanimous agreement) on validation instances only."""
        if len(nets) == 1:
            cands = [("single", ths[0])] + [("single", t) for t in a.thresholds if t != ths[0]]
        else:
            cands = [("mean", th) for th in a.thresholds] + [("agree", 0.0)]
        scores = []
        for rule, th in cands:
            net = nets[0] if rule == "single" else dt.DCLEnsemble(nets, rule, ths)
            scores.append(val_score(lambda b, e=net, th=th: dt.DCLPolicy(e, b, 0, deltas, threshold=th), val_insts))
            desc = f"{rule}, threshold {th:g}" if rule != "agree" else \
                "agree (member thresholds " + ", ".join(f"{t:g}" for t in ths) + ")"
            print(f"[{tag}] {desc}: validation (cost - table)/excess = {scores[-1]:+.4f}", flush=True)
        rule, th = cands[int(np.argmin(scores))]
        net = nets[0] if rule == "single" else dt.DCLEnsemble(nets, rule, ths)
        txt = {"single": f"threshold {th:g}", "mean": f"mean probability, threshold {th:g}",
               "agree": "unanimous agreement"}[rule]
        return net, th, txt, {f"{r}@{t:g}": sc for (r, t), sc in zip(cands, scores)}

    rows, meta_extra = [], {"n_suppliers": n_sup, "target": a.target, "search": a.search}
    if not a.fresh_train:
        prepare_all(targets)
        labelled, changed = 0, []
        for inst, key, rep in targets:
            nets, n_lab, ch = train_on([inst], tag=f"[{inst.name}] ")
            labelled += n_lab; changed.append(ch)
            b1 = bases1(inst)
            va = make_streams(inst, prep_of[inst.name]["seeds"]["val"], B.tune_T)
            cands = [(j, th, dt.DCLPolicy(n, b1, 0, deltas, threshold=th))
                     for j, n in enumerate(nets) for th in a.thresholds]
            vals = [mean_cost(inst, p, va, B.tune_warm) for _, _, p in cands]
            best, th_best, pol_best = cands[int(np.argmin(vals))]
            row = evaluate(inst, prep_of[inst.name], pol_best, a.budget, saa=a.saa)
            row["iteration"] = best + 1; row["threshold"] = th_best; row["holdout"] = False; row["replicate"] = rep
            torch.save(nets[best].state_dict(), os.path.join(RESULTS, f"dcl_{inst.name}.pt"))
            rows.append(row)
            print(f"  {inst.name}: GMI {row['GMI']['gap']:.2f}%  table {row['Status-table']['gap']:.2f}%  "
                  f"DCL {row['DCL']['gap']:.2f}% (iteration {best + 1}, threshold {th_best:g})  "
                  f"selected {row['Selected']['gap']:.2f}% ({row['selected']})", flush=True)
        tag, mode = f"dcl_per-instance{target_suffix(a.target)}", "per-instance"
        fresh_tr, fresh_va, train_seed = [], [], a.train_seed
    else:
        seeds = a.ensemble_seeds or [a.train_seed]
        members = [] if a.transfer_only else [get_member(s) for s in seeds]
        if a.train_only:
            for m in members:
                print(f"[seed {m['train_seed']}] saved (iteration {m['iteration']}, threshold {m['threshold']:g}, "
                      f"validation {m['val_score']:+.4f}); --train-only: no test evaluation", flush=True)
            return
        if members:
            fresh_tr = [t for m in members for t in m["fresh_tr"]]
            fresh_va = [t for m in members for t in m["fresh_va"]]
        else:                                  # zero-shot only: validation instances of this target, no training
            fresh_tr, fresh_va = [], [t for s_ in seeds for t in fresh_sets(s_)[1]]
            prepare_all(fresh_va)
        val_insts = [t[0] for t in fresh_va]
        prepare_all(targets)
        labelled = sum(m["labelled_states"] for m in members)
        changed = [m["share_changed"] for m in members] or [0.0]
        train_seed = seeds if len(seeds) > 1 else seeds[0]
        suffix = ("" if a.reps == list(range(8)) else f"_r{min(a.reps)}-{max(a.reps)}") + target_suffix(a.target)
        sname = "-".join(map(str, seeds))
        extras = []                           # (label, net, threshold)
        if members:
            if len(members) == 1:
                m = members[0]
                main_net, main_th, main_txt = m["net"], m["threshold"], f"threshold {m['threshold']:g}"
                tag, mode, main_label = f"dcl_generalise{a.fresh_train}_s{sname}{suffix}", "generalise", "DCL"
            else:
                main_net, main_th, main_txt, scores = choose_combination(
                    [m["net"] for m in members], [m["threshold"] for m in members], val_insts, "ensemble")
                for m in members:
                    print(f"[ensemble]   member seed {m['train_seed']} alone: {m['val_score']:+.4f} "
                          "(on its own validation instances)", flush=True)
                    extras.append((f"Seed {m['train_seed']}", m["net"], m["threshold"]))
                meta_extra["ensemble"] = {"seeds": seeds, "rule": main_txt, "validation_scores": scores,
                                          "members": [{k: m[k] for k in ("train_seed", "iteration", "threshold",
                                                                         "val_score", "share_changed")}
                                                      for m in members]}
                tag, mode = f"dcl_ensemble{a.fresh_train}_s{sname}{suffix}", "generalise (ensemble)"
                main_label = "Ensemble"
        if a.transfer_from:
            src_nets, src_ths = [], []
            for s_ in seeds:
                pt, js = member_paths(s_, a.transfer_from)
                if not (os.path.exists(pt) and os.path.exists(js)):
                    raise SystemExit(f"--transfer-from: {os.path.basename(pt)} not found; train on "
                                     f"{a.transfer_from} with the same seeds and --fresh-train first")
                src_info = json.load(open(js))
                src_nets.append(dt.load_net(pt, K, prior_of(src_info))); src_ths.append(src_info["threshold"])
            n_src = target_n(a.transfer_from)
            t_net, t_th, t_txt, t_scores = choose_combination(src_nets, src_ths, val_insts, f"zero-shot N={n_src}")
            t_label = f"Zero-shot from N = {n_src}"
            meta_extra["transfer"] = {"from": a.transfer_from, "seeds": seeds, "rule": t_txt,
                                      "validation_scores": t_scores}
            if members:
                extras.append((t_label, t_net, t_th))
            else:
                main_net, main_th, main_label = t_net, t_th, t_label
                tag = f"dcl_transfer{a.fresh_train}_{a.transfer_from}_s{sname}{suffix}"
                mode = "zero-shot transfer"
        if a.compare_variants:
            abl = []
            for v in a.compare_variants:
                for s_ in seeds:
                    pt, js = member_paths(s_, variant=v)
                    if not (os.path.exists(pt) and os.path.exists(js)):
                        raise SystemExit(f"--compare-variants: {os.path.basename(pt)} not found; run "
                                         f"run_ablations.py (or train_dcl.py --variant {v} --train-only) first")
                    info = json.load(open(js))
                    lab = f"Ablation: {v}" + (f" (seed {s_})" if len(seeds) > 1 else "")
                    extras.append((lab, dt.load_net(pt, K, prior_of(info)), info["threshold"]))
                    abl.append({"variant": v, "seed": s_, **{k: info.get(k) for k in
                                ("iteration", "threshold", "val_score", "share_changed", "labelled_states",
                                 "train_minutes")}})
            meta_extra["ablations"] = abl
            tag = f"dcl_ablation{a.fresh_train}_s{sname}{suffix}"
            mode = mode + " + ablations"
            main_label = "Full method" if len(members) == 1 else main_label
        if a.variant:
            tag += f"_{a.variant}"
        meta_extra["main_label"] = main_label
        for inst, key, rep in targets:
            b1 = bases1(inst)
            extra = {lab: dt.DCLPolicy(net, b1, 0, deltas, threshold=th) for lab, net, th in extras} or None
            row = evaluate(inst, prep_of[inst.name], dt.DCLPolicy(main_net, b1, 0, deltas, threshold=main_th),
                           a.budget, extra=extra, saa=a.saa)
            row["iteration"] = "ensemble" if len(members) > 1 else (members[0]["iteration"] if members else "zero-shot")
            row["threshold"] = main_th; row["holdout"] = True; row["replicate"] = rep
            rows.append(row)
            print(f"  {inst.name} (held out): GMI {row['GMI']['gap']:.2f}%  table {row['Status-table']['gap']:.2f}%  "
                  f"DCL {row['DCL']['gap']:.2f}%  selected {row['Selected']['gap']:.2f}% ({row['selected']})", flush=True)
    split_args = types.SimpleNamespace(seed=a.seed, fresh_train=a.fresh_train, fresh_val=a.fresh_val,
                                       train_seed=train_seed, mode=mode, holdout=[])
    meta = {"mode": mode, "iterations": a.iterations, "labelled_states": labelled,
            "share_changed": float(np.mean(changed)), "M": a.M, "H": a.H, "z": a.z, "confirm_M": a.confirm_M,
            "thresholds": a.thresholds, "deltas": list(deltas), "rounds": a.rounds,
            "episodes": a.episodes, "periods": a.periods, "every": a.every, "epochs": a.epochs, "device": device,
            "minutes": (time.time() - t_start) / 60, "instances": [t[0].name for t in targets],
            "split": split_description(split_args, targets, fresh_tr, fresh_va), **meta_extra}
    summ = write_report(rows, meta, os.path.join(RESULTS, f"{tag}.md"))
    save_json({"meta": meta, "summary": summ, "rows": rows}, f"{tag}.json")
    print(json.dumps({k: v for k, v in summ.items() if k not in ("members", "saa")}, indent=1))
    print("wrote", os.path.join(RESULTS, f"{tag}.md"))


if __name__ == "__main__":
    main()
