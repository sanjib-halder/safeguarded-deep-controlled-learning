"""Train and evaluate the state-aware neural policy (PyTorch).

Default target: the hardest factorial cells (low fixed cost, correlated
shocks; both price-premium levels), replicates 0-7 of seed 7, which are the
same instances as the factorial study run with default arguments. One shared
network is trained on replicates 0-5 and evaluated on all eight; replicates
6-7 are *held out* (never seen in training) to test generalisation.

Steps
  1. per instance (NumPy, parallel, cached): exact DP, tuned GMI heuristic,
     status-table learner, validation costs;
  2. train the network with pathwise gradients (PyTorch; GPU if available);
  3. evaluate DP, GMI, status table, neural and the validation-selected
     policy in the NumPy simulator on held-out test seeds (paired, CRN).

Usage
  python experiments/train_neural.py                         # defaults (see above)
  python experiments/train_neural.py --iters 1500 --device cuda
  python experiments/train_neural.py --target pilot --holdout   # P1-P4, no hold-out
  python experiments/train_neural.py --mode per-instance --only F00_lch F00_lcl F05_lcl F02_lcl
                                                            # diagnostic: can training beat the table at all?
  python experiments/train_neural.py --base gmi --bound 0   # the first version (GMI base, unbounded)
  python experiments/train_neural.py --fresh-train 48 --iters 1500 --jobs 8
        # generalisation test: train on 96 new instances (seed 101), select on 16 more,
        # test on the 16 seed-7 instances (never seen)
  python experiments/train_neural.py --eval-only results/neural_factorial-hard.pt
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from common import RESULTS, Budget, md_table, save_json  # noqa: E402

from supplier_pilot import dp
from supplier_pilot.instance import core_instance, factorial_instance, pilot_instances, scaled_instance
from supplier_pilot.learner import StatusTablePolicy, tune_status_table
from supplier_pilot.metrics import ci95, summarize
from supplier_pilot.policies import MultiIndexPolicy
from supplier_pilot.simulator import make_streams, mean_cost, simulate
from supplier_pilot.tuning import SeedSplit, tune_gmi

CACHE = os.path.join(RESULTS, "cache")
os.makedirs(CACHE, exist_ok=True)


# ------------------------------------------------------------- instances
SCALE_TARGETS = {"scale4": 4, "scale5": 5}     # N = 4 / 5 suppliers, hard cells, no exact DP


def target_n(target: str) -> int:
    return SCALE_TARGETS.get(target, 3)


def make_instance(target, seed, rep, k, s, p):
    if target in SCALE_TARGETS:
        return scaled_instance(seed, rep, SCALE_TARGETS[target], k, s, p)
    return factorial_instance(seed, rep, k, s, p)


def target_cells(a):
    if a.target == "factorial-all":
        import itertools
        return list(itertools.product(["low", "high"], ["none", "correlated"], ["low", "high"]))
    return [("low", "correlated", "low"), ("low", "correlated", "high")]


def build_targets(a):
    out = []          # (instance, key, rep)
    if a.target == "pilot":
        for j, inst in enumerate(pilot_instances()):
            out.append((inst, 200_000 + j, j))
    else:
        key0 = 300_000 if a.target not in SCALE_TARGETS else 500_000 + 10_000 * target_n(a.target)
        for rep in a.reps:
            for c, (k, s, p) in enumerate(target_cells(a)):
                inst = make_instance(a.target, a.seed, rep, k, s, p)
                out.append((inst, key0 + rep * 8 + c, rep))
    return out


LEGACY_TRAIN_SEED = 101     # names without the seed are kept for 101 so existing caches stay valid


def build_fresh(a, reps, prefix, key0):
    """New instances from the same cells but a different factorial seed, renamed
    (T.. = training, V.. = validation) so they never collide with test instances.
    The training seed is part of the name (e.g. T202_00_lcl), so instances drawn
    with different seeds never share a cache entry."""
    out = []
    scale = a.target in SCALE_TARGETS
    if scale:
        key0 += 200_000 + 10_000 * target_n(a.target)
    for rep in reps:
        for c, (k, s, p) in enumerate(target_cells(a)):
            inst = make_instance(a.target, a.train_seed, rep, k, s, p)
            if scale:                      # e.g. T101_N5_00_lcl
                name = f"{prefix}{a.train_seed}_{inst.name}"
            else:
                tag = "" if a.train_seed == LEGACY_TRAIN_SEED else f"{a.train_seed}_"
                name = prefix + tag + inst.name[1:]
            out.append((inst.with_(name=name), key0 + rep * 8 + c, rep))
    return out


def fingerprint(inst) -> str:
    """Hash of everything that defines an instance (not its name)."""
    import hashlib
    spec = (inst.suppliers, inst.demand_values, inst.demand_probs, inst.holding, inst.backorder,
            inst.regional_shock)
    return hashlib.sha1(repr(spec).encode()).hexdigest()


def _cache_valid(out, inst) -> bool:
    """A cached baseline is used only if it belongs to this exact instance. Old
    cache files have no fingerprint; they are checked once against the DP
    optimum and the unavoidable cost, then upgraded."""
    fp = fingerprint(inst)
    if "fingerprint" in out:
        return out["fingerprint"] == fp
    if abs(out["unavoidable"] - inst.unavoidable_cost()) > 1e-9 or inst.n > 3:
        return False
    g = dp.solve_instance(inst, -10, 26).g
    if abs(g - out["g"]) > 1e-6 * max(1.0, abs(g)):
        return False
    out["fingerprint"] = fp
    return True


# --------------------------------------------------------- baselines (NumPy)
def prepare(args):
    inst, key, budget, table_runs = args
    path = os.path.join(CACHE, f"{inst.name}_{key}_{budget}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            out = pickle.load(f)
        had_fp = "fingerprint" in out
        if _cache_valid(out, inst):
            if not had_fp:
                with open(path, "wb") as f:
                    pickle.dump(out, f)
            return out
        print(f"[prep {inst.name}] cached baseline belongs to a different instance; recomputing", flush=True)
    B = Budget.preset(budget)
    t0 = time.time()
    seeds = SeedSplit.default(B.tune_E, B.val_E, B.test_E, base=key * 10)
    tr = make_streams(inst, seeds.train, B.tune_T)
    va = make_streams(inst, seeds.val, B.tune_T)
    # exact DP only for N = 3; for N >= 4, the DP of the A-B-C core (3-supplier optimum)
    sol = dp.solve_instance(inst, -10, 26) if inst.n <= 3 else None
    g3 = dp.solve_instance(core_instance(inst), -10, 26).g if inst.n > 3 else None
    # N >= 4: no explicit subset enumeration (triggers switch suppliers off), same procedure for N = 4 and 5
    gmi, _, _ = tune_gmi(inst, tr, B.tune_warm, rng=np.random.default_rng(key), subsets_up_to=3)
    gmi.name = "GMI"
    tables = []
    for r in range(table_runs):
        trr = make_streams(inst, [key * 10 + 5_000_000 + 100 * r + j for j in range(B.tune_E)], B.tune_T)
        tab, _, _ = tune_status_table(inst, gmi, trr, B.tune_warm)
        tables.append((mean_cost(inst, tab, va, B.tune_warm), tab.table))
    best_tab = min(tables, key=lambda x: x[0])
    out = {"name": inst.name, "key": key, "fingerprint": fingerprint(inst), "n": inst.n,
           "g": None if sol is None else sol.g, "g_core3": g3,
           "unavoidable": inst.unavoidable_cost(),
           "gmi": {"S": gmi.S.tolist(), "cap": gmi.cap.tolist(), "trig": gmi.trig.tolist(),
                   "active": None if gmi.active is None else np.asarray(gmi.active).tolist()},
           "gmi_val": mean_cost(inst, gmi, va, B.tune_warm), "table": best_tab[1].tolist(),
           "table_val": best_tab[0], "seeds": {"val": seeds.val, "test": seeds.test},
           "seconds": round(time.time() - t0, 1)}
    with open(path, "wb") as f:
        pickle.dump(out, f)
    ref = f"DP g*={sol.g:.3f}" if sol is not None else f"3-supplier DP g={g3:.3f}"
    print(f"[prep {inst.name}] {ref} GMI val={out['gmi_val']:.3f} table val={out['table_val']:.3f} "
          f"({out['seconds']:.0f}s)", flush=True)
    return out


def split_description(a, targets, fresh_tr, fresh_va) -> dict:
    """Explicit record of which instances were used for training, checkpoint
    selection and testing, so 'held out' is unambiguous in every mode.
    'replicate' in result rows always refers to the factorial replicate
    within the *test* seed (--seed)."""
    test = {"seed": a.seed, "replicates": sorted({r for _, _, r in targets}), "n": len(targets)}
    if a.fresh_train:
        return {"scheme": "fresh-instance generalisation",
                "train": {"seed": a.train_seed, "replicates": [0, a.fresh_train - 1], "n": len(fresh_tr)},
                "validation": {"seed": a.train_seed, "replicates": [a.fresh_train, a.fresh_train + a.fresh_val - 1],
                               "n": len(fresh_va)},
                "test": test,
                "held_out_rule": "every test instance is held out: training and validation use a different "
                                 "seed, so no test instance (or its supplier configuration) is seen"}
    if a.mode == "per-instance":
        return {"scheme": "per-instance (one network per instance)", "train": test, "test": test,
                "held_out_rule": "nothing is held out; each network is trained and tested on its own "
                                 "instance (test uses separate random seeds)"}
    tr = [r for r in test["replicates"] if r not in a.holdout]
    return {"scheme": "replicate hold-out within one seed",
            "train": {"seed": a.seed, "replicates": tr},
            "validation": {"seed": a.seed, "replicates": tr, "note": "fresh random streams on training instances"},
            "test": test,
            "held_out_rule": f"replicates {sorted(a.holdout)} of seed {a.seed} are held out; the others are "
                             "training instances evaluated on new random seeds"}


def gmi_from(p):
    act = None if p["gmi"]["active"] is None else np.array(p["gmi"]["active"], bool)
    return MultiIndexPolicy(np.array(p["gmi"]["S"], float), np.array(p["gmi"]["cap"], float),
                            np.array(p["gmi"]["trig"], float), act, "GMI")


# --------------------------------------------------------------- evaluation
def evaluate(inst, prep, neural_pol, budget):
    B = Budget.preset(budget)
    sol = dp.solve_instance(inst, -10, 26)
    gmi = gmi_from(prep)
    tab = StatusTablePolicy(gmi, np.array(prep["table"], float), "Status-table")
    va = make_streams(inst, prep["seeds"]["val"], B.tune_T)
    val = {"GMI": prep["gmi_val"], "Status-table": prep["table_val"],
           "Neural": mean_cost(inst, neural_pol, va, B.tune_warm)}
    chosen = min(val, key=val.get)
    test = make_streams(inst, prep["seeds"]["test"], B.test_T)
    ref = simulate(inst, dp.DPPolicy(sol), streams=test, warmup=B.test_warm)
    res = {}
    for name, pol in (("GMI", gmi), ("Status-table", tab), ("Neural", neural_pol)):
        res[name] = simulate(inst, pol, streams=test, warmup=B.test_warm)
    row = {"instance": inst.name, "val": val, "selected": chosen}
    for name, r in res.items():
        s = summarize(r, inst, ref=ref, ref_cost=sol.g)
        row[name] = {"gap": s["paired_gap_excess_pct"], "gap_ci": s["paired_gap_excess_ci95"],
                     "gap_total": s["paired_gap_total_pct"], "fill": s["fill_rate"],
                     "volume_share": s["volume_share"]}
    row["Selected"] = dict(row[chosen], chosen=chosen)
    exc = sol.g - inst.unavoidable_cost()
    for other in ("GMI", "Status-table"):
        d, h = ci95(res["Neural"].episode_cost - res[other].episode_cost)
        row[f"neural_vs_{other}"] = {"diff_pct_excess": 100 * d / exc, "ci": 100 * h / exc,
                                     "better": bool(d + h < 0), "worse": bool(d - h > 0)}
    return row


def report(rows, meta, path):
    def grp(sel):
        rs = [r for r in rows if sel(r)]
        if not rs:
            return None
        g = lambda k: float(np.mean([r[k]["gap"] for r in rs]))
        return {"n": len(rs), "GMI": g("GMI"), "Status-table": g("Status-table"), "Neural": g("Neural"),
                "Selected": g("Selected"),
                "neural_beats_table": float(np.mean([r["neural_vs_Status-table"]["better"] for r in rs])),
                "neural_worse_table": float(np.mean([r["neural_vs_Status-table"]["worse"] for r in rs])),
                "neural_beats_gmi": float(np.mean([r["neural_vs_GMI"]["better"] for r in rs])),
                "selected_neural": float(np.mean([r["selected"] == "Neural" for r in rs]))}
    groups = [("training instances", lambda r: not r["holdout"]), ("held-out instances", lambda r: r["holdout"]),
              ("all", lambda r: True)]
    summ = [dict(group=n, **s) for n, f in groups if (s := grp(f))]
    L = ["# State-aware neural policy", "",
         f"**Split:** {meta['split']['scheme']}. {meta['split']['held_out_rule'][:1].upper() + meta['split']['held_out_rule'][1:]}.", "",
         f"Target `{meta['target']}`, mode `{meta['mode']}`, base `{meta['base']}`, bound ±{meta['bound']:g}, "
         f"{meta['iters']} iterations on `{meta['device']}` "
         f"({meta['train_seconds']:.0f} s training). Gaps are paired against the DP optimum on held-out test "
         "seeds (NumPy simulator), as % of excess cost. *Selected* = lowest validation cost among GMI, "
         "status-table and neural (heuristic as fallback).", "",
         "## Summary (mean gap %, share of instances)", "",
         md_table(summ, [("group", "Group", ""), ("n", "n", ""), ("GMI", "GMI", ".2f"),
                         ("Status-table", "Status-table", ".2f"), ("Neural", "Neural", ".2f"),
                         ("Selected", "Selected", ".2f"), ("neural_beats_table", "Neural sig. better than table", ".2f"),
                         ("neural_worse_table", "Neural sig. worse than table", ".2f"),
                         ("neural_beats_gmi", "Neural sig. better than GMI", ".2f"),
                         ("selected_neural", "Neural selected", ".2f")]), "",
         "## Per instance", "",
         md_table([{"instance": r["instance"] + (" (held out)" if r["holdout"] else ""),
                    "gmi": r["GMI"]["gap"], "tab": r["Status-table"]["gap"], "neu": r["Neural"]["gap"],
                    "neu_ci": r["Neural"]["gap_ci"], "sel": r["Selected"]["gap"], "chosen": r["selected"],
                    "vs_tab": r["neural_vs_Status-table"]["diff_pct_excess"],
                    "vs_tab_ci": r["neural_vs_Status-table"]["ci"]} for r in rows],
                  [("instance", "Instance", ""), ("gmi", "GMI %", ".2f"), ("tab", "Table %", ".2f"),
                   ("neu", "Neural %", ".2f"), ("neu_ci", "±", ".2f"), ("sel", "Selected %", ".2f"),
                   ("chosen", "Chosen", ""), ("vs_tab", "Neural − table (pp)", "+.2f"), ("vs_tab_ci", "±", ".2f")]),
         ""]
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="factorial-hard", choices=["factorial-hard", "factorial-all", "pilot"])
    ap.add_argument("--seed", type=int, default=7, help="factorial seed (7 = the study you ran)")
    ap.add_argument("--reps", type=int, nargs="*", default=list(range(8)))
    ap.add_argument("--holdout", type=int, nargs="*", default=[6, 7],
                    help="replicates kept out of training (empty = train on all)")
    ap.add_argument("--mode", default="shared", choices=["shared", "per-instance"])
    ap.add_argument("--budget", default="quick", choices=["smoke", "quick", "full"])
    ap.add_argument("--table-runs", type=int, default=2)
    ap.add_argument("--jobs", type=int, default=4, help="processes for the NumPy baselines")
    ap.add_argument("--iters", type=int, default=600)
    ap.add_argument("--episodes", type=int, default=None,
                    help="training episodes per instance (default 64; 16 with --fresh-train)")
    ap.add_argument("--horizon", type=int, default=48)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--eval-only", default=None, help="path of a saved .pt network")
    ap.add_argument("--base", default="table", choices=["table", "gmi"],
                    help="policy the network adjusts: tuned status table (default) or plain GMI")
    ap.add_argument("--bound", type=float, default=2.0,
                    help="max adjustment per parameter in units (<= 0: unbounded)")
    ap.add_argument("--only", nargs="*", default=None, help="restrict to these instance names")
    ap.add_argument("--fresh-train", type=int, default=0,
                    help="generalisation test: train on this many NEW replicates (seed --train-seed), select the "
                         "checkpoint on --fresh-val further new replicates, test on the --seed/--reps instances")
    ap.add_argument("--fresh-val", type=int, default=8)
    ap.add_argument("--train-seed", type=int, default=101)
    ap.add_argument("--ema", type=float, default=0.98, help="weight averaging (0 = off)")
    ap.add_argument("--val-every", type=int, default=None)
    a = ap.parse_args()
    if a.episodes is None:
        a.episodes = 16 if a.fresh_train else 64

    import torch
    from supplier_pilot.neural import (NeuralPolicy, SupplierSetPolicy, TrainConfig, active_of, pick_device,
                                       table_from_gmi, train)

    targets = build_targets(a)
    if a.only:
        targets = [t for t in targets if t[0].name in set(a.only)]
        missing = set(a.only) - {t[0].name for t in targets}
        if missing:
            raise SystemExit(f"unknown instance names: {sorted(missing)}")
    if a.target == "pilot" and a.holdout == [6, 7]:
        a.holdout = []
    fresh_tr, fresh_va = [], []
    if a.fresh_train:
        if a.target == "pilot":
            raise SystemExit("--fresh-train needs a factorial target")
        a.mode = "generalise"
        fresh_tr = build_fresh(a, range(a.fresh_train), "T", 400_000)
        fresh_va = build_fresh(a, range(a.fresh_train, a.fresh_train + a.fresh_val), "V", 400_000)
    everything = targets + fresh_tr + fresh_va
    tasks = [(inst, key, a.budget, a.table_runs) for inst, key, _ in everything]
    print(f"Preparing baselines for {len(tasks)} instances (cached in {CACHE}) ...", flush=True)
    if a.jobs > 1:
        with ProcessPoolExecutor(a.jobs) as ex:
            all_preps = list(ex.map(prepare, tasks))
    else:
        all_preps = [prepare(t) for t in tasks]
    preps = all_preps[:len(targets)]
    preps_tr = all_preps[len(targets):len(targets) + len(fresh_tr)]
    preps_va = all_preps[len(targets) + len(fresh_tr):]

    hold = [rep in a.holdout for _, _, rep in targets]
    if a.fresh_train:
        hold = [True] * len(targets)          # every test instance is unseen
    train_ids = [i for i, h in enumerate(hold) if not h] or list(range(len(targets)))
    cfg = TrainConfig(episodes_per_instance=a.episodes, horizon=a.horizon, iters=a.iters, lr=a.lr, device=a.device,
                      val_every=a.val_every or max(a.iters // 24, 5), bound=a.bound, ema=a.ema)
    device = pick_device(a.device)
    tag = f"{a.target}_{a.mode}_{a.base}"
    if a.fresh_train:
        tag = f"{a.target}_generalise{a.fresh_train}_{a.base}"

    def base_from(p):
        gmi = gmi_from(p)
        tab = np.array(p["table"], float) if a.base == "table" else table_from_gmi(gmi)
        val = p["table_val"] if a.base == "table" else p["gmi_val"]
        return tab, active_of(gmi), val - p["unavoidable"]

    def base_of(i):
        return base_from(preps[i])
    nets, train_s = {}, 0.0
    if a.eval_only:
        net = SupplierSetPolicy(cfg.d_model, layers=cfg.layers)
        net.load_state_dict(torch.load(a.eval_only, map_location="cpu"))
        nets = {i: net for i in range(len(targets))}
    elif a.fresh_train:
        btr = [base_from(p) for p in preps_tr]
        bva = [base_from(p) for p in preps_va]
        print(f"Training one shared network on {len(fresh_tr)} new instances (seed {a.train_seed}); "
              f"checkpoint selection on {len(fresh_va)} validation instances; testing on {len(targets)} "
              f"unseen instances (seed {a.seed}) on {device} ...", flush=True)
        res = train([t[0] for t in fresh_tr], [b[0] for b in btr], [b[1] for b in btr], [b[2] for b in btr], cfg,
                    val_set=([t[0] for t in fresh_va], [b[0] for b in bva], [b[1] for b in bva],
                             [b[2] for b in bva]))
        train_s = res.seconds
        torch.save(res.net.state_dict(), os.path.join(RESULTS, f"neural_{tag}.pt"))
        save_json(res.history, f"neural_{tag}_history.json")
        nets = {i: res.net for i in range(len(targets))}
    elif a.mode == "shared":
        ids = train_ids
        print(f"Training one shared network on {len(ids)} instances on {device} ...", flush=True)
        bases = [base_of(i) for i in ids]
        res = train([targets[i][0] for i in ids], [b[0] for b in bases], [b[1] for b in bases],
                    [b[2] for b in bases], cfg)
        train_s = res.seconds
        torch.save(res.net.state_dict(), os.path.join(RESULTS, f"neural_{tag}.pt"))
        nets = {i: res.net for i in range(len(targets))}
        save_json(res.history, f"neural_{tag}_history.json")
    else:
        for i in range(len(targets)):
            print(f"Training on {targets[i][0].name} ...", flush=True)
            tab, act, exc = base_of(i)
            res = train([targets[i][0]], [tab], [act], [exc], cfg)
            nets[i] = res.net; train_s += res.seconds
            hold[i] = False

    print("Evaluating in the NumPy simulator ...", flush=True)
    rows = []
    for i, (inst, key, rep) in enumerate(targets):
        tab, act, _ = base_of(i)
        pol = NeuralPolicy(nets[i].cpu(), tab, act, bound=a.bound)
        row = evaluate(inst, preps[i], pol, a.budget)
        row["holdout"] = bool(hold[i]); row["replicate"] = rep
        rows.append(row)
        print(f"  {inst.name}{' (held out)' if hold[i] else ''}: GMI {row['GMI']['gap']:.2f}%  "
              f"table {row['Status-table']['gap']:.2f}%  neural {row['Neural']['gap']:.2f}%  "
              f"selected {row['Selected']['gap']:.2f}% ({row['selected']})", flush=True)
    meta = {"target": a.target, "mode": a.mode, "iters": a.iters, "device": device, "train_seconds": train_s,
            "holdout_reps": None if a.fresh_train or a.mode == "per-instance" else a.holdout,
            "split": split_description(a, targets, fresh_tr, fresh_va),
            "base": a.base, "bound": a.bound, "lr": a.lr, "ema": a.ema,
            "episodes": a.episodes, "instances": [t[0].name for t in targets],
            "fresh_train": len(fresh_tr), "fresh_val": len(fresh_va), "train_seed": a.train_seed}
    summ = report(rows, meta, os.path.join(RESULTS, f"neural_{tag}.md"))
    save_json({"meta": meta, "summary": summ, "rows": rows}, f"neural_{tag}.json")
    print(json.dumps(summ, indent=1))
    print("wrote", os.path.join(RESULTS, f"neural_{tag}.md"))


if __name__ == "__main__":
    main()
