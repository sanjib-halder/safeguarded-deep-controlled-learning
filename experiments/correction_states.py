"""Where does the S-DCL ensemble depart from SDI? (Appendix E of the paper.)

Simulates the saved N = 3 ensemble (dcl_member48_s101/202/303) on the hard-cell
test instances and records, in every period, SDI's order b and the ensemble's
order q. It reports how often q differs from b, in which disruption statuses,
for which supplier, in which direction, and at what inventory level.

Nothing is trained and no result file of the paper is changed. Baselines come
from the existing cache (results/cache), so the run takes a few minutes.

Usage (from the project folder):
    python experiments/correction_states.py                 # H1 and H2 (reps 0-15)
    python experiments/correction_states.py --reps 0 1 2    # quick check

Writes results/correction_states.md and results/correction_states.json.
"""
from __future__ import annotations

import argparse
import json
import os
import types
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from common import RESULTS, save_json  # noqa: E402
from train_neural import build_targets, gmi_from, prepare  # noqa: E402

from supplier_pilot import dcl

SUP = ["A", "B", "C"]


def status_label(z):
    down = [SUP[i] for i in range(len(z)) if z[i] < 0.5]
    if not down:
        return "all up"
    return " and ".join(down) + " down"


def inv_bucket(i):
    if i < 0:
        return "backorders (I < 0)"
    if i <= 3:
        return "0 to 3"
    if i <= 7:
        return "4 to 7"
    return "8 or more"


def direction(b, q):
    if b > 0 and q == 0:
        return "cancel SDI's order"
    if b == 0 and q > 0:
        return "order where SDI does not"
    if q > b:
        return "order more"
    return "order less"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, nargs="*", default=list(range(16)))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--seeds", type=int, nargs="*", default=[101, 202, 303], help="ensemble member seeds")
    ap.add_argument("--episodes", type=int, default=16)
    ap.add_argument("--periods", type=int, default=1000)
    ap.add_argument("--budget", default="quick")
    ap.add_argument("--table-runs", type=int, default=2)
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()

    import torch  # noqa: F401
    from supplier_pilot import dcl_torch as dt

    nets, ths = [], []
    for s in a.seeds:
        stem = os.path.join(RESULTS, f"dcl_member48_s{s}")
        info = json.load(open(stem + ".json"))
        nets.append(dt.load_net(stem + ".pt", dcl.n_classes(dcl.DELTAS), info.get("config", {}).get("prior", 3.0)))
        ths.append(info["threshold"])
    ens = dt.DCLEnsemble(nets, "mean", ths)

    targets = build_targets(types.SimpleNamespace(target="factorial-hard", seed=a.seed, reps=a.reps))
    print(f"Loading baselines for {len(targets)} instances (from cache) ...", flush=True)
    with ProcessPoolExecutor(a.jobs) as ex:
        preps = list(ex.map(prepare, [(i, k, a.budget, a.table_runs) for i, k, _ in targets]))

    n_per = Counter(); n_corr = Counter()                 # by status
    by_dir = Counter(); by_sup = Counter(); by_inv = Counter(); n_inv = Counter()
    by_status_sup = defaultdict(Counter)
    per_inst = []
    total, total_corr = 0, 0
    for (inst, key, rep), p in zip(targets, preps):
        g = gmi_from(p)
        act = np.ones(inst.n) if g.active is None else np.asarray(g.active, float)
        bases = dcl.Bases.build([inst], [np.array(p["table"], float)], [act])
        ens_fn = dt.policy_fn_for({None: ens}, bases, dcl.DELTAS, "cpu", threshold=0.0)
        rec = {"I": [], "z": [], "b": [], "q": []}

        def fn(spec, idx, I, O, z, rec=rec):
            b, _ = bases.orders(spec, idx, I, O, z)
            q, gg = ens_fn(spec, idx, I, O, z)
            rec["I"].append(np.asarray(I).copy()); rec["z"].append(np.asarray(z).copy())
            rec["b"].append(np.rint(b).copy()); rec["q"].append(np.rint(q).copy())
            return q, gg

        dcl.collect_states(bases, [0], fn, episodes=a.episodes, burn_in=100, periods=a.periods, every=a.periods,
                           seed=900_000 + key)
        I = np.concatenate(rec["I"])[a.episodes * 100:]       # drop the warm-up periods
        Z = np.concatenate(rec["z"])[a.episodes * 100:]
        Bq = np.concatenate(rec["b"])[a.episodes * 100:]
        Q = np.concatenate(rec["q"])[a.episodes * 100:]
        diff = np.any(Q != Bq, axis=1)
        total += len(I); total_corr += int(diff.sum())
        per_inst.append({"instance": inst.name, "rep": rep, "share_corrected": float(diff.mean())})
        for t in range(len(I)):
            lab = status_label(Z[t]); n_per[lab] += 1
            ib = inv_bucket(I[t]); n_inv[ib] += 1
            if not diff[t]:
                continue
            n_corr[lab] += 1; by_inv[ib] += 1
            for i in range(inst.n):
                if Q[t, i] != Bq[t, i]:
                    by_sup[SUP[i]] += 1; by_status_sup[lab][SUP[i]] += 1
                    by_dir[direction(Bq[t, i], Q[t, i])] += 1
        print(f"  {inst.name} (rep {rep}): {100 * diff.mean():.1f}% of periods corrected", flush=True)

    lines = ["# Where the S-DCL ensemble departs from SDI", "",
             f"Hard-cell test instances, replicates {min(a.reps)}-{max(a.reps)} ({len(targets)} instances), "
             f"{a.episodes} episodes x {a.periods} periods each after 100 warm-up periods. "
             f"Overall, the ensemble's order differs from SDI's in {100 * total_corr / total:.2f}% of periods.", "",
             "## By joint disruption status", "",
             "| Status | Share of periods (%) | Share of these periods corrected (%) | Share of all corrections (%) "
             "| Corrections by supplier (A / B / C, %) |", "|---|---|---|---|---|"]
    for lab, n in n_per.most_common():
        c = n_corr[lab]
        s = by_status_sup[lab]; tot = sum(s.values()) or 1
        lines.append(f"| {lab} | {100 * n / total:.2f} | {100 * c / n:.2f} | {100 * c / max(total_corr, 1):.1f} | "
                     + " / ".join(f"{100 * s[x] / tot:.0f}" for x in SUP) + " |")
    lines += ["", "## By net inventory at the start of the period", "",
              "| Net inventory | Share of periods (%) | Share corrected (%) |", "|---|---|---|"]
    for ib in ["backorders (I < 0)", "0 to 3", "4 to 7", "8 or more"]:
        if n_inv[ib]:
            lines.append(f"| {ib} | {100 * n_inv[ib] / total:.2f} | {100 * by_inv[ib] / n_inv[ib]:.2f} |")
    nd = sum(by_dir.values()) or 1
    lines += ["", "## Direction of the corrected supplier orders", "",
              "| Direction | Share of corrected supplier orders (%) |", "|---|---|"]
    for k, v in by_dir.most_common():
        lines.append(f"| {k} | {100 * v / nd:.1f} |")
    ns = sum(by_sup.values()) or 1
    lines += ["", "## Corrected supplier orders by supplier", "", "| Supplier | Share (%) |", "|---|---|"]
    for x in SUP:
        lines.append(f"| {x} | {100 * by_sup[x] / ns:.1f} |")
    lines += ["", "## Per instance", "", "| Instance | Replicate | Periods corrected (%) |", "|---|---|---|"]
    for r in per_inst:
        lines.append(f"| {r['instance']} | {r['rep']} | {100 * r['share_corrected']:.2f} |")
    path = os.path.join(RESULTS, "correction_states.md")
    open(path, "w").write("\n".join(lines) + "\n")
    save_json({"total_periods": total, "corrected": total_corr, "by_status": dict(n_per),
               "corrected_by_status": dict(n_corr), "by_direction": dict(by_dir), "by_supplier": dict(by_sup),
               "by_inventory": dict(n_inv), "corrected_by_inventory": dict(by_inv),
               "by_status_supplier": {k: dict(v) for k, v in by_status_sup.items()}, "per_instance": per_inst},
              "correction_states.json")
    print("wrote", path)


if __name__ == "__main__":
    main()
