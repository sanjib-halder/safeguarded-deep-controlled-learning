"""Gradient check: does the training gradient point downhill on the TRUE cost?

The neural policy is trained with pathwise gradients that use straight-through
surrogates for rounding and order/no-order decisions. This script tests
whether those gradients agree with the exact simulated cost.

For each instance and each test point (the untrained network, and a randomly
perturbed one) it fixes a start state and the random numbers (common random
numbers), so the cost L(theta) is a deterministic function of the weights:

  1. Directional derivatives. For K random directions d it compares
       autograd:  g . d            (what training uses)
       true:      [L(theta + e d) - L(theta - e d)] / (2 e)   (exact hard simulation)
     for several step sizes e, and reports sign agreement and correlation.
     The true cost is piecewise constant in the weights (integer orders), so
     tiny e gives zero differences; larger e measures the smoothed slope.
  2. Descent test. It moves the weights a small step against the gradient
     (theta - s * g/|g|) and reports the change in the exact cost, next to the
     same steps along random directions as a baseline.

Reading the verdict
  * sign agreement >= 0.75 and cost decreasing along -g  -> gradient informative;
    look for the training problem elsewhere (learning rate, data, features)
  * sign agreement <= 0.60 or cost not decreasing         -> gradient biased /
    uninformative; switch to a gradient-free (simulation-based) learner

Usage (needs torch; uses the cached baselines from train_neural.py):
  python experiments/check_gradients.py                    # F00_lch and F05_lcl, ~10-20 min
  python experiments/check_gradients.py --only F00_lch F00_lcl F05_lcl F02_lcl --directions 16
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time
import types

import numpy as np

from common import RESULTS, md_table, save_json  # noqa: E402
from train_neural import build_targets, gmi_from, prepare  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=["F00_lch", "F05_lcl"])
    ap.add_argument("--episodes", type=int, default=256)
    ap.add_argument("--horizons", type=int, nargs="*", default=[48, 240],
                    help="periods per cost evaluation (48 = the training window)")
    ap.add_argument("--directions", type=int, default=12)
    ap.add_argument("--eps", type=float, nargs="*", default=[0.003, 0.01, 0.03, 0.1])
    ap.add_argument("--steps", type=float, nargs="*", default=[0.01, 0.03, 0.1, 0.3])
    ap.add_argument("--budget", default="quick")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import torch
    from supplier_pilot import batched as bt
    from supplier_pilot.neural import BaseParams, SupplierSetPolicy, active_of, rollout

    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    ops = bt.torch_ops("cpu")
    targets = [t for t in build_targets(types.SimpleNamespace(target="factorial-hard", seed=7, reps=list(range(8))))
               if t[0].name in set(a.only)]
    out_rows, t_start = [], time.time()

    class Replay:
        """Fixed random numbers, replayed identically on every evaluation."""
        def __init__(self, spec, T, seed):
            g = torch.Generator().manual_seed(seed)
            B, N, R = len(spec.inst_idx), spec.n, spec.R
            self.draws = [(torch.rand(B, generator=g), torch.rand(B, N, generator=g), torch.rand(B, R, generator=g))
                          for _ in range(T)]
            self.i = 0

        def reset(self):
            self.i = 0

        def draw(self):
            d = self.draws[self.i]; self.i += 1
            return d

    for inst, key, rep in targets:
        prep = prepare((inst, key, a.budget, 2))
        gmi = gmi_from(prep)
        table, active = np.array(prep["table"], float), active_of(gmi)
        idx = np.zeros(a.episodes, dtype=int)
        spec = bt.build_spec(ops, [inst], idx)
        base = BaseParams.build([table], [active], idx, "cpu")
        burn = Replay(spec, 300, 1234)
        net0 = SupplierSetPolicy(out_scale=1.0)
        _, _, start = rollout(net0, ops, spec, base, bt.initial(ops, spec, a.episodes), burn, 300, grad=False,
                              use_base=True)
        start = tuple(s.detach() for s in start)

        # test points: untrained network, and a perturbed one (outputs of order 0.5 units)
        pert = copy.deepcopy(net0)
        with torch.no_grad():
            torch.nn.init.normal_(pert.head[-1].weight, std=0.05)
            torch.nn.init.normal_(pert.head[-1].bias, std=0.2)
        points = [("untrained (= status table)", net0, lambda n: [n.head[-1].weight, n.head[-1].bias]),
                  ("perturbed", pert, lambda n: list(n.parameters()))]

        for T in a.horizons:
            streams = Replay(spec, T, 777 + T)

            def cost(net):
                streams.reset()
                _, per_ep, _ = rollout(net, ops, spec, base, start, streams, T, grad=False)
                return float(per_ep.mean())

            for pname, net, sel in points:
                params = sel(net)
                # autograd gradient on the same streams
                for p in net.parameters():
                    p.grad = None
                streams.reset()
                loss, _, _ = rollout(net, ops, spec, base, start, streams, T, grad=True)
                loss.backward()
                g = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten() for p in params])
                theta = torch.cat([p.detach().flatten() for p in params])
                L0 = cost(net)
                gnorm = float(g.norm())

                def set_vec(v):
                    k = 0
                    with torch.no_grad():
                        for p in params:
                            m = p.numel(); p.copy_(v[k:k + m].view_as(p)); k += m

                def cost_at(v):
                    set_vec(v); c = cost(net); set_vec(theta); return c

                # 1) directional derivatives
                dirs = [torch.as_tensor(rng.normal(size=theta.numel()), dtype=torch.float32) for _ in range(a.directions)]
                dirs = [d / d.norm() for d in dirs]
                ad = [float(g @ d) for d in dirs]
                per_eps = []
                for e in a.eps:
                    fd = [(cost_at(theta + e * d) - cost_at(theta - e * d)) / (2 * e) for d in dirs]
                    nz = [i for i, v in enumerate(fd) if abs(v) > 1e-9]
                    agree = float(np.mean([np.sign(fd[i]) == np.sign(ad[i]) for i in nz])) if nz else float("nan")
                    corr = float(np.corrcoef(np.array(ad)[nz], np.array(fd)[nz])[0, 1]) if len(nz) > 2 else float("nan")
                    per_eps.append({"eps": e, "nonzero_fd": len(nz), "sign_agreement": agree, "correlation": corr,
                                    "mean_abs_fd": float(np.mean(np.abs(fd))), "mean_abs_ad": float(np.mean(np.abs(ad)))})
                # 2) descent test
                ghat = g / max(gnorm, 1e-12)
                descent = []
                for s in a.steps:
                    dL = cost_at(theta - s * ghat) - L0
                    rand = [cost_at(theta + s * d) - L0 for d in dirs[:4]]
                    descent.append({"step": s, "dL_along_minus_grad": dL, "dL_random_mean": float(np.mean(rand)),
                                    "dL_random_min": float(np.min(rand))})
                best = max(per_eps, key=lambda r: (r["nonzero_fd"], -abs(r["eps"] - 0.03)))
                row = {"instance": inst.name, "point": pname, "horizon": T, "cost": L0, "grad_norm": gnorm,
                       "directional": per_eps, "descent": descent,
                       "sign_agreement_best_eps": best["sign_agreement"], "best_eps": best["eps"],
                       "descent_works": any(d["dL_along_minus_grad"] < 0 for d in descent[:2])}
                out_rows.append(row)
                print(f"[{inst.name} | {pname} | T={T}] cost {L0:.3f}  |grad| {gnorm:.3g}  "
                      + "  ".join(f"e={r['eps']:g}: agree {r['sign_agreement']:.2f} (n={r['nonzero_fd']}), "
                                  f"corr {r['correlation']:.2f}" for r in per_eps), flush=True)
                print("      descent: " + "  ".join(f"s={d['step']:g}: dL {d['dL_along_minus_grad']:+.4f} "
                                                    f"(random {d['dL_random_mean']:+.4f})" for d in descent), flush=True)

    agree = [r["sign_agreement_best_eps"] for r in out_rows if np.isfinite(r["sign_agreement_best_eps"])]
    desc = [r["descent_works"] for r in out_rows]
    mean_agree = float(np.mean(agree)) if agree else float("nan")
    if mean_agree >= 0.75 and np.mean(desc) >= 0.75:
        verdict = "INFORMATIVE: gradients agree with the true cost; look for the training problem elsewhere."
    elif mean_agree <= 0.60 or np.mean(desc) <= 0.5:
        verdict = "BIASED/UNINFORMATIVE: gradients do not reliably point downhill; use a gradient-free learner."
    else:
        verdict = "MIXED: gradients are partly informative; consider a lower-variance or gradient-free learner."
    summary = {"mean_sign_agreement": mean_agree, "share_descent_works": float(np.mean(desc)), "verdict": verdict,
               "minutes": round((time.time() - t_start) / 60, 1)}
    save_json({"summary": summary, "rows": out_rows, "args": vars(a)}, "gradient_check.json")
    L = ["# Gradient check", "", f"**Verdict:** {verdict}", "",
         f"Mean sign agreement (autograd vs exact finite differences, at the step size with most non-zero "
         f"differences): {mean_agree:.2f}. Descent along the negative gradient lowered the exact cost in "
         f"{100 * np.mean(desc):.0f}% of cases (two smallest steps).", "",
         md_table([{"instance": r["instance"], "point": r["point"], "T": r["horizon"], "agree": r["sign_agreement_best_eps"],
                    "eps": r["best_eps"], "d1": r["descent"][0]["dL_along_minus_grad"],
                    "d2": r["descent"][1]["dL_along_minus_grad"], "r1": r["descent"][0]["dL_random_mean"],
                    "ok": r["descent_works"]} for r in out_rows],
                  [("instance", "Instance", ""), ("point", "Point", ""), ("T", "Horizon", ""),
                   ("agree", "Sign agreement", ".2f"), ("eps", "at e", "g"), ("d1", "dL, step 1", "+.4f"),
                   ("d2", "dL, step 2", "+.4f"), ("r1", "dL random dir, step 1", "+.4f"), ("ok", "Descent works", "")]),
         "", "Negative dL = the exact cost went down. Full details in `gradient_check.json`."]
    path = os.path.join(RESULTS, "gradient_check.md")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
    print(json.dumps(summary, indent=1))
    print("wrote", path)


if __name__ == "__main__":
    main()
