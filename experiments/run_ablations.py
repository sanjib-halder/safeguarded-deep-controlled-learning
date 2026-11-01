"""Ablation study of DCL (N = 3 factorial hard cells, generalisation setting).

Each variant changes one component of the method and is trained exactly like
the full method's seed network (same 96 training instances, same validation
instances, same settings otherwise). All variants are then evaluated in one
run, on the same 16 held-out test instances and test seeds as the full method,
so every comparison is paired (common random numbers).

Variants (default: all)
  no_confirm   labels without the confirmation stage (--confirm-M 0):
               the best of the 216 candidates is accepted on the same futures
               that selected it (optimiser's curse)
  coord        coordinate-search labelling instead of joint enumeration
               (the labeller used for N = 4 / 5)
  one_iter     a single labelling round, no DAgger-style second iteration
  no_prior     no prior on the base action in the classifier: the network
               has to learn the base action itself
  short_H      look-ahead of 12 periods instead of 36 when scoring candidates
  z05, z196    confirmation threshold z = 0.5 or 1.96 instead of 1.0 (sensitivity;
               only when named: --variants z05 z196; output gets the suffix _z05-z196)

The full method's network is the saved seed network (results/dcl_member48_s{seed}),
which is reused without retraining. The confidence threshold is not ablated
separately: validation chose threshold 0 (no threshold) for all three N = 3
networks, so the full method already runs without it.

Usage
  caffeinate -i python experiments/run_ablations.py --jobs 8
  caffeinate -i python experiments/run_ablations.py --variants no_confirm coord --jobs 8
  python experiments/run_ablations.py --compare-only        # re-run only the paired evaluation

Output: results/dcl_ablation48_s{seed}.md / .json (one table: each variant vs.
the table, vs. the DP optimum and paired against the full method). Each
variant's network is saved as results/dcl_member48_s{seed}_{variant}.pt/.json
and reused if the script is run again.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

from common import RESULTS  # noqa: F401  (also cleans pasted arguments)

VARIANTS = {
    "no_confirm": ["--confirm-M", "0"],
    "coord": ["--search", "coord"],
    "one_iter": ["--iterations", "1"],
    "no_prior": ["--prior", "none"],
    "short_H": ["--H", "12"],
    # sensitivity of the confirmation threshold z (default 1.0); not run by default
    "z05": ["--z", "0.5"],
    "z196": ["--z", "1.96"],
}
DEFAULT_VARIANTS = ["no_confirm", "coord", "one_iter", "no_prior", "short_H"]

HERE = os.path.dirname(os.path.abspath(__file__))


def run(cmd):
    print("\n$ " + " ".join(cmd), flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, cwd=os.path.dirname(HERE))
    print(f"(finished in {(time.time() - t0) / 60:.0f} min, exit code {r.returncode})", flush=True)
    if r.returncode != 0:
        sys.exit(r.returncode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="*", default=DEFAULT_VARIANTS, choices=list(VARIANTS))
    ap.add_argument("--train-seed", type=int, default=101)
    ap.add_argument("--fresh-train", type=int, default=48)
    ap.add_argument("--reps", type=int, nargs="*", default=None, help="test replicates (default 0-7)")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--compare-only", action="store_true", help="skip training; evaluate saved variants")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                    help="further train_dcl.py arguments passed to every run (e.g. --extra --budget quick)")
    a = ap.parse_args()
    script = os.path.join(HERE, "train_dcl.py")
    base = [sys.executable, script, "--fresh-train", str(a.fresh_train), "--train-seed", str(a.train_seed),
            "--jobs", str(a.jobs)] + list(a.extra)
    if not a.compare_only:
        for v in a.variants:
            run(base + ["--variant", v, "--train-only"] + VARIANTS[v])
    cmp = base + ["--compare-variants"] + a.variants
    if a.reps:
        cmp += ["--reps"] + [str(r) for r in a.reps]
    run(cmp)
    extra = "" if sorted(a.variants) == sorted(DEFAULT_VARIANTS) else "_" + "-".join(a.variants)
    print(f"\nAblation table: results/dcl_ablation{a.fresh_train}_s{a.train_seed}"
          + (f"_r{min(a.reps)}-{max(a.reps)}" if a.reps and a.reps != list(range(8)) else "") + extra + ".md")


if __name__ == "__main__":
    main()
