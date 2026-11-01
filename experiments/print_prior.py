"""Print the learned base-action prior (lambda in eq. 18) of every saved network.

The prior starts at 3.0. Proposition 4 needs lambda > 0 only at the start of
training; this script reports where training left it, for the paper.

Usage:  python experiments/print_prior.py
"""
import glob
import os

import torch

from common import RESULTS  # noqa: F401

rows = []
for pt in sorted(glob.glob(os.path.join(RESULTS, "dcl_*.pt"))):
    sd = torch.load(pt, map_location="cpu")
    if "prior" in sd:
        rows.append((os.path.basename(pt), float(sd["prior"])))
    else:
        rows.append((os.path.basename(pt), None))
w = max(len(r[0]) for r in rows) if rows else 10
print(f"{'network':{w}s}  lambda")
for name, lam in rows:
    print(f"{name:{w}s}  {'no prior (ablation / original DCL)' if lam is None else f'{lam:.3f}'}")
vals = [l for _, l in rows if l is not None]
if vals:
    print(f"\n{len(vals)} networks with a prior: min {min(vals):.3f}, max {max(vals):.3f}, "
          f"negative: {sum(v < 0 for v in vals)}")
