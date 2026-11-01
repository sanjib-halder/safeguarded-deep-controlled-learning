"""Bar chart: gap to the DP optimum (% of excess cost) per pilot instance for
the tuned heuristic and the two structured learners (mean of training runs,
whiskers = min/max across runs). Usage: python experiments/plot_pilot.py [--budget quick]"""
from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import RESULTS  # noqa: E402

COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]   # validated categorical slots 1-3


def main(budget):
    outs = json.load(open(f"{RESULTS}/pilot_results_{budget}.json"))
    names = [o["instance"].replace("_", "\n", 1) for o in outs]
    groups = [("GMI heuristic", lambda p: p == "GMI heuristic"),
              ("Structured-ES (neural, ES)", lambda p: p.startswith("Structured-ES")),
              ("Status-table (direct search)", lambda p: p.startswith("Status-table"))]
    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=160)
    w = 0.26
    x = np.arange(len(outs))
    for g, (label, sel) in enumerate(groups):
        vals = [[r["paired_gap_excess_pct"] for r in o["rows"] if sel(r["policy"])] for o in outs]
        m = np.array([np.mean(v) for v in vals])
        lo = m - np.array([np.min(v) for v in vals])
        hi = np.array([np.max(v) for v in vals]) - m
        xs = x + (g - 1) * w
        ax.bar(xs, m, w - 0.03, color=COLORS[g], label=label, zorder=2)
        if g > 0:
            ax.errorbar(xs, m, yerr=[lo, hi], fmt="none", ecolor="#555", elinewidth=1, capsize=2, zorder=3)
        tops = m + (hi if g > 0 else 0)
        for xi, mi, ti in zip(xs, m, tops):
            ax.text(xi, ti + 0.12, f"{mi:.1f}", ha="center", va="bottom", fontsize=8, color="#333")
    ax.set_xticks(x, names, fontsize=9)
    ax.set_ylabel("Gap to DP optimum (% of excess cost)", fontsize=9)
    ax.set_title("Pilot (N = 3): heuristic vs. structured learners, held-out test seeds", fontsize=10,
                 loc="left")
    ax.grid(axis="y", color="#e5e5e5", zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    path = f"{RESULTS}/pilot_gaps_{budget}.png"
    fig.savefig(path)
    print("wrote", path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", default="quick")
    main(ap.parse_args().budget)
