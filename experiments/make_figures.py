"""Figures 1-5 of the paper, from the saved result files.

Usage (from the project folder):
    python experiments/make_figures.py
    python experiments/make_figures.py --results results --out results/figures

Writes fig1.pdf ... fig5.pdf (vector, for the journal) and .png copies (for
checking), plus figure_data.txt with every number plotted, so the figures can
be checked against the tables.

Needs only numpy, scipy and matplotlib. No simulation is run.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from scipy import stats
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Result files used by each figure (names as written by the experiment scripts).
F_FACTORIAL = "factorial_study_quick.json"                       # Fig. 2
F_H = ["dcl_ensemble48_s101-202-303.json",                       # Fig. 3, 5: test set H1
       "dcl_ensemble48_s101-202-303_r8-15.json"]                 #             test set H2
F_N4 = ["dcl_ensemble48_s101-202-303_scale4.json",               # Fig. 4
        "dcl_ensemble48_s101-202-303_r8-15_scale4.json"]
F_N5 = ["dcl_ensemble48_s101-202-303_scale5.json",
        "dcl_ensemble48_s101-202-303_r8-15_scale5.json"]
F_ABL = ["dcl_ablation48_s101.json", "dcl_ablation48_s202.json",  # Fig. 5
         "dcl_ablation48_s303.json"]
F_O2 = "dcl_ensemble48_s101-202-303_r8-15_test-factorial-all.json"  # Fig. 5: test set O2

SDI = "Status-table"  # key of SDI in the result files

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "lines.linewidth": 0.9,
    "savefig.dpi": 600,
    "pdf.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
})
W1, W2 = 3.54, 7.48  # single- and double-column widths (inches), Elsevier

# Colour-blind-safe palette (Okabe and Ito), used with the same meaning in every figure:
# SDI orange, GMI sky blue, S-DCL ensemble blue, gated S-DCL policy green, failures vermillion.
# Marker shapes differ as well, so the figures stay readable in greyscale print.
COLOUR = {"sdi": "#E69F00", "gmi": "#56B4E9", "ens": "#0072B2", "gate": "#009E73", "fail": "#D55E00",
          "s101": "#56B4E9", "s202": "#CC79A7", "s303": "#999933", "net": "#CC79A7", "abl": "#555555",
          "main": "#0072B2", "inter": "#8C8C8C", "bar": "#F5D9A8", "line": "#666666",
          "box_base": "#FBE3BC", "box_safe": "#CFE5F5", "box": "white", "edge": "black",
          "fallback": "#E69F00", "fallback_text": "#A86F00"}
GREY = {"sdi": "0.6", "gmi": "0.45", "ens": "black", "gate": "black", "fail": "0.3",
        "s101": "0.55", "s202": "0.55", "s303": "0.55", "net": "0.6", "abl": "0.35",
        "main": "black", "inter": "0.5", "bar": "0.8", "line": "0.4",
        "box_base": "white", "box_safe": "#e3e3e3", "box": "white", "edge": "black",
        "fallback": "0.4", "fallback_text": "0.3"}
P = dict(COLOUR)

LOG: list[str] = []
SUFFIX = ""


def log(*a):
    s = " ".join(str(x) for x in a)
    LOG.append(s)
    print(s)


def load(res, name):
    path = os.path.join(res, name)
    if not os.path.exists(path):
        raise SystemExit(f"Missing result file: {path}")
    with open(path) as f:
        return json.load(f)


def save(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{name}{SUFFIX}.{ext}"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    log(f"  -> {name}{SUFFIX}.pdf / .png")


# --------------------------------------------------------------------------- Fig. 1
def fig1(res, out):
    """Structure of S-DCL (schematic, no data)."""
    log("Fig. 1: structure of S-DCL (schematic)")
    fig, ax = plt.subplots(figsize=(W2, 3.0))
    ax.set_xlim(0, 100)
    ax.set_ylim(-1, 40)
    ax.axis("off")
    grey = P["box_safe"]
    fs = 7.2

    def box(x, y, w, h, text, fill="white"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2",
                                    fc=fill, ec="black", lw=0.7))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, linespacing=1.25)

    def arrow(p0, p1, ls="-", color="black", rad=0.0):
        ax.annotate("", xy=p1, xytext=p0,
                    arrowprops=dict(arrowstyle="-|>", lw=0.7, color=color, ls=ls, shrinkA=0, shrinkB=0,
                                    mutation_scale=8, connectionstyle=f"arc3,rad={rad}"))

    def note(x, y, t, ha="center", va="bottom", color="black"):
        ax.text(x, y, t, ha=ha, va=va, fontsize=6.6, style="italic", color=color)

    for x, t in ((1, "Base policies"), (29, "Training"), (78, "Deployment")):
        ax.text(x, 38.3, t, fontsize=7.6, fontweight="bold")
    for x in (26.5, 75.5):
        ax.plot([x, x], [5.5, 39], color="0.65", lw=0.5, ls=":")

    # Base policies (left column)
    box(1, 26, 22, 8, "GMI\n(one parameter set,\ntuned by simulation)", fill=P["box_base"])
    box(1, 11, 22, 8, "SDI\n(one parameter set per\ndisruption status)", fill=P["box_base"])
    arrow((12, 25.7), (12, 19.3))

    # Training loop: label -> confirm -> classifier -> second pass -> label
    box(29, 11, 20, 8, "Label visited states by\nrollouts on common\nrandom numbers")
    box(54, 11, 19, 8, "Confirmation test\non fresh futures\n(Section 4.3)", fill=grey)
    box(54, 26, 19, 8, "Classifier with a prior\non SDI's order\n(Section 4.4)", fill=grey)
    box(29, 26, 20, 8, "Second labelling pass\non states the trained\npolicy visits (4.5)")
    arrow((23.3, 15), (28.7, 15))
    note(26, 15.6, "states")
    arrow((49.3, 15), (53.7, 15))
    arrow((63.5, 19.3), (63.5, 25.7))
    note(64.3, 21.6, "accepted\nlabels", ha="left", va="center")
    arrow((53.7, 30), (49.3, 30))
    arrow((39, 25.7), (39, 19.3), ls="--")
    note(39.8, 22.5, "new states", ha="left", va="center")

    # Deployment (right column)
    box(78, 26, 21, 8, "Ensemble of E networks\nfrom independent\ntraining sets (4.6)", fill=grey)
    box(78, 11, 21, 8, "Deployment gate: keep\nSDI unless the policy\nwins in validation (4.7)", fill=grey)
    arrow((73.3, 30), (77.7, 30))
    arrow((88.5, 25.7), (88.5, 19.3))
    arrow((88.5, 10.7), (88.5, 6.2))
    ax.text(88.5, 4.6, "S-DCL policy", ha="center", va="center", fontsize=7.6, fontweight="bold", color=P["ens"])

    # Fallback path from SDI to the gate
    ax.plot([12, 12, 81], [10.7, 2.0, 2.0], color=P["fallback"], lw=0.6, ls=(0, (2, 1.5)))
    arrow((81, 2.0), (81, 10.7), ls=(0, (2, 1.5)), color=P["fallback"])
    note(46, 2.5, "fallback: SDI's order", color=P["fallback_text"])

    # Key
    ax.add_patch(FancyBboxPatch((1.3, 35.0), 2.6, 1.8, boxstyle="round,pad=0.2,rounding_size=0.5",
                                fc=grey, ec="black", lw=0.6))
    ax.text(5.0, 35.9, "= safeguard", fontsize=6.6, va="center")
    save(fig, out, "fig1")


# --------------------------------------------------------------------------- Fig. 2
def fig2(res, out):
    """Factorial effects on GMI's gap (linear model, replicate fixed effects, all interactions)."""
    log("Fig. 2: factorial effects on the GMI gap")
    d = load(res, F_FACTORIAL)
    rows = d["rows"]
    y = np.array([r["policies"]["GMI"]["gap_excess"] for r in rows])
    f = np.array([1.0 if r["fixed_cost"] else -1.0 for r in rows])
    s = np.array([1.0 if r["shock"] else -1.0 for r in rows])
    p = np.array([1.0 if r["premium"] else -1.0 for r in rows])
    rep = np.array([r["replicate"] for r in rows])
    reps = sorted(set(rep))
    terms = [("Fixed cost", f), ("Regional shocks", s), ("Price premium", p),
             ("Fixed cost × shocks", f * s), ("Fixed cost × premium", f * p),
             ("Shocks × premium", s * p), ("Three-way", f * s * p)]
    X = np.column_stack([t for _, t in terms] + [(rep == k).astype(float) for k in reps])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ b
    dof = len(y) - np.linalg.matrix_rank(X)
    V = (resid @ resid / dof) * np.linalg.pinv(X.T @ X)
    k = len(terms)
    eff = 2 * b[:k]                     # effect-coded: high minus low = 2 * coefficient
    se = 2 * np.sqrt(np.diag(V)[:k])
    tq = stats.t.ppf(0.975, dof)
    pv = 2 * stats.t.sf(np.abs(eff / se), dof)
    log(f"  n = {len(y)}, replicates = {len(reps)}, residual dof = {dof}")
    for (nm, _), e, s_, p_ in zip(terms, eff, se, pv):
        log(f"  {nm:24s} effect {e:+.2f} pp  SE {s_:.2f}  95% CI [{e - tq * s_:+.2f}, {e + tq * s_:+.2f}]  p = {p_:.4f}")

    fig, ax = plt.subplots(figsize=(W1, 2.5))
    ypos = np.arange(k)[::-1]
    for i in range(k):
        main = i < 3
        ax.errorbar(eff[i], ypos[i], xerr=tq * se[i], fmt="o" if main else "s",
                    ms=4 if main else 3, color=P["main"] if main else P["inter"],
                    mfc=P["main"] if main else "white", capsize=2, lw=0.8)
    ax.axvline(0, color="0.4", lw=0.6, ls="--")
    ax.axhline(k - 3.5, color="0.8", lw=0.5)  # main effects above, interactions below
    ax.set_yticks(ypos)
    ax.set_yticklabels([nm for nm, _ in terms])
    ax.set_xlabel("Effect on GMI's gap, high minus low level (pp)")
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    save(fig, out, "fig2")


# --------------------------------------------------------------------------- Fig. 3
def _diffs(row):
    """Differences to SDI (pp) of the three members and the ensemble for one instance."""
    m = row["members"]
    seeds = sorted(m)
    out = {s: m[s]["diff_vs_table"] for s in seeds}
    out["Ensemble"] = row[f"dcl_vs_{SDI}"]["diff_pct_excess"]
    sig = {s: (m[s]["better"], m[s]["worse"]) for s in seeds}
    sig["Ensemble"] = (row[f"dcl_vs_{SDI}"]["better"], row[f"dcl_vs_{SDI}"]["worse"])
    return out, sig


def fig3(res, out):
    log("Fig. 3: per-instance difference to SDI, hard cells, H1 + H2")
    rows = []
    for name, tag in zip(F_H, ("H1", "H2")):
        for r in load(res, name)["rows"]:
            r = dict(r)
            r["_set"] = tag
            rows.append(r)
    rows.sort(key=lambda r: r[SDI]["gap"])
    series = ["Seed 101", "Seed 202", "Seed 303", "Ensemble"]
    style = {"Seed 101": dict(marker="^", color=P["s101"], ms=3.6),
             "Seed 202": dict(marker="v", color=P["s202"], ms=3.6),
             "Seed 303": dict(marker="D", color=P["s303"], ms=3.0),
             "Ensemble": dict(marker="o", color=P["ens"], ms=4.0)}
    x = np.arange(len(rows))
    fig, (ax0, ax) = plt.subplots(2, 1, figsize=(W2, 3.3), sharex=True,
                                  gridspec_kw=dict(height_ratios=[1, 2.6], hspace=0.08))
    ax0.bar(x, [r[SDI]["gap"] for r in rows], color=P["bar"], edgecolor=P["sdi"], linewidth=0.4, width=0.7)
    ax0.set_ylabel("SDI gap (%)")
    ax.axhline(0, color="0.4", lw=0.6)
    vals = {s: [] for s in series}
    for i, r in enumerate(rows):
        d, sig = _diffs(r)
        for s in series:
            v = d[s]
            vals[s].append(v)
            filled = sig[s][0] or sig[s][1]
            st = style[s]
            ax.plot(i + {"Seed 101": -0.25, "Seed 202": 0.0, "Seed 303": 0.25, "Ensemble": 0.0}[s], v,
                    ls="none", marker=st["marker"], ms=st["ms"], mec=st["color"],
                    mfc=st["color"] if filled else "white", mew=0.7, zorder=3 if s == "Ensemble" else 2)
    for s in series:
        v = np.array(vals[s])
        log(f"  {s:9s} mean {v.mean():+.3f} pp, worst {v.max():+.2f} pp, best {v.min():+.2f} pp")
    big = [(i, s, vals[s][i]) for s in series[:3] for i in range(len(rows)) if vals[s][i] > 3]
    for i, s, v in big:
        right = i > len(rows) - 6
        ax.annotate(f"seed {s.split()[1]}: +{v:.2f}", (i, v), xytext=(-5 if right else 5, 0),
                    textcoords="offset points", fontsize=6.8, ha="right" if right else "left", va="center")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['instance']}" for r in rows], rotation=90, fontsize=6.2)
    ax.set_ylabel("Difference to SDI (pp)")
    ax.set_xlim(-0.8, len(rows) - 0.2)
    handles = [Line2D([], [], ls="none", marker=style[s]["marker"], ms=style[s]["ms"],
                      mec=style[s]["color"], mfc=style[s]["color"], label=f"Single network, {s.lower()}"
                      if s != "Ensemble" else "S-DCL ensemble") for s in series]
    handles.append(Line2D([], [], ls="none", marker="o", ms=4, mec=P["line"], mfc="white",
                          label="open: not significant"))
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(0.01, 0.6), frameon=False, ncol=2, fontsize=7)
    save(fig, out, "fig3")


# --------------------------------------------------------------------------- Fig. 4
def fig4(res, out):
    log("Fig. 4: gap to the restricted three-supplier optimum, N = 4 and 5")
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.6), sharey=True)
    pols = [("GMI", "GMI", dict(marker="s", mfc="white", mec=P["gmi"], ms=3.4)),
            (SDI, "SDI", dict(marker="^", mfc=P["sdi"], mec=P["sdi"], ms=3.6)),
            ("DCL", "S-DCL ensemble", dict(marker="o", mfc=P["ens"], mec=P["ens"], ms=3.4))]
    for ax, files, n in zip(axes, (F_N4, F_N5), (4, 5)):
        rows = [r for fn in files for r in load(res, fn)["rows"]]
        rows.sort(key=lambda r: r[SDI]["gap"])
        x = np.arange(len(rows))
        for key, lab, st in pols:
            v = np.array([r[key]["gap"] for r in rows])
            ax.plot(x, v, ls="none", mew=0.7, label=lab, **st)
            log(f"  N = {n} {lab:15s} mean {v.mean():+.2f}%  min {v.min():+.2f}  max {v.max():+.2f}  "
                f"(n = {len(v)})")
        ax.axhline(0, color="0.4", lw=0.6, ls="--")
        ax.set_title(f"N = {n} (test sets combined, {len(rows)} instances)")
        ax.set_xlabel("Test instance, sorted by SDI's gap")
        ax.set_xticks([])
    axes[0].set_ylabel("Gap to restricted optimum (%)")
    axes[0].legend(frameon=False, loc="lower right")
    save(fig, out, "fig4")


# --------------------------------------------------------------------------- Fig. 5
def _gain_worst(d):
    d = np.asarray(d)
    return -d.mean(), d.max(), len(d)


def fig5(res, out):
    log("Fig. 5: mean gain on SDI against worst-instance loss")
    pts = []  # (label, gain, worst, n, group)

    # Ablation: full method and five variants, 3 seeds x 16 instances of H1
    abl = [load(res, fn) for fn in F_ABL]
    full = [r[f"dcl_vs_{SDI}"]["diff_pct_excess"] for d in abl for r in d["rows"]]
    pts.append(("Full single network", *_gain_worst(full), "abl"))
    names = {"no_confirm": "No confirmation", "one_iter": "One labelling pass", "no_prior": "No prior",
             "coord": "Coordinate search", "short_H": "H = 12"}
    for key, lab in names.items():
        mk = f"Ablation: {key}"
        v = [r["members"][mk]["diff_vs_table"] for d in abl for r in d["rows"] if mk in r["members"]]
        if v:
            pts.append((lab, *_gain_worst(v), "abl"))

    # Single networks and ensemble, H1 + H2 (32 instances)
    hrows = [r for fn in F_H for r in load(res, fn)["rows"]]
    for s in ("Seed 101", "Seed 202", "Seed 303"):
        pts.append((f"Network {s.split()[1]}", *_gain_worst([r["members"][s]["diff_vs_table"] for r in hrows]), "net"))
    pts.append(("Ensemble", *_gain_worst([r[f"dcl_vs_{SDI}"]["diff_pct_excess"] for r in hrows]), "ens"))

    # Out-of-range confirmation set O2 (64 instances): ensemble without and with gate
    o2 = load(res, F_O2)["rows"]
    pts.append(("Ensemble, O2, no gate", *_gain_worst([r[f"dcl_vs_{SDI}"]["diff_pct_excess"] for r in o2]), "nogate"))
    # Deployment gate (eq. 21): keep the learned policy iff its validation cost is <= SDI's.
    # (The "Selected" field in older files is a three-way choice incl. GMI; it is not the gate.)
    gated = [r[f"dcl_vs_{SDI}"]["diff_pct_excess"] if r["val"]["DCL"] <= r["val"][SDI] else 0.0 for r in o2]
    pts.append(("S-DCL policy (gated), O2", *_gain_worst(gated), "gate"))

    for lab, g, w, n, grp in pts:
        log(f"  {lab:28s} gain {g:+.3f} pp  worst {w:+.2f} pp  (n = {n})")

    st = {"abl": dict(marker="s", mfc="white", mec=P["abl"], ms=4.2),
          "net": dict(marker="D", mfc=P["net"], mec=P["net"], ms=3.8),
          "ens": dict(marker="o", mfc=P["ens"], mec=P["ens"], ms=5),
          "nogate": dict(marker="X", mfc=P["fail"], mec=P["fail"], ms=5.5),
          "gate": dict(marker="*", mfc=P["gate"], mec=P["gate"], ms=8)}
    fig, ax = plt.subplots(figsize=(W1 * 1.35, 3.0))
    off = {"Full single network": (6, 3, "left"), "Network 303": (6, -4, "left"),
           "No prior": (6, 3, "left"), "No confirmation": (-7, 0, "right"),
           "One labelling pass": (6, 0, "left"), "Coordinate search": (6, 0, "left"),
           "H = 12": (0, 9, "center"), "Network 101": (6, 0, "left"), "Network 202": (-6, 0, "right"),
           "Ensemble": (-7, 0, "right"), "Ensemble, O2, no gate": (-7, 0, "right"),
           "S-DCL policy (gated), O2": (7, 0, "left")}
    for lab, g, w, n, grp in pts:
        ax.plot(w, g, ls="none", mew=0.7, **st[grp], zorder=3)
        dx, dy, ha = off.get(lab, (6, 0, "left"))
        ax.annotate(lab, (w, g), xytext=(dx, dy), textcoords="offset points",
                    fontsize=6.6, ha=ha, va="center")
    ax.set_xscale("symlog", linthresh=1.0)
    ax.set_xlim(-0.9, 30)
    ax.set_xticks([0, 0.5, 1, 2, 5, 10, 20])
    ax.set_xticklabels(["0", "0.5", "1", "2", "5", "10", "20"])
    ax.axvline(0, color="0.6", lw=0.5, ls="--")
    ax.axhline(0, color="0.6", lw=0.5, ls="--")
    ax.set_xlabel("Worst-instance loss to SDI, $L_{\\max}$ (pp, symmetric log scale)")
    ax.set_ylabel("Mean gain on SDI (pp)")
    handles = [Line2D([], [], ls="none", mew=0.7, **st[k], label=l) for k, l in
               (("abl", "Ablation, H1 (48 pairs)"), ("net", "Single network, H1+H2 (32)"),
                ("ens", "Ensemble, H1+H2 (32)"), ("nogate", "Ensemble, O2 (64)"),
                ("gate", "S-DCL policy, O2 (64)"))]
    ax.legend(handles=handles, frameon=False, loc="lower left", bbox_to_anchor=(0.12, 0.02), fontsize=6.6)
    save(fig, out, "fig5")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(ROOT, "results"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--only", nargs="*", type=int, help="e.g. --only 3 5")
    ap.add_argument("--grey", action="store_true", help="greyscale versions (for print), written with suffix _grey")
    a = ap.parse_args()
    out = a.out or os.path.join(a.results, "figures")
    if a.grey:
        P.clear(); P.update(GREY)
        global SUFFIX
        SUFFIX = "_grey"
    os.makedirs(out, exist_ok=True)
    figs = {1: fig1, 2: fig2, 3: fig3, 4: fig4, 5: fig5}
    for k in (a.only or figs):
        figs[k](a.results, out)
    with open(os.path.join(out, "figure_data.txt"), "w") as f:
        f.write("\n".join(LOG) + "\n")
    print(f"\nFigures and figure_data.txt written to {out}")


if __name__ == "__main__":
    main()
