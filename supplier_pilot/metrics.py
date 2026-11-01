"""Evaluation metrics. CVaR and regret are *reported*, never optimised here:
every policy in this package is trained/tuned on expected average cost."""
from __future__ import annotations

import numpy as np
from scipy import stats

from .instance import Instance
from .simulator import SimResult


def ci95(x: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return float(x.mean()), float("nan")
    half = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean()), float(half)


def block_cvar(cost: np.ndarray, alpha: float = 0.95, block: int = 50) -> float:
    """CVaR_alpha of average cost over consecutive blocks of ``block`` periods
    (e.g. a block = one planning year), pooled across episodes."""
    E, T = cost.shape
    nb = T // block
    if nb == 0:
        return float("nan")
    b = cost[:, : nb * block].reshape(E, nb, block).mean(axis=2).ravel()
    var = np.quantile(b, alpha)
    return float(b[b >= var].mean())


def summarize(res: SimResult, inst: Instance, ref: SimResult | None = None,
              ref_cost: float | None = None) -> dict:
    m, h = ci95(res.episode_cost)
    unavoid = inst.unavoidable_cost()
    active = (res.orders > 0)
    vol = res.orders.sum(axis=(0, 1)).astype(float)
    out = {
        "instance": inst.name,
        "policy": res.policy,
        "avg_cost": m,
        "ci95": h,
        "excess_cost": m - unavoid,
        "fill_rate": float(res.served.sum() / max(res.demand.sum(), 1)),
        "cvar95_block50": block_cvar(res.cost),
        "active_suppliers_per_period": float(active.sum(axis=2).mean()),
        "order_freq": [round(float(f), 4) for f in active.mean(axis=(0, 1))],
        "volume_share": [round(float(v), 4) for v in vol / max(vol.sum(), 1)],
        "holding": float(res.components["holding"].mean()),
        "backorder": float(res.components["backorder"].mean()),
        "ordering": float(res.components["order_cost"].mean()),
        "projections": res.projections,
        "sim_seconds": round(res.seconds, 2),
        "episodes": int(res.cost.shape[0]),
        "periods": int(res.cost.shape[1]),
    }
    if ref is not None:
        diff = res.episode_cost - ref.episode_cost      # paired by CRN
        dm, dh = ci95(diff)
        out["paired_diff_vs_ref"] = dm
        out["paired_diff_ci95"] = dh
        # denominators use g* when known (stable even on short test runs)
        rm = float(ref_cost) if ref_cost is not None else float(ref.episode_cost.mean())
        out["paired_gap_total_pct"] = 100 * dm / rm
        rex = rm - unavoid
        out["paired_gap_excess_pct"] = 100 * dm / rex if rex > 0 else float("nan")
        out["paired_gap_excess_ci95"] = 100 * dh / rex if rex > 0 else float("nan")
    if ref_cost is not None:
        out["gap_total_pct"] = 100 * (m - ref_cost) / ref_cost
        exc = ref_cost - unavoid
        out["gap_excess_pct"] = 100 * (m - ref_cost) / exc if exc > 0 else float("nan")
    return out


def activation_structure(res: SimResult, inst: Instance) -> dict:
    """How strongly does the *set* of activated suppliers depend on the state
    beyond the disruption status? For each joint status, measure the share of
    decisions whose activation pattern differs from that status's most common
    pattern. High values mean activation depends on inventory/pipeline state."""
    pat = (res.orders > 0).astype(int) @ (1 << np.arange(inst.n))
    zid = res.z.astype(int) @ (1 << np.arange(inst.n))
    pat, zid = pat.ravel(), zid.ravel()
    rows = []
    total_dev = 0
    for z in np.unique(zid):
        p = pat[zid == z]
        vals, cnt = np.unique(p, return_counts=True)
        dev = 1 - cnt.max() / cnt.sum()
        total_dev += dev * len(p)
        rows.append({"status": "".join("U" if (z >> i) & 1 else "D" for i in range(inst.n)),
                     "share_of_time": round(len(p) / len(pat), 4),
                     "distinct_patterns": int(len(vals)),
                     "deviation_from_modal": round(float(dev), 4),
                     "modal_pattern": "".join(str((int(vals[cnt.argmax()]) >> i) & 1)
                                              for i in range(inst.n))})
    return {"overall_deviation": round(total_dev / len(pat), 4),
            "distinct_patterns_total": int(len(np.unique(pat))),
            "by_status": rows}
