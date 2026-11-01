"""Simulation-based tuning of heuristic parameters with common random numbers.

Tuning uses a *training* set of seeds; policy selection uses a separate
*validation* set; final numbers are reported on a held-out *test* set.
"""
from __future__ import annotations

import itertools
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .instance import Instance
from .policies import INF, MultiIndexPolicy, dual_index_policy, single_source_policy
from .simulator import Streams, make_streams, mean_cost


@dataclass
class SeedSplit:
    train: list[int]
    val: list[int]
    test: list[int]

    @classmethod
    def default(cls, n_train=8, n_val=8, n_test=20, base=0):
        return cls(list(range(base + 1000, base + 1000 + n_train)),
                   list(range(base + 2000, base + 2000 + n_val)),
                   list(range(base + 3000, base + 3000 + n_test)))


class Objective:
    """Cached CRN objective f(x) = average cost of policy(x) on fixed streams."""

    def __init__(self, inst: Instance, make_policy: Callable, streams: Streams, warmup: int):
        self.inst, self.make_policy, self.streams, self.warmup = inst, make_policy, streams, warmup
        self.cache: dict[tuple, float] = {}
        self.evals = 0

    def __call__(self, x) -> float:
        key = tuple(int(v) for v in np.rint(x))
        if key not in self.cache:
            self.cache[key] = mean_cost(self.inst, self.make_policy(np.array(key)), self.streams,
                                        self.warmup)
            self.evals += 1
        return self.cache[key]


def coordinate_search(f: Objective, x0, lo, hi, steps=(8, 4, 2, 1), max_rounds=30,
                      verbose=False) -> tuple[np.ndarray, float]:
    """Integer pattern search: try +/- step on each coordinate, accept improvements."""
    x = np.clip(np.rint(x0), lo, hi).astype(int)
    fx = f(x)
    for step in steps:
        for _ in range(max_rounds):
            improved = False
            for k in range(len(x)):
                for s in (+step, -step):
                    y = x.copy()
                    y[k] = np.clip(y[k] + s, lo[k], hi[k])
                    if y[k] == x[k]:
                        continue
                    fy = f(y)
                    if fy < fx - 1e-12:
                        x, fx, improved = y, fy, True
            if not improved:
                break
        if verbose:
            print(f"  step {step}: f={fx:.4f} x={x.tolist()}")
    return x, fx


def cem_search(f: Objective, mean, std, lo, hi, iters=12, pop=24, elite=6,
               rng=None) -> tuple[np.ndarray, float]:
    """Cross-entropy method over integer vectors (global stage before
    coordinate refinement)."""
    rng = rng or np.random.default_rng(0)
    mean, std = np.asarray(mean, float), np.asarray(std, float)
    best_x, best_f = np.rint(mean).astype(int), f(mean)
    for _ in range(iters):
        X = np.clip(np.rint(rng.normal(mean, std, size=(pop, len(mean)))), lo, hi)
        F = np.array([f(x) for x in X])
        idx = np.argsort(F)[:elite]
        if F[idx[0]] < best_f:
            best_x, best_f = X[idx[0]].astype(int), F[idx[0]]
        mean = X[idx].mean(0)
        std = np.maximum(X[idx].std(0), 0.5)
    return best_x, best_f


# ------------------------------------------------------------- recipes
def base_stock_guess(inst: Instance, L: int, z: float = 2.33) -> int:
    mu = inst.mean_demand
    var = float(np.dot(np.asarray(inst.demand_values) ** 2, inst.demand_probs) - mu ** 2)
    return int(np.ceil(mu * (L + 1) + z * np.sqrt(var * (L + 1))))


def tune_single_source(inst: Instance, i: int, streams: Streams, warmup: int):
    """Best (s, S) policy using only supplier i (S - s = 0 gives base-stock)."""
    L = inst.suppliers[i].lead_time
    S0 = base_stock_guess(inst, L)
    hi_S = S0 + 25

    def mk(x):
        s, S = int(x[0]), int(x[1])
        return single_source_policy(inst, i, min(s, S), S)

    f = Objective(inst, mk, streams, warmup)
    x0 = np.array([S0, S0 + (inst.suppliers[i].min_order if inst.suppliers[i].fixed_cost else 0)])
    x, fx = coordinate_search(f, x0, np.array([-10, 0]), np.array([hi_S, hi_S]))
    return mk(x), fx, f.evals


def tune_dual_index(inst: Instance, streams: Streams, warmup: int, capped: bool):
    """DI (cap = inf) or CDI for N = 2, by exhaustive-then-local search."""
    fast, slow = inst.decision_order()
    Lr = inst.suppliers[slow].lead_time
    Sr0 = base_stock_guess(inst, Lr)
    mu = inst.mean_demand

    def mk(x):
        return dual_index_policy(inst, int(x[0]), int(x[1]), int(x[2]) if capped else INF)

    f = Objective(inst, mk, streams, warmup)
    # coarse grid over (S_e, S_r) (and cap) then local refinement
    best = None
    caps = range(max(int(mu) - 1, 1), int(np.ceil(mu)) + 4) if capped else [0]
    for Se in range(0, 8):
        for Sr in range(max(Se, Sr0 - 6), Sr0 + 6):
            for c in caps:
                v = f(np.array([Se, Sr, c]))
                if best is None or v < best[0]:
                    best = (v, np.array([Se, Sr, c]))
    lo = np.array([-5, -5, 1]); hi = np.array([40, 60, 40 if capped else 0])
    x, fx = coordinate_search(f, best[1], lo, hi, steps=(2, 1))
    return mk(x), fx, f.evals


def tune_gmi(inst: Instance, streams: Streams, warmup: int, rng=None,
             use_triggers: bool = True, verbose=False, subsets_up_to: int = 4):
    """Generalised multi-index heuristic for any N, tuned by CEM + coordinate
    search. Parameters per supplier: order-up-to S_i, cap_i, trigger r_i.
    Supplier subsets are explored implicitly (a high trigger or low S turns a
    supplier off) and explicitly via all non-empty subsets for N <= ``subsets_up_to``."""
    n = inst.n
    L = inst.lead_times
    capacity = inst.capacities(big=int(10 * inst.max_demand))
    S0 = np.array([base_stock_guess(inst, int(l), 1.5) for l in L])
    cap0 = np.minimum(capacity, max(int(np.ceil(inst.mean_demand)) + 2, 1))
    trig0 = np.maximum(inst.moqs, 1)
    lo = np.concatenate([np.full(n, -10), np.ones(n), np.ones(n)])
    hi = np.concatenate([S0 + 30, capacity, np.full(n, 30 if use_triggers else 1)])

    best = (np.inf, None, None)
    subsets = [np.array(m, bool) for m in itertools.product([0, 1], repeat=n) if any(m)] \
        if n <= subsets_up_to else [np.ones(n, bool)]
    total_evals = 0
    for act in subsets:
        def mk(x, act=act):
            return MultiIndexPolicy.from_vector(x, n, act, "GMI")
        f = Objective(inst, mk, streams, warmup)
        x0 = np.concatenate([S0, cap0, trig0])
        std = np.concatenate([np.full(n, 4.0), np.full(n, 2.0), np.full(n, 3.0 if use_triggers else 0.01)])
        x, fx = cem_search(f, x0, std, lo, hi, iters=8 if n <= 3 else 12, pop=20, elite=5,
                           rng=rng or np.random.default_rng(0))
        x, fx = coordinate_search(f, x, lo, hi, steps=(4, 2, 1))
        total_evals += f.evals
        if verbose:
            print(f"  subset {act.astype(int)}: {fx:.3f} ({f.evals} evals)")
        if fx < best[0]:
            best = (fx, x, act)
    pol = MultiIndexPolicy.from_vector(best[1], n, best[2], "GMI (tuned)")
    return pol, best[0], total_evals
