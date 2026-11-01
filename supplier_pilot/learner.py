"""A simple structured learning policy for the pilot.

A small MLP maps the observed disruption status (and optionally coarse
inventory features) to the parameters of the generalised multi-index policy:
per-supplier order-up-to level S_i, cap_i and trigger r_i. Orders then come
from :func:`multi_index_orders`, so every action is feasible by construction
and interpretable. The network's output layer is initialised at zero, so
training starts exactly at the tuned static heuristic (warm start).

Training: antithetic evolution strategies with common random numbers on the
expected average cost (the only training objective). Model selection: the
best iterate on a separate validation seed set; the static heuristic is kept
as a fallback and chosen instead if it is better on validation. No claim is
made that the learner cannot be worse than the heuristic.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .instance import Instance
from .policies import MultiIndexPolicy, multi_index_orders
from .simulator import State, Streams, make_streams, mean_cost


@dataclass
class StructuredPolicy:
    base: MultiIndexPolicy
    n_inputs: int
    hidden: int = 16
    theta: np.ndarray | None = None
    scale: tuple[float, float, float] = (2.0, 2.0, 2.0)
    use_inventory_features: bool = False
    anchored: bool = True
    name: str = "Structured-ES"

    def __post_init__(self):
        if self.theta is None:
            self.theta = self.init_theta(np.random.default_rng(0))

    # parameter vector layout: W1 (in, h), b1 (h), W2 (h, 3N), b2 (3N)
    def shapes(self, n):
        return [(self.n_inputs, self.hidden), (self.hidden,), (self.hidden, 3 * n), (3 * n,)]

    def unpack(self, theta, n):
        out, k = [], 0
        for s in self.shapes(n):
            m = int(np.prod(s))
            out.append(theta[k:k + m].reshape(s)); k += m
        return out

    def init_theta(self, rng, n=None):
        n = n if n is not None else len(self.base.S)
        W1 = rng.normal(0, 1 / np.sqrt(self.n_inputs), (self.n_inputs, self.hidden))
        parts = [W1.ravel(), np.zeros(self.hidden), np.zeros(self.hidden * 3 * n), np.zeros(3 * n)]
        return np.concatenate(parts)

    def features(self, inst: Instance, st: State) -> np.ndarray:
        x = 2.0 * st.z - 1.0
        if self.use_inventory_features:
            mu = max(inst.mean_demand, 1.0)
            pipe = sum(p.sum(1) for p in st.pipes) if st.pipes else 0
            x = np.column_stack([x, np.tanh(st.I / (5 * mu)), np.tanh(pipe / (5 * mu))])
        return x

    def params(self, inst: Instance, st: State, theta=None):
        n = inst.n
        W1, b1, W2, b2 = self.unpack(self.theta if theta is None else theta, n)
        # errstate: NumPy 2.x on Apple Silicon emits spurious matmul warnings
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            hdn = np.tanh(self.features(inst, st) @ W1 + b1)
            o = hdn @ W2 + b2                                   # (E, 3N)
            if self.anchored and not self.use_inventory_features:
                # residual anchored at the all-up status: when every supplier is
                # up the policy is exactly the tuned heuristic; the network only
                # adapts parameters in disrupted states
                h1 = np.tanh(np.ones(self.n_inputs) @ W1 + b1)
                o = o - (h1 @ W2 + b2)
        if not np.all(np.isfinite(o)):
            raise FloatingPointError("non-finite policy parameters")
        sS, sc, st_ = self.scale
        S = self.base.S + sS * o[:, :n]
        cap = np.maximum(self.base.cap + sc * o[:, n:2 * n], 1)
        trig = np.maximum(self.base.trig + st_ * o[:, 2 * n:], 1)
        return S, cap, trig

    def act(self, inst: Instance, st: State, t: int) -> np.ndarray:
        S, cap, trig = self.params(inst, st)
        return multi_index_orders(inst, st, S, cap, trig, self.base.active)

    def with_theta(self, theta, name=None):
        return StructuredPolicy(self.base, self.n_inputs, self.hidden, theta.copy(), self.scale,
                                self.use_inventory_features, self.anchored, name or self.name)


@dataclass
class StatusTablePolicy:
    """GMI parameters looked up per joint disruption status (2^N rows).
    A direct, non-neural check of how much disruption-dependent parameters
    can gain; used to tell 'little headroom' apart from 'weak optimiser'."""
    base: MultiIndexPolicy
    table: np.ndarray | None = None     # (2^N, 3N)
    name: str = "Status-table GMI"

    def __post_init__(self):
        if self.table is None:
            n = len(self.base.S)
            self.table = np.tile(self.base.vector(), (1 << n, 1))

    def act(self, inst: Instance, st: State, t: int) -> np.ndarray:
        n = inst.n
        p = self.table[inst.status_index(st.z)]
        return multi_index_orders(inst, st, p[:, :n], p[:, n:2 * n], p[:, 2 * n:], self.base.active)


def tune_status_table(inst: Instance, base: MultiIndexPolicy, train_streams: Streams, warmup: int,
                      min_share: float = 0.005, verbose: bool = False):
    """Coordinate search over the parameters of every status that occurs in at
    least ``min_share`` of periods under the base policy."""
    from .simulator import simulate
    from .tuning import Objective, coordinate_search
    n = inst.n
    res = simulate(inst, base, streams=train_streams, warmup=warmup)
    zid = inst.status_index(res.z.reshape(-1, n))
    share = np.bincount(zid, minlength=1 << n) / len(zid)
    order = [z for z in np.argsort(-share) if share[z] >= min_share]
    pol = StatusTablePolicy(base)
    capacity = inst.capacities(big=int(10 * inst.max_demand))
    lo = np.concatenate([np.full(n, -10), np.ones(n), np.ones(n)])
    hi = np.concatenate([base.S + 30, capacity, np.full(n, 30)])
    evals = 0
    for z in order:
        def mk(x, z=z):
            tab = pol.table.copy(); tab[z] = x
            return StatusTablePolicy(base, tab)
        f = Objective(inst, mk, train_streams, warmup)
        x, fx = coordinate_search(f, pol.table[z], lo, hi, steps=(4, 2, 1), max_rounds=10)
        pol.table[z] = x
        evals += f.evals
        if verbose:
            print(f"   status {z:0{n}b} (share {share[z]:.3f}): f={fx:.3f}")
    return pol, evals, {int(z): float(share[z]) for z in order}


def total_inventory_position(st: State) -> np.ndarray:
    """Net inventory plus everything in the pipeline of every supplier."""
    ip = st.I.astype(float).copy()
    for p in st.pipes:
        if p.shape[1]:
            ip += p.sum(axis=1)
    return ip


@dataclass
class StateTablePolicy:
    """Inventory-aware status table: GMI parameters per (joint disruption
    status, inventory-position band). Bands cut the total inventory position
    (net inventory + all pipeline) at ``thresholds``; with B bands there are
    B - 1 thresholds. A non-neural, directly tunable check of whether
    inventory/pipeline-dependent parameters add value beyond the status table."""
    base: MultiIndexPolicy
    table: np.ndarray            # (2^N, B, 3N)
    thresholds: np.ndarray       # (B - 1,) ascending
    name: str = "Status x inventory table"

    @classmethod
    def from_status_table(cls, st_pol: "StatusTablePolicy", thresholds):
        B = len(thresholds) + 1
        tab = np.repeat(st_pol.table[:, None, :], B, axis=1).astype(float)
        return cls(st_pol.base, tab, np.asarray(thresholds, float))

    def band(self, st: State) -> np.ndarray:
        return np.searchsorted(self.thresholds, total_inventory_position(st), side="right")

    def act(self, inst: Instance, st: State, t: int) -> np.ndarray:
        n = inst.n
        p = self.table[inst.status_index(st.z), self.band(st)]
        return multi_index_orders(inst, st, p[:, :n], p[:, n:2 * n], p[:, 2 * n:], self.base.active)


class _Recorder:
    """Wraps a policy and records (status index, total inventory position)."""

    def __init__(self, pol):
        self.pol, self.name, self.z, self.ip = pol, getattr(pol, "name", "pol"), [], []

    def reset(self, inst, E):
        if hasattr(self.pol, "reset"):
            self.pol.reset(inst, E)

    def act(self, inst, st, t):
        self.z.append(inst.status_index(st.z)); self.ip.append(total_inventory_position(st))
        return self.pol.act(inst, st, t)


def tune_state_table(inst: Instance, st_pol: "StatusTablePolicy", train_streams: Streams, warmup: int,
                     bands: int = 3, min_share: float = 0.003, steps=(2, 1), max_rounds: int = 6,
                     verbose: bool = False):
    """Start from a tuned status table, split each status row into ``bands``
    inventory-position bands (cut at quantiles observed under the status
    table) and tune every sufficiently frequent (status, band) row by
    coordinate search with common random numbers."""
    from .simulator import simulate
    from .tuning import Objective, coordinate_search
    n = inst.n
    rec = _Recorder(st_pol)
    simulate(inst, rec, streams=train_streams, warmup=0)
    z = np.concatenate(rec.z[warmup:]); ip = np.concatenate(rec.ip[warmup:])
    thr = np.unique(np.quantile(ip, np.arange(1, bands) / bands))
    pol = StateTablePolicy.from_status_table(st_pol, thr)
    B = len(thr) + 1
    band = np.searchsorted(thr, ip, side="right")
    cell = z * B + band
    share = np.bincount(cell, minlength=(1 << n) * B) / len(cell)
    order = [c for c in np.argsort(-share) if share[c] >= min_share]
    capacity = inst.capacities(big=int(10 * inst.max_demand))
    lo = np.concatenate([np.full(n, -10), np.ones(n), np.ones(n)])
    hi = np.concatenate([st_pol.base.S + 30, capacity, np.full(n, 30)])
    evals = 0
    for c in order:
        zi, bi = divmod(int(c), B)

        def mk(x, zi=zi, bi=bi):
            tab = pol.table.copy(); tab[zi, bi] = x
            return StateTablePolicy(pol.base, tab, pol.thresholds)
        f = Objective(inst, mk, train_streams, warmup)
        x, fx = coordinate_search(f, pol.table[zi, bi], lo, hi, steps=steps, max_rounds=max_rounds)
        pol.table[zi, bi] = x
        evals += f.evals
        if verbose:
            print(f"   status {zi:0{n}b} band {bi} (share {share[c]:.3f}): f={fx:.3f}")
    return pol, evals, {"thresholds": thr.tolist(), "cells": {f"{int(c) // B:0{n}b}/b{int(c) % B}": float(share[c])
                                                             for c in order}}


@dataclass
class ESResult:
    policy: StructuredPolicy
    val_cost: float
    base_val_cost: float
    chose_fallback: bool
    history: list = field(default_factory=list)
    seconds: float = 0.0


def train_es(inst: Instance, base: MultiIndexPolicy, train_seeds, val_seeds, T=1500, warmup=150,
             iters=60, pairs=8, sigma=0.1, lr=0.02, seed=0, hidden=16,
             use_inventory_features=False, anchored=True, batch=8, val_every=5,
             verbose=False) -> ESResult:
    rng = np.random.default_rng(seed)
    n_in = inst.n + (2 if use_inventory_features else 0)
    pol = StructuredPolicy(base, n_in, hidden, use_inventory_features=use_inventory_features,
                           anchored=anchored)
    pol.theta = pol.init_theta(rng, inst.n)
    theta = pol.theta.copy()
    m = np.zeros_like(theta); v = np.zeros_like(theta)
    b1, b2, eps_ = 0.9, 0.999, 1e-8
    val_streams = make_streams(inst, val_seeds, T)
    base_val = mean_cost(inst, base, val_streams, warmup)
    best_theta, best_val = theta.copy(), mean_cost(inst, pol.with_theta(theta), val_streams, warmup)
    hist = [(0, best_val)]
    t0 = time.perf_counter()
    train_seeds = list(train_seeds)
    for it in range(1, iters + 1):
        sd = rng.choice(train_seeds, size=min(batch, len(train_seeds)), replace=False)
        streams = make_streams(inst, sd.tolist(), T)       # CRN within an iteration
        eps = rng.normal(size=(pairs, theta.size))
        fp = np.array([mean_cost(inst, pol.with_theta(theta + sigma * e), streams, warmup) for e in eps])
        fm = np.array([mean_cost(inst, pol.with_theta(theta - sigma * e), streams, warmup) for e in eps])
        # centred-rank fitness shaping (robust to the piecewise-constant
        # objective created by integer rounding)
        allf = np.concatenate([fp, fm])
        ranks = np.empty(2 * pairs); ranks[np.argsort(allf)] = np.arange(2 * pairs)
        ranks = ranks / (2 * pairs - 1) - 0.5
        grad = ((ranks[:pairs] - ranks[pairs:])[:, None] * eps).sum(0) / (2 * pairs * sigma)
        m = b1 * m + (1 - b1) * grad
        v = b2 * v + (1 - b2) * grad ** 2
        theta = theta - lr * (m / (1 - b1 ** it)) / (np.sqrt(v / (1 - b2 ** it)) + eps_)
        if it % val_every == 0 or it == iters:
            vc = mean_cost(inst, pol.with_theta(theta), val_streams, warmup)
            hist.append((it, vc))
            if vc < best_val:
                best_val, best_theta = vc, theta.copy()
            if verbose:
                print(f"    ES it={it:3d} val={vc:.3f} best={best_val:.3f} base={base_val:.3f}")
    learned = pol.with_theta(best_theta, "Structured-ES")
    chose_fallback = base_val <= best_val
    return ESResult(learned, best_val, base_val, chose_fallback, hist, time.perf_counter() - t0)
