"""Vectorised simulator with common random numbers (CRN).

All policies are evaluated on identical demand and disruption streams, so
paired differences between policies have low variance. ``E`` episodes are
simulated in parallel as NumPy arrays.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np

from .instance import Instance


# --------------------------------------------------------------------- state
@dataclass
class State:
    I: np.ndarray            # (E,) net inventory after receipts
    pipes: list[np.ndarray]  # per supplier (E, max(L_i-1, 0)); col k due in t+1+k
    z: np.ndarray            # (E, N) int8, 1 = up

    def copy(self) -> "State":
        return State(self.I.copy(), [p.copy() for p in self.pipes], self.z.copy())


def initial_state(inst: Instance, E: int, I0: int = 0) -> State:
    pipes = [np.zeros((E, max(s.lead_time - 1, 0)), dtype=np.int64) for s in inst.suppliers]
    return State(np.full(E, I0, dtype=np.int64), pipes, np.ones((E, inst.n), dtype=np.int8))


class Policy(Protocol):
    name: str

    def act(self, inst: Instance, st: State, t: int) -> np.ndarray:  # (E, N) ints
        ...


# ----------------------------------------------------------------- dynamics
def project_feasible(inst: Instance, q: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Project raw orders onto the feasible set: 0 for suppliers that are
    down, otherwise 0 or [MOQ, capacity] (positive orders below the MOQ are
    rounded up). Only infeasible actions are altered."""
    q = np.rint(np.asarray(q)).astype(np.int64)
    q = np.maximum(q, 0)
    cap = inst.capacities()
    q = np.minimum(q, cap)
    moq = inst.moqs
    q = np.where((q > 0) & (q < moq), moq, q)
    q = np.where(z == 1, q, 0)
    return q


def next_status(inst: Instance, z: np.ndarray, u_ind: np.ndarray,
                u_shock: np.ndarray | None) -> np.ndarray:
    """Status transition driven by uniforms (for CRN). Matches
    :meth:`Instance.status_transition_matrix` exactly."""
    pf, pr = inst.p_fail, inst.p_recover
    up_next = np.where(z == 1, u_ind >= pf, u_ind < pr)
    if inst.regional_shock > 0 and u_shock is not None:
        shocked_region = u_shock < inst.regional_shock          # (E, R)
        shocked = shocked_region[:, inst.regions]               # (E, N)
        up_next = up_next & ~shocked
    return up_next.astype(np.int8)


def step(inst: Instance, st: State, q: np.ndarray, d: np.ndarray,
         u_ind: np.ndarray, u_shock: np.ndarray | None):
    """Advance one period in place. ``q`` must already be feasible.
    Returns a dict of per-episode period quantities."""
    L = inst.lead_times
    immediate = q[:, L == 0].sum(axis=1)
    I_pre = st.I + immediate
    I_end = I_pre - d
    order_cost = (q * inst.prices).sum(1) + ((q > 0) * inst.fixed_costs).sum(1)
    hold = inst.holding * np.maximum(I_end, 0)
    back = inst.backorder * np.maximum(-I_end, 0)
    served = np.minimum(d, np.maximum(I_pre, 0))
    arrivals = np.zeros_like(st.I)
    for i, Li in enumerate(L):
        if Li >= 1:
            tmp = np.concatenate([st.pipes[i], q[:, i:i + 1]], axis=1)
            arrivals += tmp[:, 0]
            st.pipes[i] = tmp[:, 1:]
    st.I = I_end + arrivals
    st.z = next_status(inst, st.z, u_ind, u_shock)
    return {"cost": order_cost + hold + back, "order_cost": order_cost,
            "holding": hold, "backorder": back, "served": served, "demand": d,
            "I_end": I_end}


# ------------------------------------------------------------ random streams
@dataclass
class Streams:
    D: np.ndarray        # (E, T) demand
    U: np.ndarray        # (E, T, N) uniforms for status transitions
    S: np.ndarray | None  # (E, T, R) uniforms for regional shocks


def make_streams(inst: Instance, seeds, T: int) -> Streams:
    """One independent stream per seed (episode). Same seeds -> same
    randomness for every policy (CRN)."""
    seeds = list(seeds)
    E = len(seeds)
    vals = np.asarray(inst.demand_values)
    cdf = np.cumsum(inst.demand_probs)
    D = np.empty((E, T), dtype=np.int64)
    U = np.empty((E, T, inst.n))
    S = np.empty((E, T, inst.n_regions)) if inst.regional_shock > 0 else None
    for e, sd in enumerate(seeds):
        rng = np.random.default_rng(sd)
        D[e] = vals[np.minimum(np.searchsorted(cdf, rng.random(T), side="right"), len(vals) - 1)]
        U[e] = rng.random((T, inst.n))
        if S is not None:
            S[e] = rng.random((T, inst.n_regions))
    return Streams(D, U, S)


# ---------------------------------------------------------------- simulation
@dataclass
class SimResult:
    policy: str
    instance: str
    cost: np.ndarray        # (E, T-warmup) per-period cost
    orders: np.ndarray      # (E, T-warmup, N)
    z: np.ndarray           # (E, T-warmup, N) status when ordering
    served: np.ndarray
    demand: np.ndarray
    components: dict
    projections: int        # number of order entries altered by projection
    seconds: float

    @property
    def episode_cost(self) -> np.ndarray:
        return self.cost.mean(axis=1)


def simulate(inst: Instance, policy, seeds=None, T: int = 3000, warmup: int = 200,
             streams: Streams | None = None, I0: int = 0) -> SimResult:
    import time
    if streams is None:
        streams = make_streams(inst, seeds, T)
    E, T = streams.D.shape
    st = initial_state(inst, E, I0)
    if hasattr(policy, "reset"):
        policy.reset(inst, E)
    keep = T - warmup
    cost = np.empty((E, keep))
    orders = np.empty((E, keep, inst.n), dtype=np.int64)
    zs = np.empty((E, keep, inst.n), dtype=np.int8)
    served = np.empty((E, keep), dtype=np.int64)
    comps = {k: np.zeros(E) for k in ("order_cost", "holding", "backorder")}
    nproj = 0
    t0 = time.perf_counter()
    for t in range(T):
        raw = np.asarray(policy.act(inst, st, t))
        q = project_feasible(inst, raw, st.z)
        nproj += int((q != np.rint(raw)).sum())
        z_now = st.z
        out = step(inst, st, q, streams.D[:, t], streams.U[:, t],
                   None if streams.S is None else streams.S[:, t])
        if t >= warmup:
            k = t - warmup
            cost[:, k] = out["cost"]
            orders[:, k] = q
            zs[:, k] = z_now
            served[:, k] = out["served"]
            for c in comps:
                comps[c] += out[c]
    for c in comps:
        comps[c] /= keep
    return SimResult(getattr(policy, "name", type(policy).__name__), inst.name, cost, orders,
                     zs, served, streams.D[:, warmup:], comps, nproj,
                     time.perf_counter() - t0)


def mean_cost(inst: Instance, policy, streams: Streams, warmup: int) -> float:
    """Fast objective for tuning: average cost per period over all episodes."""
    E, T = streams.D.shape
    st = initial_state(inst, E)
    if hasattr(policy, "reset"):
        policy.reset(inst, E)
    total = 0.0
    for t in range(T):
        q = project_feasible(inst, policy.act(inst, st, t), st.z)
        out = step(inst, st, q, streams.D[:, t], streams.U[:, t],
                   None if streams.S is None else streams.S[:, t])
        if t >= warmup:
            total += out["cost"].sum()
    return total / (E * (T - warmup))
