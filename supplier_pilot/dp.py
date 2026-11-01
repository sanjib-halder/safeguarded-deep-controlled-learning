"""Exact average-cost dynamic programming for small instances.

Relative value iteration on the same model as the simulator. The state is
(joint supplier status z, net inventory I, pipeline vectors). Inventory is
truncated to [I_min, I_max]; :func:`check_boundaries` reports how often the
optimal policy actually reaches the truncation so you can widen it.

At every iteration the span of (TV - V) brackets the optimal average cost:
    min(TV - V) <= g* <= max(TV - V)
so the reported bounds are rigorous for the truncated model.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from itertools import product

import numpy as np

from .instance import Instance
from .simulator import State


@dataclass
class DPModel:
    inst: Instance
    I_min: int
    I_max: int
    q_sets: list[list[int]]
    actions: np.ndarray        # (A, N)
    radix: list[int]           # rest-state mixed radix: [nI, pipe entries...]
    pipe_slices: list[slice]   # columns of the pipeline part per supplier
    R: int                     # number of rest states
    Z: int
    next_rest: np.ndarray      # (A, Dn, R) int32
    cost: np.ndarray           # (A, R) expected one-period cost
    feasible: np.ndarray       # (Z, A) bool
    P: np.ndarray              # (Z, Z)
    z_rows: np.ndarray         # (2^N,) joint status index -> row in V (-1 = unreachable)


@dataclass
class DPSolution:
    model: DPModel
    g: float
    g_lower: float
    g_upper: float
    V: np.ndarray              # (Z, R) relative values
    policy: np.ndarray         # (Z, R) action index
    iterations: int
    seconds: float
    converged: bool

    def summary(self) -> dict:
        m = self.model
        return {"instance": m.inst.name, "g": self.g, "g_lower": self.g_lower,
                "g_upper": self.g_upper, "iterations": self.iterations,
                "seconds": round(self.seconds, 1), "converged": self.converged,
                "states": m.Z * m.R, "actions": len(m.actions),
                "I_range": [m.I_min, m.I_max]}


def build_model(inst: Instance, I_min: int, I_max: int, verbose=False) -> DPModel:
    caps = []
    for s in inst.suppliers:
        if s.capacity is None:
            raise ValueError("DP needs a finite capacity for every supplier")
        caps.append(s.capacity)
    q_sets = [[0] + list(range(s.min_order, s.capacity + 1)) for s in inst.suppliers]
    actions = np.array(list(product(*q_sets)), dtype=np.int64)
    L = inst.lead_times
    nI = I_max - I_min + 1
    radix = [nI]
    pipe_slices = []
    col = 1
    for i, Li in enumerate(L):
        w = max(Li - 1, 0)
        pipe_slices.append(slice(col, col + w))
        radix += [caps[i] + 1] * w
        col += w
    R = int(np.prod(radix))
    # enumerate rest states: digits[:, 0] = I - I_min, then pipeline entries
    digits = np.stack(np.unravel_index(np.arange(R), radix), axis=1).astype(np.int64)
    inv = digits[:, 0] + I_min
    mult = np.array([int(np.prod(radix[k + 1:])) for k in range(len(radix))], dtype=np.int64)

    dv = np.asarray(inst.demand_values, dtype=np.int64)
    dp = np.asarray(inst.demand_probs)
    A, Dn = len(actions), len(dv)
    next_rest = np.empty((A, Dn, R), dtype=np.int32)
    cost = np.empty((A, R))
    h, b = inst.holding, inst.backorder
    for a, q in enumerate(actions):
        immediate = q[L == 0].sum()
        arrivals = np.zeros(R, dtype=np.int64)
        new_digits = digits.copy()
        for i, Li in enumerate(L):
            sl = pipe_slices[i]
            if Li == 1:
                arrivals += q[i]
            elif Li >= 2:
                cols = digits[:, sl]
                arrivals += cols[:, 0]
                shifted = np.concatenate([cols[:, 1:], np.full((R, 1), q[i])], axis=1)
                new_digits[:, sl] = shifted
        oc = float((q * inst.prices).sum() + ((q > 0) * inst.fixed_costs).sum())
        c = np.zeros(R)
        for k, d in enumerate(dv):
            I_end = inv + immediate - d
            c += dp[k] * (h * np.maximum(I_end, 0) + b * np.maximum(-I_end, 0))
            I_next = np.clip(I_end + arrivals, I_min, I_max)
            nd = new_digits.copy()
            nd[:, 0] = I_next - I_min
            next_rest[a, k] = (nd * mult).sum(axis=1)
        cost[a] = c + oc
    Zs = inst.all_statuses()
    z_rows = np.arange(len(Zs))
    P = inst.status_transition_matrix()
    if not inst.has_disruptions:          # only the all-up status is reachable
        z_rows = np.full(len(Zs), -1)
        z_rows[-1] = 0
        Zs = Zs[-1:]
        P = np.ones((1, 1))
    feasible = np.all((actions[None, :, :] == 0) | (Zs[:, None, :] == 1), axis=2)
    if verbose:
        print(f"[dp] {inst.name}: Z={len(Zs)} R={R} A={A} -> {len(Zs) * R} states")
    return DPModel(inst, I_min, I_max, q_sets, actions, radix, pipe_slices, R, len(Zs),
                   next_rest, cost, feasible, P, z_rows)


def _matmul(P: np.ndarray, V: np.ndarray) -> np.ndarray:
    """P @ V with NumPy floating-point warnings silenced. NumPy 2.x on Apple
    Silicon (Accelerate BLAS) raises spurious divide/overflow warnings in
    matmul; results are unaffected. Real problems are still caught below."""
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        W = P @ V
    if not np.all(np.isfinite(W)):
        raise FloatingPointError("non-finite values in DP iteration")
    return W


def _bellman(model: DPModel, V: np.ndarray, dp: np.ndarray):
    """Full Bellman operator: returns (TV, greedy policy)."""
    Z, R = V.shape
    W = _matmul(model.P, V)
    best = np.full((Z, R), np.inf)
    pol = np.zeros((Z, R), dtype=np.int32)
    for a in range(len(model.actions)):
        ev = np.tensordot(dp, W[:, model.next_rest[a]], axes=([0], [1]))  # (Z, R)
        qa = model.cost[a][None, :] + ev
        qa[~model.feasible[:, a]] = np.inf
        better = qa < best
        best = np.where(better, qa, best)
        pol = np.where(better, a, pol)
    return best, pol


def solve(model: DPModel, tol: float = 1e-6, max_iter: int = 5000, damping: float = 0.9,
          eval_sweeps: int = 40, verbose: bool = False,
          V0: np.ndarray | None = None) -> DPSolution:
    """Modified policy iteration for the average-cost criterion.

    Each outer iteration applies one full Bellman sweep (which yields the
    rigorous bounds min/max(TV - V) on g* and the greedy policy), followed by
    ``eval_sweeps`` cheap relative-value sweeps under that fixed policy.
    ``damping`` < 1 is the aperiodicity transformation (V <- d*TV + (1-d)*V),
    which leaves the optimal policy unchanged and guarantees convergence.
    """
    t0 = time.perf_counter()
    Z, R = model.Z, model.R
    dp = np.asarray(model.inst.demand_probs)
    V = np.zeros((Z, R)) if V0 is None else V0.copy()
    ref = (Z - 1, 0)
    lo = hi = np.nan
    converged = False
    ridx = np.arange(R)[None, :]
    zidx = np.arange(Z)[:, None, None]
    it = 0
    pol = np.zeros((Z, R), dtype=np.int32)
    for it in range(1, max_iter + 1):
        TV, pol = _bellman(model, V, dp)
        diff = TV - V
        lo, hi = float(diff.min()), float(diff.max())
        if verbose and (it % 10 == 0 or it == 1):
            print(f"[dp] outer it={it:4d}  g in [{lo:.6f}, {hi:.6f}]")
        if hi - lo <= tol * max(1.0, abs(hi)):
            converged = True
            break
        V = damping * TV + (1 - damping) * V
        V -= V[ref]
        if eval_sweeps > 0:
            NR = np.transpose(model.next_rest[pol, :, ridx], (0, 1, 2))  # (Z, R, Dn)
            C = model.cost[pol, ridx]                                     # (Z, R)
            for _ in range(eval_sweeps):
                W = _matmul(model.P, V)
                TVp = C + (W[zidx, NR] * dp).sum(-1)
                V = damping * TVp + (1 - damping) * V
                V -= V[ref]
    g = 0.5 * (lo + hi)
    return DPSolution(model, g, lo, hi, V, pol, it, time.perf_counter() - t0, converged)


def solve_instance(inst: Instance, I_min: int, I_max: int, **kw) -> DPSolution:
    verbose = kw.get("verbose", False)
    return solve(build_model(inst, I_min, I_max, verbose=verbose), **kw)


# ---------------------------------------------------------------- policy
class DPPolicy:
    """Looks up the optimal action; states outside the truncation are clipped
    (and counted in ``boundary_hits``)."""

    def __init__(self, sol: DPSolution, name: str = "DP-optimal"):
        self.sol = sol
        self.name = name
        m = sol.model
        self.mult = np.array([int(np.prod(m.radix[k + 1:])) for k in range(len(m.radix))],
                             dtype=np.int64)
        self.boundary_hits = 0
        self.decisions = 0

    def reset(self, inst, E):
        self.boundary_hits = 0
        self.decisions = 0

    def act(self, inst: Instance, st: State, t: int) -> np.ndarray:
        m = self.sol.model
        I = st.I
        out = (I < m.I_min) | (I > m.I_max)
        self.boundary_hits += int(out.sum())
        self.decisions += len(I)
        idx = (np.clip(I, m.I_min, m.I_max) - m.I_min) * self.mult[0]
        for i, sl in enumerate(m.pipe_slices):
            w = sl.stop - sl.start
            for k in range(w):
                idx = idx + st.pipes[i][:, k] * self.mult[sl.start + k]
        zid = m.z_rows[inst.status_index(st.z)]
        a = self.sol.policy[zid, idx]
        return m.actions[a]


def check_boundaries(sol: DPSolution, inst: Instance, T=20000, seeds=range(4)) -> dict:
    """Simulate the DP policy (no truncation in the simulator) and compare
    the simulated average cost with g*. Large boundary-hit shares or a cost
    mismatch mean the inventory truncation is too tight."""
    from .simulator import simulate
    pol = DPPolicy(sol)
    res = simulate(inst, pol, seeds=list(seeds), T=T, warmup=500)
    used = res.orders.max(axis=(0, 1))
    caps = inst.capacities()
    return {"sim_avg_cost": float(res.episode_cost.mean()),
            "g_star": sol.g,
            "rel_diff": float(res.episode_cost.mean() / sol.g - 1),
            "boundary_hit_share": pol.boundary_hits / max(pol.decisions, 1),
            "max_order_used": used.tolist(),
            "capacity_binding": [bool(u >= c) for u, c in zip(used, caps)]}
