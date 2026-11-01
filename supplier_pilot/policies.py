"""Index-type heuristic policies.

Generalised multi-index policy (GMI)
------------------------------------
Suppliers are processed in :meth:`Instance.decision_order` (ascending lead
time; ties by price, fixed cost, index). For supplier i with lead time L_i:

    IP_i = I + (outstanding orders from any supplier due within t+1..t+L_i)
             + (orders already placed this period with lead time <= L_i)
    gap_i = S_i - IP_i
    order iff supplier i is up, active, and gap_i >= r_i      (trigger r_i >= 1)
    q_i = min(gap_i, cap_i), rounded up to MOQ_i, at most capacity_i

Special cases (see tests/test_policies.py):
  * one active supplier with trigger r = S - s + 1  ->  (s, S) policy;
    with r = 1 -> base-stock policy;
  * N = 2, l_e < l_r, r = 1, cap_e = inf, cap_r = inf -> dual-index (DI);
  * same with finite cap_r -> capped dual-index (CDI, Sun & Van Mieghem 2019).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .instance import Instance
from .simulator import State

INF = 10 ** 9


def inventory_position(inst: Instance, st: State, window: int,
                       placed: np.ndarray, decided: list[int]) -> np.ndarray:
    """Net inventory plus everything that will have arrived by t+window."""
    L = inst.lead_times
    ip = st.I.astype(np.int64).copy()
    for j, Lj in enumerate(L):
        if Lj >= 2:
            k = min(window, Lj - 1)
            if k > 0:
                ip += st.pipes[j][:, :k].sum(axis=1)
    for j in decided:
        if L[j] <= window:
            ip += placed[:, j]
    return ip


def multi_index_orders(inst: Instance, st: State, S, cap, trig, active=None) -> np.ndarray:
    """Vectorised GMI orders. S, cap, trig: arrays broadcastable to (E, N)."""
    E = st.I.shape[0]
    N = inst.n
    S = np.broadcast_to(np.rint(S).astype(np.int64), (E, N))
    cap = np.broadcast_to(np.rint(cap).astype(np.int64), (E, N))
    trig = np.broadcast_to(np.maximum(np.rint(trig).astype(np.int64), 1), (E, N))
    act = np.ones((E, N), bool) if active is None else np.broadcast_to(np.asarray(active, bool), (E, N))
    capacity = inst.capacities()
    moq = inst.moqs
    q = np.zeros((E, N), dtype=np.int64)
    decided: list[int] = []
    for i in inst.decision_order():
        ip = inventory_position(inst, st, int(inst.lead_times[i]), q, decided)
        gap = S[:, i] - ip
        go = act[:, i] & (st.z[:, i] == 1) & (gap >= trig[:, i])
        qi = np.minimum(gap, np.maximum(cap[:, i], 1))
        qi = np.maximum(qi, moq[i])
        qi = np.minimum(qi, capacity[i])
        q[:, i] = np.where(go, qi, 0)
        decided.append(i)
    return q


@dataclass
class MultiIndexPolicy:
    """Static GMI policy. ``params`` = concatenation [S (N), cap (N), trig (N)]."""
    S: np.ndarray
    cap: np.ndarray
    trig: np.ndarray
    active: np.ndarray | None = None
    name: str = "GMI"

    @classmethod
    def from_vector(cls, x, n, active=None, name="GMI"):
        x = np.asarray(x, dtype=float)
        return cls(x[:n], x[n:2 * n], x[2 * n:3 * n], active, name)

    def vector(self) -> np.ndarray:
        return np.concatenate([self.S, self.cap, self.trig]).astype(float)

    def act(self, inst: Instance, st: State, t: int) -> np.ndarray:
        return multi_index_orders(inst, st, self.S, self.cap, self.trig, self.active)

    def describe(self, inst: Instance) -> list[dict]:
        out = []
        for i, s in enumerate(inst.suppliers):
            on = True if self.active is None else bool(self.active[i])
            out.append({"supplier": s.name, "active": on, "S": int(round(self.S[i])),
                        "cap": int(round(self.cap[i])) if self.cap[i] < INF / 2 else "inf",
                        "trigger": int(round(self.trig[i]))})
        return out


def single_source_policy(inst: Instance, i: int, s: int, S: int, name=None) -> MultiIndexPolicy:
    """(s, S) policy that only uses supplier i (orders up to S when IP < s)."""
    n = inst.n
    Sv = np.zeros(n); Sv[i] = S
    cap = np.full(n, INF, dtype=float)
    trig = np.ones(n); trig[i] = max(S - s + 1, 1)
    act = np.zeros(n, bool); act[i] = True
    return MultiIndexPolicy(Sv, cap, trig, act, name or f"single[{inst.suppliers[i].name}](s={s},S={S})")


def dual_index_policy(inst: Instance, S_e: int, S_r: int, cap_r: float = INF, name=None):
    """DI / CDI for a two-supplier instance (supplier order by lead time)."""
    assert inst.n == 2
    fast, slow = inst.decision_order()
    S = np.zeros(2); S[fast], S[slow] = S_e, S_r
    cap = np.full(2, INF, dtype=float); cap[slow] = cap_r
    nm = name or ("CDI" if cap_r < INF / 2 else "DI")
    return MultiIndexPolicy(S, cap, np.ones(2), None, nm)


@dataclass
class ConstantOrderPolicy:
    """Orders a fixed vector every period (used in tests and as a sanity check)."""
    q: np.ndarray
    name: str = "constant"

    def act(self, inst, st, t):
        return np.broadcast_to(np.asarray(self.q, dtype=np.int64), (st.I.shape[0], inst.n)).copy()


class SubsetPolicy:
    """Run a policy of a sub-instance (a subset of the suppliers) in the full
    instance; the other suppliers are never used. With the exact DP of the
    A-B-C core this gives the optimal cost achievable with three suppliers."""

    def __init__(self, inner, sub_inst, keep, name: str | None = None):
        self.inner, self.sub, self.keep = inner, sub_inst, list(keep)
        self.name = name or f"{getattr(inner, 'name', 'policy')} ({len(self.keep)} suppliers)"

    def reset(self, inst, E):
        if hasattr(self.inner, "reset"):
            self.inner.reset(self.sub, E)

    def act(self, inst, st, t):
        sub = State(st.I, [st.pipes[i] for i in self.keep], st.z[:, self.keep])
        a = np.asarray(self.inner.act(self.sub, sub, t))
        out = np.zeros((st.I.shape[0], inst.n), dtype=np.int64)
        out[:, self.keep] = a
        return out
