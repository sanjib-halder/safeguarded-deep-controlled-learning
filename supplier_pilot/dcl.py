"""Simulation-based labelling for deep controlled learning (DCL).

Instead of differentiating through the simulator, DCL improves a policy by
*comparing actions by simulation* and training a classifier to imitate the
winners (approximate policy iteration with classification; Temizoz et al.).

Actions are order vectors close to the base policy's order. For supplier i
with base order b_i, minimum order m_i = max(MOQ_i, 1) and reference
r_i = max(b_i, m_i), the per-supplier classes are

    class 0   : order nothing
    class 1.. : order r_i + delta, delta in DELTAS (default -2, -1, 0, +1, +2),
                projected onto the feasible set {0} U [m_i, capacity_i]

(a supplier that is down can only take class 0). The joint candidate set is
the product over suppliers ((K+1)^N combinations; 216 for N = 3).

A candidate is scored by applying it in the current period and then
following the base policy for H - 1 more periods (default H = 36), on M simulated futures
shared by all candidates (common random numbers). A terminal adjustment
credits the inventory position left at the end at the cheapest unit price,
so that ordering earlier is not penalised for costs whose benefit falls
after the horizon. The label is the best candidate, but only if a *confirmation* run on fresh,
independent futures shows it beating the base action by more than ``z``
standard errors; otherwise the base action is kept. Selecting the best of
~200 noisy estimates and testing it on the same data would inflate the
advantage (optimiser's curse), which made unconfirmed labels unreliable.

Everything here is NumPy and runs on the verified batched simulator.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np

from . import batched as bt
from .instance import Instance

DELTAS = (-2, -1, 0, 1, 2)
OPS = bt.NumpyOps


@dataclass
class Bases:
    """Per-instance base status tables, for fast lookup in big batches."""
    instances: list
    tables: np.ndarray        # (n_inst, 2^N, 3N)
    actives: np.ndarray       # (n_inst, N)
    spec1: bt.BatchSpec       # one episode per instance

    @classmethod
    def build(cls, instances, tables, actives):
        spec1 = bt.build_spec(OPS, instances, np.arange(len(instances)))
        return cls(list(instances), np.stack([np.asarray(t, float) for t in tables]),
                   np.stack([np.asarray(a, float) for a in actives]), spec1)

    @property
    def n(self):
        return self.spec1.n

    def spec(self, inst_idx) -> bt.BatchSpec:
        return bt.index_spec(self.spec1, inst_idx)

    def params(self, inst_idx, z):
        n = self.n
        zid = (np.rint(z).astype(int) * (1 << np.arange(n))).sum(1)
        row = self.tables[inst_idx, zid]
        return row[:, :n], row[:, n:2 * n], row[:, 2 * n:], self.actives[inst_idx]

    def orders(self, spec, inst_idx, I, O, z):
        S, cap, trig, act = self.params(inst_idx, z)
        q, g = bt.gmi_decode(OPS, spec, I, O, z, S, cap, trig, act)
        return np.rint(q), g


def is_absolute(deltas) -> bool:
    """deltas = ("abs", qmax) selects the full order space (original DCL):
    class k = order quantity k, k = 0..qmax, projected onto the feasible set."""
    return isinstance(deltas, (tuple, list)) and len(deltas) == 2 and deltas[0] == "abs"


def n_classes(deltas) -> int:
    return int(deltas[1]) + 1 if is_absolute(deltas) else len(deltas) + 1


def absolute_quantities(spec, b, z, qmax):
    """Full order space: class k orders k units (0 = nothing), projected onto
    {0} U [max(MOQ, 1), capacity]; a supplier that is down orders nothing.
    The base class is the base order itself."""
    m = np.maximum(spec.moq, 1.0)
    cap = spec.capacity
    k = np.arange(qmax + 1, dtype=float)
    Q = np.broadcast_to(k, b.shape + (qmax + 1,)).copy()
    Q = np.where(Q <= 0, 0.0, np.minimum(np.maximum(Q, m[:, :, None]), cap[:, :, None]))
    Q = np.where(z[:, :, None] > 0.5, Q, 0.0)
    base_cls = np.clip(np.rint(b), 0, qmax).astype(int)
    return Q, base_cls


def class_quantities(spec, b, z, deltas=DELTAS):
    """Order quantity of every (supplier, class): array (B, N, K+1).
    Also returns the base class of each supplier (B, N)."""
    if is_absolute(deltas):
        return absolute_quantities(spec, b, z, int(deltas[1]))
    m = np.maximum(spec.moq, 1.0)
    cap = spec.capacity
    r = np.maximum(b, m)
    Q = np.stack([np.zeros_like(b)] + [r + d for d in deltas], axis=2)
    Q = np.where(Q <= 0, 0.0, np.minimum(np.maximum(Q, m[:, :, None]), cap[:, :, None]))
    Q = np.where(z[:, :, None] > 0.5, Q, 0.0)
    base_cls = np.where(b > 0, 1 + list(deltas).index(0), 0)
    return Q, base_cls


def joint_classes(n, k):
    return np.array(list(product(range(k), repeat=n)), dtype=int)       # (C, N)


def rollout_candidates(bases: Bases, inst_idx, I, O, z, qcand, M=16, H=36, seed=0, max_batch=150_000,
                       terminal=True):
    """Cost of explicit candidate first actions: qcand (S, C, N) -> costs (S, C, M).
    Each candidate is applied in the current period, then the base policy is
    followed; all candidates of a state share the same M simulated futures."""
    S, C, N = qcand.shape
    out = np.empty((S, C, M))
    chunk = max(1, max_batch // (C * M))
    rng_master = np.random.default_rng(seed)
    for s0 in range(0, S, chunk):
        sl = slice(s0, min(S, s0 + chunk))
        Sc = sl.stop - sl.start
        rng = np.random.default_rng(rng_master.integers(1 << 62))
        idx = np.repeat(np.asarray(inst_idx)[sl], C * M)
        spec = bases.spec(idx)
        rep = lambda x: np.repeat(x[sl], C * M, axis=0)
        Ie, Oe, ze = rep(I).astype(float), rep(O).astype(float), rep(z).astype(float)
        q = np.repeat(qcand[sl][:, :, None, :], M, axis=2).reshape(-1, N)
        total = np.zeros(Sc * C * M)
        for t in range(H):
            # common random numbers: one draw per (state, m), shared by all candidates
            ud = rng.random((Sc, 1, M)); u = rng.random((Sc, 1, M, N)); us = rng.random((Sc, 1, M, spec.R))
            ud = np.broadcast_to(ud, (Sc, C, M)).reshape(-1)
            u = np.broadcast_to(u, (Sc, C, M, N)).reshape(-1, N)
            us = np.broadcast_to(us, (Sc, C, M, spec.R)).reshape(-1, spec.R)
            if t > 0:
                q, _ = bases.orders(spec, idx, Ie, Oe, ze)
            g = (q > 0).astype(float)
            d = bt.sample_demand(OPS, spec, ud)
            c, Ie, Oe, ze, _ = bt.step(OPS, spec, Ie, Oe, ze, q, g, d, u, us)
            total += c
        if terminal:
            cmin = spec.price.min(1)
            total -= cmin * (Ie + Oe.sum((1, 2)))
        out[sl] = total.reshape(Sc, C, M)
    return out


def rollout_scores(bases: Bases, inst_idx, I, O, z, Q, M=16, H=36, seed=0, max_batch=150_000,
                   terminal=True):
    """Cost of every joint candidate for every state: array (S, C, M).
    States: I (S,), O (S, N, Lmax), z (S, N); Q (S, N, K+1)."""
    S, N, K1 = Q.shape
    combos = joint_classes(N, K1)
    qcand = Q[:, np.arange(N)[None, :], combos]                 # (S, C, N): Q[s, i, combos[c, i]]
    return rollout_candidates(bases, inst_idx, I, O, z, qcand, M, H, seed, max_batch, terminal), combos


@dataclass
class Labels:
    classes: np.ndarray       # (S, N) chosen class per supplier
    base_classes: np.ndarray  # (S, N)
    advantage: np.ndarray     # (S,) estimated cost saving of the label vs the base action (H periods)
    changed: np.ndarray       # (S,) bool, label differs from base
    q_label: np.ndarray       # (S, N) labelled order vector
    q_base: np.ndarray        # (S, N)


def coordinate_candidates(cur, k1):
    """Current class vectors (S, N) and every single-supplier change of them:
    (S, 1 + N (K+1 - 1), N); candidate 0 is ``cur`` itself."""
    S, N = cur.shape
    cands = [cur]
    for i in range(N):
        for c in range(k1):
            v = cur.copy()
            v[:, i] = c
            keep = cur[:, i] != c                      # skip the no-change copy
            v[~keep] = cur[~keep]
            cands.append(v)
    return np.stack(cands, axis=1)


def sequential_halving(bases, inst_idx, I, O, z, Q, base_cls, budget=3584, H=36, seed=0, max_batch=150_000):
    """Sequential halving over the joint order space, as in original DCL
    (Temizoz et al.): arms = distinct joint order vectors; each round splits
    budget / rounds simulations evenly over the surviving arms on common
    random numbers, accumulates their costs and keeps the better half.
    Returns the winner's classes (S, N) and its estimated saving over the
    base order (S, 1); no acceptance test is applied here."""
    S, N, K1 = Q.shape
    combos = joint_classes(N, K1)                                   # (C0, N)
    qv = Q[:, np.arange(N)[None, :], combos]                        # (S, C0, N)
    base_q = np.take_along_axis(Q, base_cls[:, :, None], 2)[..., 0]
    arms, arm_cls = [], []
    for s_ in range(S):
        u, first = np.unique(qv[s_], axis=0, return_index=True)
        # the base order is always an arm, listed first
        is_base = np.all(u == base_q[s_], axis=1)
        order = np.concatenate([np.flatnonzero(is_base), np.flatnonzero(~is_base)])
        arms.append(u[order]); arm_cls.append(combos[first[order]])
    C = max(len(a) for a in arms)
    pad = lambda a: np.concatenate([a, np.repeat(a[:1], C - len(a), axis=0)]) if len(a) < C else a
    A = np.stack([pad(a) for a in arms])                            # (S, C, N) quantities
    AC = np.stack([pad(c) for c in arm_cls])                         # (S, C, N) classes
    rounds = max(1, int(np.ceil(np.log2(C))))
    idx = np.tile(np.arange(C), (S, 1))
    sums = np.zeros((S, C)); counts = np.zeros((S, C))
    for r in range(rounds):
        Cr = idx.shape[1]
        m = max(1, budget // (rounds * Cr))
        qa = np.take_along_axis(A, idx[:, :, None], 1)
        c = rollout_candidates(bases, inst_idx, I, O, z, qa, m, H, seed + 7_919 * (r + 1), max_batch)
        np.put_along_axis(sums, idx, np.take_along_axis(sums, idx, 1) + c.sum(2), 1)
        np.put_along_axis(counts, idx, np.take_along_axis(counts, idx, 1) + m, 1)
        mean = np.take_along_axis(sums, idx, 1) / np.take_along_axis(counts, idx, 1)
        order = np.argsort(mean, axis=1, kind="stable")
        keep = int(np.ceil(Cr / 2)) if r < rounds - 1 else 1
        idx = np.take_along_axis(idx, order[:, :keep], 1)
    best = idx[:, 0]
    mean_all = sums / np.maximum(counts, 1)
    # estimated saving of the winner over the base order (arm 0), from all simulations of each
    diff = (mean_all[:, 0] - mean_all[np.arange(S), best])[:, None]
    return AC[np.arange(S), best], diff


def label_states(bases: Bases, inst_idx, I, O, z, deltas=DELTAS, M=16, H=36, z_conf=1.0, seed=0,
                 max_batch=150_000, confirm_M: int = 64, search: str = "joint", rounds: int = 2,
                 sh_budget: int = 3584) -> Labels:
    """Two-stage labelling.
    Stage 1 (selection): find the best candidate on M shared futures.
      search = "joint": score all (K+1)^N joint class vectors (216 for N = 3);
      search = "coord": coordinate search from the base action, ``rounds``
      rounds of all single-supplier changes (1 + N K candidates per round;
      26 for N = 5), each round on fresh futures; scales linearly in N;
      search = "sh": sequential halving over all distinct joint orders with
      ``sh_budget`` simulations per state (original DCL; use with the full
      order space, deltas = ("abs", qmax), and confirm_M = 0).
    Stage 2 (confirmation): re-score only the winner against the base action on
    ``confirm_M`` *fresh* futures and keep the winner only if its advantage
    exceeds ``z_conf`` standard errors there. This removes the optimiser's curse
    (picking the luckiest of many noisy estimates); confirm_M = 0 disables it."""
    spec = bases.spec(inst_idx)
    b, _ = bases.orders(spec, inst_idx, I.astype(float), O.astype(float), z.astype(float))
    Q, base_cls = class_quantities(spec, b, z.astype(float), deltas)
    S, N, K1 = Q.shape
    take = lambda cls: np.take_along_axis(Q, cls[:, :, None], 2)[..., 0]
    if search == "joint":
        costs, combos = rollout_scores(bases, inst_idx, I, O, z, Q, M, H, seed, max_batch)
        mean = costs.mean(2)
        best_cls = combos[mean.argmin(1)]
        if confirm_M <= 0:
            c_base = np.ravel_multi_index(base_cls.T, (K1,) * N)
            diff = costs[np.arange(S), c_base] - costs[np.arange(S), mean.argmin(1)]
    elif search == "coord":
        cur = base_cls.copy()
        for r in range(rounds):
            cand = coordinate_candidates(cur, K1)                      # (S, C, N)
            qc = np.take_along_axis(Q[:, None, :, :], cand[..., None], 3)[..., 0]
            costs = rollout_candidates(bases, inst_idx, I, O, z, qc, M, H, seed + 104_729 * r, max_batch)
            mean = costs.mean(2)
            cur = cand[np.arange(S), mean.argmin(1)]
            if confirm_M <= 0 and r == rounds - 1:
                diff = costs[:, 0] - costs[np.arange(S), mean.argmin(1)]   # only meaningful for rounds = 1
        best_cls = cur
    elif search == "sh":
        best_cls, diff_sh = sequential_halving(bases, inst_idx, I, O, z, Q, base_cls, budget=sh_budget, H=H,
                                               seed=seed, max_batch=max_batch)
        if confirm_M <= 0:
            diff = diff_sh
    else:
        raise ValueError(search)
    if search == "sh" and confirm_M <= 0:
        # original DCL: the halving winner is taken as the label, with no acceptance test
        differs = np.any(take(best_cls) != take(base_cls), axis=1)
        cls = np.where(differs[:, None], best_cls, base_cls)
        q_lab, q_base = take(cls), take(base_cls)
        return Labels(cls, base_cls, np.where(differs, diff[:, 0], 0.0), differs, q_lab, q_base)
    if confirm_M > 0:
        c2 = rollout_candidates(bases, inst_idx, I, O, z, np.stack([take(base_cls), take(best_cls)], axis=1),
                                confirm_M, H, seed + 7_919, max_batch)
        diff = c2[:, 0] - c2[:, 1]
        m_eff = confirm_M
    else:
        m_eff = M
    adv = diff.mean(1)
    se = diff.std(1, ddof=1) / np.sqrt(m_eff) if m_eff > 1 else np.zeros(S)
    # joint: as in the original labeller (winner's classes differ from the base);
    # coord: the winner's order quantities must differ (classes can coincide in quantity)
    differs = (np.any(best_cls != base_cls, axis=1) if search == "joint"
               else np.any(take(best_cls) != take(base_cls), axis=1))
    accept = (adv > 0) & (adv > z_conf * se) & differs
    cls = np.where(accept[:, None], best_cls, base_cls)
    q_lab, q_base = take(cls), take(base_cls)
    changed = np.any(q_lab != q_base, axis=1)
    return Labels(cls, base_cls, np.where(accept, adv, 0.0), changed, q_lab, q_base)


class RolloutPolicy:
    """Online rollout policy: at every period, label the current states by
    simulation and take the labelled action. Expensive, but it is exactly the
    target DCL learns to imitate, so it tells us whether DCL can beat the base."""

    def __init__(self, bases: Bases, inst_k: int, deltas=DELTAS, M=16, H=36, z_conf=1.0, seed=0,
                 confirm_M=64, name="Rollout (status table + simulation)", search="joint", rounds=2):
        self.bases, self.k, self.deltas, self.M, self.H, self.z_conf = bases, inst_k, deltas, M, H, z_conf
        self.search, self.rounds = search, rounds
        self.confirm_M = confirm_M
        self.seed, self.name, self.calls, self.changed = seed, name, 0, 0

    def act(self, inst, st, t):
        E = st.I.shape[0]
        spec = self.bases.spec(np.full(E, self.k))
        I, O, z = bt.state_from_numpy(OPS, spec, st)
        lab = label_states(self.bases, np.full(E, self.k), I, O, z, self.deltas, self.M, self.H, self.z_conf,
                           seed=self.seed * 1_000_003 + t, confirm_M=self.confirm_M, search=self.search,
                           rounds=self.rounds)
        self.calls += E; self.changed += int(lab.changed.sum())
        return lab.q_label.astype(np.int64)


def collect_states(bases: Bases, inst_ids, policy_fn, episodes=64, burn_in=200, periods=400, every=8, seed=0):
    """Sample states visited by a policy (``policy_fn(spec, idx, I, O, z) -> q, g``)."""
    idx = np.repeat(np.asarray(inst_ids), episodes)
    spec = bases.spec(idx)
    B = len(idx)
    I, O, z = bt.initial(OPS, spec, B)
    rng = np.random.default_rng(seed)
    out = {"I": [], "O": [], "z": [], "inst": []}
    for t in range(burn_in + periods):
        if t >= burn_in and (t - burn_in) % every == 0:
            out["I"].append(I.copy()); out["O"].append(O.copy()); out["z"].append(z.copy()); out["inst"].append(idx.copy())
        q, g = policy_fn(spec, idx, I, O, z)
        d = bt.sample_demand(OPS, spec, rng.random(B))
        _, I, O, z, _ = bt.step(OPS, spec, I, O, z, q, g, d, rng.random((B, spec.n)), rng.random((B, spec.R)))
    return {k: np.concatenate(v) for k, v in out.items()}
