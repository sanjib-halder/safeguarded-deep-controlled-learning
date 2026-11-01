"""Backend-agnostic batched dynamics and a differentiable GMI decoder.

The same code runs on NumPy (used in tests to check it against the validated
simulator) and on PyTorch (used to train the neural policy with pathwise
gradients). Only the tiny ``Ops`` adapters differ.

Batch layout
------------
B parallel episodes, each tied to one instance (instances may differ but
must have the same number of suppliers N). Pipelines are padded to
``Lmax = max lead time`` slots:

    O[b, i, k] = quantity from supplier i due at the start of period t+1+k

which matches the simulator's ``State.pipes[i][:, k]`` for k < L_i - 1.
An order with lead time L >= 1 enters slot L-1; lead time 0 arrives at once.

Straight-through estimation
---------------------------
Forward passes use the *exact* integer policy (rounded order-up-to levels,
caps and triggers, hard order/no-order gates), so simulated costs equal the
real system's. Backward passes use surrogate gradients: identity for
rounding, and a sigmoid of (gap - trigger) for the order gate. On NumPy the
surrogates are inactive and everything is exact.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .instance import Instance


# ------------------------------------------------------------------- ops
class NumpyOps:
    name = "numpy"

    @staticmethod
    def asarray(x, dtype=float):
        return np.asarray(x, dtype=dtype)

    @staticmethod
    def zeros(shape, like=None):
        return np.zeros(shape, dtype=float)

    where = staticmethod(np.where)
    minimum = staticmethod(np.minimum)
    maximum = staticmethod(np.maximum)

    @staticmethod
    def relu(x):
        return np.maximum(x, 0.0)

    @staticmethod
    def cumsum(x, dim):
        return np.cumsum(x, axis=dim)

    @staticmethod
    def cat(xs, dim):
        return np.concatenate(xs, axis=dim)

    @staticmethod
    def stack(xs, dim):
        return np.stack(xs, axis=dim)

    @staticmethod
    def take_row(x, idx):
        """x (B, K), idx (B,) int -> x[b, idx[b]]"""
        return x[np.arange(x.shape[0]), idx]

    @staticmethod
    def einsum(eq, *xs):
        return np.einsum(eq, *xs)

    @staticmethod
    def st_round(x):
        return np.rint(x)

    @staticmethod
    def st_gate(hard, arg):
        return hard.astype(float)

    @staticmethod
    def to_float(b):
        return b.astype(float)


def torch_ops(device="cpu", dtype=None):
    """PyTorch adapter (imported lazily so NumPy users need no torch)."""
    import torch

    dt = dtype or torch.float32

    class TorchOps:
        name = "torch"
        dev = device

        @staticmethod
        def asarray(x, dtype=None):
            if isinstance(x, torch.Tensor):
                return x.to(device)
            return torch.as_tensor(np.asarray(x), dtype=dtype or dt, device=device)

        @staticmethod
        def zeros(shape, like=None):
            return torch.zeros(shape, dtype=dt, device=device)

        where = staticmethod(torch.where)
        minimum = staticmethod(torch.minimum)
        maximum = staticmethod(torch.maximum)

        @staticmethod
        def relu(x):
            return torch.clamp(x, min=0.0)

        @staticmethod
        def cumsum(x, dim):
            return torch.cumsum(x, dim=dim)

        @staticmethod
        def cat(xs, dim):
            return torch.cat(xs, dim=dim)

        @staticmethod
        def stack(xs, dim):
            return torch.stack(xs, dim=dim)

        @staticmethod
        def take_row(x, idx):
            return x.gather(1, idx.long().unsqueeze(1)).squeeze(1)

        @staticmethod
        def einsum(eq, *xs):
            return torch.einsum(eq, *xs)

        @staticmethod
        def st_round(x):
            return x + (torch.round(x) - x).detach()

        @staticmethod
        def st_gate(hard, arg):
            soft = torch.sigmoid(arg)
            return hard.to(dt) + soft - soft.detach()

        @staticmethod
        def to_float(b):
            return b.to(dt)

    return TorchOps


# ------------------------------------------------------------------ spec
@dataclass
class BatchSpec:
    """Per-episode instance parameters, all as backend arrays."""
    n: int
    Lmax: int
    R: int
    inst_idx: np.ndarray        # (B,) numpy, which instance each episode uses
    price: object               # (B, N)
    K: object
    moq: object
    capacity: object
    pf: object
    pr: object
    imm: object                 # (B, N) 1 if lead time 0
    slot: object                # (B, N, Lmax) 1 at slot L_i - 1 (if L_i >= 1)
    perm: object                # (B, N, N) perm[b, p, i] = 1 if supplier i is p-th in order
    L_ord: object               # (B, N) lead time of the p-th supplier (int index)
    region_oh: object           # (B, N, R)
    holding: object             # (B,)
    backorder: object           # (B,)
    shock: object               # (B,)
    dem_vals: object            # (B, D) padded demand values
    dem_cdf: object             # (B, D) padded cdf (padding = 2.0 so never <= u)
    unavoidable: np.ndarray     # (B,) numpy: mean demand x cheapest price
    mu: object                  # (B,) mean demand


def build_spec(ops, instances: list[Instance], inst_idx, big_cap: int = 10_000) -> BatchSpec:
    inst_idx = np.asarray(inst_idx, dtype=int)
    n = instances[0].n
    assert all(i.n == n for i in instances), "all instances must have the same N"
    Lmax = max(1, max(int(i.lead_times.max()) for i in instances))
    R = max(i.n_regions for i in instances)
    D = max(len(i.demand_values) for i in instances)

    def per(f):
        return np.stack([f(instances[k]) for k in inst_idx])

    L = per(lambda i: i.lead_times.astype(int))
    order = per(lambda i: np.array(i.decision_order()))
    perm = np.zeros((len(inst_idx), n, n))
    for b in range(len(inst_idx)):
        perm[b, np.arange(n), order[b]] = 1.0
    slot = np.zeros((len(inst_idx), n, Lmax))
    for b in range(len(inst_idx)):
        for i in range(n):
            if L[b, i] >= 1:
                slot[b, i, L[b, i] - 1] = 1.0
    reg = per(lambda i: i.regions)
    region_oh = np.zeros((len(inst_idx), n, R))
    for b in range(len(inst_idx)):
        region_oh[b, np.arange(n), reg[b]] = 1.0
    vals = np.zeros((len(inst_idx), D)); cdf = np.full((len(inst_idx), D), 2.0)
    for b, k in enumerate(inst_idx):
        v = np.asarray(instances[k].demand_values, float)
        vals[b, :len(v)] = v
        vals[b, len(v):] = v[-1]
        cdf[b, :len(v)] = np.cumsum(instances[k].demand_probs)
    L_ord = np.take_along_axis(L, order, axis=1)
    A = ops.asarray
    return BatchSpec(
        n=n, Lmax=Lmax, R=R, inst_idx=inst_idx,
        price=A(per(lambda i: i.prices)), K=A(per(lambda i: i.fixed_costs)),
        moq=A(per(lambda i: i.moqs.astype(float))),
        capacity=A(per(lambda i: np.minimum(i.capacities(big=big_cap), big_cap).astype(float))),
        pf=A(per(lambda i: i.p_fail)), pr=A(per(lambda i: i.p_recover)),
        imm=A((L == 0).astype(float)), slot=A(slot), perm=A(perm),
        L_ord=ops.asarray(L_ord, dtype=None) if ops.name == "numpy" else _long(ops, L_ord),
        region_oh=A(region_oh),
        holding=A(np.array([instances[k].holding for k in inst_idx])),
        backorder=A(np.array([instances[k].backorder for k in inst_idx])),
        shock=A(np.array([instances[k].regional_shock for k in inst_idx])),
        dem_vals=A(vals), dem_cdf=A(cdf),
        unavoidable=np.array([instances[k].unavoidable_cost() for k in inst_idx]),
        mu=A(np.array([max(instances[k].mean_demand, 1e-6) for k in inst_idx])))


def _long(ops, x):
    import torch
    return torch.as_tensor(np.asarray(x), dtype=torch.long, device=ops.dev)


# -------------------------------------------------------------- dynamics
def sample_demand(ops, spec: BatchSpec, u):
    """Inverse-CDF demand, identical to simulator.make_streams."""
    idx = ops.to_float(spec.dem_cdf <= u[:, None]).sum(1)
    D = spec.dem_vals.shape[1]
    idx = ops.minimum(idx, ops.asarray(np.full(1, D - 1.0)))
    oh = ops.to_float(ops.asarray(np.arange(D)[None, :] * 1.0) == idx[:, None])
    return (oh * spec.dem_vals).sum(1)


def gmi_decode(ops, spec: BatchSpec, I, O, z, S, cap, trig, active, tau: float = 1.0):
    """Generalised multi-index orders for a batch (see policies.py).

    S, cap, trig: (B, N) in supplier order (continuous; rounded with
    straight-through); active: (B, N) 0/1; z: (B, N) 0/1 status.
    Returns orders q (B, N) and order gates g (B, N) (1 = order placed)."""
    B, N = I.shape[0], spec.n
    to_ord = lambda X: ops.einsum("bi,bpi->bp", X, spec.perm)
    S_o = to_ord(ops.st_round(S))
    cap_o = to_ord(ops.maximum(ops.st_round(cap), ops.asarray(np.ones(1))))
    trig_o = to_ord(ops.maximum(ops.st_round(trig), ops.asarray(np.ones(1))))
    up_o = to_ord(z)
    act_o = to_ord(active)
    moq_o = to_ord(spec.moq)
    capy_o = to_ord(spec.capacity)
    Ctot = ops.cumsum(O.sum(1), dim=1)                       # (B, Lmax): due within t+1..t+1+k
    Ctot0 = ops.cat([ops.zeros((B, 1)), Ctot], dim=1)        # Ctot0[L] = due within t+1..t+L
    placed = ops.zeros((B,))
    qs, gs = [], []
    for p in range(N):
        IP = I + ops.take_row(Ctot0, spec.L_ord[:, p]) + placed
        gap = S_o[:, p] - IP
        ok = (up_o[:, p] > 0.5) & (act_o[:, p] > 0.5)
        hard = ok & (gap >= trig_o[:, p])
        g = ops.st_gate(hard, (gap - trig_o[:, p] + 0.5) / tau) * ops.to_float(ok)
        qc = ops.minimum(gap, cap_o[:, p])
        qc = ops.maximum(qc, moq_o[:, p])
        qc = ops.minimum(qc, capy_o[:, p])
        qc = ops.maximum(qc, ops.zeros((B,)))
        q = g * qc
        placed = placed + q
        qs.append(q); gs.append(g)
    q_o, g_o = ops.stack(qs, 1), ops.stack(gs, 1)
    from_ord = lambda X: ops.einsum("bp,bpi->bi", X, spec.perm)
    return from_ord(q_o), from_ord(g_o)


def step(ops, spec: BatchSpec, I, O, z, q, g, d, u, us):
    """One period, identical to simulator.step. Returns (cost, I, O, z, extras)."""
    B, N = I.shape[0], spec.n
    immediate = (q * spec.imm).sum(1)
    I_pre = I + immediate
    I_end = I_pre - d
    cost = ((q * spec.price).sum(1) + (g * spec.K).sum(1)
            + spec.holding * ops.relu(I_end) + spec.backorder * ops.relu(-I_end))
    O = O + q[:, :, None] * spec.slot
    arrivals = O[:, :, 0].sum(1)
    O = ops.cat([O[:, :, 1:], ops.zeros((B, N, 1))], dim=2)
    I_next = I_end + arrivals
    up = ops.where(z > 0.5, u >= spec.pf, u < spec.pr)
    if us is not None:
        shocked_r = ops.to_float(us < spec.shock[:, None])                 # (B, R)
        shocked = ops.einsum("br,bnr->bn", shocked_r, spec.region_oh) > 0.5
        up = up & ~shocked
    return cost, I_next, O, ops.to_float(up), {"I_end": I_end, "I_pre": I_pre}


def initial(ops, spec: BatchSpec, B: int):
    return ops.zeros((B,)), ops.zeros((B, spec.n, spec.Lmax)), ops.asarray(np.ones((B, spec.n)))


def state_from_numpy(ops, spec: BatchSpec, st):
    """Convert a simulator.State (single instance) into batched arrays."""
    B = st.I.shape[0]
    O = np.zeros((B, spec.n, spec.Lmax))
    for i, p in enumerate(st.pipes):
        if p.shape[1]:
            O[:, i, :p.shape[1]] = p
    return ops.asarray(st.I.astype(float)), ops.asarray(O), ops.asarray(st.z.astype(float))


def index_spec(spec: BatchSpec, idx) -> BatchSpec:
    """Select / repeat episodes of a (NumPy) spec: new episode b = old episode idx[b].
    Fast way to build very large batches from a small per-instance spec."""
    idx = np.asarray(idx, dtype=int)
    take = lambda x: x[idx]
    return BatchSpec(
        n=spec.n, Lmax=spec.Lmax, R=spec.R, inst_idx=spec.inst_idx[idx],
        price=take(spec.price), K=take(spec.K), moq=take(spec.moq), capacity=take(spec.capacity),
        pf=take(spec.pf), pr=take(spec.pr), imm=take(spec.imm), slot=take(spec.slot), perm=take(spec.perm),
        L_ord=take(spec.L_ord), region_oh=take(spec.region_oh), holding=take(spec.holding),
        backorder=take(spec.backorder), shock=take(spec.shock), dem_vals=take(spec.dem_vals),
        dem_cdf=take(spec.dem_cdf), unavoidable=take(spec.unavoidable), mu=take(spec.mu))
