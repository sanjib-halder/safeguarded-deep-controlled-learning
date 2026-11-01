"""Rolling-horizon stochastic MILP baselines (sample average approximation).

At each period the policy samples scenarios of demand and supplier status
over a horizon of ``H`` periods, solves a MILP, implements the first-period
orders and repeats (rolling horizon). Solved with HiGHS through
``scipy.optimize.milp`` (no external solver).

Two variants share one formulation, which assigns every (scenario, period)
to a *decision node*; scenarios that share a node must take the same order:

* **Two-stage** (``tree=None``): only the first-period decision is shared by
  all ``M`` scenarios. Later orders see each scenario's whole future, which
  makes the look-ahead optimistic (it under-buys safety stock).
* **Multi-stage** (``tree=(b1, b2, ...)``): a scenario tree. Decisions in
  periods 0..K-1 (K = len(tree)) are non-anticipative: scenarios with the
  same history of demand and supplier status share the decision. Period k
  branches into b_{k+1} children (b1 children of the root, etc.); after depth K
  each leaf continues along one sampled path. M = prod(tree) scenarios.

Notation inside the model: order of supplier i placed in period tau arrives
at the start of tau + L_i; end-of-period inventory = I0 + arrivals - demand;
cost = sum K_i y + c_i q + h I+ + b I-; terminal on-hand stock is salvaged
at the cheapest unit price.
"""
from __future__ import annotations

import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from .instance import Instance
from .simulator import State, next_status


class RollingSAAPolicy:
    def __init__(self, H: int | None = None, M: int = 10, tree: tuple[int, ...] | None = None,
                 relax_future_q: bool = True, relax_future_y: bool = False,
                 time_limit: float = 5.0, seed: int = 12345, salvage: bool = True,
                 name: str | None = None):
        self.H = H
        self.tree = tuple(int(x) for x in tree) if tree else None
        self.M = int(np.prod(self.tree)) if self.tree else M
        self.relax_future_q, self.relax_future_y = relax_future_q, relax_future_y
        self.time_limit, self.seed, self.salvage = time_limit, seed, salvage
        if name is None:
            name = (f"RH-SAA multi-stage(tree={'x'.join(map(str, self.tree))})" if self.tree
                    else f"RH-SAA two-stage(M={self.M})")
        self.name = name
        self.solve_times: list[float] = []
        self.failures = 0

    def reset(self, inst, E):
        self.solve_times = []
        self.failures = 0

    # ------------------------------------------------------------ scenarios
    def _sample_paths(self, inst: Instance, z0: np.ndarray, H: int, rng):
        """Returns demand D (M, H), statuses Z (M, H, N) with Z[:, 0] = z0, and
        node ids (M, H): the decision node used by scenario s in period tau."""
        vals = np.asarray(inst.demand_values)
        p = np.asarray(inst.demand_probs)
        n, M = inst.n, self.M
        D = np.empty((M, H), dtype=np.int64)
        Z = np.empty((M, H, n), dtype=np.int8)
        node = np.empty((M, H), dtype=np.int64)
        shock = inst.regional_shock > 0

        cdf = np.cumsum(p)

        def draw(k, z):
            # stratified (Latin-hypercube) demand sampling: the k draws cover the
            # whole distribution, so small branchings still contain the tail
            u = (rng.permutation(k) + rng.random(k)) / k
            d = vals[np.minimum(np.searchsorted(cdf, u, side="right"), len(vals) - 1)]
            u = rng.random((k, n))
            us = rng.random((k, inst.n_regions)) if shock else None
            return d, next_status(inst, z, u, us)

        Z[:, 0] = z0
        if self.tree is None:
            d, z = draw(M, np.repeat(z0[None], M, 0))
            D[:, 0] = d
            if H > 1:
                Z[:, 1] = z
            node[:, 0] = 0
            for tau in range(1, H):
                node[:, tau] = 1 + (tau - 1) * M + np.arange(M)
                d, z = draw(M, Z[:, tau])
                D[:, tau] = d
                if tau + 1 < H:
                    Z[:, tau + 1] = z
            return D, Z, node
        # scenario tree: scenario s has digits (k_1..k_K) in the mixed radix tree
        K = min(len(self.tree), H)
        digits = np.stack(np.unravel_index(np.arange(M), self.tree), axis=1)  # (M, K)
        next_id = 0
        for tau in range(H):
            if tau < K:
                # node = history prefix of length tau (digits[:, :tau])
                pref = digits[:, :tau]
                keys = [tuple(r) for r in pref]
                uniq = {}
                for s, kk in enumerate(keys):
                    if kk not in uniq:
                        uniq[kk] = next_id; next_id += 1
                    node[s, tau] = uniq[kk]
                # outcome of period tau (demand d_tau and z_{tau+1}) is shared by
                # scenarios with the same prefix of length tau+1
                # children of each node are drawn together (stratified siblings)
                b = self.tree[tau]
                outcome = {}
                for kk_parent, nid in uniq.items():
                    s0 = int(np.flatnonzero(node[:, tau] == nid)[0])
                    d, z = draw(b, np.repeat(Z[s0:s0 + 1, tau], b, 0))
                    for c in range(b):
                        outcome[kk_parent + (c,)] = (d[c], z[c])
                for s, kk in enumerate(tuple(r) for r in digits[:, :tau + 1]):
                    D[s, tau] = outcome[kk][0]
                    if tau + 1 < H:
                        Z[s, tau + 1] = outcome[kk][1]
            else:
                node[:, tau] = next_id + np.arange(M); next_id += M
                d, z = draw(M, Z[:, tau])
                D[:, tau] = d
                if tau + 1 < H:
                    Z[:, tau + 1] = z
        return D, Z, node

    # ----------------------------------------------------------------- solve
    def _solve_one(self, inst: Instance, I0: int, pipes: list[np.ndarray], z0: np.ndarray,
                   rng) -> np.ndarray:
        n, M = inst.n, self.M
        L = inst.lead_times
        H = self.H or int(L.max() + 5)
        D, Zs, node = self._sample_paths(inst, z0, H, rng)
        n_nodes = int(node.max()) + 1
        bigU = inst.capacities(big=int(inst.max_demand * H * 2)).astype(float)
        moq = np.maximum(inst.moqs, 1).astype(float)
        c, K = inst.prices, inst.fixed_costs
        h, b = inst.holding, inst.backorder

        # node availability: a supplier is available at a node if it is up there
        # (all scenarios sharing a node share the status history)
        avail = np.zeros((n_nodes, n))
        avail[node.ravel()] = Zs.reshape(-1, n)
        # node weight = share of scenarios passing through it
        w = np.bincount(node.ravel(), minlength=n_nodes) / M

        # variables: q (n_nodes, n), y (n_nodes, n), Ip (M, H), Im (M, H)
        qv = np.arange(n_nodes * n).reshape(n_nodes, n)
        yv = qv.size + qv
        Ip = 2 * qv.size + np.arange(M * H).reshape(M, H)
        Im = Ip.size + Ip
        nv = 2 * qv.size + 2 * M * H

        cost = np.zeros(nv)
        cost[qv] = w[:, None] * c[None, :]
        cost[yv] = w[:, None] * K[None, :]
        cost[Ip] = h / M
        cost[Im] = b / M
        if self.salvage:
            cost[Ip[:, -1]] -= c.min() / M

        known = np.zeros(H)
        for j, Lj in enumerate(L):
            for kk in range(max(Lj - 1, 0)):
                if kk + 1 < H:
                    known[kk + 1] += pipes[j][kk]
        cum_known = np.cumsum(known)

        rows, cols, vals = [], [], []
        lb, ub = [], []
        r = 0
        for s in range(M):
            cumd = np.cumsum(D[s])
            for tau in range(H):
                rows += [r, r]; cols += [Ip[s, tau], Im[s, tau]]; vals += [1.0, -1.0]
                for i in range(n):
                    for tp in range(0, tau - L[i] + 1):
                        rows.append(r); cols.append(qv[node[s, tp], i]); vals.append(-1.0)
                rhs = I0 + cum_known[tau] - cumd[tau]
                lb.append(rhs); ub.append(rhs)
                r += 1
        for nd in range(n_nodes):
            for i in range(n):
                rows += [r, r]; cols += [qv[nd, i], yv[nd, i]]; vals += [1.0, -moq[i]]
                lb.append(0.0); ub.append(np.inf); r += 1
                rows += [r, r]; cols += [qv[nd, i], yv[nd, i]]; vals += [1.0, -bigU[i]]
                lb.append(-np.inf); ub.append(0.0); r += 1
        A = coo_matrix((vals, (rows, cols)), shape=(r, nv)).tocsr()

        vub = np.full(nv, np.inf)
        vub[qv] = np.broadcast_to(bigU, qv.shape)
        vub[yv] = avail
        integrality = np.zeros(nv)
        integrality[qv[0]] = 1
        integrality[yv[0]] = 1
        if not self.relax_future_q:
            integrality[qv[1:]] = 1
        if not self.relax_future_y:
            integrality[yv[1:]] = 1
        t0 = time.perf_counter()
        res = milp(cost, constraints=LinearConstraint(A, np.array(lb), np.array(ub)),
                   integrality=integrality, bounds=Bounds(np.zeros(nv), vub),
                   options={"time_limit": self.time_limit, "mip_rel_gap": 1e-4, "disp": False})
        self.solve_times.append(time.perf_counter() - t0)
        if res.x is None:
            self.failures += 1
            return np.zeros(n, dtype=np.int64)
        return np.rint(res.x[qv[0]]).astype(np.int64)

    def act(self, inst: Instance, st: State, t: int) -> np.ndarray:
        E = st.I.shape[0]
        out = np.zeros((E, inst.n), dtype=np.int64)
        for e in range(E):
            rng = np.random.default_rng((self.seed, t, e))
            out[e] = self._solve_one(inst, int(st.I[e]), [p[e] for p in st.pipes], st.z[e], rng)
        return out
