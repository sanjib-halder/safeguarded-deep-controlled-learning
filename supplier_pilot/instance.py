"""Problem instances for dynamic supplier activation and order allocation.

Model (periodic review, single non-perishable item, backorders)
----------------------------------------------------------------
Each period t:
  1. Observe the state: net inventory ``I`` (after this period's receipts),
     the pipeline of each supplier (orders due in t+1 ... t+L_i-1) and the
     up/down status ``z_i`` of each supplier.
  2. Choose an order q_i for every supplier. A supplier that is down cannot
     accept orders (q_i = 0). Otherwise q_i = 0 or MOQ_i <= q_i <= capacity_i.
  3. Orders with lead time 0 arrive immediately; demand D_t is realised and
     backordered if unmet.
  4. Cost: sum_i [K_i 1(q_i>0) + c_i q_i] + h (I_end)^+ + b (I_end)^-.
  5. Orders due in t+1 arrive (deliveries in transit are never lost), and
     supplier statuses evolve by a two-state Markov chain, optionally with
     regional common shocks that take down every supplier of a region.

An order placed in t with lead time L arrives at the start of t+L, i.e. it
is available to meet the demand of period t+L.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import cached_property
from itertools import product
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class Supplier:
    name: str
    price: float                 # unit purchase cost c_i
    lead_time: int               # L_i >= 0 periods
    fixed_cost: float = 0.0      # K_i, charged when q_i > 0
    moq: int = 0                 # minimum order quantity (0 or 1 = none)
    capacity: int | None = None  # max order per period (None = unbounded)
    p_fail: float = 0.0          # P(up -> down) per period
    p_recover: float = 1.0       # P(down -> up) per period
    region: int = 0              # used by regional common shocks

    def __post_init__(self):
        if self.lead_time < 0:
            raise ValueError("lead_time must be >= 0")
        if self.capacity is not None and self.moq > self.capacity:
            raise ValueError(f"{self.name}: MOQ exceeds capacity")
        if not (0 <= self.p_fail <= 1 and 0 <= self.p_recover <= 1):
            raise ValueError("probabilities must lie in [0, 1]")

    @property
    def min_order(self) -> int:
        return max(self.moq, 1)


@dataclass(frozen=True)
class Instance:
    name: str
    suppliers: tuple[Supplier, ...]
    demand_values: tuple[int, ...]
    demand_probs: tuple[float, ...]
    holding: float = 5.0
    backorder: float = 495.0
    regional_shock: float = 0.0   # P(shock) per region per period
    meta: dict = field(default_factory=dict, compare=False)

    def __post_init__(self):
        p = np.asarray(self.demand_probs, dtype=float)
        if len(self.demand_values) != len(p) or abs(p.sum() - 1) > 1e-9:
            raise ValueError("demand pmf invalid")

    # ------------------------------------------------------------------ basics
    @property
    def n(self) -> int:
        return len(self.suppliers)

    @cached_property
    def lead_times(self) -> np.ndarray:
        return np.array([s.lead_time for s in self.suppliers], dtype=int)

    @cached_property
    def prices(self) -> np.ndarray:
        return np.array([s.price for s in self.suppliers], dtype=float)

    @cached_property
    def fixed_costs(self) -> np.ndarray:
        return np.array([s.fixed_cost for s in self.suppliers], dtype=float)

    @cached_property
    def moqs(self) -> np.ndarray:
        return np.array([s.moq for s in self.suppliers], dtype=int)

    @cached_property
    def p_fail(self) -> np.ndarray:
        return np.array([s.p_fail for s in self.suppliers], dtype=float)

    @cached_property
    def p_recover(self) -> np.ndarray:
        return np.array([s.p_recover for s in self.suppliers], dtype=float)

    @cached_property
    def _caps_default(self) -> np.ndarray:
        return np.array([s.capacity if s.capacity is not None else 10 ** 9
                         for s in self.suppliers], dtype=np.int64)

    def capacities(self, big: int | None = None) -> np.ndarray:
        """Per-supplier capacity; ``big`` replaces unbounded capacity."""
        if big is None:
            return self._caps_default
        return np.array([s.capacity if s.capacity is not None else big
                         for s in self.suppliers], dtype=np.int64)

    @property
    def mean_demand(self) -> float:
        return float(np.dot(self.demand_values, self.demand_probs))

    @property
    def max_demand(self) -> int:
        return int(max(self.demand_values))

    @cached_property
    def regions(self) -> np.ndarray:
        return np.array([s.region for s in self.suppliers], dtype=int)

    @property
    def n_regions(self) -> int:
        return int(self.regions.max()) + 1

    @property
    def has_disruptions(self) -> bool:
        return self.regional_shock > 0 or any(s.p_fail > 0 for s in self.suppliers)

    @cached_property
    def _order(self) -> list[int]:
        return sorted(range(self.n), key=lambda i: (self.suppliers[i].lead_time,
                                                    self.suppliers[i].price,
                                                    self.suppliers[i].fixed_cost, i))

    def decision_order(self) -> list[int]:
        """Deterministic supplier ordering used by index policies:
        ascending lead time, ties broken by price, fixed cost, then index."""
        return self._order

    def unavoidable_cost(self) -> float:
        """Mean demand times the cheapest unit price: a lower bound on the
        long-run purchase cost. 'Excess cost' = average cost - this value."""
        return self.mean_demand * float(self.prices.min())

    # ---------------------------------------------------- disruption dynamics
    def status_index(self, z: np.ndarray) -> np.ndarray:
        """Joint status index sum_i z_i 2^i (z_i = 1 means up)."""
        z = np.asarray(z, dtype=np.int64)
        return (z * (1 << np.arange(self.n))).sum(axis=-1)

    def all_statuses(self) -> np.ndarray:
        """(2^N, N) matrix of statuses in index order."""
        idx = np.arange(1 << self.n)
        return ((idx[:, None] >> np.arange(self.n)) & 1).astype(np.int8)

    def status_transition_matrix(self) -> np.ndarray:
        """Exact joint transition matrix P[z, z'] consistent with
        :func:`supplier_pilot.simulator.next_status`."""
        Z = self.all_statuses()
        nz = len(Z)
        pf = np.array([s.p_fail for s in self.suppliers])
        pr = np.array([s.p_recover for s in self.suppliers])
        reg = self.regions
        R = self.n_regions
        rho = self.regional_shock
        P = np.zeros((nz, nz))
        shock_patterns = list(product([0, 1], repeat=R)) if rho > 0 else [(0,) * R]
        for a, z in enumerate(Z):
            # independent marginals: probability next status is up
            p_up = np.where(z == 1, 1 - pf, pr)
            for pat in shock_patterns:
                pat = np.array(pat)
                w = np.prod(np.where(pat == 1, rho, 1 - rho)) if rho > 0 else 1.0
                shocked = pat[reg] == 1
                for b, z2 in enumerate(Z):
                    if np.any(shocked & (z2 == 1)):
                        continue
                    pu = np.where(shocked, 0.0, p_up)
                    P[a, b] += w * np.prod(np.where(z2 == 1, pu, 1 - pu))
        assert np.allclose(P.sum(1), 1.0)
        return P

    def with_(self, **kw) -> "Instance":
        return replace(self, **kw)


# ---------------------------------------------------------------- generators
def uniform_demand(lo: int, hi: int) -> tuple[tuple[int, ...], tuple[float, ...]]:
    vals = tuple(range(lo, hi + 1))
    return vals, tuple([1.0 / len(vals)] * len(vals))


def poisson_demand(mean: float, tail: float = 1e-4) -> tuple[tuple[int, ...], tuple[float, ...]]:
    from scipy.stats import poisson
    hi = int(poisson.ppf(1 - tail, mean))
    vals = np.arange(hi + 1)
    p = poisson.pmf(vals, mean)
    p[-1] += 1 - p.sum()
    return tuple(int(v) for v in vals), tuple(float(x) for x in p)


def dual_sourcing_instance(c_e: float = 105, c_r: float = 100, l_e: int = 0,
                           l_r: int = 2, capacity: int | None = 8,
                           h: float = 5, b: float = 495) -> Instance:
    """Classic dual-sourcing setting in the style of Gijsbrechts et al. (2022):
    demand U{0..4}, h = 5, b = 495, c_r = 100, l_e = 0. ``capacity`` keeps the
    DP finite; it should be (and is checked to be) non-binding."""
    vals, probs = uniform_demand(0, 4)
    sup = (Supplier("expedited", c_e, l_e, capacity=capacity),
           Supplier("regular", c_r, l_r, capacity=capacity))
    return Instance(f"dual_ce{int(c_e)}_lr{l_r}", sup, vals, probs, h, b,
                    meta={"family": "dual_validation"})


def pilot_instances(lead_times: Sequence[int] = (0, 1, 3)) -> list[Instance]:
    """Four N = 3 pilot instances of increasing difficulty.

    A: fast, expensive local supplier (L=0, c=118); B: regional (L=1, c=108);
    C: cheap offshore (L=3, c=100, capacity 8). Costs are calibrated so that the
    DP-optimal policy uses all three suppliers in the disrupted instances.
    P1  no fixed costs / MOQ, mild independent disruptions
    P2  + fixed ordering costs (10/20/40) and an MOQ of 4 at C
    P3  + long, frequent outages at C (and more frequent at B)
    P4  + a regional shock taking down B and C together (correlated extension)
    """
    vals, probs = uniform_demand(0, 4)
    la, lb, lc = lead_times

    def sup(K=(0, 0, 0), moq_c=0, pf=(0.01, 0.01, 0.01), pr=(0.5, 0.5, 0.5)):
        return (Supplier("A_local", 118, la, K[0], 0, 6, pf[0], pr[0], region=0),
                Supplier("B_regional", 108, lb, K[1], 0, 6, pf[1], pr[1], region=1),
                Supplier("C_offshore", 100, lc, K[2], moq_c, 8, pf[2], pr[2], region=1))

    K = (10, 20, 40)
    out = (0.01, 0.02, 0.05), (0.5, 0.4, 0.15)
    return [
        Instance("P1_base", sup(), vals, probs, 5, 495,
                 meta={"family": "pilot", "desc": "no fixed costs, mild disruptions"}),
        Instance("P2_fixed_moq", sup(K=K, moq_c=4), vals, probs, 5, 495,
                 meta={"family": "pilot", "desc": "fixed costs + MOQ"}),
        Instance("P3_long_outages", sup(K=K, moq_c=4, pf=out[0], pr=out[1]), vals, probs, 5, 495,
                 meta={"family": "pilot", "desc": "fixed costs + MOQ + long outages at C"}),
        Instance("P4_correlated", sup(K=K, moq_c=4, pf=out[0], pr=out[1]), vals, probs, 5, 495,
                 regional_shock=0.02,
                 meta={"family": "pilot", "desc": "P3 + regional shock on B and C"}),
    ]


def random_instance(rng: np.random.Generator, n: int, name: str = "rand",
                    lead_range=(0, 4), price_premium=(0.0, 0.25),
                    fixed_cost_scale=(0.0, 1.0), cap_ratio=(1.0, 3.0),
                    p_fail=(0.0, 0.08), p_recover=(0.1, 0.6),
                    demand=(0, 4)) -> Instance:
    """Random instance generator for later experiments (training
    distributions, transfer tests). Faster suppliers are more expensive."""
    vals, probs = uniform_demand(*demand)
    mu = float(np.dot(vals, probs))
    L = np.sort(rng.integers(lead_range[0], lead_range[1] + 1, size=n))
    span = max(L.max() - L.min(), 1)
    prem = rng.uniform(*price_premium)
    sups = []
    for i in range(n):
        price = 100 * (1 + prem * (L.max() - L[i]) / span)
        K = rng.uniform(*fixed_cost_scale) * 100 * (0.5 + L[i] / span)
        cap = int(np.ceil(rng.uniform(*cap_ratio) * mu))
        sups.append(Supplier(f"S{i}", round(price, 2), int(L[i]), round(K, 1), 0, cap,
                             round(rng.uniform(*p_fail), 3), round(rng.uniform(*p_recover), 3),
                             region=int(rng.integers(0, 2))))
    return Instance(name, tuple(sups), vals, probs, 5, 495, meta={"family": "random"})


DP_TRACTABLE_LEADS = ((0, 1, 2), (0, 1, 3), (0, 2, 2), (1, 1, 2), (0, 0, 2))


def random_study_instance(rng: np.random.Generator, name: str = "R") -> Instance:
    """Random N = 3 instance that the exact DP can still solve (<= 2 pipeline
    entries, capacities <= 8), sampled over a wide but realistic range:

    * lead times: one of :data:`DP_TRACTABLE_LEADS` (fast -> slow)
    * prices: slowest = 100; premium of the fastest in [5%, 25%], middle in between
    * fixed costs K_i ~ U[0, 40], increasing with lead time on average
    * MOQ of the slowest supplier in {0, 2, 4}; capacities: fast/mid 6, slow 8
    * disruptions: p_fail ~ U[0, 0.08], p_recover ~ U[0.1, 0.6] per supplier
    * regional shock (B and C share a region) with prob. 0.5, rate ~ U[0.005, 0.03]
    Demand U{0..4}, h = 5, b = 495 as in the pilot.
    """
    vals, probs = uniform_demand(0, 4)
    L = DP_TRACTABLE_LEADS[rng.integers(len(DP_TRACTABLE_LEADS))]
    prem = rng.uniform(0.05, 0.25)
    mid = rng.uniform(0.25, 0.75)
    prices = (round(100 * (1 + prem), 2), round(100 * (1 + mid * prem), 2), 100.0)
    K = np.sort(rng.uniform(0, 40, 3))
    moq_c = int(rng.choice([0, 2, 4]))
    pf = rng.uniform(0, 0.08, 3).round(3)
    pr = rng.uniform(0.1, 0.6, 3).round(3)
    shock = float(round(rng.uniform(0.005, 0.03), 4)) if rng.random() < 0.5 else 0.0
    caps = (6, 6, 8)
    names = ("A_fast", "B_mid", "C_slow")
    regions = (0, 1, 1)
    sups = tuple(Supplier(names[i], prices[i], int(L[i]), round(float(K[i]), 1),
                          moq_c if i == 2 else 0, caps[i], float(pf[i]), float(pr[i]), regions[i])
                 for i in range(3))
    down = [p / (p + q) if p + q > 0 else 0.0 for p, q in zip(pf, pr)]
    meta = {"family": "random_study", "premium": prem, "K_max": float(K.max()), "moq_c": moq_c,
            "lead_times": list(L), "shock": shock,
            "down_share": [round(float(d), 4) for d in down],
            "mean_outage": [round(float(1 / q), 2) for q in pr]}
    return Instance(name, sups, vals, probs, 5, 495, regional_shock=shock, meta=meta)


FACTOR_LEVELS = {
    # fixed ordering costs K_i (sorted so slower suppliers cost more per order)
    "fixed_cost": {"low": (0.0, 10.0), "high": (25.0, 45.0)},
    # regional shock rate hitting B and C together
    "shock": {"none": (0.0, 0.0), "correlated": (0.015, 0.03)},
    # price premium of the fastest supplier over the slowest
    "premium": {"low": (0.05, 0.10), "high": (0.18, 0.25)},
}


def factorial_instance(seed: int, replicate: int, fixed_cost: str, shock: str, premium: str) -> Instance:
    """One cell of the blocked 2x2x2 factorial design.

    All random draws come from ``(seed, replicate)`` and are *shared across the
    eight cells* of that replicate: lead times, MOQ, disruption probabilities
    and the uniform numbers that place each factor within its level range.
    Cells of the same replicate therefore differ only in the three factors,
    which makes within-replicate (paired) comparisons possible.
    Other parameters as in :func:`random_study_instance`, with moderate
    disruption ranges (p_fail in [0.01, 0.06], p_recover in [0.15, 0.5]).
    """
    rng = np.random.default_rng((seed, replicate))
    L = DP_TRACTABLE_LEADS[rng.integers(len(DP_TRACTABLE_LEADS))]
    uK = np.sort(rng.random(3))
    u_prem, u_mid, u_shock = rng.random(3)
    moq_c = int(rng.choice([0, 2, 4]))
    pf = rng.uniform(0.01, 0.06, 3).round(3)
    pr = rng.uniform(0.15, 0.5, 3).round(3)

    def lvl(f, level, u):
        lo, hi = FACTOR_LEVELS[f][level]
        return lo + (hi - lo) * u

    K = np.array([lvl("fixed_cost", fixed_cost, u) for u in uK])
    prem = lvl("premium", premium, u_prem)
    rate = round(float(lvl("shock", shock, u_shock)), 4)
    mid = 0.25 + 0.5 * u_mid
    prices = (round(float(100 * (1 + prem)), 2), round(float(100 * (1 + mid * prem)), 2), 100.0)
    vals, probs = uniform_demand(0, 4)
    caps = (6, 6, 8)
    names = ("A_fast", "B_mid", "C_slow")
    regions = (0, 1, 1)
    sups = tuple(Supplier(names[i], prices[i], int(L[i]), round(float(K[i]), 1),
                          moq_c if i == 2 else 0, caps[i], float(pf[i]), float(pr[i]), regions[i])
                 for i in range(3))
    down = [p / (p + q) for p, q in zip(pf, pr)]
    prem = float(prem)
    cell = f"K-{fixed_cost}|S-{shock}|P-{premium}"
    meta = {"family": "factorial", "cell": cell, "replicate": replicate,
            "factors": {"fixed_cost": fixed_cost, "shock": shock, "premium": premium},
            "premium": prem, "K_max": float(K.max()), "moq_c": moq_c, "lead_times": list(L),
            "shock": rate, "down_share": [round(float(d), 4) for d in down],
            "mean_outage": [round(float(1 / q), 2) for q in pr]}
    return Instance(f"F{replicate:02d}_{fixed_cost[0]}{shock[0]}{premium[0]}", sups, vals, probs, 5, 495,
                    regional_shock=rate, meta=meta)


SCALE_ROLES = {
    # name, region, capacity; prices/leads are drawn below
    "A_fast": (0, 6), "B_mid": (1, 6), "C_slow": (1, 8), "D_alt": (2, 6), "E_alt_slow": (2, 8),
}


def scaled_instance(seed: int, replicate: int, n: int, fixed_cost: str, shock: str, premium: str) -> Instance:
    """N = 4 or 5 suppliers, same design logic as :func:`factorial_instance`.

    Suppliers A, B, C are drawn exactly as in the N = 3 factorial (A fast and
    expensive in region 0; B mid and C slow/cheapest sharing region 1, C with an
    MOQ), so the first three suppliers form a DP-tractable sub-instance. Added:

    * D_alt (N >= 4): region 2, lead time between B's and C's, price premium a
      fraction 0.1-0.5 of A's premium (a second regional source);
    * E_alt_slow (N = 5): region 2, lead time of C, premium 0-0.15 of A's,
      MOQ in {0, 2, 4} (a cheap source outside C's region).

    The regional shock hits regions 1 and 2 independently at the same rate, so
    D and E diversify against shocks to B and C. Fixed costs are sorted by lead
    time (slower suppliers cost more per order). All draws come from
    (seed, replicate, n) and are shared across the factor cells, as in the
    factorial design. The joint state space (inventory x pipelines x 2^N
    statuses) is far too large for exact DP.
    """
    if n not in (4, 5):
        raise ValueError("scaled_instance supports n = 4 or 5")
    rng = np.random.default_rng((seed, replicate, n))
    L3 = DP_TRACTABLE_LEADS[rng.integers(len(DP_TRACTABLE_LEADS))]
    uK = np.sort(rng.random(n))
    u_prem, u_mid, u_shock = rng.random(3)
    moq_c = int(rng.choice([0, 2, 4]))
    pf = rng.uniform(0.01, 0.06, n).round(3)
    pr = rng.uniform(0.15, 0.5, n).round(3)
    u_d, u_e = rng.random(2)
    L_d = int(rng.integers(L3[1], L3[2] + 1))
    moq_e = int(rng.choice([0, 2, 4]))

    def lvl(f, level, u):
        lo, hi = FACTOR_LEVELS[f][level]
        return lo + (hi - lo) * u

    prem = float(lvl("premium", premium, u_prem))
    rate = round(float(lvl("shock", shock, u_shock)), 4)
    mid = 0.25 + 0.5 * u_mid
    names = list(SCALE_ROLES)[:n]
    leads = [L3[0], L3[1], L3[2], L_d, L3[2]][:n]
    fracs = [1.0, mid, 0.0, 0.1 + 0.4 * u_d, 0.15 * u_e][:n]
    prices = [round(float(100 * (1 + f * prem)), 2) for f in fracs]
    moqs = [0, 0, moq_c, 0, moq_e][:n]
    rank = sorted(range(n), key=lambda i: (leads[i], -prices[i], i))
    K = np.empty(n)
    K[rank] = [lvl("fixed_cost", fixed_cost, u) for u in uK]
    sups = tuple(Supplier(names[i], prices[i], int(leads[i]), round(float(K[i]), 1), moqs[i],
                          SCALE_ROLES[names[i]][1], float(pf[i]), float(pr[i]), SCALE_ROLES[names[i]][0])
                 for i in range(n))
    vals, probs = uniform_demand(0, 4)
    cell = f"K-{fixed_cost}|S-{shock}|P-{premium}"
    meta = {"family": f"scale{n}", "cell": cell, "replicate": replicate,
            "factors": {"fixed_cost": fixed_cost, "shock": shock, "premium": premium},
            "premium": prem, "lead_times": [int(x) for x in leads], "shock": rate, "moq": moqs}
    return Instance(f"N{n}_{replicate:02d}_{fixed_cost[0]}{shock[0]}{premium[0]}", sups, vals, probs, 5, 495,
                    regional_shock=rate, meta=meta)


def core_instance(inst: Instance) -> Instance:
    """The first three suppliers (A, B, C) of a scaled instance: an instance
    the exact DP can solve. Its optimal policy is feasible in the full
    instance (never order from D or E), so its cost is an upper bound on the
    full instance's optimal cost."""
    return inst.with_(name=inst.name + "_ABC", suppliers=inst.suppliers[:3])
