"""Correctness tests. Run with `pytest -q` or `python tests/test_core.py`."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from supplier_pilot.instance import (Instance, Supplier, dual_sourcing_instance, pilot_instances,
                                     uniform_demand)
from supplier_pilot.simulator import (State, initial_state, make_streams, next_status,
                                      project_feasible, simulate, step)
from supplier_pilot.policies import (INF, MultiIndexPolicy, dual_index_policy,
                                     multi_index_orders, single_source_policy)


def _random_state(inst, E, rng, Ilo=-6, Ihi=15):
    st = initial_state(inst, E)
    st.I = rng.integers(Ilo, Ihi, size=E)
    caps = inst.capacities(big=8)
    st.pipes = [rng.integers(0, caps[i] + 1, size=p.shape) for i, p in enumerate(st.pipes)]
    st.z = rng.integers(0, 2, size=(E, inst.n)).astype(np.int8)
    return st


def test_transition_matrix_matches_simulator():
    inst = pilot_instances()[3]          # includes a regional shock
    P = inst.status_transition_matrix()
    rng = np.random.default_rng(1)
    Zs = inst.all_statuses()
    n = 200_000
    for a in [0, 3, 7]:
        z = np.repeat(Zs[a][None], n, 0)
        z2 = next_status(inst, z, rng.random((n, inst.n)), rng.random((n, inst.n_regions)))
        freq = np.bincount(inst.status_index(z2), minlength=len(Zs)) / n
        assert np.allclose(freq, P[a], atol=4e-3), (a, freq, P[a])


def test_projection_only_changes_infeasible():
    inst = pilot_instances()[1]          # C has MOQ 4, capacities 6/6/8
    z = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.int8)
    q = np.array([[3, 2, 2], [3, 5, 9], [0, 0, 6]])
    out = project_feasible(inst, q, z)
    assert out.tolist() == [[3, 2, 4], [3, 0, 8], [0, 0, 6]]


def test_flow_conservation():
    inst = pilot_instances()[2]
    rng = np.random.default_rng(0)
    E, T = 5, 400
    st = initial_state(inst, E)
    streams = make_streams(inst, range(E), T)
    ordered = np.zeros(E, dtype=np.int64)
    for t in range(T):
        q = project_feasible(inst, rng.integers(0, 6, size=(E, inst.n)), st.z)
        ordered += q.sum(1)
        step(inst, st, q, streams.D[:, t], streams.U[:, t], None)
    in_pipe = sum(p.sum(1) for p in st.pipes)
    assert np.array_equal(st.I + in_pipe, ordered - streams.D.sum(1))


def _di_reference(inst, st, Se, Sr, cap_r):
    """Independent textbook DI/CDI implementation for l_e = 0."""
    fast, slow = inst.decision_order()
    assert inst.suppliers[fast].lead_time == 0
    ipe = st.I
    qe = np.where(st.z[:, fast] == 1, np.maximum(Se - ipe, 0), 0)
    ipr = st.I + st.pipes[slow].sum(1) + qe
    qr = np.where(st.z[:, slow] == 1, np.minimum(np.maximum(Sr - ipr, 0), cap_r), 0)
    out = np.zeros((len(st.I), 2), dtype=np.int64)
    out[:, fast], out[:, slow] = qe, qr
    return out


def test_gmi_reduces_to_di_and_cdi():
    """Proposition check: with N = 2, triggers 1, uncapped fast supplier, the
    GMI policy equals DI (cap_r = inf) and CDI (finite cap_r)."""
    inst = dual_sourcing_instance(105, 100, 0, 3, capacity=None)
    rng = np.random.default_rng(3)
    st = _random_state(inst, 2000, rng)
    for Se, Sr, cap in [(4, 11, INF), (2, 9, 3), (0, 14, 2), (6, 13, 5)]:
        got = dual_index_policy(inst, Se, Sr, cap).act(inst, st, 0)
        ref = _di_reference(inst, st, Se, Sr, cap if cap < INF else 10 ** 9)
        assert np.array_equal(got, ref), (Se, Sr, cap)


def test_single_source_sS():
    inst = pilot_instances()[1]
    rng = np.random.default_rng(4)
    st = _random_state(inst, 2000, rng)
    i, s, S = 2, 9, 16
    got = single_source_policy(inst, i, s, S).act(inst, st, 0)
    ip = st.I + st.pipes[i].sum(1)
    q = np.where(ip < s, S - ip, 0)
    q = np.where(q > 0, np.minimum(np.maximum(q, inst.moqs[i]), 8), 0)
    q = np.where(st.z[:, i] == 1, q, 0)
    assert np.array_equal(got[:, i], q)
    assert np.all(got[:, [0, 1]] == 0)


def test_gmi_permutation_equivariance():
    """Relabelling suppliers relabels the orders (distinct lead times/prices
    make the decision order label-free)."""
    base = pilot_instances()[2]
    perm = [2, 0, 1]
    inst_p = base.with_(suppliers=tuple(base.suppliers[i] for i in perm))
    rng = np.random.default_rng(5)
    st = _random_state(base, 500, rng)
    st_p = State(st.I.copy(), [st.pipes[i].copy() for i in perm], st.z[:, perm].copy())
    S, cap, trig = np.array([5, 9, 14]), np.array([6, 3, 8]), np.array([1, 2, 4])
    q = multi_index_orders(base, st, S, cap, trig)
    q_p = multi_index_orders(inst_p, st_p, S[perm], cap[perm], trig[perm])
    assert np.array_equal(q[:, perm], q_p)


def test_dp_matches_simulation_and_beats_heuristic():
    from supplier_pilot import dp
    inst = dual_sourcing_instance(105, 100, 0, 2)
    sol = dp.solve_instance(inst, -10, 20)
    assert sol.converged and sol.g_upper - sol.g_lower < 1e-4
    chk = dp.check_boundaries(sol, inst, T=20000)
    assert abs(chk["rel_diff"]) < 3e-3 and chk["boundary_hit_share"] == 0
    res = simulate(inst, dual_index_policy(inst, 3, 9), seeds=range(4), T=20000, warmup=500)
    assert res.episode_cost.mean() >= sol.g_lower - 0.3


def test_saa_feasible_orders():
    from supplier_pilot.saa import RollingSAAPolicy
    inst = pilot_instances()[1]
    pol = RollingSAAPolicy(M=4, H=6)
    pol.reset(inst, 2)
    st = _random_state(inst, 2, np.random.default_rng(6))
    q = pol.act(inst, st, 0)
    assert np.array_equal(project_feasible(inst, q, st.z), q)


def test_multistage_tree_nonanticipativity():
    """Scenarios share a decision node in period tau < K exactly when they share
    the history (demand and status) up to tau; and orders are feasible."""
    from supplier_pilot.saa import RollingSAAPolicy
    inst = pilot_instances()[3]
    pol = RollingSAAPolicy(tree=(3, 2), H=6)
    rng = np.random.default_rng(7)
    D, Z, node = pol._sample_paths(inst, np.array([1, 0, 1], dtype=np.int8), 6, rng)
    assert pol.M == 6 and len(set(node[:, 0])) == 1
    assert len(set(node[:, 1])) == 3 and len(set(node[:, 2])) == 6
    for s in range(6):
        for s2 in range(6):
            if node[s, 1] == node[s2, 1]:
                assert D[s, 0] == D[s2, 0] and np.array_equal(Z[s, 1], Z[s2, 1])
    assert len(set(node[:, 3])) == 6 and set(node[:, 3]).isdisjoint(set(node[:, 2]))
    pol.reset(inst, 2)
    st = _random_state(inst, 2, np.random.default_rng(8))
    q = pol.act(inst, st, 0)
    assert np.array_equal(project_feasible(inst, q, st.z), q)


def test_random_study_instances_dp_tractable():
    from supplier_pilot.instance import random_study_instance
    from supplier_pilot.dp import build_model
    rng = np.random.default_rng(11)
    for k in range(5):
        inst = random_study_instance(rng, f"R{k}")
        m = build_model(inst, -10, 26)
        assert m.Z * m.R <= 30000, (inst.meta, m.Z * m.R)


def test_factorial_blocking():
    """Cells of one replicate share every non-factor parameter; factors move
    within their level ranges."""
    from itertools import product
    from supplier_pilot.instance import FACTOR_LEVELS, factorial_instance
    insts = [factorial_instance(3, 1, k, s, p) for k, s, p in product(*[list(FACTOR_LEVELS[f]) for f in
                                                                       ("fixed_cost", "shock", "premium")])]
    ref = insts[0]
    for inst in insts:
        assert [s.lead_time for s in inst.suppliers] == [s.lead_time for s in ref.suppliers]
        assert [(s.moq, s.capacity, s.p_fail, s.p_recover) for s in inst.suppliers] == \
               [(s.moq, s.capacity, s.p_fail, s.p_recover) for s in ref.suppliers]
        f = inst.meta["factors"]
        lo, hi = FACTOR_LEVELS["fixed_cost"][f["fixed_cost"]]
        assert all(lo <= s.fixed_cost <= hi + 0.05 for s in inst.suppliers)
        lo, hi = FACTOR_LEVELS["shock"][f["shock"]]
        assert lo <= inst.regional_shock <= hi
        lo, hi = FACTOR_LEVELS["premium"][f["premium"]]
        assert lo <= inst.meta["premium"] <= hi
    assert len({i.name for i in insts}) == 8


def test_gym_env_runs():
    from supplier_pilot.gym_env import SupplierEnv
    env = SupplierEnv(pilot_instances()[2], horizon=50, seed=0)
    obs, info = env.reset(seed=0)
    total = 0.0
    for _ in range(50):
        obs, r, term, trunc, info = env.step(np.array([1, 1, 4]))
        total += r
    assert trunc and obs.shape == (env.obs_dim,) and total < 0


if __name__ == "__main__":
    import inspect
    fns = [f for n, f in sorted(globals().items()) if n.startswith("test_") and inspect.isfunction(f)]
    for f in fns:
        f()
        print(f"PASS {f.__name__}")
    print(f"{len(fns)} tests passed")
