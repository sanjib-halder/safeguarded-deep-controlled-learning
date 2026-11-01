"""The batched core (NumPy backend) must reproduce the validated simulator
exactly. Run with `pytest -q` or `python tests/test_batched.py`."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from supplier_pilot import batched as bt
from supplier_pilot.instance import factorial_instance, pilot_instances, random_study_instance
from supplier_pilot.policies import MultiIndexPolicy
from supplier_pilot.simulator import make_streams, simulate

OPS = bt.NumpyOps


def _run_batched(insts, inst_idx, params, streams_list, T):
    """Simulate GMI with the batched core. streams_list[k] are the simulator
    streams of the k-th episode (E=1 each)."""
    spec = bt.build_spec(OPS, insts, inst_idx)
    B = len(inst_idx)
    I, O, z = bt.initial(OPS, spec, B)
    S = np.stack([params[k][0] for k in inst_idx]); cap = np.stack([params[k][1] for k in inst_idx])
    trig = np.stack([params[k][2] for k in inst_idx]); act = np.stack([params[k][3] for k in inst_idx])
    costs = np.zeros((B, T)); orders = np.zeros((B, T, spec.n))
    for t in range(T):
        q, g = bt.gmi_decode(OPS, spec, I, O, z, S, cap, trig, act)
        d = np.array([s.D[0, t] for s in streams_list], float)
        u = np.stack([s.U[0, t] for s in streams_list])
        Rm = spec.R
        us = np.stack([np.pad(s.S[0, t], (0, Rm - len(s.S[0, t])), constant_values=2.0) if s.S is not None
                       else np.full(Rm, 2.0) for s in streams_list])
        c, I, O, z, _ = bt.step(OPS, spec, I, O, z, q, g, d, u, us)
        costs[:, t] = c; orders[:, t] = q
    return costs, orders


def test_batched_matches_simulator_mixed_instances():
    rng = np.random.default_rng(5)
    insts = pilot_instances() + [random_study_instance(rng, f"R{k}") for k in range(3)] + \
        [factorial_instance(7, 0, "low", "correlated", "low")]
    params = []
    for inst in insts:
        S = rng.integers(2, 14, inst.n).astype(float)
        cap = rng.integers(1, 7, inst.n).astype(float)
        trig = rng.integers(1, 5, inst.n).astype(float)
        act = (rng.random(inst.n) < 0.85).astype(float); act[0] = 1
        params.append((S, cap, trig, act))
    T = 400
    inst_idx = list(range(len(insts))) * 2
    streams = [make_streams(insts[k], [900 + b], T) for b, k in enumerate(inst_idx)]
    costs, orders = _run_batched(insts, inst_idx, params, streams, T)
    for b, k in enumerate(inst_idx):
        pol = MultiIndexPolicy(*params[k][:3], params[k][3].astype(bool))
        res = simulate(insts[k], pol, streams=streams[b], warmup=0)
        assert np.allclose(res.cost[0], costs[b]), (insts[k].name, np.abs(res.cost[0] - costs[b]).max())
        assert np.array_equal(res.orders[0], orders[b].astype(int))


def test_sample_demand_matches_streams():
    from supplier_pilot.instance import poisson_demand, Instance, Supplier
    vals, probs = poisson_demand(3.0)
    inst = Instance("pois", (Supplier("a", 100, 1, capacity=8),), vals, probs)
    spec = bt.build_spec(OPS, [inst], [0] * 5000)
    u = np.random.default_rng(0).random(5000)
    d = bt.sample_demand(OPS, spec, u)
    cdf = np.cumsum(probs)
    ref = np.asarray(vals)[np.minimum(np.searchsorted(cdf, u, side="right"), len(vals) - 1)]
    assert np.array_equal(d, ref)


if __name__ == "__main__":
    test_batched_matches_simulator_mixed_instances(); print("PASS batched == simulator (mixed instances)")
    test_sample_demand_matches_streams(); print("PASS demand sampling")
