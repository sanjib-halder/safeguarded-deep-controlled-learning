"""PyTorch tests. Run these FIRST on a machine with torch:
    python tests/test_neural.py
They check that the PyTorch path reproduces the validated NumPy simulator,
that the untrained policy equals the tuned heuristic, that the network is
permutation-equivariant, that gradients flow, and that training runs."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from supplier_pilot import batched as bt
from supplier_pilot.instance import factorial_instance, pilot_instances, random_study_instance
from supplier_pilot.policies import MultiIndexPolicy
from supplier_pilot.simulator import make_streams, simulate


def _skip():
    if torch is None:
        print("SKIP: torch not installed")
        return True
    return False


def _instances():
    rng = np.random.default_rng(5)
    return pilot_instances() + [random_study_instance(rng, f"R{k}") for k in range(2)] + \
        [factorial_instance(7, 0, "low", "correlated", "low")]


def test_torch_core_matches_numpy():
    if _skip():
        return
    insts = _instances()
    rng = np.random.default_rng(1)
    idx = np.arange(len(insts)).repeat(3)
    B = len(idx)
    S = rng.integers(2, 14, (B, 3)).astype(float); cap = rng.integers(1, 7, (B, 3)).astype(float)
    trig = rng.integers(1, 5, (B, 3)).astype(float); act = np.ones((B, 3))
    tops = bt.torch_ops("cpu", torch.float64)
    results = {}
    for name, ops in (("np", bt.NumpyOps), ("torch", tops)):
        spec = bt.build_spec(ops, insts, idx)
        I, O, z = bt.initial(ops, spec, B)
        g = np.random.default_rng(3)
        costs = []
        for t in range(300):
            u_d, u, us = g.random(B), g.random((B, 3)), g.random((B, spec.R))
            A = ops.asarray
            q, gt = bt.gmi_decode(ops, spec, I, O, z, A(S), A(cap), A(trig), A(act))
            d = bt.sample_demand(ops, spec, A(u_d))
            c, I, O, z, _ = bt.step(ops, spec, I, O, z, q, gt, d, A(u), A(us))
            costs.append(np.asarray(c.detach().cpu() if hasattr(c, "detach") else c, float))
        results[name] = np.array(costs)
    assert np.allclose(results["np"], results["torch"]), np.abs(results["np"] - results["torch"]).max()


def _gmi_for(inst):
    return MultiIndexPolicy(np.array([4., 8., 13.]), np.array([5., 6., 7.]), np.array([1., 3., 5.]),
                            np.array([True, True, True]), "GMI")


def test_untrained_policy_equals_base_in_numpy_simulator():
    """Zero-initialised network == its base policy, for a GMI base and for a
    status-table base (exact orders in the NumPy simulator)."""
    if _skip():
        return
    from supplier_pilot.learner import StatusTablePolicy
    from supplier_pilot.neural import NeuralPolicy, SupplierSetPolicy, active_of, table_from_gmi
    rng = np.random.default_rng(2)
    for inst in _instances()[2:5]:
        gmi = _gmi_for(inst)
        streams = make_streams(inst, range(4), 600)
        r1 = simulate(inst, gmi, streams=streams, warmup=0)
        r2 = simulate(inst, NeuralPolicy(SupplierSetPolicy(), table_from_gmi(gmi), active_of(gmi)),
                      streams=streams, warmup=0)
        assert np.array_equal(r1.orders, r2.orders), inst.name
        assert np.allclose(r1.cost, r2.cost)
        table = table_from_gmi(gmi) + rng.integers(-2, 3, size=(8, 9))
        table[:, 3:] = np.maximum(table[:, 3:], 1)
        tab = StatusTablePolicy(gmi, table)
        r3 = simulate(inst, tab, streams=streams, warmup=0)
        r4 = simulate(inst, NeuralPolicy(SupplierSetPolicy(), table, active_of(gmi)), streams=streams, warmup=0)
        assert np.array_equal(r3.orders, r4.orders), inst.name


def test_adjustments_are_bounded():
    if _skip():
        return
    from supplier_pilot.neural import BaseParams, SupplierSetPolicy, active_of, policy_params, table_from_gmi
    inst = _instances()[3]
    gmi = _gmi_for(inst)
    net = SupplierSetPolicy()
    torch.nn.init.normal_(net.head[-1].weight, std=50.0)     # deliberately extreme outputs
    ops = bt.torch_ops("cpu")
    spec = bt.build_spec(ops, [inst], np.zeros(32, dtype=int))
    base = BaseParams.build([table_from_gmi(gmi)], [active_of(gmi)], np.zeros(32, dtype=int), "cpu")
    I = torch.randint(-5, 15, (32,)).float(); O = torch.randint(0, 5, (32, 3, spec.Lmax)).float()
    z = torch.randint(0, 2, (32, 3)).float()
    with torch.no_grad():
        S, cap, trig = policy_params(net, spec, base, I, O, z, bound=2.0)
    S0, cap0, trig0 = base.at(z)
    assert torch.all((S - S0).abs() <= 2.0 + 1e-5)
    assert torch.all(cap >= 1) and torch.all(trig >= 1)


def test_network_permutation_equivariance():
    if _skip():
        return
    from supplier_pilot.neural import F_GLOB, F_SUP, SupplierSetPolicy
    torch.manual_seed(0)
    net = SupplierSetPolicy()
    torch.nn.init.normal_(net.head[-1].weight, std=0.1)
    sup, glob = torch.randn(5, 3, F_SUP), torch.randn(5, F_GLOB)
    perm = torch.tensor([2, 0, 1])
    out = net(sup, glob)
    out_p = net(sup[:, perm], glob)
    assert torch.allclose(out[:, perm], out_p, atol=1e-5)


def test_gradients_flow_and_training_runs():
    if _skip():
        return
    from supplier_pilot.neural import TrainConfig, active_of, table_from_gmi, train
    inst = factorial_instance(7, 0, "low", "correlated", "low")
    gmi = _gmi_for(inst)
    cfg = TrainConfig(episodes_per_instance=16, horizon=12, burn_in=30, iters=6, val_every=3,
                      val_episodes=4, val_periods=60, log_every=1, device="cpu")
    res = train([inst], [table_from_gmi(gmi)], [active_of(gmi)], [30.0], cfg, log=lambda *a: None)
    gn = [h["grad_norm"] for h in res.history if "grad_norm" in h]
    assert len(gn) == 6 and gn[0] > 0, gn
    assert any("val_rel_to_base" in h for h in res.history)
    # separate validation instances (generalisation mode)
    inst2 = factorial_instance(7, 1, "low", "correlated", "high")
    res2 = train([inst], [table_from_gmi(gmi)], [active_of(gmi)], [30.0], cfg, log=lambda *a: None,
                 val_set=([inst2], [table_from_gmi(gmi)], [active_of(gmi)], [30.0]))
    assert res2.best_val_rel <= 0.0


if __name__ == "__main__":
    tests = (test_torch_core_matches_numpy, test_untrained_policy_equals_base_in_numpy_simulator,
             test_adjustments_are_bounded, test_network_permutation_equivariance,
             test_gradients_flow_and_training_runs)
    for f in tests:
        f()
        print(f"PASS {f.__name__}" if torch is not None else "")
    if torch is not None:
        print(f"{len(tests)} torch tests passed")
