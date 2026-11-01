"""Tests for simulation-based training (DCL).
NumPy tests run anywhere; the PyTorch tests are skipped without torch.
Run: python tests/test_dcl.py"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from supplier_pilot import batched as bt
from supplier_pilot import dcl
from supplier_pilot.instance import factorial_instance
from supplier_pilot.learner import StatusTablePolicy
from supplier_pilot.policies import MultiIndexPolicy
from supplier_pilot.simulator import make_streams, simulate

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def _setup(rep=0, cell=("low", "correlated", "high")):
    inst = factorial_instance(7, rep, *cell)
    gmi = MultiIndexPolicy(np.array([4., 8., 13.]), np.array([5., 6., 7.]), np.array([1., 3., 5.]), None, "GMI")
    rng = np.random.default_rng(rep)
    table = np.tile(gmi.vector(), (8, 1)) + rng.integers(-1, 2, (8, 9))
    table[:, 3:] = np.maximum(table[:, 3:], 1)
    return inst, gmi, table


def _states(bases, S, seed=0):
    return dcl.collect_states(bases, [0], lambda spec, idx, I, O, z: bases.orders(spec, idx, I, O, z),
                              episodes=8, burn_in=50, periods=S // 8 * 4, every=4, seed=seed)


def test_candidates_feasible_and_base_reproduced():
    inst, gmi, table = _setup()
    bases = dcl.Bases.build([inst], [table], [np.ones(3)])
    st = _states(bases, 64)
    S = len(st["I"])
    spec = bases.spec(np.zeros(S, dtype=int))
    b, _ = bases.orders(spec, np.zeros(S, dtype=int), st["I"], st["O"], st["z"])
    Q, bc = dcl.class_quantities(spec, b, st["z"])
    m, cap = np.maximum(spec.moq, 1), spec.capacity
    assert np.all((Q == 0) | ((Q >= m[:, :, None]) & (Q <= cap[:, :, None])))
    assert np.all(Q[st["z"] < 0.5] == 0)
    assert np.array_equal(np.take_along_axis(Q, bc[:, :, None], 2)[..., 0], b)


def test_base_orders_match_status_table_policy():
    """Bases.orders == StatusTablePolicy in the validated simulator."""
    inst, gmi, table = _setup(1)
    bases = dcl.Bases.build([inst], [table], [np.ones(3)])
    tab = StatusTablePolicy(gmi, table)
    streams = make_streams(inst, range(3), 300)
    r1 = simulate(inst, tab, streams=streams, warmup=0)

    class P:
        name = "b"
        def act(self, inst_, st, t):
            spec = bases.spec(np.zeros(st.I.shape[0], dtype=int))
            I, O, z = bt.state_from_numpy(bt.NumpyOps, spec, st)
            return bases.orders(spec, np.zeros(len(I), dtype=int), I, O, z)[0].astype(np.int64)
    r2 = simulate(inst, P(), streams=streams, warmup=0)
    assert np.array_equal(r1.orders, r2.orders)


def test_common_random_numbers_and_labels():
    """Identical candidates get identical costs (CRN), labels are feasible, and
    non-base labels only appear with a positive estimated advantage."""
    inst, gmi, table = _setup(2)
    bases = dcl.Bases.build([inst], [table], [np.ones(3)])
    st = _states(bases, 32, seed=3)
    S = len(st["I"])
    idx = np.zeros(S, dtype=int)
    spec = bases.spec(idx)
    b, _ = bases.orders(spec, idx, st["I"], st["O"], st["z"])
    Q, bc = dcl.class_quantities(spec, b, st["z"])
    costs, combos = dcl.rollout_scores(bases, idx, st["I"], st["O"], st["z"], Q, M=4, H=12, seed=1)
    qv = Q[:, np.arange(3)[None, :], combos]                       # (S, C, N)
    for s in range(min(S, 8)):
        _, first = np.unique(qv[s], axis=0, return_inverse=True)
        first = first.ravel()
        for grp in np.unique(first):
            members = np.flatnonzero(first == grp)
            assert np.allclose(costs[s, members], costs[s, members[0]])
    lab = dcl.label_states(bases, idx, st["I"], st["O"], st["z"], M=4, H=12, seed=1)
    m, cap = np.maximum(spec.moq, 1), spec.capacity
    q = lab.q_label
    assert np.all((q == 0) | ((q >= m) & (q <= cap)))
    assert np.all(lab.advantage[lab.changed] > 0) and np.all(lab.advantage[~lab.changed] == 0)


def test_mixed_instances_batching():
    """Labelling a mixed-instance batch equals labelling each instance alone."""
    i1, g1, t1 = _setup(0)
    i2, g2, t2 = _setup(3, ("low", "correlated", "low"))
    both = dcl.Bases.build([i1, i2], [t1, t2], [np.ones(3), np.ones(3)])
    b1 = dcl.Bases.build([i1], [t1], [np.ones(3)])
    st = _states(b1, 16, seed=5)
    O = np.zeros((len(st["I"]), 3, both.spec1.Lmax)); O[:, :, :st["O"].shape[2]] = st["O"]
    idx = np.zeros(len(st["I"]), dtype=int)
    q_one, _ = b1.orders(b1.spec(idx), idx, st["I"], st["O"], st["z"])
    q_two, _ = both.orders(both.spec(idx), idx, st["I"], O, st["z"])
    assert np.array_equal(q_one, q_two)


def test_fresh_instances_never_share_cache():
    """Training instances drawn with different seeds get different names, and a
    cached baseline is rejected if it belongs to another instance."""
    import types
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
    from train_neural import _cache_valid, build_fresh, fingerprint
    f = lambda s: build_fresh(types.SimpleNamespace(target="factorial-hard", train_seed=s), [0], "T", 400_000)[0][0]
    i101, i202 = f(101), f(202)
    assert i101.name != i202.name and fingerprint(i101) != fingerprint(i202)
    assert _cache_valid({"fingerprint": fingerprint(i101)}, i101)
    assert not _cache_valid({"fingerprint": fingerprint(i101)}, i202)
    assert not _cache_valid({"unavoidable": i101.unavoidable_cost() + 1.0, "g": 0.0}, i101)


def test_scaled_instances_and_core_reference():
    """N = 4 / 5 instances: D and E sit in region 2; the A-B-C core's DP policy
    runs in the full instance and never orders from D or E."""
    from supplier_pilot.instance import core_instance, scaled_instance
    from supplier_pilot import dp
    from supplier_pilot.policies import SubsetPolicy
    for n in (4, 5):
        inst = scaled_instance(7, 2, n, "low", "correlated", "high")
        assert inst.n == n and inst.name == f"N{n}_02_lch"
        assert list(inst.regions[:3]) == [0, 1, 1] and all(r == 2 for r in inst.regions[3:])
        same_cells = scaled_instance(7, 2, n, "low", "correlated", "low")
        assert [s.lead_time for s in same_cells.suppliers] == [s.lead_time for s in inst.suppliers]
    core = core_instance(inst)
    sol = dp.solve_instance(core, -10, 26)
    r = simulate(inst, SubsetPolicy(dp.DPPolicy(sol), core, [0, 1, 2]), streams=make_streams(inst, range(4), 300),
                 warmup=50)
    assert r.orders[..., 3:].sum() == 0 and r.orders[..., :3].sum() > 0


def test_coordinate_search_labels_n5():
    """Coordinate-search labelling at N = 5: labels are feasible, a non-base label
    needs a positive advantage, and the candidate set grows linearly in N."""
    from supplier_pilot.instance import scaled_instance
    inst = scaled_instance(7, 0, 5, "low", "correlated", "low")
    gmi = MultiIndexPolicy(np.array([4., 8., 13., 9., 14.]), np.array([3., 4., 5., 3., 6.]),
                           np.array([1., 3., 5., 2., 5.]), None, "GMI")
    table = np.tile(gmi.vector(), (32, 1))
    bases = dcl.Bases.build([inst], [table], [np.ones(5)])
    st = _states(bases, 32, seed=2)
    idx = np.zeros(len(st["I"]), dtype=int)
    assert dcl.coordinate_candidates(np.zeros((3, 5), dtype=int), 6).shape == (3, 1 + 5 * 6, 5)
    lab = dcl.label_states(bases, idx, st["I"], st["O"], st["z"], M=4, H=12, seed=1, search="coord")
    spec = bases.spec(idx)
    m, cap = np.maximum(spec.moq, 1), spec.capacity
    assert np.all((lab.q_label == 0) | ((lab.q_label >= m) & (lab.q_label <= cap)))
    assert np.all(lab.q_label[st["z"] < 0.5] == 0)
    assert np.all(lab.advantage[lab.changed] > 0) and np.all(lab.advantage[~lab.changed] == 0)


# ------------------------------------------------------------------ torch
def test_untrained_dcl_equals_status_table():
    if torch is None:
        return
    from supplier_pilot import dcl_torch as dt
    inst, gmi, table = _setup(4)
    bases = dcl.Bases.build([inst], [table], [np.ones(3)])
    streams = make_streams(inst, range(3), 400)
    r1 = simulate(inst, StatusTablePolicy(gmi, table), streams=streams, warmup=0)
    r2 = simulate(inst, dt.DCLPolicy(dt.DCLNet(len(dcl.DELTAS) + 1), bases, 0), streams=streams, warmup=0)
    assert np.array_equal(r1.orders, r2.orders)


def test_dcl_net_equivariance():
    if torch is None:
        return
    from supplier_pilot import dcl_torch as dt
    torch.manual_seed(0)
    net = dt.DCLNet(6)
    torch.nn.init.normal_(net.body.head[-1].weight, std=0.2)
    sup, glob = torch.randn(5, 3, dt.F_SUP_DCL), torch.randn(5, dt.F_GLOB)
    bc = torch.randint(0, 6, (5, 3))
    perm = torch.tensor([2, 0, 1])
    assert torch.allclose(net(sup, glob, bc)[:, perm], net(sup[:, perm], glob, bc[:, perm]), atol=1e-5)


def test_dcl_fit_and_iteration_policy():
    if torch is None:
        return
    from supplier_pilot import dcl_torch as dt
    inst, gmi, table = _setup(5)
    bases = dcl.Bases.build([inst], [table], [np.ones(3)])
    st = _states(bases, 96, seed=7)
    lab = dcl.label_states(bases, np.zeros(len(st["I"]), dtype=int), st["I"], st["O"], st["z"], M=4, H=12)
    st["inst"] = np.zeros(len(st["I"]), dtype=int)
    tr = dt.tensorize(bases, st, lab, "cpu")
    res = dt.fit(tr, tr, 6, epochs=5, log=lambda *a: None)
    assert res.val_acc >= res.history[0]["val_acc"] - 1e-9
    fn = dt.policy_fn_for({None: res.net}, bases, device="cpu")
    st2 = dcl.collect_states(bases, [0], fn, episodes=4, burn_in=10, periods=20, every=5)
    assert len(st2["I"]) == 16


def test_dcl_confidence_threshold():
    if torch is None:
        return
    from supplier_pilot.dcl_torch import confident_classes
    logits = torch.log(torch.tensor([[[0.1, 0.6, 0.3], [0.47, 0.1, 0.43], [0.9, 0.05, 0.05]]]))
    base = torch.tensor([[2, 2, 1]])
    assert confident_classes(logits, base, 0.0).tolist() == [[1, 0, 0]]
    assert confident_classes(logits, base, 0.5).tolist() == [[1, 2, 0]]
    assert confident_classes(logits, base, 0.95).tolist() == [[2, 2, 1]]


def test_dcl_ensemble():
    """Ensemble of identical nets = that net; 'agree' needs all members to pick
    the same non-base class; an untrained ensemble = the status table."""
    if torch is None:
        return
    from supplier_pilot import dcl_torch as dt
    torch.manual_seed(1)
    nets = []
    for _ in range(3):
        n = dt.DCLNet(6)
        torch.nn.init.normal_(n.body.head[-1].weight, std=0.5)
        nets.append(n.eval())
    sup, glob = torch.randn(64, 3, dt.F_SUP_DCL), torch.randn(64, dt.F_GLOB)
    bc = torch.randint(0, 6, (64, 3))
    with torch.no_grad():
        single = dt.confident_classes(nets[0](sup, glob, bc), bc, 0.5)
        same = dt.DCLEnsemble([nets[0]] * 3, "mean")
        assert torch.equal(dt.confident_classes(same(sup, glob, bc), bc, 0.5), single)
        agree1 = dt.DCLEnsemble([nets[0]], "agree", [0.5])
        assert torch.equal(dt.confident_classes(agree1(sup, glob, bc), bc, 0.0), single)
        each = torch.stack([dt.confident_classes(n(sup, glob, bc), bc, 0.0) for n in nets])
        agree = dt.confident_classes(dt.DCLEnsemble(nets, "agree")(sup, glob, bc), bc, 0.0)
        unanimous = (each == each[0]).all(0)
        assert torch.equal(agree[unanimous], each[0][unanimous])
        assert torch.equal(agree[~unanimous], bc[~unanimous])
    inst, gmi, table = _setup(6)
    bases = dcl.Bases.build([inst], [table], [np.ones(3)])
    streams = make_streams(inst, range(3), 300)
    r1 = simulate(inst, StatusTablePolicy(gmi, table), streams=streams, warmup=0)
    ens = dt.DCLEnsemble([dt.DCLNet(6), dt.DCLNet(6)], "mean")
    r2 = simulate(inst, dt.DCLPolicy(ens, bases, 0, threshold=0.5), streams=streams, warmup=0)
    assert np.array_equal(r1.orders, r2.orders)


def test_dcl_without_prior():
    """Ablation network: no base-action prior, so the untrained net does not
    reproduce the table (it orders nothing), and it still trains and saves/loads."""
    if torch is None:
        return
    import tempfile
    from supplier_pilot import dcl_torch as dt
    net = dt.DCLNet(6, prior=None)
    assert net.prior is None
    sup, glob = torch.randn(4, 3, dt.F_SUP_DCL), torch.randn(4, dt.F_GLOB)
    bc = torch.full((4, 3), 3)
    assert torch.equal(net(sup, glob, bc).argmax(2), torch.zeros(4, 3, dtype=torch.long))
    path = os.path.join(tempfile.mkdtemp(), "n.pt")
    torch.save(net.state_dict(), path)
    assert dt.load_net(path, 6, prior=None).prior is None


if __name__ == "__main__":
    import inspect
    fns = [f for n, f in sorted(globals().items()) if n.startswith("test_") and inspect.isfunction(f)]
    for f in fns:
        f()
        is_torch = "dcl" in f.__name__ and any(k in f.__name__ for k in ("untrained", "net", "fit", "confidence", "ensemble", "prior"))
        print(("SKIP " if (is_torch and torch is None) else "PASS ") + f.__name__)
    print("done")
