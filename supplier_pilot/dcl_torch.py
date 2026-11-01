"""DCL classifier (PyTorch): imitate the simulation-labelled best actions.

The network is the equivariant set transformer of ``neural.py`` with K+1
outputs per supplier (one per order class, see ``dcl.py``) and one extra
per-supplier input (the base policy's order). Logits get a learnable prior
on the base action's class:

    logits_i = net_i(state) + prior * onehot(base class_i)

The head starts at zero, so the untrained classifier picks the base action
(the status table) everywhere; training moves it only where the simulation
labels say so. Orders are always taken from the feasible class quantities,
so every action is feasible by construction.
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from . import batched as bt
from . import dcl
from .neural import F_GLOB, F_SUP, BaseParams, SupplierSetPolicy, features, pick_device

F_SUP_DCL = F_SUP + 1


class DCLNet(nn.Module):
    def __init__(self, k_classes: int, d: int = 64, layers: int = 2, prior: float | None = 3.0):
        """prior = None removes the base-action prior (ablation): the network must
        then learn the base action itself and, untrained, orders nothing."""
        super().__init__()
        self.body = SupplierSetPolicy(d, layers=layers, n_out=k_classes, f_sup=F_SUP_DCL)
        self.prior = None if prior is None else nn.Parameter(torch.tensor(float(prior)))
        self.k = k_classes

    def forward(self, sup, glob, base_cls):
        out = self.body(sup, glob)
        if self.prior is None:
            return out
        return out + self.prior * nn.functional.one_hot(base_cls.long(), self.k).float()


def dcl_features(tspec, tbase, I, O, z, b):
    """Network inputs: the features of neural.py plus the base order (B, N)."""
    sup, glob = features(tspec, tbase, I, O, z)
    return torch.cat([sup, (b / tspec.mu[:, None] / 2)[:, :, None]], dim=2), glob


@dataclass
class StateBatch:
    """Labelled states as torch tensors (inputs + targets)."""
    sup: torch.Tensor
    glob: torch.Tensor
    base_cls: torch.Tensor
    target: torch.Tensor
    weight: torch.Tensor


def tensorize(bases: dcl.Bases, states: dict, labels: dcl.Labels, device: str, adv_weight: float = 0.0):
    ops = bt.torch_ops(device)
    idx = states["inst"]
    tspec = bt.build_spec(ops, bases.instances, idx)
    tbase = BaseParams.build(list(bases.tables), list(bases.actives), idx, device)
    T = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)
    sup, glob = dcl_features(tspec, tbase, T(states["I"]), T(states["O"]), T(states["z"]), T(labels.q_base))
    w = np.ones(len(idx))
    if adv_weight > 0:          # optionally emphasise states where the choice matters
        w = 1 + adv_weight * labels.advantage / max(labels.advantage.mean(), 1e-9)
    return StateBatch(sup, glob, torch.as_tensor(labels.base_classes, device=device),
                      torch.as_tensor(labels.classes, device=device), T(w))


@dataclass
class FitResult:
    net: DCLNet
    history: list = field(default_factory=list)
    val_acc: float = 0.0
    seconds: float = 0.0


def fit(train: StateBatch, val: StateBatch, k_classes: int, net: DCLNet | None = None, epochs: int = 40,
        batch: int = 512, lr: float = 1e-3, weight_decay: float = 1e-5, d: int = 64, layers: int = 2,
        seed: int = 0, log=print, prior: float | None = 3.0) -> FitResult:
    """Supervised classification (cross-entropy per supplier). Keeps the epoch
    with the best validation loss."""
    torch.manual_seed(seed)
    device = train.sup.device
    net = net or DCLNet(k_classes, d, layers, prior=prior).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    n = train.sup.shape[0]
    t0 = time.time()

    def loss_on(sb, idx=None):
        s = slice(None) if idx is None else idx
        logits = net(sb.sup[s], sb.glob[s], sb.base_cls[s])                 # (B, N, K)
        ce = nn.functional.cross_entropy(logits.reshape(-1, k_classes), sb.target[s].reshape(-1).long(),
                                         reduction="none").view(logits.shape[0], -1).mean(1)
        acc = (logits.argmax(2) == sb.target[s]).all(1).float().mean()
        return (ce * sb.weight[s]).mean(), float(acc)

    with torch.no_grad():
        best_loss, best_acc = loss_on(val)
    best = (float(best_loss), copy.deepcopy(net.state_dict()))
    hist = [{"epoch": 0, "val_loss": float(best_loss), "val_acc": best_acc}]
    for ep in range(1, epochs + 1):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch):
            loss, _ = loss_on(train, perm[i:i + batch])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        with torch.no_grad():
            vl, va = loss_on(val)
        hist.append({"epoch": ep, "val_loss": float(vl), "val_acc": va})
        if float(vl) < best[0]:
            best = (float(vl), copy.deepcopy(net.state_dict()))
            best_acc = va
        if ep % 10 == 0 or ep == epochs:
            log(f"    epoch {ep:3d}  val loss {float(vl):.4f}  val joint accuracy {va:.3f}  ({time.time() - t0:.0f}s)")
    net.load_state_dict(best[1])
    return FitResult(net, hist, best_acc, time.time() - t0)


def confident_classes(logits, base_cls, threshold: float):
    """Argmax class per supplier, but keep the base class unless the predicted
    class has probability above ``threshold`` (0 = plain argmax)."""
    prob = torch.softmax(logits, dim=2)
    p, cls = prob.max(2)
    if threshold > 0:
        cls = torch.where((cls != base_cls) & (p <= threshold), base_cls.long(), cls)
    return cls


class DCLEnsemble(nn.Module):
    """Several trained classifiers (e.g. trained on different instance sets)
    combined into one. Its output plugs into DCLPolicy like a single DCLNet.

    mode "mean":  average the members' class probabilities (soft vote); the
                  policy's confidence threshold then applies to the average.
    mode "agree": each member picks its class with its own threshold; deviate
                  from the base class only if all members pick the same class.
    """

    def __init__(self, nets, mode: str = "mean", thresholds=None):
        super().__init__()
        if mode not in ("mean", "agree"):
            raise ValueError(mode)
        self.members = nn.ModuleList(nets)
        self.mode = mode
        self.k = nets[0].k
        self.thresholds = [0.0] * len(nets) if thresholds is None else [float(t) for t in thresholds]

    def forward(self, sup, glob, base_cls):
        outs = [m(sup, glob, base_cls) for m in self.members]
        if self.mode == "mean":
            p = torch.stack([torch.softmax(o, dim=2) for o in outs]).mean(0)
            return torch.log(p.clamp_min(1e-12))
        cls = torch.stack([confident_classes(o, base_cls, th) for o, th in zip(outs, self.thresholds)])
        pick = torch.where((cls == cls[0]).all(0), cls[0], base_cls.long())
        return 30.0 * nn.functional.one_hot(pick, self.k).float()


def load_net(path: str, k_classes: int, prior: float | None = 3.0) -> DCLNet:
    net = DCLNet(k_classes, prior=prior)
    net.load_state_dict(torch.load(path, map_location="cpu"))
    return net.eval()


class DCLPolicy:
    """Trained classifier as a policy for the NumPy simulator (exact, feasible)."""

    def __init__(self, net: DCLNet, bases: dcl.Bases, inst_k: int, deltas=dcl.DELTAS,
                 name: str = "DCL (state-aware)", device: str = "cpu", threshold: float = 0.0):
        self.net = net.to(device).eval()
        self.bases, self.k, self.deltas, self.name, self.device = bases, inst_k, deltas, name, device
        self.threshold = threshold      # deviate from the base action only if P(class) > threshold
        self.ops = bt.torch_ops(device)
        self._E = None

    def reset(self, inst, E):
        self._E = E
        self.idx = np.full(E, self.k)
        self.spec = self.bases.spec(self.idx)
        self.tspec = bt.build_spec(self.ops, self.bases.instances, self.idx)
        self.tbase = BaseParams.build(list(self.bases.tables), list(self.bases.actives), self.idx, self.device)

    def orders_from_arrays(self, I, O, z):
        """I (E,), O (E, N, Lmax), z (E, N) NumPy -> order vectors (E, N)."""
        b, _ = self.bases.orders(self.spec, self.idx, I, O, z)
        Q, base_cls = dcl.class_quantities(self.spec, b, z, self.deltas)
        T = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            sup, glob = dcl_features(self.tspec, self.tbase, T(I), T(O), T(z), T(b))
            bc = torch.as_tensor(base_cls, device=self.device)
            cls = confident_classes(self.net(sup, glob, bc), bc, self.threshold).cpu().numpy()
        return np.take_along_axis(Q, cls[:, :, None], 2)[..., 0]

    def act(self, inst, st, t):
        if self._E != st.I.shape[0]:
            self.reset(inst, st.I.shape[0])
        I, O, z = bt.state_from_numpy(bt.NumpyOps, self.spec, st)
        return np.rint(self.orders_from_arrays(I, O, z)).astype(np.int64)


def policy_fn_for(nets: dict, bases: dcl.Bases, deltas=dcl.DELTAS, device: str = "cpu", threshold: float = 0.0):
    """``policy_fn(spec, idx, I, O, z)`` for dcl.collect_states using trained
    networks (``nets[instance index]``, or one shared net under key None)."""
    ops = bt.torch_ops(device)
    cache = {}

    def fn(spec, idx, I, O, z):
        key = len(idx)
        if key not in cache:
            cache[key] = (bt.build_spec(ops, bases.instances, idx),
                          BaseParams.build(list(bases.tables), list(bases.actives), idx, device))
        tspec, tbase = cache[key]
        b, _ = bases.orders(spec, idx, I, O, z)
        Q, base_cls = dcl.class_quantities(spec, b, z, deltas)
        T = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)
        with torch.no_grad():
            sup, glob = dcl_features(tspec, tbase, T(I), T(O), T(z), T(b))
            bc = torch.as_tensor(base_cls, device=device)
            if None in nets:
                cls = confident_classes(nets[None](sup, glob, bc), bc, threshold)
            else:
                cls = torch.empty_like(bc)
                for k, net in nets.items():
                    m = torch.as_tensor(idx == k, device=device)
                    if m.any():
                        cls[m] = confident_classes(net(sup[m], glob[m], bc[m]), bc[m], threshold)
            cls = cls.cpu().numpy()
        q = np.take_along_axis(Q, cls[:, :, None], 2)[..., 0]
        return q, (q > 0).astype(float)
    return fn
