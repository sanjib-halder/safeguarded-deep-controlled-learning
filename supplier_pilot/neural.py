"""State-aware neural policy (PyTorch), trained by pathwise gradients.

Policy
------
An equivariant set encoder reads every supplier's features (static data,
disruption status, its own pipeline, the base policy's parameters) plus a
global token (inventory, total pipeline). Self-attention without positional
encodings makes the per-supplier outputs permutation-equivariant. For each
supplier the head outputs *bounded residual adjustments* to a base policy's
parameters (order-up-to level S_i, cap_i, trigger r_i):

    S_i = S0_i(z) + b*tanh(o_i1/b),  cap_i = max(cap0_i(z) + b*tanh(o_i2/b), 1), ...

The base is a status table (parameters per joint disruption status z, e.g.
the tuned status-table learner) or a plain GMI heuristic (identical rows).
The bound b (default 2 units) keeps adjustments small and stops wild
extrapolation on unfamiliar instances. The head's last layer starts at zero,
so the untrained policy *is* the base policy. Orders come from the exact GMI decoder in ``batched.py``, so every
action is feasible by construction and stays interpretable as a state-
dependent multi-index rule. Unlike the status-table learner, the adjustments
depend on inventory and pipeline, not only on which suppliers are up.

Training (model-exploiting)
---------------------------
Pathwise gradients through the known dynamics (as in Alvo et al.'s HDPO):
the batched simulator is differentiable with straight-through estimators for
rounding and order gates. Episodes continue across iterations (truncated
backpropagation through time). The loss is the average period cost, scaled
per instance by the heuristic's excess cost so instances count equally.
The best iterate is chosen on fixed validation streams; final policy
selection against the heuristic happens on the NumPy simulator.
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from . import batched as bt
from .instance import Instance
from .policies import INF, MultiIndexPolicy

PIPE_SLOTS = 4      # pipeline slots fed to the network (lead times up to 5)
F_SUP = 1 + PIPE_SLOTS + 13
F_GLOB = 4


def pick_device(name: str = "auto") -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"      # MPS works too (pass --device mps) but is rarely faster at this size


# --------------------------------------------------------------- base params
def table_from_gmi(gmi: MultiIndexPolicy) -> np.ndarray:
    """A status table whose rows all equal the GMI parameters."""
    n = len(gmi.S)
    return np.tile(np.concatenate([gmi.S, gmi.cap, gmi.trig]).astype(float), (1 << n, 1))


def active_of(gmi: MultiIndexPolicy) -> np.ndarray:
    return np.ones(len(gmi.S)) if gmi.active is None else np.asarray(gmi.active, float)


@dataclass
class BaseParams:
    """Base policy the network adjusts: a status table (one row of GMI
    parameters [S | cap | trig] per joint disruption status), broadcast to the
    batch. A plain GMI heuristic is the special case with identical rows."""
    table: torch.Tensor      # (B, 2^N, 3N)
    active: torch.Tensor     # (B, N)
    pow2: torch.Tensor       # (N,)

    @classmethod
    def build(cls, tables: list[np.ndarray], actives: list[np.ndarray], inst_idx, device,
              cap_clip: float = 50.0):
        tab = np.stack([np.asarray(tables[k], float) for k in inst_idx]).copy()
        n = tab.shape[2] // 3
        tab[:, :, n:2 * n] = np.minimum(tab[:, :, n:2 * n], cap_clip)
        act = np.stack([np.asarray(actives[k], float) for k in inst_idx])
        f = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)
        return cls(f(tab), f(act), f(2.0 ** np.arange(n)))

    def at(self, z):
        """Base (S, cap, trig), each (B, N), for the current statuses z (B, N)."""
        n = self.active.shape[1]
        zid = (z * self.pow2).sum(1).round().long()
        row = self.table[torch.arange(z.shape[0], device=z.device), zid]
        return row[:, :n], row[:, n:2 * n], row[:, 2 * n:]


# ------------------------------------------------------------------ features
def features(spec: bt.BatchSpec, base: BaseParams, I, O, z):
    mu = spec.mu[:, None]                                        # (B, 1)
    B, N = z.shape
    S0, cap0, trig0 = base.at(z)
    pipe = O[:, :, :PIPE_SLOTS]
    if pipe.shape[2] < PIPE_SLOTS:
        pipe = torch.cat([pipe, torch.zeros(B, N, PIPE_SLOTS - pipe.shape[2], device=O.device)], 2)
    lead = torch.einsum("bp,bpi->bi", spec.L_ord.float(), spec.perm)
    static = torch.stack([
        spec.price / 100.0 - 1.0, lead / 4.0, spec.K / 50.0, spec.moq / mu, spec.capacity.clamp(max=50) / mu / 4,
        spec.pf * 10.0, spec.pr, spec.shock[:, None].expand(B, N) * 20.0,
        spec.imm,
        S0 / mu / 4, cap0 / mu / 4, trig0 / mu / 4, base.active], dim=2)
    sup = torch.cat([z[:, :, None], pipe / mu[:, :, None] / 2, static], dim=2)    # (B, N, F_SUP)
    tot_pipe = O.sum((1, 2))
    glob = torch.stack([I / spec.mu / 5, tot_pipe / spec.mu / 5, (I + tot_pipe) / spec.mu / 5,
                        z.mean(1)], dim=1)                       # (B, F_GLOB)
    return sup, glob


# ---------------------------------------------------------------------- net
class SupplierSetPolicy(nn.Module):
    def __init__(self, d: int = 64, heads: int = 4, layers: int = 2, out_scale: float = 1.0,
                 n_out: int = 3, f_sup: int = F_SUP):
        super().__init__()
        self.out_scale = out_scale
        self.sup = nn.Sequential(nn.Linear(f_sup, d), nn.ReLU(), nn.Linear(d, d))
        self.glob = nn.Sequential(nn.Linear(F_GLOB, d), nn.ReLU(), nn.Linear(d, d))
        self.type_emb = nn.Parameter(torch.zeros(2, d))          # [global token, supplier token]
        layer = nn.TransformerEncoderLayer(d, heads, dim_feedforward=2 * d, dropout=0.0, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Linear(d, n_out))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, sup, glob):
        """sup (B, N, F_SUP), glob (B, F_GLOB) -> residuals (B, N, 3)."""
        hs = self.sup(sup) + self.type_emb[1]
        hg = self.glob(glob)[:, None, :] + self.type_emb[0]
        h = self.enc(torch.cat([hg, hs], dim=1))
        g, s = h[:, :1, :], h[:, 1:, :]
        return self.out_scale * self.head(torch.cat([s, g.expand_as(s)], dim=2))


def policy_params(net, spec, base: BaseParams, I, O, z, bound: float = 2.0):
    """Base parameters for the current status plus bounded adjustments:
    each adjustment lies in (-bound, +bound) units (bound <= 0: unbounded)."""
    raw = net(*features(spec, base, I, O, z))
    o = bound * torch.tanh(raw / bound) if bound > 0 else raw
    S0, cap0, trig0 = base.at(z)
    S = S0 + o[:, :, 0]
    cap = torch.clamp(cap0 + o[:, :, 1], min=1.0)
    trig = torch.clamp(trig0 + o[:, :, 2], min=1.0)
    return S, cap, trig


# --------------------------------------------------------------- randomness
class StreamSampler:
    """Uniform random numbers generated on the CPU (reproducible on any device)."""

    def __init__(self, spec: bt.BatchSpec, seed: int, device: str):
        self.g = torch.Generator().manual_seed(seed)
        self.spec, self.device = spec, device
        self.B = len(spec.inst_idx)

    def draw(self):
        B, N, R = self.B, self.spec.n, self.spec.R
        r = torch.rand(B * (1 + N + R), generator=self.g).to(self.device)
        u_d, u, us = r[:B], r[B:B + B * N].view(B, N), r[B + B * N:].view(B, R)
        return u_d, u, us


def rollout(net, ops, spec, base, state, sampler, T, grad: bool, scale=None, use_base: bool = False,
            bound: float = 2.0):
    """Simulate T periods. ``use_base`` runs the base policy (no network).
    Returns (mean scaled cost, per-episode mean cost, state)."""
    I, O, z = state
    total = torch.zeros(len(spec.inst_idx), device=I.device)
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        for _ in range(T):
            if use_base:
                S, cap, trig = base.at(z)
            else:
                S, cap, trig = policy_params(net, spec, base, I, O, z, bound)
            q, g = bt.gmi_decode(ops, spec, I, O, z, S, cap, trig, base.active)
            u_d, u, us = sampler.draw()
            d = bt.sample_demand(ops, spec, u_d)
            cost, I, O, z, _ = bt.step(ops, spec, I, O, z, q, g, d, u, us)
            total = total + cost
    per_ep = total / T
    loss = (per_ep / scale).mean() if scale is not None else per_ep.mean()
    return loss, per_ep, (I, O, z)


@dataclass
class TrainConfig:
    episodes_per_instance: int = 64
    horizon: int = 48            # backprop window (periods)
    burn_in: int = 300
    iters: int = 600
    lr: float = 3e-4
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    tau: float = 1.0
    bound: float = 2.0           # max adjustment per parameter, in units (<= 0: unbounded)
    ema: float = 0.98            # Polyak/EMA weight averaging (0 = off); validation uses the average
    val_every: int = 25
    val_episodes: int = 32
    val_periods: int = 1500
    d_model: int = 64
    layers: int = 2
    seed: int = 0
    device: str = "auto"
    log_every: int = 10


@dataclass
class TrainResult:
    net: SupplierSetPolicy
    history: list = field(default_factory=list)
    best_val_rel: float = 0.0
    seconds: float = 0.0
    device: str = "cpu"


def _val_score(net, ops, spec, base, sampler_seed, cfg, base_only: bool):
    sampler = StreamSampler(spec, sampler_seed, ops.dev)
    state = bt.initial(ops, spec, len(spec.inst_idx))
    net.eval()
    _, _, state = rollout(net, ops, spec, base, state, sampler, cfg.burn_in, grad=False, use_base=True)
    _, per_ep, _ = rollout(net, ops, spec, base, state, sampler, cfg.val_periods, grad=False,
                           use_base=base_only, bound=cfg.bound)
    net.train()
    return per_ep


def train(instances: list[Instance], tables: list[np.ndarray], actives: list[np.ndarray],
          excess_scale: list[float], cfg: TrainConfig = TrainConfig(), log=print,
          val_set: tuple | None = None) -> TrainResult:
    """Train one shared network on ``instances``. ``tables[k]`` is the base
    status table of instance k (use :func:`table_from_gmi` for a plain GMI
    base); the untrained network reproduces the base policy exactly.

    ``val_set = (instances, tables, actives, excess_scale)`` selects the best
    checkpoint on *separate* validation instances (generalisation); by default
    validation uses fresh random streams on the training instances.

    Stabilisation: small learning rate, exponential moving average of the
    weights (Polyak averaging; the averaged network is validated and
    returned), and a large fixed validation set with common random numbers."""
    torch.manual_seed(cfg.seed)
    device = pick_device(cfg.device)
    ops = bt.torch_ops(device)
    n_inst = len(instances)
    idx = np.repeat(np.arange(n_inst), cfg.episodes_per_instance)
    spec = bt.build_spec(ops, instances, idx)
    base = BaseParams.build(tables, actives, idx, device)
    scale = torch.as_tensor(np.asarray(excess_scale, float)[idx], dtype=torch.float32, device=device).clamp(min=1e-3)
    v_inst, v_tab, v_act, v_exc = val_set if val_set is not None else (instances, tables, actives, excess_scale)
    vidx = np.repeat(np.arange(len(v_inst)), cfg.val_episodes)
    vspec = bt.build_spec(ops, v_inst, vidx)
    vbase = BaseParams.build(v_tab, v_act, vidx, device)
    vscale = torch.as_tensor(np.asarray(v_exc, float)[vidx], dtype=torch.float32, device=device).clamp(min=1e-3)

    net = SupplierSetPolicy(cfg.d_model, layers=cfg.layers, out_scale=1.0).to(device)
    ema_net = copy.deepcopy(net)
    for p in ema_net.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sampler = StreamSampler(spec, cfg.seed + 1, device)
    t0 = time.time()
    # burn-in with the base policy to start near the stationary distribution
    state = bt.initial(ops, spec, len(idx))
    _, _, state = rollout(net, ops, spec, base, state, sampler, cfg.burn_in, grad=False, use_base=True)
    base_val = _val_score(net, ops, vspec, vbase, 10_000 + cfg.seed, cfg, base_only=True)

    def rel(per_ep):          # mean over instances of (cost - base cost) / excess scale
        return float(((per_ep - base_val) / vscale).mean())

    best = (0.0, copy.deepcopy(net.state_dict()), 0)       # zero-init net == base policy (rel = 0)
    hist = []
    for it in range(1, cfg.iters + 1):
        state = tuple(s.detach() for s in state)
        loss, per_ep, state = rollout(net, ops, spec, base, state, sampler, cfg.horizon, grad=True,
                                      scale=scale, bound=cfg.bound)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip))
        opt.step()
        with torch.no_grad():
            d = cfg.ema if cfg.ema > 0 else 0.0
            for pe, p in zip(ema_net.parameters(), net.parameters()):
                pe.mul_(d).add_(p.detach(), alpha=1 - d)
        if it % cfg.log_every == 0:
            hist.append({"iter": it, "train_loss": loss.detach().item(), "grad_norm": gn})
        if it % cfg.val_every == 0 or it == cfg.iters:
            v = rel(_val_score(ema_net, ops, vspec, vbase, 10_000 + cfg.seed, cfg, base_only=False))
            hist.append({"iter": it, "val_rel_to_base": v})
            if v < best[0]:
                best = (v, copy.deepcopy(ema_net.state_dict()), it)
            log(f"  iter {it:5d}  train loss {loss.detach().item():.4f}  val (cost - base)/excess = {v:+.4f}  "
                f"best {best[0]:+.4f} @ {best[2]}  ({time.time() - t0:.0f}s)")
    net.load_state_dict(best[1])
    return TrainResult(net, hist, best[0], time.time() - t0, device)


# ------------------------------------------------- policy for the simulator
class NeuralPolicy:
    """Wraps a trained net for the NumPy simulator (exact, hard decoding)."""

    def __init__(self, net: SupplierSetPolicy, table: np.ndarray, active: np.ndarray, bound: float = 2.0,
                 name: str = "Neural (state-aware)", device: str = "cpu"):
        self.net = net.to(device).eval()
        self.table, self.active = np.asarray(table, float), np.asarray(active, float)
        self.bound, self.name, self.device = bound, name, device
        self.ops = bt.torch_ops(device)
        self.spec = None

    def reset(self, inst, E):
        self.spec = bt.build_spec(self.ops, [inst], np.zeros(E, dtype=int))
        self.base = BaseParams.build([self.table], [self.active], np.zeros(E, dtype=int), self.device)

    def _params(self, inst, st):
        if self.spec is None or len(self.spec.inst_idx) != st.I.shape[0]:
            self.reset(inst, st.I.shape[0])
        I, O, z = bt.state_from_numpy(self.ops, self.spec, st)
        return (I, O, z), policy_params(self.net, self.spec, self.base, I, O, z, self.bound)

    def act(self, inst, st, t):
        with torch.no_grad():
            (I, O, z), (S, cap, trig) = self._params(inst, st)
            q, _ = bt.gmi_decode(self.ops, self.spec, I, O, z, S, cap, trig, self.base.active)
        return np.rint(q.cpu().numpy()).astype(np.int64)

    def explain(self, inst, st):
        """State-dependent parameters (S, cap, trigger) per supplier, for interpretation."""
        with torch.no_grad():
            _, (S, cap, trig) = self._params(inst, st)
        return {"S": S.cpu().numpy(), "cap": cap.cpu().numpy(), "trig": trig.cpu().numpy()}
