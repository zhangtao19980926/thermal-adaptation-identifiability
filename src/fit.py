# -*- coding: utf-8 -*-
"""
Estimators for the identifiability experiment.

Stage A uses the *correctly specified* parametric model: two parallel first-order low-pass
filters (fast / slow adaptation states) driven by the dense ambient-temperature grid,
followed by an ordered-logit observation head for the 7-point TSV. Hierarchical
(subject-level) log-time-constants with a Gaussian partial-pooling prior.

The low-pass filters are evaluated as causal grouped 1-D convolutions with truncated
exponential kernels: one GPU call gives every subject's hidden state trajectory, with no
Python time loop, so thousands of training iterations are cheap.

Baselines:
  * single  - one exponential memory (one tau)
  * fixed7  - 7-day rectangular running mean (ASHRAE-style short window)
  * fixed30 - 30-day rectangular running mean
  * static  - no memory (operative deviation only)
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _vote_indices(t_grid: torch.Tensor, subj: torch.Tensor, t_vote: torch.Tensor):
    g = torch.searchsorted(t_grid, t_vote).clamp(1, len(t_grid) - 1)
    w = (t_vote - t_grid[g - 1]) / (t_grid[g] - t_grid[g - 1])
    return subj.long(), g.long(), w


def _gather_states(S: torch.Tensor, subj, g, w):
    """Interpolate hidden states (N, G, C) at vote times."""
    s0 = S[subj, g - 1]
    s1 = S[subj, g]
    return s0 * (1 - w)[:, None] + s1 * w[:, None]


def exp_kernel(log_tau: torch.Tensor, dt: float, K: int):
    """(N, K) normalised causal exponential kernels; alpha = exp(-dt/tau)."""
    tau = torch.exp(log_tau)
    alpha = torch.exp(-dt / tau)                          # (N,)
    m = torch.arange(K, device=log_tau.device, dtype=torch.float32)
    kern = (1.0 - alpha)[:, None] * alpha[:, None] ** m[None, :]
    return kern, alpha


def causal_conv(x: torch.Tensor, kern: torch.Tensor, G: int):
    """Per-series CAUSAL convolution. x:(N,G), kern:(N,K) with kern[:,0] the newest
    weight -> out[t]=sum_m kern[:,m] x[t-m] (zero state before t=0).

    conv1d is cross-correlation, so left-pad by K-1 and flip the kernel."""
    N, K = kern.shape
    w = kern.flip(dims=[1])
    xp = F.pad(x, (K - 1, 0))
    out = F.conv1d(xp.unsqueeze(0), w.unsqueeze(1), groups=N, padding=0)
    return out[0, :, :G]


def rect_kernel(window_days: float, dt: float, device):
    K = max(1, int(round(window_days / dt)))
    kern = torch.full((1, K), 1.0 / K, device=device, dtype=torch.float32)
    return kern, K


class OrdinalHead(nn.Module):
    """Ordered cutpoints for an ncat-point ordinal vote centred at 0.
    ncat=7 -> TSV in {-3..+3}; ncat=3 -> preference {-1 cooler?,0 no change,+1}."""
    def __init__(self, ncat: int = 7, init=None):
        super().__init__()
        self.ncat = ncat
        if init is None:
            if ncat == 7:
                init = torch.tensor([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5])
            elif ncat == 3:
                init = torch.tensor([-0.5, 0.5])
            else:
                init = torch.linspace(-1.0, 1.0, ncat - 1)
        self.base = float(init[0] - 3.0)
        gaps = torch.diff(torch.cat([torch.tensor([self.base]), init]))
        self.gaps = nn.Parameter(gaps.clamp_min(0.05).clone())

    def cutpoints(self):
        return torch.cumsum(self.gaps, 0) + self.base

    def nll(self, z, y):
        """Negative log-likelihood; y in {-floor(ncat/2) .. +floor(ncat/2)}."""
        c = self.cutpoints()                       # (ncat-1,)
        cdf = torch.sigmoid(c[None, :] - z[:, None])
        eps = 1e-6
        p_lo = cdf[:, 0]
        p_hi = 1.0 - cdf[:, -1]
        p_mid = cdf[:, 1:] - cdf[:, :-1]
        probs = torch.cat([p_lo[:, None], p_mid, p_hi[:, None]], dim=1).clamp_min(eps)
        idx = (y + self.ncat // 2).long()
        return F.nll_loss(torch.log(probs), idx)


class MemoryModel(nn.Module):
    def __init__(self, n_subjects: int, kind: str = "dual", dt: float = 0.2,
                 k_frac: float = 7.0, sigma_logtau: float = 0.35,
                 init_tau_fast: float = 4.0, init_tau_slow: float = 40.0, ncat: int = 7):
        super().__init__()
        assert kind in ("dual", "single", "fixed7", "fixed30", "static")
        self.kind = kind
        self.n = n_subjects
        self.dt = dt
        self.k_frac = k_frac
        self.sigma = sigma_logtau
        self.head = OrdinalHead(ncat)
        self.k0 = nn.Parameter(torch.tensor(1.0))
        self.b = nn.Parameter(torch.zeros(n_subjects))
        if kind == "dual":
            self.log_tf_pop = nn.Parameter(torch.tensor(float(np.log(init_tau_fast))))
            self.log_ts_pop = nn.Parameter(torch.tensor(float(np.log(init_tau_slow))))
            self.log_tf_i = nn.Parameter(self.log_tf_pop.detach().clone().repeat(n_subjects))
            self.log_ts_i = nn.Parameter(self.log_ts_pop.detach().clone().repeat(n_subjects))
            # cascade band gains: slow level / mid band / fast residual (free, linear)
            self.bs = nn.Parameter(torch.tensor(0.20))
            self.bf = nn.Parameter(torch.tensor(0.55))
            self.bi = nn.Parameter(torch.tensor(1.00))
        elif kind == "single":
            self.log_t_pop = nn.Parameter(torch.tensor(float(np.log(init_tau_fast))))
            self.log_t_i = nn.Parameter(self.log_t_pop.detach().clone().repeat(n_subjects))
            self.bs = nn.Parameter(torch.tensor(0.5))
            self.bi = nn.Parameter(torch.tensor(0.9))
        if kind in ("fixed7", "fixed30"):
            self.bs = nn.Parameter(torch.tensor(0.5))
            self.bi = nn.Parameter(torch.tensor(0.9))
        if kind == "static":
            self.bi = nn.Parameter(torch.tensor(1.0))

    def states(self, xgrid: torch.Tensor):
        """xgrid:(N,G) temperature deviations on the shared grid -> (N,G,C) or None."""
        N, G = xgrid.shape
        dev = xgrid.device
        if self.kind == "static":
            return None
        if self.kind in ("fixed7", "fixed30"):
            w = 7.0 if self.kind == "fixed7" else 30.0
            kern, K = rect_kernel(w, self.dt, dev)
            a = causal_conv(xgrid, kern.expand(N, -1), G)
            return a.unsqueeze(-1)
        if self.kind == "single":
            K = int(self.k_frac * np.exp(self.log_t_pop.detach().item()) / self.dt) + 2
            K = min(max(K, 4), G)
            kern, _ = exp_kernel(self.log_t_i, self.dt, K)
            a = causal_conv(xgrid, kern, G)
            return a.unsqueeze(-1)
        # dual
        Kf = int(self.k_frac * np.exp(self.log_tf_pop.detach().item()) / self.dt) + 2
        Ks = int(self.k_frac * np.exp(self.log_ts_pop.detach().item()) / self.dt) + 2
        K = min(max(Kf, Ks, 4), G)
        kf, _ = exp_kernel(self.log_tf_i, self.dt, K)
        ks, _ = exp_kernel(self.log_ts_i, self.dt, K)
        af = causal_conv(xgrid, kf, G)
        asl = causal_conv(xgrid, ks, G)
        return torch.stack([af, asl], dim=-1)

    def latent(self, xgrid, subj, g, w):
        xv = xgrid[subj, (g - 1).clamp(max=xgrid.shape[1] - 1)] * (1 - w) + \
             xgrid[subj, g.clamp(max=xgrid.shape[1] - 1)] * w
        z = self.b[subj]
        S = self.states(xgrid)
        if self.kind == "dual":
            sv = _gather_states(S, subj, g, w)
            Lf, Ls = sv[:, 0], sv[:, 1]
            # frequency-orthogonal cascade: slow level, mid band, fast residual
            z = (z + self.bs * Ls + self.bf * (Lf - Ls) + self.bi * (xv - Lf))
        elif self.kind in ("single", "fixed7", "fixed30"):
            sv = _gather_states(S, subj, g, w)
            L = sv[:, 0]
            z = z + self.bs * L + self.bi * (xv - L)
        else:  # static
            z = z + self.bi * xv
        return z

    def hier_penalty(self):
        if self.kind == "dual":
            pf = ((self.log_tf_i - self.log_tf_pop) ** 2).mean() / (2 * self.sigma ** 2)
            ps = ((self.log_ts_i - self.log_ts_pop) ** 2).mean() / (2 * self.sigma ** 2)
            return pf + ps
        if self.kind == "single":
            return ((self.log_t_i - self.log_t_pop) ** 2).mean() / (2 * self.sigma ** 2)
        return torch.tensor(0.0, device=self.b.device)


def prep_inputs(meta, votes, device, dt=0.2):
    """Resample shared ambient grid and build vote index tensors."""
    t_fine = np.arange(0.0, meta["t"][-1] + dt, dt)
    Tamb = np.interp(t_fine, meta["t"], meta["T_amb"])
    t_grid = torch.tensor(t_fine, dtype=torch.float32, device=device)
    tamb = torch.tensor(Tamb, dtype=torch.float32, device=device)
    subj = torch.tensor(votes["subject"].values, dtype=torch.long, device=device)
    tv = torch.tensor(votes["t_day"].values, dtype=torch.float32, device=device)
    # use true exposure when available for the grid; neutral temps are latent params
    Tn_true = torch.tensor(meta["Tn"], dtype=torch.float32, device=device)
    s, g, w = _vote_indices(t_grid, subj, tv)
    y = torch.tensor(votes["TSV"].values, dtype=torch.long, device=device)
    return dict(t_grid=t_grid, tamb=tamb, Tn_true=Tn_true, subj=s, g=g, w=w, y=y, dt=dt)


def fit_model(kind, data, n_subjects, epochs=1800, lr=0.03, seed=0, verbose=False,
              init_tau_fast=4.0, init_tau_slow=40.0, ncat=7, tn_init=22.0):
    torch.manual_seed(seed)
    dev = data["tamb"].device
    dt = data["dt"]
    model = MemoryModel(n_subjects, kind=kind, dt=dt, ncat=ncat,
                        init_tau_fast=init_tau_fast, init_tau_slow=init_tau_slow).to(dev)
    # per-subject absolute neutral temperature as free parameters, init near climate mean
    Tn = nn.Parameter(torch.full((n_subjects,), float(tn_init), device=dev))
    opt = torch.optim.Adam(list(model.parameters()) + [Tn], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    G = len(data["t_grid"])
    subj, g, w, y = data["subj"], data["g"], data["w"], data["y"]
    tamb = data["tamb"]
    def _xg():
        base = tamb if tamb.dim() == 2 else tamb[None, :]
        return base - Tn[:, None]
    for ep in range(epochs):
        opt.zero_grad()
        xgrid = _xg()
        z = model.latent(xgrid, subj, g, w)
        nll = model.head.nll(z, y)
        loss = nll + 0.05 * model.hier_penalty()
        loss.backward()
        opt.step()
        sched.step()
        if verbose and (ep % 300 == 0 or ep == epochs - 1):
            print(f"  [{kind}] ep{ep:4d} nll={nll.item():.4f}")
    with torch.no_grad():
        xgrid = _xg()
        z = model.latent(xgrid, subj, g, w)
        nll = model.head.nll(z, y).item()
        out = dict(kind=kind, nll=nll, model=model, Tn=Tn.detach().cpu().numpy())
        if kind == "dual":
            out.update(tau_fast_pop=float(np.exp(model.log_tf_pop.item())),
                       tau_slow_pop=float(np.exp(model.log_ts_pop.item())),
                       tau_fast_i=np.exp(model.log_tf_i.detach().cpu().numpy()),
                       tau_slow_i=np.exp(model.log_ts_i.detach().cpu().numpy()),
                       b_slow=float(model.bs),
                       b_mid=float(model.bf),
                       b_fast=float(model.bi))
        if kind == "single":
            out.update(tau_pop=float(np.exp(model.log_t_pop.item())),
                       tau_i=np.exp(model.log_t_i.detach().cpu().numpy()))
    return out
