# -*- coding: utf-8 -*-
"""
Layer-by-layer confound stripping for tau identification.

Cell I  : dense, noise-free latent z, true Tn/gains/cutpoints, ONLY tau varies (analytic
          MSE, no estimation) — the information-theoretic upper bound. Must peak at the
          true pair; otherwise the filtering math is wrong.
Cell II : sparse ordinal votes, but Tn fixed at truth and no bias terms — tests whether
          ordinal noise alone destroys slow-tau identification.
"""
import os, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from p1mk.synth import simulate_dataset
from p1mk.fit import prep_inputs, exp_kernel, causal_conv

OUT = r"D:\thermal-adaptation-series\p1\outputs\identifiability"
os.makedirs(OUT, exist_ok=True)
TRUE_TF, TRUE_TS, DT = 2.0, 21.0, 0.2


def resample(meta):
    t_new = np.arange(0.0, meta["t"][-1] + DT, DT)
    def rs(A):
        if A.ndim == 1:
            return np.interp(t_new, meta["t"], A)
        return np.stack([np.interp(t_new, meta["t"], row) for row in A])
    Tamb = rs(meta["T_amb"])
    return t_new, Tamb


def lp(x_np, tau, dev):
    N, G = x_np.shape
    K = min(int(7.0 * tau / DT) + 2, G)
    alpha = float(np.exp(-DT / tau))
    m = np.arange(K)
    kern = torch.tensor((1 - alpha) * alpha ** m, dtype=torch.float32, device=dev).repeat(N, 1)
    xg = torch.tensor(x_np, dtype=torch.float32, device=dev)
    return causal_conv(xg, kern, G).cpu().numpy()


def cell_I(dev):
    votes, meta = simulate_dataset(n_subjects=40, n_votes=150, study_days=180.0,
                                   tau_fast_pop=TRUE_TF, tau_slow_pop=TRUE_TS,
                                   signal="multisine", seed=1, return_states=True)
    t_new, Tamb = resample(meta)
    Tn = meta["Tn"][:, None]
    x = Tamb[None, :] - Tn
    # recompute true states/z on the 0.2 d grid with true taus
    Lf_true = lp(x, TRUE_TF, dev); Ls_true = lp(x, TRUE_TS, dev)
    z_true = x - 0.45 * Lf_true - 0.35 * Ls_true

    tf_grid = np.round(np.logspace(np.log10(0.5), np.log10(8.0), 16), 3)
    ts_grid = np.round(np.logspace(np.log10(2.0), np.log10(300.0), 18), 2)
    mse = np.zeros((len(ts_grid), len(tf_grid)))
    for i, ts in enumerate(ts_grid):
        Ls = lp(x, float(ts), dev)
        for j, tf in enumerate(tf_grid):
            Lf = lp(x, float(tf), dev)
            z_pred = 0.20 * Ls + 0.55 * (Lf - Ls) + 1.0 * (x - Lf)
            mse[i, j] = np.mean((z_pred - z_true) ** 2)
    return tf_grid, ts_grid, mse


class KnownTnModel(nn.Module):
    """No free Tn (fixed at truth), no per-subject bias: only gains + cutpoints."""
    def __init__(self, n, Tn, dev):
        super().__init__()
        self.Tn = torch.tensor(Tn, dtype=torch.float32, device=dev)
        self.bs = nn.Parameter(torch.tensor(0.3, device=dev))
        self.bf = nn.Parameter(torch.tensor(0.6, device=dev))
        self.bi = nn.Parameter(torch.tensor(0.9, device=dev))
        self.gaps = nn.Parameter(torch.tensor([3.5, 1., 1., 1., 1., 1.], device=dev))

    def cuts(self):
        return torch.cumsum(self.gaps.clamp_min(0.05), 0) - 6.0

    def nll_at(self, tamb, subj, g, w, y, tf, ts):
        xg = tamb[None, :] - self.Tn[:, None]
        G = xg.shape[1]; N = xg.shape[0]
        K = min(max(int(7.0 * tf / DT) + 2, int(7.0 * ts / DT) + 2, 4), G)
        kf, _ = exp_kernel(torch.full((N,), np.log(tf), device=xg.device), DT, K)
        ks, _ = exp_kernel(torch.full((N,), np.log(ts), device=xg.device), DT, K)
        Lf = causal_conv(xg, kf, G); Ls = causal_conv(xg, ks, G)

        def at(A):
            return A[subj, g - 1] * (1 - w) + A[subj, g.clamp(max=G - 1)] * w
        xv, lf, ls = at(xg), at(Lf), at(Ls)
        z = (F.softplus(self.bs) * ls + F.softplus(self.bf) * (lf - ls)
             + F.softplus(self.bi) * (xv - lf))
        c = self.cuts()
        cdf = torch.sigmoid(c[None, :] - z[:, None])
        probs = torch.cat([cdf[:, :1], cdf[:, 1:] - cdf[:, :-1], 1 - cdf[:, -1:]], 1).clamp_min(1e-6)
        return F.nll_loss(torch.log(probs), (y + 3).long())


def cell_II(dev, epochs=300):
    votes, meta = simulate_dataset(n_subjects=40, n_votes=150, study_days=180.0,
                                   tau_fast_pop=TRUE_TF, tau_slow_pop=TRUE_TS,
                                   signal="multisine", seed=1, return_states=True)
    data = prep_inputs(meta, votes, dev, dt=DT)
    tf_grid = np.round(np.logspace(np.log10(0.5), np.log10(8.0), 12), 3)
    ts_grid = np.round(np.logspace(np.log10(5.0), np.log10(120.0), 12), 2)
    nll = np.zeros((len(ts_grid), len(tf_grid)))
    for i, ts in enumerate(ts_grid):
        for j, tf in enumerate(tf_grid):
            torch.manual_seed(0)
            m = KnownTnModel(meta["Tn"].shape[0], meta["Tn"], dev)
            opt = torch.optim.Adam(m.parameters(), lr=0.04)
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
            for _ in range(epochs):
                opt.zero_grad()
                loss = m.nll_at(data["tamb"], data["subj"], data["g"], data["w"], data["y"],
                                float(tf), float(ts))
                loss.backward(); opt.step(); sch.step()
            with torch.no_grad():
                nll[i, j] = m.nll_at(data["tamb"], data["subj"], data["g"], data["w"],
                                     data["y"], float(tf), float(ts)).item()
        print(f"  Cell II row tau_s={ts:6.1f} best={nll[:i+1].min():.4f}", flush=True)
    return tf_grid, ts_grid, nll


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", dev)
    print("Cell I: analytic dense noise-free ...")
    tf1, ts1, mse = cell_I(dev)
    i, j = np.unravel_index(np.argmin(mse), mse.shape)
    print(f"  Cell I best: tau_f={tf1[j]} tau_s={ts1[i]} MSE={mse.min():.5f} (truth 2, 21)")

    print("Cell II: ordinal, Tn known, no bias ...", flush=True)
    tf2, ts2, nll = cell_II(dev)
    i2, j2 = np.unravel_index(np.argmin(nll), nll.shape)
    print(f"  Cell II best: tau_f={tf2[j2]} tau_s={ts2[i2]} NLL={nll.min():.4f} (truth 2, 21)")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, (Tg, Sg, Z, zlab, ttl) in zip(axes, (
            (tf1, ts1, mse, "MSE of latent z", "I: dense, noise-free, everything else TRUE"),
            (tf2, ts2, nll, "NLL / vote", "II: ordinal EMA, Tn known & no bias"))):
        im = ax.contourf(Tg, Sg, Z, levels=25, cmap="viridis_r")
        ax.contour(Tg, Sg, Z, levels=12, colors="white", linewidths=0.4, alpha=0.6)
        ax.scatter([TRUE_TF], [TRUE_TS], marker="*", s=260, c="red", edgecolors="white",
                   linewidths=1.0, zorder=5)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("tau_fast (d)"); ax.set_ylabel("tau_slow (d)"); ax.set_title(ttl, fontsize=10)
        plt.colorbar(im, ax=ax, label=zlab)
    fig.tight_layout()
    png = os.path.join(OUT, "diagnostic_stripping.png")
    fig.savefig(png, dpi=145)
    pd.DataFrame(mse, index=ts1, columns=tf1).to_csv(os.path.join(OUT, "cellI_mse.csv"))
    pd.DataFrame(nll, index=ts2, columns=tf2).to_csv(os.path.join(OUT, "cellII_nll.csv"))
    print("saved:", png)


if __name__ == "__main__":
    main()
