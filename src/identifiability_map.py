# -*- coding: utf-8 -*-
"""
Slow-tau identifiability map.

One-dimensional profile over tau_slow (tau_fast held at truth) under crossed designs:
cohort size N x study length T x excitation signal x confound regime.

Regimes:
  known : Tn fixed at truth, no reporting bias in DGP, no bias parameter (information
          ceiling for the ordinal EMA design)
  free  : Tn latent, reporting bias ~ N(0,0.25) in DGP, both estimated (realistic)

Metrics per cell:
  argmin tau_s, and total delta-NLL = [NLL(tau_s=boundary) - NLL(tau_s=true)] * n_votes
  (positive + large => truth preferred over the static-slow limit; chi2_1 95% ~= 1.92)
"""
import os, sys, json, argparse
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
DT = 0.2
TS_GRID = np.array([5, 7, 9, 12, 15, 21, 28, 38, 50, 67, 90, 120, 150], dtype=float)
TS_BOUNDARY = 150.0


class CellModel(nn.Module):
    def __init__(self, n, dev, free_Tn, Tn_true=None):
        super().__init__()
        self.free = free_Tn
        if free_Tn:
            self.Tn = nn.Parameter(torch.tensor(Tn_true, dtype=torch.float32, device=dev) * 0 + 22.0)
            self.bias = nn.Parameter(torch.zeros(n, device=dev))
        else:
            self.register_buffer("Tn", torch.tensor(Tn_true, dtype=torch.float32, device=dev))
        # free linear gains, initialised at the true cascade values (softplus was
        # double-transforming the init and collapsing the model into degenerate basins)
        self.bs = nn.Parameter(torch.tensor(0.20, device=dev))
        self.bf = nn.Parameter(torch.tensor(0.55, device=dev))
        self.bi = nn.Parameter(torch.tensor(1.00, device=dev))
        self.gaps = nn.Parameter(torch.tensor([3.5, 1., 1., 1., 1., 1.], device=dev))

    def cuts(self):
        return torch.cumsum(self.gaps.clamp_min(0.05), 0) - 6.0

    def nll_at(self, tamb, subj, g, w, y, tf, ts):
        xg = tamb[None, :] - self.Tn[:, None]
        G, N = xg.shape[1], xg.shape[0]
        K = min(max(int(7.0 * tf / DT) + 2, int(7.0 * ts / DT) + 2, 4), G)
        kf, _ = exp_kernel(torch.full((N,), np.log(tf), device=xg.device), DT, K)
        ks, _ = exp_kernel(torch.full((N,), np.log(ts), device=xg.device), DT, K)
        Lf = causal_conv(xg, kf, G); Ls = causal_conv(xg, ks, G)

        def at(A):
            return A[subj, g - 1] * (1 - w) + A[subj, g.clamp(max=G - 1)] * w
        xv, lf, ls = at(xg), at(Lf), at(Ls)
        z = (self.bs * ls + self.bf * (lf - ls) + self.bi * (xv - lf))
        if self.free:
            z = z + self.bias[subj]
        c = self.cuts()
        cdf = torch.sigmoid(c[None, :] - z[:, None])
        probs = torch.cat([cdf[:, :1], cdf[:, 1:] - cdf[:, :-1], 1 - cdf[:, -1:]], 1).clamp_min(1e-6)
        return F.nll_loss(torch.log(probs), (y + 3).long())


def profile_cell(N, T, nv, signal, regime, true_ts=21.0, true_tf=2.0, epochs=250, seed=1, dev="cuda"):
    votes, meta = simulate_dataset(
        n_subjects=N, n_votes=nv, study_days=float(T),
        tau_fast_pop=true_tf, tau_slow_pop=true_ts,
        signal=signal, seed=seed, return_states=True,
        report_bias_sd=(0.25 if regime == "free" else 0.0))
    data = prep_inputs(meta, votes, dev, dt=DT)
    free = regime == "free"
    nll = []
    for ts in TS_GRID:
        torch.manual_seed(0)
        m = CellModel(meta["Tn"].shape[0], dev, free_Tn=free, Tn_true=meta["Tn"])
        opt = torch.optim.Adam(m.parameters(), lr=0.04)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        for _ in range(epochs):
            opt.zero_grad()
            loss = m.nll_at(data["tamb"], data["subj"], data["g"], data["w"], data["y"],
                            true_tf, float(ts))
            loss.backward(); opt.step(); sch.step()
        with torch.no_grad():
            nll.append(m.nll_at(data["tamb"], data["subj"], data["g"], data["w"],
                                data["y"], true_tf, float(ts)).item())
    nll = np.array(nll)
    n_votes = len(votes)
    i_true = int(np.argmin(np.abs(TS_GRID - true_ts)))
    i_bnd = int(np.argmin(np.abs(TS_GRID - TS_BOUNDARY)))
    return dict(N=N, T=T, nv=nv, signal=signal, regime=regime, true_ts=true_ts,
                argmin_ts=float(TS_GRID[int(np.argmin(nll))]),
                nll_true=float(nll[i_true]), nll_boundary=float(nll[i_bnd]),
                delta_nll_total=float((nll[i_bnd] - nll[i_true]) * n_votes),
                n_votes=n_votes, profile=nll.tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["quick", "full"], default="quick")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", dev, flush=True)

    if args.mode == "quick":
        for regime in ("known", "free"):
            r = profile_cell(40, 180, 150, "multisine", regime, dev=dev)
            print(json.dumps({k: v for k, v in r.items() if k != "profile"}, indent=1), flush=True)
            print("  profile:", dict(zip(TS_GRID.astype(str),
                                         np.round(r["profile"], 4))), flush=True)
        return

    rows = []
    for regime in ("known", "free"):
        for N in (20, 40, 100):
            for T in (90, 180, 365):
                r = profile_cell(N, T, 150, "weather", regime, dev=dev, epochs=400)
                rows.append(r)
                print(f"{regime:5s} weather N={N:3d} T={T:3d} "
                      f"argmin_ts={r['argmin_ts']:6.1f} dNLL={r['delta_nll_total']:8.1f}",
                      flush=True)
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "profile"} for r in rows])
    df.to_csv(os.path.join(OUT, "ident_map.csv"), index=False)
    with open(os.path.join(OUT, "ident_map_profiles.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f)

    # heatmaps: rows=N, cols=T, one panel per regime (weather excitation)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True, sharey=True)
    for ax, regime in zip(axes, ("known", "free")):
        sub = df[(df.regime == regime) & (df.signal == "weather")]
        piv = sub.pivot(index="N", columns="T", values="delta_nll_total")
        im = ax.imshow(piv.values, aspect="auto", origin="lower", cmap="RdYlGn",
                       vmin=-20, vmax=60)
        for i, N in enumerate(piv.index):
            for j, T in enumerate(piv.columns):
                v = piv.values[i, j]
                cell = sub[(sub.N == N) & (sub.T == T)].iloc[0]
                ax.text(j, i, f"{v:.0f}\n{cell.argmin_ts:.0f}d", ha="center", va="center",
                        fontsize=9)
        ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns)
        ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index)
        ax.set_xlabel("study length (days)"); ax.set_ylabel("cohort N")
        ax.set_title(f"{regime} / weather\nΔNLL (truth vs τs→∞), argmin τs", fontsize=10)
        plt.colorbar(im, ax=ax, label="ΔNLL total")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "ident_map_weather.png"), dpi=145)
    print("saved ident_map.csv / ident_map_weather.png", flush=True)


if __name__ == "__main__":
    main()
