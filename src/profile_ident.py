# -*- coding: utf-8 -*-
"""
Profile-likelihood map over (tau_fast, tau_slow).

For each grid pair we FIX both population time constants (common to all subjects) and
optimise everything else (cascade gains, ordinal cutpoints, per-subject neutral temp and
bias). The resulting NLL surface shows whether the true pair sits in an identifiable
basin or whether a ridge connects it to the degenerate static / infinite-tau boundary.
"""
import os, sys, json
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


class ProfileModel(nn.Module):
    def __init__(self, n, dev):
        super().__init__()
        self.bs = nn.Parameter(torch.tensor(0.3, device=dev))
        self.bf = nn.Parameter(torch.tensor(0.6, device=dev))
        self.bi = nn.Parameter(torch.tensor(0.9, device=dev))
        gaps = torch.tensor([3.5, 1.0, 1.0, 1.0, 1.0, 1.0], device=dev)
        self.gaps = nn.Parameter(gaps)
        self.Tn = nn.Parameter(torch.full((n,), 22.0, device=dev))
        self.bias = nn.Parameter(torch.zeros(n, device=dev))

    def cuts(self):
        return torch.cumsum(self.gaps.clamp_min(0.05), 0) - 6.0

    def nll_at(self, tamb, subj, g, w, y, tf, ts, dt):
        N, G = tamb.shape[0] + 0, tamb.shape[0]
        xg = tamb[None, :] - self.Tn[:, None]
        Kf = int(7.0 * tf / dt) + 2
        Ks = int(7.0 * ts / dt) + 2
        K = min(max(Kf, Ks, 4), xg.shape[1])
        ltf = torch.full((xg.shape[0],), np.log(tf), device=xg.device)
        lts = torch.full((xg.shape[0],), np.log(ts), device=xg.device)
        kf, _ = exp_kernel(ltf, dt, K)
        ks, _ = exp_kernel(lts, dt, K)
        Lf = causal_conv(xg, kf, xg.shape[1])
        Ls = causal_conv(xg, ks, xg.shape[1])

        def at(A):
            return A[subj, g - 1] * (1 - w) + A[subj, g.clamp(max=A.shape[1] - 1)] * w
        xv, lf, ls = at(xg), at(Lf), at(Ls)
        z = (self.bias[subj] + F.softplus(self.bs) * ls
             + F.softplus(self.bf) * (lf - ls) + F.softplus(self.bi) * (xv - lf))
        c = self.cuts()
        cdf = torch.sigmoid(c[None, :] - z[:, None])
        probs = torch.cat([cdf[:, :1], cdf[:, 1:] - cdf[:, :-1], 1 - cdf[:, -1:]], 1).clamp_min(1e-6)
        return F.nll_loss(torch.log(probs), (y + 3).long())


def profile_cell(signal, dev, true_tf=2.0, true_ts=21.0, n_subjects=40, n_votes=150,
                 study=180.0, epochs=350, seed=1):
    votes, meta = simulate_dataset(n_subjects=n_subjects, n_votes=n_votes, study_days=study,
                                   tau_fast_pop=true_tf, tau_slow_pop=true_ts,
                                   signal=signal, seed=seed, return_states=True)
    data = prep_inputs(meta, votes, dev, dt=0.2)
    tf_grid = np.round(np.logspace(np.log10(0.5), np.log10(8.0), 12), 3)
    ts_grid = np.round(np.logspace(np.log10(5.0), np.log10(120.0), 12), 2)
    rows = []
    for ts in ts_grid:
        for tf in tf_grid:
            torch.manual_seed(0)
            m = ProfileModel(n_subjects, dev)
            opt = torch.optim.Adam(m.parameters(), lr=0.04)
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
            for _ in range(epochs):
                opt.zero_grad()
                loss = m.nll_at(data["tamb"], data["subj"], data["g"], data["w"], data["y"],
                                float(tf), float(ts), 0.2)
                loss.backward(); opt.step(); sch.step()
            with torch.no_grad():
                v = m.nll_at(data["tamb"], data["subj"], data["g"], data["w"], data["y"],
                             float(tf), float(ts), 0.2).item()
            rows.append(dict(signal=signal, tau_fast=tf, tau_slow=ts, nll=v))
        print(f"  {signal}: tau_slow row {ts:6.1f} done, best NLL={min(r['nll'] for r in rows):.4f}",
              flush=True)
    return pd.DataFrame(rows), (true_tf, true_ts)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", dev)
    all_df = []
    for sig in ("multisine", "weather"):
        print(f"profiling {sig} ...", flush=True)
        df, truth = profile_cell(sig, dev)
        all_df.append(df)
    out = pd.concat(all_df, ignore_index=True)
    out.to_csv(os.path.join(OUT, "profile_nll.csv"), index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, sig in zip(axes, ("multisine", "weather")):
        d = out[out.signal == sig]
        piv = d.pivot(index="tau_slow", columns="tau_fast", values="nll")
        Tf = piv.columns.values; Ts = piv.index.values
        nll = piv.values
        im = ax.contourf(Tf, Ts, nll, levels=25, cmap="viridis_r")
        ax.contour(Tf, Ts, nll, levels=12, colors="white", linewidths=0.4, alpha=0.6)
        ax.scatter([2.0], [21.0], marker="*", s=260, c="red", edgecolors="white",
                   linewidths=1.0, zorder=5, label="true (2, 21)")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("tau_fast (days)"); ax.set_ylabel("tau_slow (days)")
        ax.set_title(f"Profile NLL — {sig}")
        ax.legend(loc="upper left", fontsize=8)
        plt.colorbar(im, ax=ax, label="NLL / vote")
    fig.tight_layout()
    png = os.path.join(OUT, "profile_likelihood.png")
    fig.savefig(png, dpi=145)
    print("saved:", png)
    for sig in ("multisine", "weather"):
        d = out[out.signal == sig].sort_values("nll")
        print(sig, "best 5:\n", d.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
