# -*- coding: utf-8 -*-
"""Visualise the post-step relaxation: empirical binned TSV vs fitted tau_s models,
and decompose NLL by epoch since the step to locate where tau_s=150 'wins'."""
import os, sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from p1mk.synth import simulate_dataset
from p1mk.fit import prep_inputs, exp_kernel, causal_conv
from p1mk.identifiability_map import CellModel, DT

OUT = r"D:\thermal-adaptation-series\p1\outputs\identifiability"
T_STEP = 10.0


def fit(ts, data, meta, dev, epochs=900):
    torch.manual_seed(0)
    m = CellModel(meta["Tn"].shape[0], dev, free_Tn=False, Tn_true=meta["Tn"])
    opt = torch.optim.Adam(m.parameters(), lr=0.03)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for _ in range(epochs):
        opt.zero_grad()
        lo = m.nll_at(data["tamb"], data["subj"], data["g"], data["w"], data["y"], 2.0, ts)
        lo.backward(); opt.step(); sch.step()
    return m


def z_series(m, ts, data):
    with torch.no_grad():
        tamb = data["tamb"]
        Tn = m.Tn
        xg = tamb[None, :] - Tn[:, None]
        G, N = xg.shape[1], xg.shape[0]
        K = min(int(7.0 * ts / DT) + 2, G)
        kf, _ = exp_kernel(torch.full((N,), np.log(2.0), device=xg.device), DT,
                           min(int(7 * 2 / DT) + 2, G))
        ks, _ = exp_kernel(torch.full((N,), np.log(ts), device=xg.device), DT, K)
        Lf = causal_conv(xg, kf, G); Ls = causal_conv(xg, ks, G)
        import torch.nn.functional as F
        z = (F.softplus(m.bs) * Ls + F.softplus(m.bf) * (Lf - Ls)
             + F.softplus(m.bi) * (xg - Lf))
    return z.detach().cpu().numpy()


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    vtr, metr = simulate_dataset(20, 150, 180.0, 2.0, 21.0, signal="step", seed=1,
                                 return_states=True, report_bias_sd=0.0)
    dtr = prep_inputs(metr, vtr, dev, dt=DT)
    vte, mete = simulate_dataset(1000, 150, 180.0, 2.0, 21.0, signal="step", seed=77,
                                 return_states=True, report_bias_sd=0.0)
    dte = prep_inputs(mete, vte, dev, dt=DT)

    models = {}
    for ts in (21.0, 150.0):
        m = fit(ts, dtr, metr, dev)
        mt = CellModel(mete["Tn"].shape[0], dev, free_Tn=False, Tn_true=mete["Tn"])
        mt.load_state_dict({k: v for k, v in m.state_dict().items() if k != "Tn"}, strict=False)
        models[ts] = mt

    # empirical binned mean TSV vs days since step
    tgrid = np.arange(0, 180 + DT, DT)
    u = vte["t_day"].values - T_STEP
    bins = [(-10, 0), (0, 2), (2, 5), (5, 10), (10, 20), (20, 40), (40, 80), (80, 170)]
    bx, by, bse = [], [], []
    for a, b0 in bins:
        msk = (u >= a) & (u < b0)
        bx.append((a + b0) / 2 - (0 if a < 0 else 0))
        by.append(vte.loc[msk, "TSV"].mean())
        bse.append(vte.loc[msk, "TSV"].std() / np.sqrt(msk.sum()))
    # true and predicted mean z over subjects over time
    z_true_fine = (mete["x"] - 0.45 * mete["af"] - 0.35 * mete["as_"]).mean(0)
    z_true = np.interp(tgrid, mete["t"], z_true_fine)
    preds = {ts: z_series(mt, ts, dte).mean(0) for ts, mt in models.items()}

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    ax = axes[0]
    ax.errorbar(bx, by, yerr=bse, fmt="ko", ms=5, capsize=3, label="empirical mean TSV")
    tt = tgrid - T_STEP
    ax.plot(tt, z_true, color="black", lw=1.5, label="true latent z")
    ax.plot(tt, preds[21.0], color="#2c7fb8", lw=1.6, label="fit tau_s=21")
    ax.plot(tt, preds[150.0], color="#d95f0e", lw=1.6, ls="--", label="fit tau_s=150")
    ax.axvline(0, color="grey", lw=0.8); ax.axhline(0, color="grey", lw=0.8)
    ax.set_xlim(-10, 120); ax.set_xlabel("days since step"); ax.set_ylabel("sensation (z / TSV)")
    ax.set_title("Post-step relaxation (N=1000 test)"); ax.legend(fontsize=8)

    # NLL by epoch
    ax = axes[1]
    edges = np.array([-10, 0, 2, 5, 10, 20, 40, 80, 170.0])
    width = 0.35
    for k, ts in enumerate((21.0, 150.0)):
        mt = models[ts]
        with torch.no_grad():
            import torch.nn.functional as F
            xg = dte["tamb"][None, :] - mt.Tn[:, None]
            G, N = xg.shape[1], xg.shape[0]
            Kf = min(int(7 * 2 / DT) + 2, G); Ks = min(int(7 * ts / DT) + 2, G)
            kf, _ = exp_kernel(torch.full((N,), np.log(2.0), device=xg.device), DT, Kf)
            ks, _ = exp_kernel(torch.full((N,), np.log(ts), device=xg.device), DT, Ks)
            Lf = causal_conv(xg, kf, G); Ls = causal_conv(xg, ks, G)
            s, g, w, y = dte["subj"], dte["g"], dte["w"], dte["y"]

            def at(A):
                return A[s, g - 1] * (1 - w) + A[s, g.clamp(max=G - 1)] * w
            xv, lf, ls = at(xg), at(Lf), at(Ls)
            z = (F.softplus(mt.bs) * ls + F.softplus(mt.bf) * (lf - ls)
                 + F.softplus(mt.bi) * (xv - lf))
            c = mt.cuts(); cdf = torch.sigmoid(c[None, :] - z[:, None])
            P = torch.cat([cdf[:, :1], cdf[:, 1:] - cdf[:, :-1], 1 - cdf[:, -1:]], 1).clamp_min(1e-6)
            nll_i = -torch.log(P[torch.arange(len(y)), (y + 3).long()]).cpu().numpy()
        uv = vte["t_day"].values - T_STEP
        seg = [nll_i[(uv >= a) & (uv < b)].mean() for a, b in zip(edges[:-1], edges[1:])]
        xpos = np.arange(len(seg)) + (k - 0.5) * width
        ax.bar(xpos, seg, width=width, color="#2c7fb8" if ts == 21 else "#d95f0e",
               label=f"tau_s={ts:g}")
    ax.set_xticks(range(len(edges) - 1))
    ax.set_xticklabels([f"{int(a)}-{int(b)}" for a, b in zip(edges[:-1], edges[1:])], rotation=30)
    ax.set_xlabel("days since step"); ax.set_ylabel("mean NLL")
    ax.set_title("NLL by epoch (lower=better)"); ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "relax_diagnostic.png"), dpi=145)
    print("saved relax_diagnostic.png")
    for ts, mt in models.items():
        print(f"tau_s={ts}: gains bs={torch.nn.functional.softplus(mt.bs):.3f} "
              f"bf={torch.nn.functional.softplus(mt.bf):.3f} bi={torch.nn.functional.softplus(mt.bi):.3f} "
              f"cuts={np.round(mt.cuts().detach().cpu().numpy(),2)}")


if __name__ == "__main__":
    main()
