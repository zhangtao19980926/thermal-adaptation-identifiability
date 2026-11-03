# -*- coding: utf-8 -*-
"""Analytic OLS profile on noise-free latent z at vote times; gains fit per tau_s."""
import os, sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from p1mk.synth import simulate_dataset
from p1mk.diagnostic_ident import lp, resample

OUT = r"D:\thermal-adaptation-series\p1\outputs\identifiability"
TS = np.array([5, 7, 9, 12, 15, 21, 28, 38, 50, 67, 90, 120, 150], float)


def run(signal, N, T, dev, nv=300, seed=1):
    votes, meta = simulate_dataset(N, nv, float(T), 2.0, 21.0, signal=signal, seed=seed,
                                   return_states=True, report_bias_sd=0.0)
    t_new, Tamb = resample(meta)
    x = Tamb[None, :] - meta["Tn"][:, None]
    Lf = lp(x, 2.0, dev)
    tv = votes["t_day"].values
    gi = np.clip(np.searchsorted(t_new, tv), 1, len(t_new) - 1)
    w = (tv - t_new[gi - 1]) / (t_new[gi] - t_new[gi - 1])
    sj = votes["subject"].values

    def at(A):
        return A[sj, gi - 1] * (1 - w) + A[sj, gi] * w

    xv, lf = at(x), at(Lf)
    ztrue = votes["z_latent"].values
    sse = []
    for ts in TS:
        ls = at(lp(x, float(ts), dev))
        X = np.column_stack([ls, lf - ls, xv - lf, np.ones_like(xv)])
        beta, *_ = np.linalg.lstsq(X, ztrue, rcond=None)
        sse.append(float(np.mean((X @ beta - ztrue) ** 2)))
    return TS, np.array(sse)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for signal, N, T in (("step", 20, 180), ("multisine", 40, 180),
                         ("events", 40, 90), ("weather", 40, 365)):
        TSg, mse = run(signal, N, T, dev)
        i21 = list(TSg).index(21.0)
        print(f"{signal:9s} N={N} T={T}: best tau_s={TSg[int(np.argmin(mse))]:6.1f} "
              f"MSE21={mse[i21]:.5f} MSE150={mse[-1]:.5f}", flush=True)
        ax.plot(TSg, mse, "o-", ms=4, label=f"{signal} N={N} T={T}")
    ax.axvline(21, color="red", ls="--", lw=1, label="true 21")
    ax.set_xscale("log"); ax.set_xlabel("tau_s (d)")
    ax.set_ylabel("MSE / vote (noise-free z)")
    ax.set_title("Analytic OLS profile on noise-free latent z"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "linear_profile.png"), dpi=145)
    print("saved linear_profile.png")


if __name__ == "__main__":
    main()
