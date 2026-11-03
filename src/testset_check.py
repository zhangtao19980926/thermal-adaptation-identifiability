# -*- coding: utf-8 -*-
"""
Decisive checks for the slow-tau bias.

X: fit tau_s profile on a training cohort, then evaluate each fitted model on a large
   INDEPENDENT cohort (true tau_s=21) — separates finite-sample over-fitting (bias
   variance / shrinkage) from systematic misspecification.
Z: same profile with near-dense ordinal voting (1000 votes/subject) — tests whether
   7-level interval censoring plus sparse EMA is the information bottleneck.
"""
import os, sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from p1mk.synth import simulate_dataset
from p1mk.fit import prep_inputs
from p1mk.identifiability_map import CellModel, DT

OUT = r"D:\thermal-adaptation-series\p1\outputs\identifiability"
TS = [10.0, 15.0, 21.0, 28.0, 38.0, 60.0, 100.0, 150.0]


def fit_one(ts, data, meta, dev, epochs=800, seed=0):
    torch.manual_seed(seed)
    m = CellModel(meta["Tn"].shape[0], dev, free_Tn=False, Tn_true=meta["Tn"])
    opt = torch.optim.Adam(m.parameters(), lr=0.03)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for _ in range(epochs):
        opt.zero_grad()
        lo = m.nll_at(data["tamb"], data["subj"], data["g"], data["w"], data["y"], 2.0, ts)
        lo.backward(); opt.step(); sch.step()
    return m


def eval_model(m, ts, data):
    with torch.no_grad():
        return m.nll_at(data["tamb"], data["subj"], data["g"], data["w"], data["y"],
                        2.0, ts).item()


def curve(nv, N, T, dev, epochs, seed_train=1, seed_test=99, dense_test_nv=None):
    votes, meta = simulate_dataset(N, nv, float(T), 2.0, 21.0, signal="multisine",
                                   seed=seed_train, return_states=True, report_bias_sd=0.0)
    data = prep_inputs(meta, votes, dev, dt=DT)
    test_nv = dense_test_nv or nv
    vte, mete = simulate_dataset(1000, test_nv, float(T), 2.0, 21.0, signal="multisine",
                                 seed=seed_test, return_states=True, report_bias_sd=0.0)
    # NOTE: test cohort Tn differs; rebuild a model carrying TEST Tn but trained weights
    dtest = prep_inputs(mete, vte, dev, dt=DT)
    tr, te = [], []
    for ts in TS:
        m = fit_one(ts, data, meta, dev, epochs=epochs)
        tr.append(eval_model(m, ts, data))
        mt = CellModel(mete["Tn"].shape[0], dev, free_Tn=False, Tn_true=mete["Tn"])
        mt.load_state_dict({k: v for k, v in m.state_dict().items() if k != "Tn"},
                           strict=False)
        te.append(eval_model(mt, ts, dtest))
    return tr, te


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", dev, flush=True)

    print("[X] N=200 T=365 nv=150 ...", flush=True)
    trX, teX = curve(150, 200, 365, dev, epochs=800)
    print("  train:", dict(zip(TS, np.round(trX, 4))), flush=True)
    print("  test :", dict(zip(TS, np.round(teX, 4))), flush=True)

    print("[Z] N=200 T=365 nv=1000 (near-dense ordinal) ...", flush=True)
    trZ, teZ = curve(1000, 200, 365, dev, epochs=600, dense_test_nv=1000)
    print("  train:", dict(zip(TS, np.round(trZ, 4))), flush=True)
    print("  test :", dict(zip(TS, np.round(teZ, 4))), flush=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharex=True)
    for ax, tr, te, ttl in ((axes[0], trX, teX, "X: 150 votes/subject (EMA)"),
                            (axes[1], trZ, teZ, "Z: 1000 votes/subject (near-dense)")):
        ax.plot(TS, tr, "o-", label="train NLL")
        ax.plot(TS, te, "s-", label="independent-test NLL (N=1000)")
        ax.axvline(21, color="red", ls="--", lw=1, label="true tau_s=21")
        ax.set_xscale("log"); ax.set_xlabel("tau_slow (days)"); ax.set_ylabel("NLL / vote")
        ax.set_title(ttl); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "testset_check.png"), dpi=145)
    print("saved testset_check.png", flush=True)


if __name__ == "__main__":
    main()
