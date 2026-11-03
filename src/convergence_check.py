# -*- coding: utf-8 -*-
"""Check whether the slow-tau profile preference is real or an optimisation artefact.

For a high-information cell (N=200, T=365, multisine, known regime), train tau_s=21 vs
tau_s=150 across increasing epoch budgets and from two inits; also finish with L-BFGS.
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


def train_curve(ts, epochs_list, data, meta, dev, seed=0):
    torch.manual_seed(seed)
    m = CellModel(meta["Tn"].shape[0], dev, free_Tn=False, Tn_true=meta["Tn"])
    opt = torch.optim.Adam(m.parameters(), lr=0.03)
    tot = max(epochs_list)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tot)
    curve = {}
    for ep in range(tot):
        opt.zero_grad()
        loss = m.nll_at(data["tamb"], data["subj"], data["g"], data["w"], data["y"], 2.0, ts)
        loss.backward(); opt.step(); sch.step()
        if ep + 1 in epochs_list:
            with torch.no_grad():
                v = m.nll_at(data["tamb"], data["subj"], data["g"], data["w"],
                             data["y"], 2.0, ts).item()
            curve[ep + 1] = v
    # L-BFGS polish
    lbfgs = torch.optim.LBFGS(m.parameters(), max_iter=200, lr=0.5,
                              line_search_fn="strong_wolfe")

    def closure():
        lbfgs.zero_grad()
        lo = m.nll_at(data["tamb"], data["subj"], data["g"], data["w"], data["y"], 2.0, ts)
        lo.backward()
        return lo
    lbfgs.step(closure)
    with torch.no_grad():
        curve["lbfgs"] = m.nll_at(data["tamb"], data["subj"], data["g"], data["w"],
                                  data["y"], 2.0, ts).item()
    return curve


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    votes, meta = simulate_dataset(n_subjects=200, n_votes=150, study_days=365.0,
                                   tau_fast_pop=2.0, tau_slow_pop=21.0,
                                   signal="multisine", seed=1, return_states=True,
                                   report_bias_sd=0.0)
    data = prep_inputs(meta, votes, dev, dt=DT)
    epochs_list = [100, 250, 500, 1000, 2000, 4000]
    out = {}
    for ts in (21.0, 150.0):
        out[ts] = train_curve(ts, epochs_list, data, meta, dev)
        print(f"tau_s={ts:5.0f}: " + "  ".join(f"{k}->{v:.5f}" for k, v in out[ts].items()),
              flush=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for ts, c in out.items():
        eps = [e for e in c if isinstance(e, int)]
        ax.plot(eps, [c[e] for e in eps], "o-", label=f"tau_s={ts:g}")
        ax.axhline(c["lbfgs"], ls="--", lw=1, alpha=0.6)
    ax.set_xscale("log"); ax.set_xlabel("Adam epochs"); ax.set_ylabel("NLL / vote")
    ax.set_title("Profile convergence (N=200, T=365, multisine, known regime)\n"
                 "dashed = L-BFGS polished value")
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(OUT, "convergence_check.png"), dpi=145)
    print("saved convergence_check.png")


if __name__ == "__main__":
    main()
