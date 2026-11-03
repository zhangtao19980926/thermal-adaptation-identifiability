# -*- coding: utf-8 -*-
"""
Smoke tests for tau identifiability.

Cell A (multisine): persistently exciting multi-frequency exposure — with the cascade
band-pass parameterisation and ample data, both population time constants MUST be
recovered; otherwise model/code are wrong.

Cell B (weather): realistic seasonal + OU + heat-wave exposure — tests how strongly a
realistic 180-day longitudinal design identifies the slow constant.
"""
import os, sys, json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from p1mk.synth import simulate_dataset
from p1mk.fit import prep_inputs, fit_model

OUT = r"D:\thermal-adaptation-series\p1\outputs\identifiability"
os.makedirs(OUT, exist_ok=True)
KINDS = ("dual", "single", "fixed7", "fixed30", "static")


def run_cell(name, signal, TRUE_TF, TRUE_TS, dev, n_subjects=40, n_votes=150,
             study_days=180.0, seed=1):
    votes, meta = simulate_dataset(
        n_subjects=n_subjects, n_votes=n_votes, study_days=study_days,
        tau_fast_pop=TRUE_TF, tau_slow_pop=TRUE_TS,
        miss_cov=0.0, regular=False, signal=signal, seed=seed,
        return_states=True)
    dist = np.bincount(votes.TSV + 3, minlength=7).tolist()
    print(f"\n=== Cell {name} ({signal}) === votes={len(votes)} TSV dist={dist}")
    data = prep_inputs(meta, votes, dev, dt=0.2)
    results = {}
    for kind in KINDS:
        r = fit_model(kind, data, n_subjects=n_subjects, epochs=1500, lr=0.03,
                      seed=0, init_tau_fast=4.0, init_tau_slow=40.0, verbose=False)
        results[kind] = r
        msg = f"{kind:8s} NLL/vote={r['nll']:.4f}"
        if kind == "dual":
            msg += (f" | tau_f={r['tau_fast_pop']:.2f} (true {TRUE_TF})"
                    f"  tau_s={r['tau_slow_pop']:.2f} (true {TRUE_TS})"
                    f" | gains slow/mid/fast=({r['b_slow']:.2f},{r['b_mid']:.2f},{r['b_fast']:.2f})")
        elif kind == "single":
            msg += f" | tau={r['tau_pop']:.2f}"
        print(msg)
    d = results["dual"]
    summ = {
        "signal": signal, "true_tau_fast": TRUE_TF, "true_tau_slow": TRUE_TS,
        "est_tau_fast": d["tau_fast_pop"], "est_tau_slow": d["tau_slow_pop"],
        "rel_err_fast": abs(d["tau_fast_pop"] - TRUE_TF) / TRUE_TF,
        "rel_err_slow": abs(d["tau_slow_pop"] - TRUE_TS) / TRUE_TS,
        "corr_tau_fast_i": float(np.corrcoef(d["tau_fast_i"], meta["tau_f"])[0, 1]),
        "corr_tau_slow_i": float(np.corrcoef(d["tau_slow_i"], meta["tau_s"])[0, 1]),
        "nll": {k: results[k]["nll"] for k in KINDS},
        "tsv_dist": dist,
    }
    return summ, results, meta, votes


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", dev, "|", torch.cuda.get_device_name(0) if dev == "cuda" else "CPU")

    summA, resA, metaA, votesA = run_cell("A-multisine", "multisine", 2.0, 21.0, dev)
    summB, resB, metaB, votesB = run_cell("B-weather", "weather", 2.0, 21.0, dev)

    with open(os.path.join(OUT, "smoke_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"A_multisine": summA, "B_weather": summB}, f, indent=2, ensure_ascii=False)

    # ---- figure ----
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for row, (summ, meta, votes, title) in enumerate(
            ((summA, metaA, votesA, "A: multisine (persistent excitation)"),
             (summB, metaB, votesB, "B: realistic weather"))):
        d = resA["dual"] if row == 0 else resB["dual"]
        ax = axes[row, 0]
        ax.scatter(meta["tau_f"], d["tau_fast_i"], c="#2c7fb8", label="fast", alpha=0.8)
        ax.scatter(meta["tau_s"], d["tau_slow_i"], c="#d95f0e", marker="s", label="slow", alpha=0.8)
        lim = [0.5, 500]
        ax.plot(lim, lim, "k--", lw=1)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("true tau (d)"); ax.set_ylabel("estimated tau (d)")
        ax.set_title(f"{title}\npop tau_f={summ['est_tau_fast']:.2f} (true {summ['true_tau_fast']}), "
                     f"tau_s={summ['est_tau_slow']:.1f} (true {summ['true_tau_slow']})", fontsize=9)
        ax.legend(fontsize=8)

        ax = axes[row, 1]
        nlls = [summ["nll"][k] for k in KINDS]
        colors = ["#d95f0e", "#2c7fb8", "#7fbc41", "#7fbc41", "#969696"]
        ax.bar(KINDS, nlls, color=colors)
        ax.set_ylabel("NLL / vote"); ax.tick_params(axis="x", rotation=30)
        ax.set_title(f"model fit (dual={nlls[0]:.3f}, static={nlls[-1]:.3f})", fontsize=9)

        ax = axes[row, 2]
        t = meta["t"]
        ax.plot(t, meta["x"][0], color="#bdbdbd", lw=0.7, label="x(t)")
        ax.plot(t, meta["af"][0], color="#2c7fb8", lw=1.0, label="fast (true)")
        ax.plot(t, meta["as_"][0], color="#d95f0e", lw=1.0, label="slow (true)")
        ax.set_xlabel("day"); ax.set_ylabel("deg C"); ax.legend(fontsize=7)
        ax.set_title("example subject", fontsize=9)
    fig.tight_layout()
    out_png = os.path.join(OUT, "smoke_diagnostic.png")
    fig.savefig(out_png, dpi=140)
    print("\nsaved:", out_png)
    print(json.dumps({"A": {k: v for k, v in summA.items() if k != "nll"},
                      "B": {k: v for k, v in summB.items() if k != "nll"}}, indent=2))


if __name__ == "__main__":
    main()
