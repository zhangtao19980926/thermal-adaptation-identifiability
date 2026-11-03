# -*- coding: utf-8 -*-
"""Fig S2: model hierarchy, controlled constant recovery and seed stability."""
import os, sys, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from figstyle import apply_style, save, export_panels, C

apply_style()
R = r"D:\thermal-adaptation-series\p1\outputs"
inj = pd.read_csv(os.path.join(R, "identifiability", "realband_injection.csv"))
dw = pd.read_csv(os.path.join(R, "standards", "realband_dual_window.csv"))
smk = json.load(open(os.path.join(R, "identifiability", "smoke_summary.json"), encoding="utf-8"))

fig = plt.figure(figsize=(7.0, 4.7), layout="constrained")
gs = fig.add_gridspec(2, 2, hspace=.26, wspace=.24)
axA = fig.add_subplot(gs[0, 0]); axB = fig.add_subplot(gs[0, 1])
axC = fig.add_subplot(gs[1, 0]); axD = fig.add_subplot(gs[1, 1])

# (a) five-model held-out NLL, N=100 rows
models = ["static", "fixed7", "fixed30", "single", "dual"]
labels = ["Static", "Fixed 7 d", "Fixed 30 d", "Single-exp", "Dual (ours)"]
mcols = [C["grey"], C["lblue"], C["blue"], C["orange"], C["green"]]
sub = inj[inj.N == 100].set_index("band")
bands = ["InGauge_strong", "DORN_weak"]
blab = ["In-Gauge strong band", "DORN weak band"]
x = np.arange(len(bands)); w = .15
for j, m in enumerate(models):
    vals = [sub.loc[b, "nll_" + m] for b in bands]
    axA.bar(x + (j - 2) * w, vals, w, color=mcols[j], label=labels[j],
            edgecolor="none", zorder=3)
axA.set_xticks(x); axA.set_xticklabels(blab, fontsize=8.6)
axA.set_ylabel("Held-out NLL per vote", fontsize=9.5)
axA.set_title("(a) Five-model hierarchy\n(N = 100)", loc="left", fontsize=10)
axA.legend(fontsize=7.6, ncol=1, frameon=False, loc="upper left")
axA.grid(axis="y", alpha=.25, lw=.7)
axA.set_ylim(0, 1.85)

# (b) dual improvement over fixed30 and static, all 4 rows
xx = np.arange(len(inj)); wb = .38
axB.bar(xx - wb/2, inj.dNLL_dual_vs_fixed30.values, wb, color=C["blue"],
        label="Dual vs fixed 30 d", edgecolor="white", lw=.5, zorder=3)
axB.bar(xx + wb/2, inj.dNLL_dual_vs_static.values, wb, color=C["green"],
        label="Dual vs static", edgecolor="white", lw=.5, zorder=3)
axB.axhline(0, color=C["dgrey"], lw=1)
axB.set_xticks(xx)
axB.set_xticklabels(["In-Gauge\nN=40", "In-Gauge\nN=100", "DORN\nN=40", "DORN\nN=100"], fontsize=8)
axB.set_ylabel("\u0394 NLL improvement", fontsize=9.5)
axB.set_title("(b) Dual-kernel\nlikelihood gain", loc="left", fontsize=10)
axB.legend(fontsize=8, frameon=False, loc="upper right")
axB.grid(axis="y", alpha=.25, lw=.7)

# (c) controlled recovery of tau_s
keys = [k for k in smk if "multisine" in k or "weather" in k]
names = ["Multi-sine", "Weather-like"]
est = [smk[k]["est_tau_slow"] for k in keys]
ferr = [smk[k].get("rel_err_slow", np.nan) for k in keys]
xc = np.arange(len(keys))
bars = axC.bar(xc, est, .5, color=[C["purple"], C["orange"]], edgecolor="white", lw=.8, zorder=3)
axC.axhline(21.0, color=C["green"], lw=1.5, ls="--", zorder=2)
axC.text(0.03, 22.3, "true \u03c4_s = 21 d", color=C["green"], fontsize=8.4, va="bottom", ha="left")
for i, v in enumerate(est):
    axC.text(i, v + 1.2, "%.1f d" % v, ha="center", fontsize=9, color=C["dgrey"])
axC.set_xticks(xc); axC.set_xticklabels(names, fontsize=9)
axC.set_ylabel("Recovered \u03c4_s (days)", fontsize=9.5)
axC.set_title("(c) Recovery of the\nslow constant τ_s", loc="left", fontsize=10)
axC.set_ylim(0, 46); axC.grid(axis="y", alpha=.25, lw=.7)

# (d) seed stability M0-M3
dwi = dw[dw.band == "InGauge_strong"]
mm = ["M0_inst_ll", "M1_short_ll", "M2_long_ll", "M3_dual_ll"]
mlab = ["M0 instant", "M1 short", "M2 long", "M3 dual"]
xd = np.arange(4)
for _, row in dwi.iterrows():
    axD.plot(xd, [row[m] for m in mm], "-o", color=C["grey"], ms=4, lw=1, alpha=.7, zorder=2)
means = [dwi[m].mean() for m in mm]
axD.plot(xd, means, "-s", color=C["red"], ms=8, lw=2, label="mean over seeds", zorder=3)
for i, v in enumerate(means):
    axD.text(i, v + .006, "%.3f" % v, ha="center", va="bottom", fontsize=8, color=C["red"])
axD.set_xticks(xd); axD.set_xticklabels(mlab, fontsize=8.4)
axD.set_ylim(1.35, 1.55)
axD.set_ylabel("NLL per vote", fontsize=9.5)
axD.set_title("(d) Seed stability\n(In-Gauge, %d seeds)" % len(dwi), loc="left", fontsize=10)
axD.legend(fontsize=8, frameon=False, loc="upper right")
axD.grid(axis="y", alpha=.25, lw=.7)

fig.suptitle("Model hierarchy, controlled recovery and estimation stability",
             fontsize=12.5, fontweight="bold")
export_panels(fig, [(axA, "r0c0"), (axB, "r0c1"), (axC, "r1c0"), (axD, "r1c2")], "FigS2_models")
save(fig, "FigS2_models")
print("FigS2 done; smoke keys:", list(smk.keys()))
