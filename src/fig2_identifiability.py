# -*- coding: utf-8 -*-
"""Figure 2: slow time-constant identifiability - synthetic critical surface (known/free),
real-band tau_s recovery, real-band window scan, and long-window incremental gain."""
import os, sys, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
sys.path.insert(0, os.path.dirname(__file__))
from figstyle import apply_style, panel_label, save, export_panels, C

apply_style()
ROOT = r"D:\thermal-adaptation-series\p1"

im = pd.read_csv(os.path.join(ROOT, "outputs", "identifiability", "ident_map.csv"))
inj = pd.read_csv(os.path.join(ROOT, "outputs", "identifiability", "realband_injection.csv"))
scan = pd.read_csv(os.path.join(ROOT, "outputs", "standards", "realband_window_scan.csv"))
dual = pd.read_csv(os.path.join(ROOT, "outputs", "standards", "realband_dual_window.csv"))

Ns = [20, 40, 100]; Ts = [90, 180, 365]
def grid(regime, col):
    d = im[im.regime == regime]
    return d.pivot(index="N", columns="T", values=col).reindex(index=Ns, columns=Ts).values

fig = plt.figure(figsize=(7.0, 3.94))
gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0], width_ratios=[1.06, 1.06, 0.88],
                      hspace=0.62, wspace=0.46)

def heat(ax, regime, title):
    Z = grid(regime, "delta_nll_total"); A = grid(regime, "argmin_ts")
    vmax = np.nanmax(np.abs(Z))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    cmap = plt.get_cmap("RdYlGn", 9)   # discrete steps so the colour bar stays vector in SVG
    for i in range(3):
        for j in range(3):
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1,
                                   facecolor=cmap(norm(Z[i, j])),
                                   edgecolor="white", lw=0.5, zorder=1))
    ax.set_xlim(-0.5, 2.5); ax.set_ylim(-0.5, 2.5); ax.invert_yaxis()
    ax.set_xticks(range(3)); ax.set_xticklabels(Ts)
    ax.set_yticks(range(3)); ax.set_yticklabels(Ns)
    ax.set_xlabel("Study duration T (days)"); ax.set_ylabel("Subjects N")
    ax.set_title(title)
    for i in range(3):
        for j in range(3):
            z, a = Z[i, j], A[i, j]
            col = "white" if abs(z) > 0.55 * vmax else "#1a1a1a"
            ax.text(j, i, "%.1f\nτ_s=%.0fd" % (z, a), ha="center", va="center",
                    fontsize=7.6, color=col)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.06, pad=0.04)
    cb.solids.set_edgecolor("face")
    cb.solids.set_rasterized(False)
    return cb

axa = fig.add_subplot(gs[0, 0]); cba = heat(axa, "known", "(a) Information ceiling")
axb = fig.add_subplot(gs[0, 1]); cbb = heat(axb, "free", "(b) Realistic\n(latent Tn, bias)")

# (c) real-band tau_s recovery
axc = fig.add_subplot(gs[0, 2])
labels = ["In-Gauge\n(strong)", "DORN\n(weak)"]
x = np.arange(2); w = 0.36
n40 = [inj[(inj.band == "InGauge_strong") & (inj.N == 40)].tau_slow.iloc[0],
       inj[(inj.band == "DORN_weak") & (inj.N == 40)].tau_slow.iloc[0]]
n100 = [inj[(inj.band == "InGauge_strong") & (inj.N == 100)].tau_slow.iloc[0],
        inj[(inj.band == "DORN_weak") & (inj.N == 100)].tau_slow.iloc[0]]
axc.bar(x - w/2, n40, w, label="N=40", color=[C["lgreen"], "#f5b041"])
axc.bar(x + w/2, n100, w, label="N=100", color=[C["green"], C["orange"]])
axc.axhline(21, color=C["red"], ls="--", lw=1.6, label="true τ_s=21 d")
axc.set_xticks(x); axc.set_xticklabels(labels); axc.set_ylabel("Estimated τ_s (days)")
axc.set_ylim(0, 85); axc.set_title("(c) τ_s recovery on\nREAL slow bands")
for xi, v in zip(x - w/2, n40): axc.text(xi, v + 1.5, "%.0f" % v, ha="center", fontsize=8)
for xi, v in zip(x + w/2, n100): axc.text(xi, v + 1.5, "%.0f" % v, ha="center", fontsize=8)
axc.legend(frameon=False, fontsize=7.8)

# (d) window scan
axd = fig.add_subplot(gs[1, 0:2])
for band, col, lab, mk in [("InGauge_strong", C["green"], "In-Gauge strong slow band", "o"),
                          ("DORN_weak", C["orange"], "DORN weak slow band", "s")]:
    q = scan[scan.band == band]
    axd.plot(q.w, q.logloss, marker=mk, color=col, label=lab)
    imn = q.loc[q.logloss.idxmin()]
    axd.annotate("optimum w=%gd" % int(imn.w), (imn.w, imn.logloss),
                 textcoords="offset points", xytext=(8, 8 if band.startswith("In") else -14),
                 fontsize=8, color=col, fontweight="bold")
axd.set_xlabel("Causal rectangular memory window w (days)")
axd.set_ylabel("Held-out-subject log-loss")
axd.set_title("(d) Window loss profile\n(sharp V = identifiable; flat = not)")
axd.legend(frameon=False)

# (e) long-window incremental gain
axe = fig.add_subplot(gs[1, 2])
g = dual.groupby("band").dLong_ll.agg(["mean", "std"]).reindex(["InGauge_strong", "DORN_weak"])
bars = axe.bar(range(2), g["mean"], yerr=g["std"], color=[C["green"], C["orange"]],
               capsize=4, width=0.55)
axe.set_xticks(range(2)); axe.set_xticklabels(["In-Gauge\nstrong", "DORN\nweak"])
axe.set_ylabel("Δ log-loss from adding 30-d memory")
axe.set_title("(e) Independent\nLONG-memory gain")
for i, v in enumerate(g["mean"]): axe.text(i, v + 0.001, "%.3f" % v, ha="center", fontsize=8.5, fontweight="bold")
axe.set_ylim(0, max(g["mean"]) * 1.35)

export_panels(fig, [(axa, "r0c0", [cba.ax]), (axb, "r0c1", [cbb.ax]), (axc, "r0c2"),
                    (axd, "r1c0_span2"), (axe, "r1c2")], "Fig2_identifiability")
save(fig, "Fig2_identifiability")
