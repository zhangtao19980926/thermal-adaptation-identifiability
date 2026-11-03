# -*- coding: utf-8 -*-
"""Figure 5: re-estimating the standards' running mean - CTCD four memory scales by mode and
climate zone, DB II global adaptive slopes by Koppen band, tenure effect, and memory gain."""
import os, sys, numpy as np, pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from figstyle import apply_style, save, export_panels, C, SEQ

apply_style()
ROOT = r"D:\thermal-adaptation-series\p1"
STD = os.path.join(ROOT, "outputs", "standards")
sc = pd.read_csv(os.path.join(STD, "ctcd_window_scales.csv"))
db = pd.read_csv(os.path.join(STD, "db2_window.csv"))
sl = pd.read_csv(os.path.join(STD, "db2_adaptive_slope.csv"))
ly = pd.read_csv(os.path.join(STD, "ctcd_living_years.csv"))

cfgs = ["base(no outdoor memory)", "+m1", "+m7", "+m15", "+m30"]
xlab = ["none", "1 d", "7 d", "15 d", "30 d"]

def row(stratum, metric="mse"):
    d = sc[sc.stratum == stratum].set_index("config")
    return [d.loc[c, metric] for c in cfgs]

fig = plt.figure(figsize=(7.0, 4.25))
gs = fig.add_gridspec(2, 3, hspace=0.80, wspace=0.54)
x = np.arange(5)

# (a) all-sample four scales
axa = fig.add_subplot(gs[0, 0])
y = row("ALL")
axa.plot(x, y, "-o", color=C["blue"])
bi = int(np.argmin(y)); axa.plot(bi, y[bi], "*", ms=15, color=C["red"])
axa.annotate("optimum 15 d", (bi, y[bi]), textcoords="offset points", xytext=(-10, 14),
             fontsize=8.5, color=C["red"], fontweight="bold")
axa.set_xticks(x); axa.set_xticklabels(xlab); axa.set_xlabel("Outdoor memory scale")
axa.set_ylabel("Held-out-city MSE"); axa.set_title("(a) CTCD all\n(n=23,461)")

# (b) by operation mode
axb = fig.add_subplot(gs[0, 1])
for st, col, lab in [("NV", C["green"], "Naturally ventilated"),
                     ("AC", C["orange"], "Air-conditioned"),
                     ("Heated", C["grey"], "Centrally heated")]:
    yy = row(st); axb.plot(x, yy, "-o", color=col, label=lab)
    bi = int(np.argmin(yy)); axb.plot(bi, yy[bi], "*", ms=12, color=col)
axb.set_xticks(x); axb.set_xticklabels(xlab); axb.set_xlabel("Outdoor memory scale")
axb.set_ylabel("Held-out-city MSE"); axb.set_title("(b) By operation\nmode")
axb.text(0.02, 0.06, "memory helps NV/AC,\nhurts centrally heated", transform=axb.transAxes,
         fontsize=7.4, color=C["grey"], style="italic",
         bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.72))
axb.legend(frameon=False, fontsize=7.8)

# (c) by climate zone
axc = fig.add_subplot(gs[0, 2])
zones = [("zone:Cold zone", "Cold", C["blue"]),
         ("zone:Severe cold zone", "Severe cold", C["purple"]),
         ("zone:Hot summer and cold wi", "Hot-sum./cold-w.", C["teal"]),
         ("zone:Hot summer and warm wi", "Hot-sum./warm-w.", C["red"])]
for st, lab, col in zones:
    yy = np.array(row(st)); yy = (yy - yy[0])  # change vs no-memory
    axc.plot(x, yy, "-o", color=col, label=lab, ms=4)
axc.axhline(0, color="k", lw=.8)
axc.set_xticks(x); axc.set_xticklabels(xlab); axc.set_xlabel("Outdoor memory scale")
axc.set_ylabel("ΔMSE vs no memory")
axc.set_title("(c) By climate\nzone")
axc.legend(frameon=False, fontsize=7.0, ncol=2, columnspacing=0.7, handlelength=1.1,
           loc="upper center", bbox_to_anchor=(0.5, -0.24))

# (d) DB II adaptive slopes by Koppen band
axd = fig.add_subplot(gs[1, 0])
bandcol = {"A:Tropical": C["red"], "B:Arid": C["orange"], "C:Temperate": C["blue"], "D:Continental": C["purple"]}
clean = sl[(sl.Tneutral.between(15, 32)) & (sl.out_mon.between(-5, 35))]
xx = np.linspace(-5, 35, 50)
for b in ["B:Arid", "C:Temperate", "D:Continental", "A:Tropical"]:
    d = clean[clean.kband == b]
    col = bandcol[b]
    dplot = d if len(d) <= 180 else d.sample(180, random_state=0)
    if b == "A:Tropical":
        axd.scatter(dplot.out_mon, dplot.Tneutral, s=14, color=col, alpha=.35, edgecolor="none")
        continue
    b1, b0 = np.polyfit(d.out_mon, d.Tneutral, 1)
    axd.scatter(dplot.out_mon, dplot.Tneutral, s=16, color=col, alpha=.75, edgecolor="white", lw=.3,
                label="%s %.2f (n=%d)" % (b[0], b1, len(d)))
    axd.plot(xx, b1 * xx + b0, color=col, lw=1.8)
axd.set_xlabel("Monthly mean outdoor T (°C)"); axd.set_ylabel("Neutral T (°C)")
axd.set_title("(d) DB II: neutral T vs\noutdoor T")
axd.legend(frameon=True, facecolor="white", framealpha=0.82, edgecolor="none",
           fontsize=7.0, loc="upper left")
axd.text(2, 16.5, "tropical: narrow range,\nslope not identifiable", fontsize=7.3, color=C["red"],
         bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.72))

# (e) tenure / long acclimatisation
axe = fig.add_subplot(gs[1, 1])
labels = ["≤2 y", "3-10 y", ">10 y"]
vals = ly.slope_residTSV_on_monthlyOut.values * 100  # per 100? keep per-deg -> x100 for visibility
axe.bar(range(3), ly.slope_residTSV_on_monthlyOut.values, color=[C["lblue"], C["blue"], C["dgrey"]], width=.6)
for i, (v, n) in enumerate(zip(ly.slope_residTSV_on_monthlyOut, ly.n)):
    axe.text(i, v - 0.002, "%.3f\n(n=%d)" % (v, n), ha="center", va="top", fontsize=8)
axe.axhline(0, color="k", lw=.8); axe.set_xticks(range(3)); axe.set_xticklabels(labels)
axe.set_ylabel("Residual TSV–monthly T slope")
axe.set_title("(e) By years of\nresidence")
axe.set_ylim(min(ly.slope_residTSV_on_monthlyOut) * 1.5, 0.006)

# (f) DB II memory gain by band (both vs base)
axf = fig.add_subplot(gs[1, 2])
bands = ["A:Tropical", "B:Arid", "C:Temperate", "D:Continental"]
pct = [(db[db.stratum == b]["+both_mse"].iloc[0] / db[db.stratum == b].base_mse.iloc[0] - 1) * 100 for b in bands]
cols = [bandcol[b] for b in bands]
bars = axf.bar(range(4), pct, color=cols, width=.62)
axf.axhline(0, color="k", lw=.8)
axf.set_xticks(range(4)); axf.set_xticklabels(["Trop.", "Arid", "Temp.", "Cont."])
axf.set_ylabel("Δ MSE, memory added (%)")
axf.set_title("(f) Outdoor-memory\ngain by climate")
for i, v in enumerate(pct):
    axf.text(i, v + (0.15 if v >= 0 else -0.15), "%+.1f%%" % v, ha="center",
             va="bottom" if v >= 0 else "top", fontsize=8.5, fontweight="bold")

export_panels(fig, [(axa, "r0c0"), (axb, "r0c1"), (axc, "r0c2"),
                    (axd, "r1c0"), (axe, "r1c1"), (axf, "r1c2")], "Fig5_standards")
save(fig, "Fig5_standards")
