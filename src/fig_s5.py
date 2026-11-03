# -*- coding: utf-8 -*-
"""Fig S5: full window scans under both MSE and log-loss, CTCD and DB II.
Honestly shows where the two metrics disagree (MSE improves but log-loss does not).
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from figstyle import apply_style, save, export_panels, C

apply_style()
R = r"D:\thermal-adaptation-series\p1\outputs\standards"
ct = pd.read_csv(os.path.join(R, "ctcd_window_scales.csv"))
db = pd.read_csv(os.path.join(R, "db2_window.csv"))
sl = pd.read_csv(os.path.join(R, "db2_adaptive_slope.csv"))

fig = plt.figure(figsize=(7.0, 4.7), layout="constrained")
gs = fig.add_gridspec(2, 3, hspace=.36, wspace=.22)
axA = fig.add_subplot(gs[0, 0]); axB = fig.add_subplot(gs[0, 1]); axC = fig.add_subplot(gs[0, 2])
axD = fig.add_subplot(gs[1, 0]); axE = fig.add_subplot(gs[1, 1]); axF = fig.add_subplot(gs[1, 2])

cfg = ["base(no outdoor memory)", "+m1", "+m7", "+m15", "+m30"]
wlab = ["base", "1d", "7d", "15d", "30d"]
xw = np.arange(5)

def ct_row(stratum, col):
    s = ct[ct.stratum == stratum].set_index("config")
    return np.array([s.loc[c, col] for c in cfg], float)

# (a) CTCD ALL: percent change vs base, both metrics
mse = ct_row("ALL", "mse"); ll = ct_row("ALL", "logloss")
axA.plot(xw, (mse / mse[0] - 1) * 100, "-o", color=C["blue"], lw=2, ms=5, label="MSE")
axA.plot(xw, (ll / ll[0] - 1) * 100, "-s", color=C["orange"], lw=2, ms=5, label="Log-loss")
axA.axhline(0, color=C["dgrey"], lw=1.2)
axA.set_xticks(xw); axA.set_xticklabels(wlab, fontsize=7.2, rotation=40, ha="right")
axA.set_ylabel("Change vs base (%)", fontsize=9)
axA.set_title("(a) CTCD all votes:\ntwo metrics disagree", loc="left", fontsize=9.2)
axA.legend(fontsize=7, frameon=True, framealpha=.85, edgecolor="none", loc="center right")
axA.grid(axis="y", alpha=.25, lw=.7)

# (b) CTCD climate zones MSE
zones = [("zone:Cold zone", "Cold", C["blue"]),
         ("zone:Severe cold zone", "Severe", C["lblue"]),
         ("zone:Hot summer and cold wi", "HSCW", C["orange"]),
         ("zone:Hot summer and warm wi", "HSWW", C["red"])]
for key, lab, col in zones:
    v = ct_row(key, "mse")
    axB.plot(xw, (v / v[0] - 1) * 100, "-o", color=col, lw=1.7, ms=4, label=lab)
axB.axhline(0, color=C["dgrey"], lw=1.2)
axB.set_xticks(xw); axB.set_xticklabels(wlab, fontsize=7.2, rotation=40, ha="right")
# ylabel shared with left column (a)/(d)
axB_set_ylabel = None
axB.set_title("(b) CTCD climate\nzones (MSE)", loc="left", fontsize=9.2)
axB.legend(fontsize=6.8, ncol=2, frameon=True, framealpha=.85, edgecolor="none", loc="best")
axB.grid(axis="y", alpha=.25, lw=.7)

# (c) CTCD operation modes MSE
ops = [("NV", "Nat. vent.", C["green"]),
       ("AC", "Air-cond.", C["blue"]),
       ("Heated", "Heated", C["red"])]
for key, lab, col in ops:
    v = ct_row(key, "mse")
    axC.plot(xw, (v / v[0] - 1) * 100, "-o", color=col, lw=1.8, ms=4.5, label=lab)
axC.axhline(0, color=C["dgrey"], lw=1.2)
axC.set_xticks(xw); axC.set_xticklabels(wlab, fontsize=7.2, rotation=40, ha="right")
# ylabel shared with left column (a)/(d)
axC_set_ylabel = None
axC.set_title("(c) CTCD operation\nmodes (MSE)", loc="left", fontsize=9.2)
axC.legend(fontsize=6.8, frameon=True, framealpha=.85, edgecolor="none", loc="center right")
axC.grid(axis="y", alpha=.25, lw=.7)

# (d)(e) DB II bands: now/mon/both percent change vs base
bands = ["ALL", "A:Tropical", "B:Arid", "C:Temperate", "D:Continental"]
blab = ["ALL", "Trop", "Arid", "Temp", "Cont"]
xb = np.arange(len(bands)); w = .25
metrics = [("mse", "+out_now(instant)_mse", "+out_mon(monthly)_mse", "+both_mse", axD, "MSE"),
           ("ll", "+out_now(instant)_ll", "+out_mon(monthly)_ll", "+both_ll", axE, "Log-loss")]
for basecol, cnow, cmon, cboth, ax, name in metrics:
    base = db.set_index("stratum").loc[bands, "base_" + basecol].values.astype(float)
    now = db.set_index("stratum").loc[bands, cnow].values.astype(float)
    mon = db.set_index("stratum").loc[bands, cmon].values.astype(float)
    both = db.set_index("stratum").loc[bands, cboth].values.astype(float)
    ax.bar(xb - w, (now / base - 1) * 100, w, color=C["lblue"], label="inst.", edgecolor="none")
    ax.bar(xb, (mon / base - 1) * 100, w, color=C["blue"], label="mon.", edgecolor="none")
    ax.bar(xb + w, (both / base - 1) * 100, w, color=C["green"], label="both", edgecolor="none")
    ax.axhline(0, color=C["dgrey"], lw=1.2)
    ax.set_xticks(xb); axB_t = blab
    ax.set_xticklabels(blab, fontsize=7.2, rotation=40, ha="right")
    if name == "MSE":
        ax.set_ylabel("%s change vs base (%%)" % name, fontsize=9)
    ax.set_title("(%s) DB II bands (%s)" % ("d" if name == "MSE" else "e", name),
                 loc="left", fontsize=9.2)
    ax.legend(fontsize=6.6, frameon=True, framealpha=.85, edgecolor="none", loc="upper center", ncol=3)
    ax.grid(axis="y", alpha=.25, lw=.7)

# (f) city neutral temperature vs monthly outdoor, by band
bcol = {"A:Tropical": C["red"], "B:Arid": C["orange"], "C:Temperate": C["blue"], "D:Continental": C["purple"]}
# drop cities with non-physical (non-converged) neutral-temperature fits
sl = sl[(sl.Tneutral > 5) & (sl.Tneutral < 35) & (sl.out_mon > -20) & (sl.out_mon < 35)].copy()
for bnd, col in bcol.items():
    s0 = sl[sl.kband == bnd]
    axF.scatter(s0.out_mon, s0.Tneutral, s=22, color=col, alpha=.7, edgecolor="white", lw=.4,
                label="%s (n=%d)" % (bnd.split(":")[1], len(s0)), zorder=3)
lo = min(sl.Tneutral.min(), sl.out_mon.min()) - 1
hi = max(sl.Tneutral.max(), sl.out_mon.max()) + 1
axF.plot([lo, hi], [lo, hi], color=C["grey"], ls="--", lw=1.3)
axF.set_xlim(lo, hi); axF.set_ylim(lo, hi)
axF.set_xlabel("Monthly mean outdoor T (\u00b0C)", fontsize=9)
# f y-axis meaning moved into panel title
axF.set_title("(f) Estimated neutral vs\noutdoor T (slope 1, \u00b0C)", loc="left", fontsize=9.2)
axF.legend(fontsize=6.8, frameon=True, framealpha=.92, edgecolor="none",
           loc="best")
axF.grid(alpha=.25, lw=.7)

fig.suptitle("Full running-window scans under both error metrics (CTCD and ASHRAE DB II)",
             fontsize=12.5, fontweight="bold")
export_panels(fig, [(axA, "r0c0"), (axB, "r0c1"), (axC, "r0c2"),
                    (axD, "r1c0"), (axE, "r1c1"), (axF, "r1c2")], "FigS5_windows")
save(fig, "FigS5_windows")
print("FigS5 done")
