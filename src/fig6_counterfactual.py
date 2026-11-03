# -*- coding: utf-8 -*-
"""Figure 6: standards implications / counterfactual - adaptive set-point trajectories under
fixed 7-d, fixed 30-d and learned tau_s=21 d memory; transition bias, recovery, indicative
cooling-degree-hours, and the proposed excitation-adaptive (graded) memory recommendation."""
import os, sys, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
sys.path.insert(0, os.path.dirname(__file__))
from figstyle import apply_style, save, export_panels, C

apply_style()
ROOT = r"D:\thermal-adaptation-series\p1"
cf = pd.read_csv(os.path.join(ROOT, "outputs", "standards", "ingauge_counterfactual.csv"))
cf["d7"] = cf.Tcomf7 - cf.Tcomf_tau21
cf["d30"] = cf.Tcomf30 - cf.Tcomf_tau21
tr = (cf.day >= 15) & (cf.day <= 90)
bias7, int7 = cf.d7[tr].mean(), cf.d7[tr].sum()

fig = plt.figure(figsize=(7.0, 4.9))
gs = fig.add_gridspec(2, 3, hspace=0.72, wspace=0.42)

# (a) full trajectories
axa = fig.add_subplot(gs[0, 0])
axa.plot(cf.day, cf.Tout, color=C["grey"], lw=1.0, alpha=.7, label="Daily outdoor T")
axa.plot(cf.day, cf.Tcomf7, color=C["red"], lw=2.0, label="Fixed 7-d window")
axa.plot(cf.day, cf.Tcomf30, color=C["purple"], lw=1.8, ls="--", label="Fixed 30-d window")
axa.plot(cf.day, cf.Tcomf_tau21, color=C["green"], lw=2.4, label="Learned τ_s=21 d")
axa.axvspan(83, 86, color=C["orange"], alpha=.18)
axa.set_xlabel("Study day (Melbourne)"); axa.set_ylabel("Temperature (°C)")
axa.set_title("(a) Adaptive set-point\nby memory rule")
axa.legend(frameon=True, facecolor="white", framealpha=0.82, edgecolor="none", fontsize=7.3)

# (b) zoom
axb = fig.add_subplot(gs[0, 1])
z = (cf.day >= 70) & (cf.day <= 115)
axb.plot(cf.day[z], cf.Tout[z], color=C["grey"], lw=1.2)
axb.plot(cf.day[z], cf.Tcomf7[z], color=C["red"], lw=2.0)
axb.plot(cf.day[z], cf.Tcomf30[z], color=C["purple"], lw=1.8, ls="--")
axb.plot(cf.day[z], cf.Tcomf_tau21[z], color=C["green"], lw=2.4)
axb.axvspan(83, 86, color=C["orange"], alpha=.25)
axb.annotate("warm spell", xy=(84.5, 27.6), xytext=(96, 28.4), fontsize=7.6, color=C["orange"],
             arrowprops=dict(arrowstyle="->", color=C["orange"], lw=1.2))
axb.set_xlabel("Study day"); axb.set_ylabel("Temperature (°C)")
axb.set_title("(b) Spring-to-summer\ntransition zoom")

# (c) transition bias
axc = fig.add_subplot(gs[0, 2])
dd = cf[tr]
axc.fill_between(dd.day, 0, dd.d7, color=C["red"], alpha=.35, label="7-d − τ_s21")
axc.plot(dd.day, dd.d7, color=C["red"], lw=1.4)
axc.plot(dd.day, dd.d30, color=C["purple"], lw=1.6, ls="--", label="30-d − τ_s21")
axc.axhline(0, color="k", lw=.8)
axc.set_xlabel("Study day (transition window 15-90)"); axc.set_ylabel("Set-point bias (°C)")
axc.set_title("(c) 7-d window overshoots\nin warming")
axc.text(0.97, 0.06, "mean %+.2f °C\nintegral %+.1f °C·day" % (bias7, int7),
         transform=axc.transAxes, ha="right", va="bottom", fontsize=7.6,
         bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.75))
axc.legend(frameon=True, facecolor="white", framealpha=0.82, edgecolor="none",
           fontsize=7.6, loc="upper right")

# (d) recovery half-time
axd = fig.add_subplot(gs[1, 0])
rec = [5, 84, 84]; labs = ["7-d", "30-d", "τ_s=21 d"]; cols = [C["red"], C["purple"], C["green"]]
axd.bar(range(3), rec, color=cols, width=.55)
for i, v in enumerate(rec):
    if v < 80:
        axd.text(i, v + 2, "%d d" % v, ha="center", fontsize=8.2, fontweight="bold")
    else:
        ylab = 74 if i == 1 else 60
        axd.text(i, ylab, "not recovered\nwithin study", ha="center", va="center",
                 fontsize=7.6, fontweight="bold")
axd.set_xticks(range(3)); axd.set_xticklabels(labs); axd.set_ylim(0, 95)
axd.set_ylabel("Set-point recovery time after warm spell (days)")
axd.set_title("(d) Warm-spell\nrecovery time")

# (e) indicative CDH
axe = fig.add_subplot(gs[1, 1])
cdh = [259, 260, 256]; pct = [0.0, 0.2, -1.4]
axe.bar(range(3), cdh, color=cols, width=.55)
for i, (v, p) in enumerate(zip(cdh, pct)):
    axe.text(i, v + 2, "%d\n(%+.1f%%)" % (v, p), ha="center", fontsize=8.2)
axe.set_xticks(range(3)); axe.set_xticklabels(labs); axe.set_ylim(240, 272)
axe.set_ylabel("Indicative occupied cooling-degree-hours")
axe.set_title("(e) Cooling-degree-hours\n(cooling 0.86% of time)")

# (f) excitation-adaptive graded recommendation
axf = fig.add_subplot(gs[1, 2]); axf.axis("off")
axf.set_xlim(0, 10); axf.set_ylim(0, 10)
axf.text(5, 9.95, "(f) Excitation-adaptive\nmemory rule",
         ha="center", va="top", fontsize=10.5, fontweight="bold")
grades = [
    (6.20, "Weak: steady trop.", "heated → instantaneous", C["lblue"]),
    (4.50, "Moderate: temperate", "mixed → 7–15 d mean", C["orange"]),
    (2.80, "Strong: NV, cold/", "warm-winter → 21–30 d", C["green"]),
]
for yb, cond, recm, col in grades:
    axf.add_patch(Rectangle((0.2, yb), 9.6, 1.40, facecolor=col, alpha=.18, edgecolor=col, lw=1.6))
    axf.text(0.5, yb + 0.96, cond, ha="left", va="center", fontsize=7.4, fontweight="bold")
    axf.text(0.5, yb + 0.42, recm, ha="left", va="center", fontsize=7.2, color="#1a1a1a")
axf.annotate("", xy=(9.72, 3.2), xytext=(9.72, 7.4),
             arrowprops=dict(arrowstyle="-|>", color=C["dgrey"], lw=2.2))
axf.text(5, 2.45, "Weak: τ_s drifts 73/681 d\nstrong band: 20–23 d\nCTCD optimum 7 d→30 d\n"
                  "heated buildings decouple",
         ha="center", va="top", fontsize=6.6, color=C["dgrey"])

export_panels(fig, [(axa, "r0c0"), (axb, "r0c1"), (axc, "r0c2"),
                    (axd, "r1c0"), (axe, "r1c1"), (axf, "r1c2")], "Fig6_counterfactual")
save(fig, "Fig6_counterfactual")
