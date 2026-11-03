# -*- coding: utf-8 -*-
"""Fig S6: full-season In-Gauge counterfactual: running means, adaptive set-points,
set-point bias of fixed windows, and the realised cooling schedule."""
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from figstyle import apply_style, save, export_panels, C

apply_style()
R = r"D:\thermal-adaptation-series\p1\outputs\standards"
df = pd.read_csv(os.path.join(R, "ingauge_counterfactual.csv"))
d = df.day.values

fig = plt.figure(figsize=(7.0, 4.7), layout="constrained")
gs = fig.add_gridspec(2, 2, hspace=.26, wspace=.24)
axA = fig.add_subplot(gs[0, 0]); axB = fig.add_subplot(gs[0, 1])
axC = fig.add_subplot(gs[1, 0]); axD = fig.add_subplot(gs[1, 1])

# (a) outdoor temperature and three running means
axA.plot(d, df.Tout, color=C["grey"], lw=.8, alpha=.8, label="Outdoor T")
axA.plot(d, df.Trm7, color=C["blue"], lw=1.5, label="7-d mean")
axA.plot(d, df.Trm30, color=C["orange"], lw=1.5, label="30-d mean")
axA.plot(d, df.Trm_tau21, color=C["green"], lw=2.2, label="\u03c4_s = 21 d memory")
axA.set_xlabel("Study day", fontsize=9.5); axA.set_ylabel("Temperature (\u00b0C)", fontsize=9.5)
axA.set_title("(a) Forcing & three\nthermal-memory windows", loc="left", fontsize=10)
axA.legend(fontsize=7.6, ncol=2, frameon=True, framealpha=.85, edgecolor="none", loc="upper right")
axA.grid(alpha=.25, lw=.7)

# (b) three adaptive comfort set-points
axB.plot(d, df.Tcomf7, color=C["blue"], lw=1.4, label="7-d set-point")
axB.plot(d, df.Tcomf30, color=C["orange"], lw=1.4, label="30-d set-point")
axB.plot(d, df.Tcomf_tau21, color=C["green"], lw=2.2, label="\u03c4_s = 21 d set-point")
axB.set_xlabel("Study day", fontsize=9.5); axB.set_ylabel("Adaptive set-point (\u00b0C)", fontsize=9.5)
axB.set_title("(b) Resulting\nadaptive set-points", loc="left", fontsize=10)
axB.legend(fontsize=7.8, frameon=True, framealpha=.85, edgecolor="none", loc="best")
axB.grid(alpha=.25, lw=.7)

# (c) set-point bias relative to the 21-day memory rule
b7 = df.Tcomf7 - df.Tcomf_tau21
b30 = df.Tcomf30 - df.Tcomf_tau21
axC.fill_between(d, 0, b7, color=C["blue"], alpha=.35, label="7-d minus \u03c421")
axC.fill_between(d, 0, b30, color=C["orange"], alpha=.35, label="30-d minus \u03c421")
axC.axhline(0, color=C["dgrey"], lw=1.2)
axC.set_xlabel("Study day", fontsize=9.5)
axC.set_ylabel("Set-point bias (\u00b0C)", fontsize=9.5)
axC.set_title("(c) Fixed-window bias\nover the full season", loc="left", fontsize=10)
axC.legend(fontsize=7.8, frameon=True, framealpha=.85, edgecolor="none", loc="upper right")
axC.grid(alpha=.25, lw=.7)

# (d) realised cooling fraction
axD.fill_between(d, 0, df.cool_frac, color=C["lblue"], alpha=.7, zorder=3)
axD.plot(d, df.cool_frac, color=C["blue"], lw=1.2, zorder=4)
axD.set_xlabel("Study day", fontsize=9.5)
axD.set_ylabel("Cooling fraction of occupied hours", fontsize=9.5)
axD.set_title("(d) Cooling schedule\n(no heatwave in record)", loc="left", fontsize=10)
axD.text(0.03, 0.92, "mean = %.3f%%   peak = %.2f%%" %
         (df.cool_frac.mean() * 100, df.cool_frac.max() * 100),
         transform=axD.transAxes, fontsize=8.4, color=C["dgrey"], va="top")
axD.grid(alpha=.25, lw=.7)
axD.set_ylim(0, max(.05, df.cool_frac.max() * 1.25))

fig.suptitle("Full-season counterfactual: memory windows, adaptive set-points and cooling",
             fontsize=12.5, fontweight="bold")
export_panels(fig, [(axA, "r0c0"), (axB, "r0c1"), (axC, "r1c0"), (axD, "r1c2")], "FigS6_counter")
save(fig, "FigS6_counter")
print("FigS6 done; cool mean=%.4f peak=%.4f" % (df.cool_frac.mean(), df.cool_frac.max()))
