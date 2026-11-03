# -*- coding: utf-8 -*-
"""Fig S1: profile-likelihood identifiability of the slow constant.
Delta-NLL profiles over tau_s for known vs freely-estimated nuisance parameters,
across cohort size N and study duration T. 95% profile-likelihood threshold = 1.92.
"""
import os, sys, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from figstyle import apply_style, save, export_panels, C

apply_style()
R = r"D:\thermal-adaptation-series\p1\outputs"
profs = json.load(open(os.path.join(R, "identifiability", "ident_map_profiles.json"), encoding="utf-8"))
tau_grid = list(pd.read_csv(os.path.join(R, "identifiability", "cellII_nll.csv")).iloc[:, 0]) + [150.0]
tau_grid = np.array(tau_grid, float)
THR = 1.92

def find(regime, N, T):
    for e in profs:
        if e["regime"] == regime and e["N"] == N and e["T"] == T:
            return e
    raise KeyError((regime, N, T))

combos = [
    ("known", 100, 365), ("known", 40, 180), ("known", 20, 90),
    ("free", 100, 365), ("free", 40, 180), ("free", 20, 90),
]
fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.7), layout="constrained",
                         gridspec_kw=dict(hspace=.30, wspace=.30))
axes = axes.ravel()
panel_names = ["r0c0", "r0c1", "r0c2", "r1c0", "r1c1", "r1c2"]
for k, (regime, N, T) in enumerate(combos):
    ax = axes[k]
    e = find(regime, N, T)
    p = np.array(e["profile"], float)
    nv = float(e.get("n_votes", 1))
    dnll = (p - p.min()) * nv     # per-vote profile -> summed log-likelihood scale
    top = max(2.3, min(dnll.max(), 4.0) * 1.12)
    dplot = np.minimum(dnll, top)   # clamp off-scale arms so they never leave the axes
    col = C["blue"] if regime == "known" else C["orange"]
    ax.semilogx(tau_grid, dplot, "-o", color=col, ms=4.2, lw=1.7, zorder=3)
    ax.axhline(THR, color=C["grey"], ls="--", lw=1.2, zorder=2)
    ax.axvline(21.0, color=C["green"], ls="-", lw=1.3, alpha=.75, zorder=1)
    im = int(np.argmin(dnll))
    ax.plot(tau_grid[im], dplot[im], "o", ms=8, mfc=C["red"], mec="white", mew=1.0, zorder=4)
    tag = "Known nuisance" if regime == "known" else "Free nuisance"
    ax.set_title("%s \u00b7 N=%d, T=%d d" % (tag, N, T), fontsize=10)
    ax.set_ylim(-0.1, top * 1.07)
    note = "argmin = %g d" % e["argmin_ts"]
    if dnll[-1] > 4.0:
        note += "\n\u0394NLL(150 d) = %.0f (off scale)" % dnll[-1]
    ax.text(0.97, 0.94, note, transform=ax.transAxes,
            ha="right", va="top", fontsize=7.6, color=C["dgrey"])
    if k % 3 == 0:
        ax.set_ylabel("\u0394 NLL (summed over votes)", fontsize=9.5)
    if k >= 3:
        ax.set_xlabel("Slow constant \u03c4_s (days)", fontsize=9.5)
    ax.text(0.03, 0.06, "95% = 1.92", transform=ax.transAxes, fontsize=7.4,
            color=C["grey"], ha="left", va="bottom")
    ax.text(21.0, 0.40, "true 21 d", color=C["green"], fontsize=7.0,
            ha="center", va="bottom")
    ax.set_xticks([10, 100])
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda v, _: "%g" % v))
    ax.grid(alpha=.25, lw=.7)

fig.suptitle("Profile-likelihood recovery of \u03c4_s: information ceiling across cohort size and duration",
             fontsize=12.5, fontweight="bold")
panels = [(axes[k], panel_names[k]) for k in range(6)]
export_panels(fig, panels, "FigS1_profile")
save(fig, "FigS1_profile")
print("FigS1 done")
