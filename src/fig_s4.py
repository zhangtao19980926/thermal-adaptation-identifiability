# -*- coding: utf-8 -*-
"""Fig S4: inter-individual heterogeneity of the fast constant (n = 20)."""
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(__file__))
from figstyle import apply_style, save, export_panels, C

apply_style()
R = r"D:\thermal-adaptation-series\p1\outputs"
h = pd.read_csv(os.path.join(R, "eda", "dorn_tau_heterogeneity.csv"))

fig = plt.figure(figsize=(7.0, 4.7), layout="constrained")
gs = fig.add_gridspec(2, 2, hspace=.26, wspace=.24)
axA = fig.add_subplot(gs[0, 0]); axB = fig.add_subplot(gs[0, 1])
axC = fig.add_subplot(gs[1, 0]); axD = fig.add_subplot(gs[1, 1])

def sp(x, y):
    d = pd.DataFrame({"x": x, "y": y}).dropna()
    r, p = spearmanr(d.x, d.y)
    return r, p

# (a) tau vs age
rA, pA = sp(h.Age_n, h.tau)
axA.scatter(h.Age_n, h.tau, s=42, color=C["blue"], alpha=.75, edgecolor="white", lw=.6, zorder=3)
b1, b0 = np.polyfit(h.Age_n, h.tau, 1)
xx = np.linspace(h.Age_n.min(), h.Age_n.max(), 50)
axA.plot(xx, b1 * xx + b0, color=C["red"], lw=1.8, zorder=2)
axA.set_xlabel("Age (years)", fontsize=9.5)
axA.set_ylabel("\u03c4_fast (min)", fontsize=9.5)
axA.set_title("(a) \u03c4_fast vs age\n(Spearman \u03c1=%.2f, p=%.3f)" % (rA, pA),
              loc="left", fontsize=9.6)
axA.grid(alpha=.25, lw=.7)

# (b) warm vs cool paired
axB.scatter(h.tau_warm, h.tau_cool, s=42, color=C["purple"], alpha=.75, edgecolor="white", lw=.6, zorder=3)
lim = [min(h.tau_warm.min(), h.tau_cool.min()) - 5, max(h.tau_warm.max(), h.tau_cool.max()) + 5]
axB.plot(lim, lim, color=C["grey"], ls="--", lw=1.3)
axB.set_xlim(lim); axB.set_ylim(lim)
axB.set_xlabel("Warming \u03c4_fast (min)", fontsize=9.5)
axB.set_ylabel("Cooling \u03c4_fast (min)", fontsize=9.5)
axB.set_title("(b) Per-person warming\nvs cooling constants", loc="left", fontsize=9.6)
axB.text(0.05, 0.95, "dashed: equal", transform=axB.transAxes, fontsize=7.8,
         color=C["grey"], va="top")
axB.grid(alpha=.25, lw=.7)

# (c) Spearman rho of tau with each covariate
covs = [("gain", "Response gain"), ("BMI", "BMI"), ("Age_n", "Age"), ("female", "Female"),
        ("coldhand", "Cold hands"), ("exercise", "Exercise (d/wk)"), ("years_SG", "Years in SG")]
rhos, ps, names = [], [], []
for col, lab in covs:
    r, p = sp(h[col], h.tau)
    rhos.append(r); ps.append(p); names.append(lab)
y = np.arange(len(covs))[::-1]
cols = [C["red"] if p < .05 else C["lblue"] for p in ps]
axC.barh(y, rhos, color=cols, edgecolor="none", height=.62, zorder=3)
axC.axvline(0, color=C["dgrey"], lw=1)
for yi, r, p in zip(y, rhos, ps):
    star = " *" if p < .05 else ""
    axC.text(r + (0.03 if r >= 0 else -0.03), yi, "\u03c1=%.2f%s" % (r, star),
             va="center", ha="left" if r >= 0 else "right", fontsize=8,
             color=C["red"] if p < .05 else C["dgrey"])
axC.set_yticks(y); axC.set_yticklabels(names, fontsize=8.6)
axC.set_xlabel("Spearman \u03c1 with \u03c4_fast", fontsize=9.5)
axC.set_title("(c) Covariate\ncorrelations (* p<0.05)", loc="left", fontsize=9.6)
axC.set_xlim(-.75, .75); axC.grid(axis="x", alpha=.25, lw=.7)

# (d) tau by sex
rng = np.random.default_rng(7)
for xv, val, col, lab in [(0, h.loc[h.female == 0, "tau"], C["blue"], "male"),
                          (1, h.loc[h.female == 1, "tau"], C["red"], "female")]:
    jit = rng.normal(0, .05, len(val))
    axD.scatter(np.full(len(val), xv) + jit, val, s=40, color=col, alpha=.75,
                edgecolor="white", lw=.6, label="%s (n=%d)" % (lab, len(val)), zorder=3)
    axD.plot([xv - .12, xv + .12], [val.median(), val.median()], color=col, lw=2.4, zorder=4)
rS, pS = sp(h.female, h.tau)
axD.set_xticks([0, 1]); axD.set_xticklabels(["Male", "Female"], fontsize=9)
axD.set_ylabel("\u03c4_fast (min)", fontsize=9.5)
axD.set_title("(d) \u03c4_fast by sex\n(\u03c1=%.2f, p=%.3f, n.s.)" % (rS, pS),
              loc="left", fontsize=9.6)
axD.legend(fontsize=8, frameon=False, loc="upper right")
axD.grid(axis="y", alpha=.25, lw=.7)

fig.suptitle("Inter-individual heterogeneity of the fast constant (n = 20 occupants)",
             fontsize=12.5, fontweight="bold")
export_panels(fig, [(axA, "r0c0"), (axB, "r0c1"), (axC, "r1c0"), (axD, "r1c2")], "FigS4_hetero")
save(fig, "FigS4_hetero")
print("FigS4 done; age rho=%.3f p=%.3f; sex rho=%.3f p=%.3f" % (rA, pA, rS, pS))
