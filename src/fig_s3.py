# -*- coding: utf-8 -*-
"""Fig S3: fast-constant event selection, warm/cool asymmetry and method comparison."""
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from figstyle import apply_style, save, export_panels, C

apply_style()
R = r"D:\thermal-adaptation-series\p1\outputs"
ev = pd.read_csv(os.path.join(R, "eda", "dorn_event_tau.csv"))
het = pd.read_csv(os.path.join(R, "eda", "dorn_tau_heterogeneity.csv"))
trf = pd.read_csv(os.path.join(R, "eda", "dorn_fast_transfer_tau.csv"))

A_TH, R2_TH = 0.15, 0.20
ev["absA"] = ev.A.abs()
keep = (ev.absA >= A_TH) & (ev.r2 >= R2_TH)
kept = ev[keep]
rej = ev[~keep]

fig = plt.figure(figsize=(7.0, 4.7), layout="constrained")
gs = fig.add_gridspec(2, 2, hspace=.26, wspace=.24)
axA = fig.add_subplot(gs[0, 0]); axB = fig.add_subplot(gs[0, 1])
axC = fig.add_subplot(gs[1, 0]); axD = fig.add_subplot(gs[1, 1])

# (a) robustness of the pooled median to the fit-quality threshold
r2grid = [0.0, .1, .2, .3, .4, .5]
meds, cnts = [], []
for t in r2grid:
    s = ev[(ev.r2 >= t) & (ev.absA >= A_TH)]
    meds.append(s.tau_min.median()); cnts.append(len(s))
axA.plot(r2grid, meds, "-o", color=C["red"], lw=2, ms=6, zorder=3)
for t, md, cn in zip(r2grid, meds, cnts):
    axA.text(t, md + 2.2, "%.0f min\nn=%d" % (md, cn), ha="center", va="bottom",
             fontsize=7.4, color=C["dgrey"])
axA.axvline(R2_TH, color=C["grey"], ls=":", lw=1.4)
axA.text(R2_TH + .012, axA.get_ylim()[1] if False else 58, "chosen R\u00b2 = 0.20",
         color=C["grey"], fontsize=7.8, ha="left", va="top")
axA.set_xlabel("Minimum single-event fit R\u00b2 threshold", fontsize=9.5)
axA.set_ylabel("Pooled median \u03c4_fast (min)", fontsize=9.5)
axA.set_title("(a) Robustness of \u03c4_fast\nto screening threshold", loc="left", fontsize=9.8)
axA.set_ylim(0, 70); axA.grid(alpha=.25, lw=.7)

# (b) tau histogram warm/cool
kw = kept[kept.direction == "warm"].tau_min
kc = kept[kept.direction == "cool"].tau_min
CAP = 299.9
bins = np.linspace(0, 300, 50)
ceil_n = int((kept.tau_min >= CAP).sum())
kc0, kw0 = kc[kc < CAP], kw[kw < CAP]
axB.hist(kc0, bins=bins, color=C["blue"], alpha=.65,
         label="cooling, converged (n=%d)" % len(kc0), zorder=3)
axB.hist(kw0, bins=bins, color=C["red"], alpha=.7,
         label="warming, converged (n=%d)" % len(kw0), zorder=4)
axB.bar(297, ceil_n, width=6, color=C["grey"], alpha=.85, zorder=5,
        label="_ceiling")
axB.annotate("unconverged (n=%d) events\n(\u03c4 capped at 300 min)" % ceil_n, xy=(297, ceil_n), xytext=(175, 60),
             fontsize=7.4, color=C["dgrey"], ha="center",
             arrowprops=dict(arrowstyle="->", color=C["dgrey"], lw=1))
axB.axvline(kept.tau_min.median(), color=C["dgrey"], lw=1.8)
axB.text(120, 100, "pooled median %.1f min" % kept.tau_min.median(),
         fontsize=8, color=C["dgrey"], ha="left", va="top")
axB.axvline(kw.median(), color=C["red"], ls=":", lw=1.4)
axB.axvline(kc.median(), color=C["blue"], ls=":", lw=1.4)
axB.set_xlabel("Fast constant \u03c4_fast (min)", fontsize=9.5)
axB.set_ylabel("Number of events", fontsize=9.5)
axB.set_title("(b) Retained fast-\nconstant distribution", loc="left", fontsize=9.8)
axB.legend(fontsize=7.2, frameon=True, framealpha=.88, edgecolor="none",
           loc="upper center", bbox_to_anchor=(0.64, 1.0))
axB.set_xlim(0, 312); axB.grid(axis="y", alpha=.25, lw=.7)

# (c) warm vs cool box
data = [kw.values, kc.values]
bp = axC.boxplot(data, patch_artist=True, widths=.5, showfliers=False,
                 medianprops=dict(color=C["dgrey"], lw=1.6),
                 boxprops=dict(edgecolor=C["dgrey"]), whiskerprops=dict(color=C["dgrey"]),
                 capprops=dict(color=C["dgrey"]))
for patch, col in zip(bp["boxes"], [C["red"], C["blue"]]):
    patch.set_facecolor(col); patch.set_alpha(.55)
for i, d in enumerate(data, start=1):
    axC.text(i + .16, np.median(d), "%.1f" % np.median(d), fontsize=9, va="center", color=C["dgrey"])
axC.set_xticks([1, 2]); axC.set_xticklabels(["Warming", "Cooling"], fontsize=9)
axC.set_ylabel("\u03c4_fast (min)", fontsize=9.5)
axC.set_title("(c) Warming vs.\ncooling constants", loc="left", fontsize=9.8)
axC.grid(axis="y", alpha=.25, lw=.7)

# (d) event method vs continuous transfer-function method, paired
m = het[["user", "tau"]].merge(trf[["user", "tau_min"]], on="user")
xs = [1, 2]
for _, r0 in m.iterrows():
    axD.plot(xs, [r0.tau, r0.tau_min], "-", color=C["grey"], alpha=.4, lw=.8, zorder=2)
bpd = axD.boxplot([m.tau.values, m.tau_min.values], positions=xs, patch_artist=True, widths=.4,
                  showfliers=False, medianprops=dict(color=C["dgrey"], lw=1.6),
                  boxprops=dict(edgecolor=C["dgrey"]), whiskerprops=dict(color=C["dgrey"]),
                  capprops=dict(color=C["dgrey"]))
for patch, col in zip(bpd["boxes"], [C["blue"], C["purple"]]):
    patch.set_facecolor(col); patch.set_alpha(.55)
axD.set_yscale("log")
axD.set_yticks([10, 100])
axD.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda v, _: "%g" % v))
axD.set_ylim(10, 400)
axD.set_xticks(xs); axD.set_xticklabels(["Aligned step events\n(piecewise fit)",
                                         "Continuous transfer\nfunction (whole record)"], fontsize=8.2)
axD.set_ylabel("\u03c4_fast (min, log scale)", fontsize=9.5)
axD.set_title("(d) Two estimators of\nthe fast constant", loc="left", fontsize=9.8)
axD.text(1, m.tau.max() * 1.12, "median %.0f min" % m.tau.median(), ha="center", va="bottom",
         fontsize=8, color=C["blue"])
axD.text(2, m.tau_min.max() * 1.06, "median %.0f min" % m.tau_min.median(), ha="center", va="bottom",
         fontsize=8, color=C["purple"])
axD.grid(axis="y", alpha=.25, which="both", lw=.7)

fig.suptitle("Fast-constant event screening, direction asymmetry and estimator comparison",
             fontsize=12.5, fontweight="bold")
export_panels(fig, [(axA, "r0c0"), (axB, "r0c1"), (axC, "r1c0"), (axD, "r1c2")], "FigS3_fast")
save(fig, "FigS3_fast")
print("FigS3 done; kept=%d rej=%d med=%.1f warm=%.1f cool=%.1f" %
      (len(kept), len(rej), kept.tau_min.median(), kw.median(), kc.median()))
