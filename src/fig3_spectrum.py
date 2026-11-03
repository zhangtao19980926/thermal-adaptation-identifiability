# -*- coding: utf-8 -*-
"""Figure 3: excitation-spectrum diagnosis - daily outdoor T for DORN (weak slow band) and
In-Gauge (strong slow band), band-SD decomposition, seasonal component, the degenerate real-data
dual fit (tau_s -> 681 d), and slow-band excitation vs tau_s identifiability."""
import os, sys, json, numpy as np, pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from figstyle import apply_style, save, export_panels, C
from ingauge_inject import ingauge_daily, dorn_daily

apply_style()
ROOT = r"D:\thermal-adaptation-series\p1"
dorn = dorn_daily(); ing = ingauge_daily()

def bands(v):
    s = pd.Series(v)
    ma7 = s.rolling(7, min_periods=1, center=True).mean()
    ma30 = s.rolling(30, min_periods=1, center=True).mean()
    syn = s - ma7; slow = ma7 - ma30; seas = ma30 - s.mean()
    return dict(syn=syn, slow=slow, seas=seas,
                sd=[np.nanstd(syn), np.nanstd(slow), np.nanstd(seas)],
                seas_range=float(np.nanmax(seas) - np.nanmin(seas)))

bd, bi = bands(dorn), bands(ing)
print("DORN band SD syn/slow/seas:", np.round(bd["sd"], 3), "seas range %.2f" % bd["seas_range"])
print("InGauge band SD syn/slow/seas:", np.round(bi["sd"], 3), "seas range %.2f" % bi["seas_range"])

fig = plt.figure(figsize=(7.0, 4.25))
gs = fig.add_gridspec(2, 3, hspace=0.78, wspace=0.42)

# (a) DORN daily
axa = fig.add_subplot(gs[0, 0])
axa.plot(np.arange(len(dorn)), dorn, color=C["orange"], lw=1.3)
axa.set_xlabel("Study day"); axa.set_ylabel("Daily outdoor T (°C)")
axa.set_title("(a) DORN, Singapore\n(278 d)")
axa.text(0.03, 0.05, "daily range %.1f °C\n(narrow tropical band)" % (dorn.max() - dorn.min()),
         transform=axa.transAxes, fontsize=7.2, color=C["orange"], va="bottom",
         bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.72))

# (b) In-Gauge daily
axb = fig.add_subplot(gs[0, 1])
axb.plot(np.arange(len(ing)), ing, color=C["green"], lw=1.3)
axb.set_xlabel("Study day"); axb.set_ylabel("Daily outdoor T (°C)")
axb.set_title("(b) In-Gauge, Melbourne\n(169 d)")
axb.text(0.03, 0.05, "daily range %.1f °C\n(strong seasonal swing)" % (ing.max() - ing.min()),
         transform=axb.transAxes, fontsize=7.2, color=C["green"], va="bottom",
         bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.72))

# (c) band SD bars
axc = fig.add_subplot(gs[0, 2])
bands_lab = ["synoptic\n<7 d", "slow\n7-30 d", "seasonal\n>30 d"]
x = np.arange(3); w = 0.38
axc.bar(x - w/2, bd["sd"], w, color=C["orange"], label="DORN (weak)")
axc.bar(x + w/2, bi["sd"], w, color=C["green"], label="In-Gauge (strong)")
axc.set_xticks(x); axc.set_xticklabels(bands_lab); axc.set_ylabel("Band SD (°C)")
axc.set_title("(c) Difference is\nin the SLOW band"); axc.legend(frameon=False)
for i, (v1, v2) in enumerate(zip(bd["sd"], bi["sd"])):
    axc.text(i - w/2, v1 + 0.03, "%.2f" % v1, ha="center", fontsize=7.4)
    axc.text(i + w/2, v2 + 0.03, "%.2f" % v2, ha="center", fontsize=7.4)

# (d) seasonal component overlay
axd = fig.add_subplot(gs[1, 0])
axd.plot(np.arange(len(dorn)), bd["seas"], color=C["orange"], lw=1.8, label="DORN seasonal (>30 d)")
axd.plot(np.arange(len(ing)), bi["seas"], color=C["green"], lw=1.8, label="In-Gauge seasonal (>30 d)")
axd.axhline(0, color="k", lw=.8)
axd.set_xlabel("Study day"); axd.set_ylabel("Seasonal component (°C)")
axd.set_title("(d) Slow excitation that\nmust drive τ_s"); axd.legend(frameon=False, fontsize=7.8)

# (e) DORN real-data five-model NLL
axe = fig.add_subplot(gs[1, 1])
fit = json.load(open(os.path.join(ROOT, "outputs", "eda", "dorn_fit_results.json")))
models = ["static", "fixed7", "fixed30", "single", "dual"]
nll = [fit[m]["nll"] for m in models]
cols = [C["grey"], C["lblue"], C["blue"], C["purple"], C["green"]]
axe.bar(range(5), nll, color=cols, width=.62)
axe.set_xticks(range(5)); axe.set_xticklabels(["static", "7-d", "30-d", "single-τ", "dual-τ"],
                                              rotation=25, ha="right")
axe.set_ylabel("Mean NLL (real DORN votes)"); axe.set_ylim(0.635, 0.660)
axe.set_title("(e) Real DORN: dual fit\nwins only by escaping\n(τ_s=681 d)")
axe.axhline(fit["static"]["nll"], color=C["grey"], ls=":", lw=1)
for i, v in enumerate(nll): axe.text(i, v - 0.0006, "%.3f" % v, ha="center", va="top", fontsize=7.2)

# (f) slow-band excitation vs recovered tau_s
axf = fig.add_subplot(gs[1, 2])
inj = pd.read_csv(os.path.join(ROOT, "outputs", "identifiability", "realband_injection.csv"))
xd = bd["seas_range"]; xi = bi["seas_range"]
for band, xv, col in [("DORN_weak", xd, C["orange"]), ("InGauge_strong", xi, C["green"])]:
    for N, mk, dx in [(40, "o", -0.12), (100, "s", 0.12)]:
        y = inj[(inj.band == band) & (inj.N == N)].tau_slow.iloc[0]
        axf.scatter(xv + dx, y, marker=mk, s=70, color=col, edgecolor="white", zorder=3,
                    label=("%s N=%d" % (band.split("_")[0], N)))
axf.axhline(21, color=C["red"], ls="--", lw=1.6, label="true τ_s=21 d")
axf.annotate("free real-data fit\nescapes to 681 d", xy=(xd + 0.12, 73),
             xytext=(0.02, 0.97), textcoords="axes fraction",
             fontsize=7.6, color=C["orange"], ha="left", va="top",
             arrowprops=dict(arrowstyle="->", color=C["orange"], lw=1.3))
axf.set_xlim(min(xd, xi) - 1.2, max(xd, xi) + 1.2)
axf.set_ylim(0, 90); axf.set_xticks([xd, xi]); axf.set_xticklabels(["DORN\nweak", "In-Gauge\nstrong"])
axf.set_xlabel("Seasonal slow-band swing (°C)")
axf.set_ylabel("Recovered τ_s (days)")
axf.set_title("(f) τ_s recovered\nonly with a strong\nslow band")
axf.legend(frameon=False, fontsize=7.0, ncol=1, loc="center", bbox_to_anchor=(0.50, 0.30))

export_panels(fig, [(axa, "r0c0"), (axb, "r0c1"), (axc, "r0c2"),
                    (axd, "r1c0"), (axe, "r1c1"), (axf, "r1c2")], "Fig3_spectrum")
save(fig, "Fig3_spectrum")
