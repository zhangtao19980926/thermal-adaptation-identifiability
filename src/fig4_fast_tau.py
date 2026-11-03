# -*- coding: utf-8 -*-
"""Figure 4: fast-process physiological anchor - warm/cool step responses, event-level tau
distribution, passive-vs-event R^2 gain, age heterogeneity, warm/cool asymmetry."""
import os, sys, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(__file__))
from figstyle import apply_style, save, export_panels, C

apply_style()
ROOT = r"D:\thermal-adaptation-series\p1"
EDA = os.path.join(ROOT, "outputs", "eda")
curv = pd.read_csv(os.path.join(EDA, "dorn_event_curves.csv"))
ev = pd.read_csv(os.path.join(EDA, "dorn_event_tau.csv"))
het = pd.read_csv(os.path.join(EDA, "dorn_tau_heterogeneity.csv"))

def step(t, A, tau): return A * (1 - np.exp(-t / tau))

fig = plt.figure(figsize=(7.0, 4.25))
gs = fig.add_gridspec(2, 3, hspace=0.84, wspace=0.40)

def response_panel(ax, dr, color, letter, title):
    d = curv[curv.direction == dr]
    t = d.tmin.values
    air = d.air.values - d[d.tmin < 0].air.mean()
    skin = d.skin.values - d[d.tmin < 0].skin.mean()
    m = t >= 0
    popt, _ = curve_fit(step, t[m], skin[m], p0=[0.8, 30], bounds=([0, 1], [5, 600]))
    A, tau = popt
    ax.plot(t, air, color=C["red"], lw=1.6, label="Near-body air T (step)")
    ax.plot(t, skin, color=color, lw=2.0, label="Skin T (mean response)")
    tt = np.linspace(0, t.max(), 200)
    ax.plot(tt, step(tt, A, tau), color=C["green"], ls="--", lw=1.8,
            label="1st-order fit  τ=%.0f min" % tau)
    ax.axvline(0, color="k", lw=0.8, alpha=.4)
    ax.set_xlabel("Time from step onset (min)"); ax.set_ylabel("ΔT (°C)")
    ax.set_title("%s %s" % (letter, title)); ax.legend(frameon=False, fontsize=7.8)

axa = fig.add_subplot(gs[0, 0]); response_panel(axa, "warm", C["lred"], "(a)", "Warm step (n=%d)" % (ev.direction == "warm").sum())
axb = fig.add_subplot(gs[0, 1]); response_panel(axb, "cool", C["blue"], "(b)", "Cool step (n=%d)" % (ev.direction == "cool").sum())

# (c) event tau distribution
axc = fig.add_subplot(gs[0, 2])
bins = np.arange(0, 155, 8)
for dr, col, lab in [("warm", C["lred"], "warm"), ("cool", C["blue"], "cool")]:
    x = ev[ev.direction == dr].tau_min.values
    axc.hist(x, bins=bins, color=col, alpha=.65, label="%s (median %.0f)" % (lab, np.median(x)))
axc.set_xlabel("Event-level skin-response τ_fast (min)"); axc.set_ylabel("Number of events")
axc.set_title("(c) Distribution of τ_fast\n(1550 events)"); axc.legend(frameon=False)

# (d) R^2 passive vs event
axd = fig.add_subplot(gs[1, 0])
r2_event = ev.r2.median()
r2_passive = 0.014   # whole-period passive skin->air transfer (dorn_fast_transfer)
axd.bar(["Passive\nwhole-period", "Event-aligned\n(this study)"], [r2_passive, r2_event],
        color=[C["grey"], C["green"]], width=0.55)
for i, v in enumerate([r2_passive, r2_event]):
    axd.text(i, v + 0.012, "%.3f" % v, ha="center", fontsize=9.5, fontweight="bold")
axd.set_ylabel("Median fit R²"); axd.set_ylim(0, 0.55)
axd.set_title("(d) Event alignment\nreveals the fast gain")

# (e) tau vs age
axe = fig.add_subplot(gs[1, 1])
d2 = het.dropna(subset=["Age_n", "tau"])
rho, p = spearmanr(d2.Age_n, d2.tau)
mfc = np.where(d2.female == 1, C["red"], C["blue"])
axe.scatter(d2.Age_n, d2.tau, c=mfc, s=42, edgecolor="white", lw=.6)
xx = np.linspace(d2.Age_n.min(), d2.Age_n.max(), 50)
b1, b0 = np.polyfit(d2.Age_n, d2.tau, 1)
axe.plot(xx, b1 * xx + b0, color=C["dgrey"], ls="--", lw=1.4)
axe.set_xlabel("Age (years)"); axe.set_ylabel("Individual median τ_fast (min)")
axe.set_title("(e) Slower skin response\nwith age\nρ=%.2f, p=%.3f (n=%d)" % (rho, p, len(d2)))
axe.scatter([], [], c=C["blue"], label="male"); axe.scatter([], [], c=C["red"], label="female")
axe.legend(frameon=False, fontsize=8)

# (f) warm/cool tau and steady-state gain
axf = fig.add_subplot(gs[1, 2])
groups = ["warm", "cool"]
rng = np.random.default_rng(0)
def boot_ci(vals, n_boot=1000):
    v = np.asarray(vals, dtype=float); v = v[np.isfinite(v)]; n = len(v)
    med = np.median(v)
    bs = np.array([np.median(rng.choice(v, n, replace=True)) for _ in range(n_boot)])
    return med, np.percentile(bs, 2.5), np.percentile(bs, 97.5)
tst = [boot_ci(ev[ev.direction == g].tau_min) for g in groups]
gst = [boot_ci(ev[ev.direction == g].gain) for g in groups]
tau_med = [s[0] for s in tst]; gain_med = [s[0] for s in gst]
tau_err = [[s[0] - s[1] for s in tst], [s[2] - s[0] for s in tst]]
gain_err = [[s[0] - s[1] for s in gst], [s[2] - s[0] for s in gst]]
ekw = dict(ecolor="#333333", elinewidth=1.1, capthick=1.1, capsize=3)
x = np.arange(2)
axf.bar(x - 0.2, tau_med, 0.4, color=[C["lred"], C["blue"]], yerr=tau_err, error_kw=ekw, label="τ_fast (min)")
axf.set_xticks(x); axf.set_xticklabels(["Warm", "Cool"]); axf.set_ylabel("τ_fast (min)")
for i, s in enumerate(tst): axf.text(i - 0.2, s[2] + 1.5, "%.0f" % s[0], ha="center", fontsize=9)
ax2 = axf.twinx(); ax2.grid(False)
ax2.bar(x + 0.2, gain_med, 0.4, color=[C["orange"], C["purple"]], alpha=.75,
        yerr=gain_err, error_kw=ekw, label="steady gain")
ax2.set_ylabel("Skin/air steady-state gain"); ax2.set_ylim(0, 1.1)
for i, s in enumerate(gst): ax2.text(i + 0.2, s[2] + 0.02, "%.2f" % s[0], ha="center", fontsize=9)
axf.set_title("(f) Cool steps:\nslower, higher gain\n(median, 95% CI)")
# twin-axis bars are read against their own axis (left: integer tau minutes; right: decimal gain)
h1, l1 = axf.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()

export_panels(fig, [(axa, "r0c0"), (axb, "r0c1"), (axc, "r0c2"),
                    (axd, "r1c0"), (axe, "r1c1"), (axf, "r1c2", [ax2])], "Fig4_fast_tau")
save(fig, "Fig4_fast_tau")
