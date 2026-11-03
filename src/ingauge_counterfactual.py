# -*- coding: utf-8 -*-
"""
Result E (In-Gauge counterfactual, standard implication). Build the adaptive comfort
set-point  Tcomf = Tn0 + beta * Trm  under three outdoor-memory specifications:
  * fixed 7-day rectangular running mean (short ASHRAE-style)
  * fixed 30-day rectangular window (long)
  * learned slow kernel, exponential running mean tau=21 d (our identified slow constant)
Using REAL Melbourne spring->summer outdoor temperature and real occupied indoor
temperature, quantify:
  (1) season-transition set-point mistracking (deg C * day) and its direction
  (2) post-heatwave set-point recovery days
  (3) indicative occupied cooling-degree-hours (CDH) balance change (NOT an energy model)
"""
import os, glob, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"D:\thermal-adaptation-series\p1"
OUT = os.path.join(ROOT, "outputs", "standards"); os.makedirs(OUT, exist_ok=True)
BETA = 0.25          # adaptive slope (C comfort / C running-mean outdoor), DBII temperate~0.20

def load_daily():
    rows = []
    for p in glob.glob(os.path.join(ROOT, "data", "raw", "ingauge", "longitudinal", "*.csv")):
        d = pd.read_csv(p, low_memory=False)
        rows.append(d[["Day", "Occupied", "IndoorTemperature", "OutdoorTemperature",
                       "CoolingState", "HeatingState"]])
    df = pd.concat(rows, ignore_index=True)
    for c in ["Occupied", "IndoorTemperature", "OutdoorTemperature", "CoolingState", "HeatingState"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    print("CoolingState values:", df.CoolingState.value_counts(dropna=False).head().to_dict())
    print("HeatingState values:", df.HeatingState.value_counts(dropna=False).head().to_dict())
    daily = df.groupby("Day").agg(
        Tout=("OutdoorTemperature", "mean"),
        Tin_occ=("IndoorTemperature", lambda s: s[df.loc[s.index, "Occupied"] > 0].mean()),
        Tin_all=("IndoorTemperature", "mean"),
        cool_frac=("CoolingState", "mean"),
        occ_frac=("Occupied", "mean")).reset_index().sort_values("Day")
    return daily

def rect_rm(x, w):
    return pd.Series(x).rolling(w, min_periods=1).mean().values

def exp_rm(x, tau):
    x = np.asarray(x, float); out = np.zeros_like(x); a = 1.0 / tau   # daily step
    out[0] = x[0]
    for k in range(1, len(x)):
        out[k] = out[k - 1] + a * (x[k] - out[k - 1])
    return out

def main():
    d = load_daily()
    Tout = d.Tout.values; days = d.Day.values.astype(float)
    Tin = d.Tin_occ.fillna(d.Tin_all).values
    Trm = {"7d": rect_rm(Tout, 7), "30d": rect_rm(Tout, 30), "tau21d": exp_rm(Tout, 21.0)}
    # calibrate Tn0 per spec so each set-point has the SAME mean as the real occupied indoor
    # temperature (isolates trajectory-shape differences)
    Tcomf = {}
    for k, r in Trm.items():
        Tn0 = np.nanmean(Tin) - BETA * np.nanmean(r)
        Tcomf[k] = Tn0 + BETA * r

    # (1) season-transition mistracking: warming limb where daily Tout rises persistently
    #     define transition = days 15..90 (spring->summer). short window lags slow kernel.
    trans = (days >= 15) & (days <= 90)
    diff7 = Tcomf["7d"] - Tcomf["tau21d"]
    diff30 = Tcomf["30d"] - Tcomf["tau21d"]
    mistrack7 = np.trapz(diff7[trans], days[trans])
    mistrack30 = np.trapz(diff30[trans], days[trans])
    print("\n(1) TRANSITION (day15-90) set-point vs learned tau21:")
    print("    fixed7d  mean bias=%+.3f C, integral=%+.1f C*day (POS on warming = short window"
          " OVERSHOOTS: assumes adaptation before it is complete -> set-point too high)"
          % (np.nanmean(diff7[trans]), mistrack7))
    print("    fixed30d mean bias=%+.3f C, integral=%+.1f C*day (30d ~ learned tau21)"
          % (np.nanmean(diff30[trans]), mistrack30))

    # (2) heatwaves: >=3 consecutive days above the 85th-percentile daily outdoor T (>=22 C);
    #     recovery = days for set-point to fall half-way from peak to pre-event 14-d baseline
    hot_thr = max(float(np.quantile(Tout, 0.85)), 22.0)
    print("\nheatwave daily-T threshold = %.1f C" % hot_thr)
    hot = Tout >= hot_thr
    rec = {k: [] for k in Tcomf}
    evs = []
    i = 0
    while i < len(Tout) - 3:
        if hot[i:i + 3].all():
            j = i + 3
            while j < len(Tout) and Tout[j] >= hot_thr - 1.5: j += 1
            evs.append((i, j)); i = j
        else: i += 1
    for (i, j) in evs:
        base_i = max(0, i - 14)
        for k in Tcomf:
            Tc = Tcomf[k]
            base = np.nanmean(Tc[base_i:i]); peak = np.nanmax(Tc[i:j])
            half = peak - 0.5 * (peak - base)
            aft = np.where(Tc[j:] <= half)[0]
            rec[k].append(aft[0] if len(aft) else (len(Tc) - j))
    print("\n(2) HEATWAVES detected: %d at day-ranges %s" %
          (len(evs), [(int(days[a]), int(days[min(b, len(days)-1)])) for a, b in evs]))
    for k in Tcomf:
        if rec[k]: print("    %-7s recovery days: %s (median %.1f)" %
                         (k, [int(x) for x in rec[k]], np.median(rec[k])))

    # (3) indicative occupied cooling-degree-hours relative to each set-point, daily proxy
    #     CDH_proxy = sum_d max(0, Tin_d - Tcomf_d) * occupied_hours(=24*occ_frac)
    occ_h = d.occ_frac.fillna(0.5).values * 24.0
    cdh = {k: np.nansum(np.maximum(0, Tin - Tcomf[k]) * occ_h) for k in Tcomf}
    print("\n(3) INDICATIVE occupied cooling-degree-hours (higher = more cooling need):")
    for k in ["7d", "30d", "tau21d"]:
        print("    %-7s CDH=%8.0f  (vs fixed7d %+.1f%%)" %
              (k, cdh[k], 100 * (cdh[k] - cdh["7d"]) / cdh["7d"]))

    # save table
    pd.DataFrame({"day": days, "Tout": Tout, "Tin_occ": Tin,
                  "Trm7": Trm["7d"], "Trm30": Trm["30d"], "Trm_tau21": Trm["tau21d"],
                  "Tcomf7": Tcomf["7d"], "Tcomf30": Tcomf["30d"], "Tcomf_tau21": Tcomf["tau21d"],
                  "cool_frac": d.cool_frac.values}).to_csv(
        os.path.join(OUT, "ingauge_counterfactual.csv"), index=False)

    # figure: 2 panels (set-point trajectories + heatwave zoom), compact
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), gridspec_kw={"width_ratios": [1.55, 1.0]})
    ax = axes[0]
    ax.plot(days, Tout, color="grey", lw=1.2, alpha=.8, label="Daily outdoor T")
    ax.plot(days, Tcomf["7d"], color="#c0392b", lw=1.8, label="Set-point, fixed 7-d window")
    ax.plot(days, Tcomf["30d"], color="#7d3c98", lw=1.6, ls="--", label="Set-point, fixed 30-d window")
    ax.plot(days, Tcomf["tau21d"], color="#1e8449", lw=2.2, label="Set-point, learned $\\tau_s$=21 d")
    for (a, b) in evs:
        ax.axvspan(days[a], days[min(b, len(days)-1)], color="orange", alpha=.15)
    ax.set_xlabel("Study day (Melbourne, spring->summer)"); ax.set_ylabel("Temperature (deg C)")
    ax.set_title("Adaptive set-point under three memory specifications")
    ax.legend(frameon=False, fontsize=8); ax.grid(alpha=.25)
    ax2 = axes[1]
    if evs:
        a, b = max(evs, key=lambda e: e[1] - e[0]); c0, c1 = max(0, a - 12), min(len(days), b + 28)
        dd = days[c0:c1]
        ax2.plot(dd, Tout[c0:c1], color="grey", lw=1.4, label="Outdoor T")
        ax2.plot(dd, Tcomf["7d"][c0:c1], color="#c0392b", lw=1.8, label="7-d")
        ax2.plot(dd, Tcomf["30d"][c0:c1], color="#7d3c98", lw=1.6, ls="--", label="30-d")
        ax2.plot(dd, Tcomf["tau21d"][c0:c1], color="#1e8449", lw=2.2, label="$\\tau_s$=21 d")
        ax2.axvspan(days[a], days[b], color="orange", alpha=.15)
        ax2.set_title("Heat-wave set-point recovery zoom")
        ax2.set_xlabel("Study day"); ax2.set_ylabel("Set-point (deg C)")
        ax2.legend(frameon=False, fontsize=8); ax2.grid(alpha=.25)
    plt.tight_layout()
    fp = os.path.join(OUT, "ingauge_counterfactual.png")
    plt.savefig(fp, dpi=200); print("\nsaved", fp)

if __name__ == "__main__":
    main()
