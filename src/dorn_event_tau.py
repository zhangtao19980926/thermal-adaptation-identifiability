# -*- coding: utf-8 -*-
"""
Result B (DORN, fast-tau by EVENT ALIGNMENT). Instead of passive all-time regression
(R2=0.014, swamped by activity/wearing), detect rapid near-body air-temperature steps
(entering/leaving air-conditioned spaces), align the skin-temperature traces at step onset,
and fit the first-order step response  dTskin(t) = A*(1-exp(-t/tau)).  Reports per-event
tau distribution and pooled warm/cold responses with proper R2.
"""
import os, sys, types, zipfile, pickle
import numpy as np, pandas as pd
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = r"D:\thermal-adaptation-series\p1\data\raw\dorn\dorn-longitudinal-tc-study-main\data"
OUT = r"D:\thermal-adaptation-series\p1\outputs\eda"; os.makedirs(OUT, exist_ok=True)
DT = 3.0  # minutes

def shim():
    sh = types.ModuleType("pandas.core.indexes.numeric")
    class _I(pd.Index): pass
    for nm in ("Int64Index", "Float64Index", "UInt64Index", "NumericIndex"):
        setattr(sh, nm, getattr(pd, nm, _I))
    sys.modules["pandas.core.indexes.numeric"] = sh
shim()
def load(name):
    with zipfile.ZipFile(os.path.join(DATA, name)) as z:
        t = [n for n in z.namelist() if not n.endswith("/")][0]
        with z.open(t) as f: return pickle.load(f)

def paired():
    skin = load("df_ib_skin.pkl.zip")[["Temperature", "userid"]].rename(columns={"Temperature": "Tskin"})
    air = load("df_ib_air.pkl.zip")[["Temperature", "userid"]].rename(columns={"Temperature": "Tair"})
    skin.index = pd.to_datetime(skin.index); air.index = pd.to_datetime(air.index)
    def rs3(df, val):
        parts = []
        for u, g in df.groupby("userid"):
            r = g[val].resample("3min").mean().to_frame(val); r["userid"] = u; parts.append(r)
        o = pd.concat(parts).reset_index().rename(columns={lambda c: True: "time"})
        o.columns = ["time", val, "userid"]
        return o[["time", "userid", val]]
    a, s = rs3(air, "Tair"), rs3(skin, "Tskin")
    m = a.merge(s, on=["time", "userid"]).dropna().sort_values(["userid", "time"])
    m = m[(m.Tskin.between(27, 38)) & (m.Tair.between(15, 40))].copy()
    m["dtmin"] = m.groupby("userid")["time"].diff().dt.total_seconds() / 60.0
    m = m[(m.dtmin >= 2.5) & (m.dtmin <= 3.5)].copy()
    m["seg"] = (m.dtmin.isna() | (m.dtmin > 3.5)).cumsum()
    return m

def detect_events(g, step_min=2.0, wp=4, suppr=20, hold=10):
    """g: one continuous segment sorted by time. Returns list of onset iloc indices."""
    ta = g.Tair.values
    pre = pd.Series(ta).rolling(wp, min_periods=wp).mean().values
    post = pd.Series(ta).shift(-wp).rolling(wp, min_periods=1).mean().values
    d = post - pre
    cand = np.where(np.abs(d) >= step_min)[0]
    # require post-step plateau (air stays at new level over next ~30 min)
    keep = []
    for i in cand:
        if i + hold + 10 >= len(ta): continue
        early = ta[i + 2:i + wp + 2].mean()
        late = ta[i + hold:i + hold + 8].mean()
        if abs(late - early) > 0.8:        # not a sustained step (drifting / spike)
            continue
        keep.append(i)
    # non-max suppression
    keep.sort(key=lambda i: -abs(d[i]))
    picked = []
    for i in keep:
        if all(abs(i - j) > suppr for j in picked): picked.append(i)
    return sorted(picked), d

def step_model(t, A, tau):
    return A * (1.0 - np.exp(-t / tau))

def main():
    m = paired()
    print("paired 3-min rows:", len(m), "users:", m.userid.nunique())
    events = []
    PRE, POST = 10, 40  # 30 min before, 120 min after
    for (u, seg), g in m.groupby(["userid", "seg"]):
        g = g.reset_index(drop=True)
        if len(g) < PRE + POST + 5: continue
        onsets, d = detect_events(g)
        for i in onsets:
            lo, hi = i - PRE, i + POST
            if lo < 0 or hi >= len(g): continue
            win = g.iloc[lo:hi]
            if win.Tskin.notna().mean() < 0.7: continue
            base = g.Tskin.iloc[lo:i].mean()
            base_sd = g.Tskin.iloc[lo:i].std()
            if not np.isfinite(base_sd) or base_sd > 0.8: continue
            trel = np.arange(-PRE, POST) * DT
            ys = (win.Tskin.values - base)
            step = d[i]
            # fit only post-onset
            tp = trel[trel >= 0]; yp = ys[trel >= 0]
            ok = np.isfinite(yp)
            tp, yp = tp[ok], yp[ok]
            if len(tp) < 20: continue
            a0 = np.clip(0.35 * step, -3, 3) * 0.4   # expect attenuated skin response
            try:
                popt, _ = curve_fit(step_model, tp, yp, p0=[a0, 30.0],
                                    bounds=([-3.0, 3.0], [3.0, 300.0]), maxfev=4000)
                A, tau = popt
                pred = step_model(tp, A, tau)
                ss_res = np.sum((yp - pred) ** 2); ss_tot = np.sum((yp - yp.mean()) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 1e-9 else np.nan
            except Exception:
                continue
            # require response in step direction and a meaningful, well-fit response
            if np.sign(A) != np.sign(step) or abs(A) < 0.15 or not np.isfinite(r2) or r2 < 0.2:
                continue
            events.append(dict(user=u, seg=seg, i=i, time=g.time.iloc[i],
                               direction="warm" if step > 0 else "cool",
                               dTair=step, A=A, tau_min=tau, r2=r2,
                               gain=A / step))
    ev = pd.DataFrame(events)
    print("valid step events:", len(ev), "| warm:", (ev.direction == "warm").sum(),
          "| cool:", (ev.direction == "cool").sum(), "| users:", ev.user.nunique())
    if len(ev):
        print("\nevent-level tau (min): median=%.1f mean=%.1f IQR=[%.1f,%.1f]" %
              (ev.tau_min.median(), ev.tau_min.mean(), ev.tau_min.quantile(.25), ev.tau_min.quantile(.75)))
        print("tau hours median=%.2f | skin/air gain median=%.3f | fit R2 median=%.3f" %
              (ev.tau_min.median() / 60, ev.gain.median(), ev.r2.median()))
        for dr in ["warm", "cool"]:
            s = ev[ev.direction == dr]
            if len(s) >= 5:
                print("  %-5s n=%3d  tau median=%.1f min (%.2f h), gain=%.3f, R2=%.3f" %
                      (dr, len(s), s.tau_min.median(), s.tau_min.median() / 60, s.gain.median(), s.r2.median()))
        ev.to_csv(os.path.join(OUT, "dorn_event_tau.csv"), index=False)

        # pooled aligned response for the figure (normalised by step sign)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        curves = []
        for ax, dr in zip(axes, ["warm", "cool"]):
            sgn = 1 if dr == "warm" else -1
            # rebuild pooled aligned curves from event windows
            ca, cs = [], []
            for _, e in ev[ev.direction == dr].iterrows():
                g = m[(m.userid == e.user) & (m.seg == e.seg)].reset_index(drop=True)
                ii = int(e.i); lo, hi = ii - PRE, ii + POST
                if lo < 0 or hi >= len(g): continue
                ab = g.Tair.iloc[lo:ii].mean(); sb = g.Tskin.iloc[lo:ii].mean()
                ca.append((g.Tair.iloc[lo:hi].values - ab) * sgn)
                cs.append((g.Tskin.iloc[lo:hi].values - sb) * sgn)
            if ca:
                Ca = np.nanmean(np.vstack(ca), 0); Cs = np.nanmean(np.vstack(cs), 0)
                tt = np.arange(-PRE, POST) * DT
                ax.plot(tt, Ca, color="#c0392b", lw=2, label="Near-body air (step)")
                ax.plot(tt, Cs, color="#2471a3", lw=2, label="Skin temperature")
                ok = tt >= 0
                popt, _ = curve_fit(step_model, tt[ok], Cs[ok], p0=[Cs[ok].max(), 30],
                                    bounds=([0, 3], [5, 300]), maxfev=4000)
                tt2 = np.linspace(0, POST * DT, 200)
                ax.plot(tt2, step_model(tt2, *popt), "--", color="#1e8449", lw=1.8,
                        label="1st-order fit, tau=%.0f min" % popt[1])
                ax.axvline(0, color="grey", ls=":", lw=1)
                ax.set_title("%s steps (n=%d)" % (dr.capitalize(), len(ca)))
                ax.set_xlabel("Time relative to step onset (min)")
                ax.set_ylabel("Signed temperature change (deg C)")
                ax.legend(frameon=False, fontsize=8); ax.grid(alpha=.25)
                curves.append(pd.DataFrame({"direction": dr, "tmin": tt, "air": Ca, "skin": Cs}))
        if curves:
            pd.concat(curves).to_csv(os.path.join(OUT, "dorn_event_curves.csv"), index=False)
        plt.tight_layout()
        fp = os.path.join(OUT, "dorn_event_aligned.png")
        plt.savefig(fp, dpi=200); print("saved", fp)

if __name__ == "__main__":
    main()
