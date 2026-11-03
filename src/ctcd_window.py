# -*- coding: utf-8 -*-
"""
Result C (main, CTCD): data-driven re-estimation of the outdoor-memory window behind the
adaptive comfort standard. Chinese Thermal Comfort Dataset Class I+II (n=33,430).

The dataset ships four outdoor-memory scales per vote:
  G2 daily (1d), G4 7-day running mean, G4 15-day running mean, G3 monthly (~30d).
We ask WHICH scale best predicts thermal sensation out of sample (grouped 5-fold by city,
controls = operative temp, clo, met), overall and stratified by building operation mode
and climate zone. Also: campaign-level continuous causal window scan, and Living-Years
(acclimatisation) analysis.
"""
import os, numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss, mean_squared_error

ROOT = r"D:\thermal-adaptation-series\p1"
OUT = os.path.join(ROOT, "outputs", "standards"); os.makedirs(OUT, exist_ok=True)

def load():
    frames = []
    for tag, p in [("I", r"data\raw\china_ctcd\Chinese Thermal Comfort Dataset Class I.csv"),
                   ("II", r"data\raw\china_ctcd\Chinese Thermal Comfort Dataset Class II.csv")]:
        d = pd.read_csv(os.path.join(ROOT, p), encoding="gbk", low_memory=False)
        d["cls"] = tag; frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    ren = {"A2.Date": "date", "A5.City": "city", "A6.Climate Zone": "zone",
           "B4.Building Operation Mode": "mode", "C1.Sex": "sex", "C2.Age": "age",
           "C5.Living Years": "live_y", "D1.TSV": "tsv",
           "D5.Clothing Insulation (clo)": "clo", "D6.Metabolic Rate (met)": "met",
           "F1.Operative Temperature (℃)": "top",
           "G2.Mean Daily Outdoor Temperature (℃)": "m1",
           "G3.Monthly Mean Outdoor Temperature (℃)": "m30",
           "G4.7-Day Running Mean Outdoor Temperature (℃)": "m7",
           "G4.15-Day Running Mean Outdoor Temperature (℃)": "m15"}
    df = df.rename(columns=ren)
    keep = list(ren.values()) + ["cls"]
    df = df[[c for c in keep if c in df.columns]].copy()
    for c in ["live_y", "tsv", "clo", "met", "top", "m1", "m7", "m15", "m30"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], format="mixed", errors="coerce")
    return df

def mode_group(m):
    if not isinstance(m, str): return "Other"
    m = m.lower()
    if "natural" in m or "naturally" in m or m.strip() == "nv" or "ventilat" in m and "natural" in m: return "NV"
    if "mixed" in m or m.strip() == "mm": return "MM"
    if "air" in m and "condition" in m or "ac" in m or "cooling" in m and "natural" not in m: return "AC"
    if "heat" in m or "radiator" in m or "convection" in m or "boiler" in m: return "Heated"
    return "Other"

def continuity_diag(df):
    lines = []
    daily = df.dropna(subset=["date", "m1"]).groupby(["city", df["date"].dt.date])["m1"].mean().reset_index()
    daily["date"] = pd.to_datetime(daily["date"])
    lines.append("city-campaign daily outdoor observations: %d city-days across %d cities" %
                 (len(daily), daily.city.nunique()))
    seg_lens = []
    for city, g in daily.sort_values("date").groupby("city"):
        days = g["date"].dt.normalize().drop_duplicates().sort_values().values
        if len(days) < 2: continue
        gaps = np.diff(days).astype("timedelta64[D]").astype(int)
        # contiguous segments
        run = 1
        for gp in gaps:
            if gp == 1: run += 1
            else: seg_lens.append(run); run = 1
        seg_lens.append(run)
    seg_lens = np.array(seg_lens)
    lines.append("contiguous campaign segments: n=%d, median len=%.1f d, p75=%.1f, max=%d, >=14d:%.1f%%, >=30d:%.1f%%" %
                 (len(seg_lens), np.median(seg_lens), np.percentile(seg_lens, 75), seg_lens.max(),
                  100 * (seg_lens >= 14).mean(), 100 * (seg_lens >= 30).mean()))
    return "\n".join(lines)

def cv_metrics(d, scales):
    d = d.dropna(subset=["tsv", "top", "clo", "met"] + scales).copy()
    d = d[np.abs(d["tsv"]) <= 3]
    d["grp"] = pd.factorize(d["city"])[0]
    ng = d["grp"].nunique()
    if ng < 2:
        return None
    y_neu = (d["tsv"] == 0).astype(int).values
    y = d["tsv"].values
    ctrl = ["top", "clo", "met"]
    rows = []
    configs = [("base(no outdoor memory)", [])] + [("+%s" % s, [s]) for s in scales]
    nfold = min(5, ng)
    gkf = GroupKFold(n_splits=nfold)
    for name, mem in configs:
        feats = [c for c in ctrl + mem if c in d.columns]
        X = d[feats].values
        ll, mse, n = [], [], 0
        for tr, te in gkf.split(X, y, groups=d["grp"].values):
            sc = StandardScaler().fit(X[tr])
            Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
            clf = LogisticRegression(max_iter=3000, C=1.0).fit(Xtr, y_neu[tr])
            p = clf.predict_proba(Xte)
            ll.append(log_loss(y_neu[te], p, labels=[0, 1]) * len(te))
            reg = LinearRegression().fit(Xtr, y[tr])
            mse.append(mean_squared_error(y[te], reg.predict(Xte)) * len(te))
            n += len(te)
        rows.append(dict(config=name, logloss=sum(ll) / n, mse=sum(mse) / n, n=n))
    return pd.DataFrame(rows)

def main():
    df = load()
    df["zone"] = df["zone"].astype(str).str.strip().str.replace("Sever cold", "Severe cold", regex=False)
    df["mode_g"] = df["mode"].map(mode_group)
    print("n=%d, cities=%d, zones=%s" % (len(df), df.city.nunique(), df.zone.dropna().unique()[:8]))
    print("mode groups:\n", df.mode_g.value_counts())
    diag = continuity_diag(df)
    print("\nCONTINUITY:\n" + diag)

    scales = ["m1", "m7", "m15", "m30"]
    all_rows = []
    strata = [("ALL", df)]
    for mg in ["NV", "MM", "AC", "Heated"]:
        sub = df[df.mode_g == mg]
        if len(sub) > 400: strata.append((mg, sub))
    for zn in df.zone.dropna().unique():
        sub = df[df.zone == zn]
        if len(sub) > 800: strata.append(("zone:" + str(zn)[:22], sub))
    for name, sub in strata:
        m = cv_metrics(sub, scales)
        if m is None:
            print("\n[%s] skipped (<2 cities)" % name); continue
        m.insert(0, "stratum", name)
        all_rows.append(m)
        b = m.set_index("config")
        print("\n[%s n=%d]" % (name, m["n"].iloc[0]))
        print(m[["config", "logloss", "mse"]].round(4).to_string(index=False))
    res = pd.concat(all_rows, ignore_index=True)
    res.to_csv(os.path.join(OUT, "ctcd_window_scales.csv"), index=False)

    # Living years (slow acclimatisation): does long residence strengthen coupling to the
    # local slow outdoor climate? Compare neutral-comfort odds vs monthly outdoor by tenure.
    d = df.dropna(subset=["tsv", "top", "m30", "live_y"]).copy()
    d["tenure"] = pd.cut(d.live_y, [-1, 2, 10, 200], labels=["<=2y", "3-10y", ">10y"])
    ly = []
    for tg, g in d.groupby("tenure", observed=True):
        if len(g) < 200: continue
        # adaptive slope: TSV residual after operative temp regressed on monthly outdoor
        X = StandardScaler().fit_transform(g[["top"]].values)
        r = LinearRegression().fit(X, g.tsv.values)
        resid = g.tsv.values - r.predict(X)
        slope = LinearRegression().fit(g[["m30"]].values, resid).coef_[0]
        ly.append(dict(tenure=str(tg), n=len(g), slope_residTSV_on_monthlyOut=slope,
                       live_y_med=g.live_y.median()))
    lyd = pd.DataFrame(ly)
    lyd.to_csv(os.path.join(OUT, "ctcd_living_years.csv"), index=False)
    print("\nLIVING YEARS (residual TSV slope on monthly outdoor, after removing operative temp):")
    print(lyd.round(4).to_string(index=False))
    print("\nsaved", os.path.join(OUT, "ctcd_window_scales.csv"))

if __name__ == "__main__":
    main()
