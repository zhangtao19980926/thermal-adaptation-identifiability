# -*- coding: utf-8 -*-
"""Figure: DORN excitation lives in the intra-day band; the 7-30d band that
informs tau_slow is <1.4 C -> locate DORN on the identifiability surface."""
import os, sys, types, zipfile, pickle
import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "DejaVu Sans"; plt.rcParams["font.size"] = 9
DATA = r"D:\thermal-adaptation-series\p1\data\raw\dorn\dorn-longitudinal-tc-study-main\data"
FIG = r"D:\thermal-adaptation-series\p1\outputs\eda"

def shim():
    sh = types.ModuleType("pandas.core.indexes.numeric")
    class _I(pd.Index): pass
    for nm in ("Int64Index", "Float64Index", "UInt64Index", "NumericIndex"):
        setattr(sh, nm, getattr(pd, nm, _I))
    sys.modules["pandas.core.indexes.numeric"] = sh
shim()
def load_pkl_zip(name):
    with zipfile.ZipFile(os.path.join(DATA, name)) as z:
        t = [n for n in z.namelist() if not n.endswith("/")][0]
        with z.open(t) as f: return pickle.load(f)

def bands(x):
    x = x.dropna()
    ma1 = x.rolling(24, min_periods=1).mean()
    ma7 = x.rolling(24*7, min_periods=1).mean()
    ma30 = x.rolling(24*30, min_periods=1).mean()
    return x, {"<1d": x-ma1, "1-7d": ma1-ma7, "7-30d": ma7-ma30, ">30d": ma30-ma30.mean()}

w = load_pkl_zip("df_weather_sg.pkl.zip"); w.index = pd.to_datetime(w.index)
env = pd.read_csv(os.path.join(DATA, "df_cozie_env.csv"))
env["dt"] = pd.to_datetime(env["time"], utc=True, format="mixed").dt.tz_convert("Asia/Singapore")
for c in ("t-net", "t-env", "t-skin"): env[c] = pd.to_numeric(env[c], errors="coerce")
inb = env.set_index("dt")
series = {
    "Outdoor t-out": w["t-out"].resample("1h").mean(),
    "Indoor t-net": inb["t-net"].resample("1h").mean(),
    "Indoor t-env": inb["t-env"].resample("1h").mean(),
    "Skin t-skin": inb["t-skin"].resample("1h").mean(),
}
B = ["<1d", "1-7d", "7-30d", ">30d"]
res = {k: bands(v) for k, v in series.items()}

fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.05, 1.25]})
ax = axes[0]
names = list(series.keys())
sd = np.array([[res[n][1][b].std() for b in B] for n in names])
colors = ["#2c7fb8", "#7fcdbb", "#f0a830", "#d7301f"]
x = np.arange(len(names)); bottom = np.zeros(len(names))
for j, b in enumerate(B):
    ax.bar(x, sd[:, j], bottom=bottom, color=colors[j], label=b, width=0.62)
    for i, v in enumerate(sd[:, j]):
        if v > 0.12:
            ax.text(i, bottom[i] + v/2, f"{v:.2f}", ha="center", va="center", fontsize=7, color="k")
    bottom += sd[:, j]
ax.set_xticks(x); ax.set_xticklabels([n.replace(" ", "\n") for n in names])
ax.set_ylabel("band SD of temperature (C)")
ax.set_title("Excitation variance by timescale\n(DORN, Singapore, ~7 months)")
ax.legend(title="band", fontsize=7, title_fontsize=8)

ax = axes[1]
for n, c in [("Outdoor t-out", "#2c7fb8"), ("Indoor t-env", "#d7301f")]:
    xx, bd = res[n]
    ax.plot(bd["7-30d"].index, bd["7-30d"].values, lw=1.1, color=c, label=n + " 7-30d band")
ax.axhline(0, color="k", lw=0.6)
for y in (2, -2): ax.axhline(y, color="grey", ls="--", lw=0.8)
ax.text(bd["7-30d"].index[10], 2.05, "+/-2 C event threshold", color="grey", fontsize=7)
ax.set_ylabel("7-30d band temperature (C)")
ax.set_title("Slow band never reaches +/-2 C outdoors (max 1.38 C)\n"
             "-> tau_slow not identifiable from DORN alone", fontsize=9)
ax.legend(fontsize=8)
fig.autofmt_xdate()
fig.tight_layout()
out = os.path.join(FIG, "dorn_excitation_spectrum.png")
fig.savefig(out, dpi=150); print("saved", out)
