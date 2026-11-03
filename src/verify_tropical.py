# -*- coding: utf-8 -*-
import numpy as np, pandas as pd
from scipy import stats

d = pd.read_csv(r"D:\thermal-adaptation-series\p1\outputs\standards\db2_adaptive_slope.csv")
d = d[(d.Tneutral.between(15, 32)) & (d.out_mon.between(-5, 35))]
print("cleaned cities:", len(d))
print("%-14s %5s %8s %8s %8s %8s" % ("band", "n", "slope", "SE", "p", "R2"))
for k, g in d.groupby("kband"):
    x, y = g.out_mon.values, g.Tneutral.values
    lr = stats.linregress(x, y)
    print("%-14s %5d %8.3f %8.3f %8.3f %8.3f" %
          (k, len(g), lr.slope, lr.stderr, lr.pvalue, lr.rvalue ** 2))

t = d[d.kband == "A:Tropical"].copy()
full = stats.linregress(t.out_mon.values, t.Tneutral.values).slope
loo = []
for idx in t.index:
    g = t.drop(idx)
    loo.append((t.loc[idx, "city"], round(stats.linregress(g.out_mon.values, g.Tneutral.values).slope, 3)))
sl = [v for _, v in loo]
print("\nTropical full-data slope = %.3f" % full)
print("Tropical leave-one-city-out slope: min=%.3f max=%.3f" % (min(sl), max(sl)))
print("most influential single-city removals:")
for city, v in sorted(loo, key=lambda z: abs(z[1] - full), reverse=True)[:4]:
    print("   drop %-14s -> slope %.3f" % (city, v))
