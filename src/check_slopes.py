# -*- coding: utf-8 -*-
import numpy as np, pandas as pd
d = pd.read_csv(r"D:\thermal-adaptation-series\p1\outputs\standards\db2_adaptive_slope.csv")
print("raw rows:", len(d), "columns:", list(d.columns))
d = d[(d.Tneutral.between(15, 32)) & (d.out_mon.between(-5, 35))]
print("after FIGURE-5d cleaning rows (cities):", len(d))
for k, g in d.groupby("kband"):
    if len(g) >= 2:
        slope = np.polyfit(g.out_mon, g.Tneutral, 1)[0]
    else:
        slope = np.nan
    print("  %-14s n_cities=%2d  slope(Tneutral on out_mon)=%.3f" % (k, len(g), slope))
