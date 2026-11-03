# -*- coding: utf-8 -*-
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, r"D:\thermal-adaptation-series\p1\src\p1mk")
from ingauge_inject import ingauge_daily, dorn_daily

def bands(v):
    s = pd.Series(v)
    ma7 = s.rolling(7, min_periods=1, center=True).mean()
    ma30 = s.rolling(30, min_periods=1, center=True).mean()
    syn = s-ma7; slow = ma7-ma30; seas = ma30-s.mean()
    return dict(sd=[np.nanstd(syn), np.nanstd(slow), np.nanstd(seas)],
                seas_range=float(seas.max()-seas.min()))

for name, v in [("DORN", dorn_daily()), ("InGauge", ingauge_daily())]:
    b = bands(v)
    print("%s days=%d daily min %.2f max %.2f range %.2f" % (name, len(v), v.min(), v.max(), v.max()-v.min()))
    print("   band SD synoptic %.3f  slow %.3f  seasonal %.3f | seasonal-component range %.2f"
          % (b["sd"][0], b["sd"][1], b["sd"][2], b["seas_range"]))
