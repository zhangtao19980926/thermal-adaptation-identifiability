# -*- coding: utf-8 -*-
import json, glob, os
import numpy as np
import pandas as pd

OUT = r'D:\thermal-adaptation-series\p1\outputs'
def line(*s): print(' '.join(str(x) for x in s))

# ---- event-level by direction, and converged subset ----
ev = pd.read_csv(os.path.join(OUT,'eda','dorn_event_tau.csv'))
line('tau_min max=%.2f  values >=299 count=%d' % (ev.tau_min.max(), (ev.tau_min>=299).sum()))
line('tau min=%.2f' % ev.tau_min.min())
for label, sub in [('ALL', ev), ('converged(tau<299)', ev[ev.tau_min<299])]:
    line('--- %s n=%d ---' % (label, len(sub)))
    line('  direction counts:', sub.direction.value_counts().to_dict())
    line('  tau med=%.2f IQR=%.2f-%.2f' % (sub.tau_min.median(), sub.tau_min.quantile(.25), sub.tau_min.quantile(.75)))
    for d in ['warm','cool']:
        s=sub[sub.direction==d]
        line('  %s n=%d  tau med=%.2f IQR=%.2f-%.2f | gain med=%.3f | r2 med=%.3f' % (
            d, len(s), s.tau_min.median(), s.tau_min.quantile(.25), s.tau_min.quantile(.75),
            s.gain.median(), s.r2.median()))
    line('  r2 med=%.3f gain med=%.3f' % (sub.r2.median(), sub.gain.median()))

# passive vs event R2 maybe in another file
line('='*20, 'search numeric strings across outputs')
targets = ['1.93','0.57','0.158','18.7','5.6','seasonal','half-life','halflife','half_life','12.0','recovery','7.0','2.2']
files = []
for root,_,fs in os.walk(OUT):
    for f in fs:
        if f.lower().endswith(('.csv','.json','.txt')):
            files.append(os.path.join(root,f))
for t in ['1.93','0.57','0.158','18.7','half','recover','12.0','seasonal sd','band_sd','swing']:
    line('### target %r' % t)
    hits=0
    for fp in files:
        try:
            txt=open(fp,encoding='utf-8',errors='ignore').read()
        except Exception:
            continue
        for ln in txt.splitlines():
            if t.lower() in ln.lower():
                line('  [%s] %s' % (os.path.basename(fp), ln.strip()[:180]))
                hits+=1
                if hits>=12: break
        if hits>=12: break

# list eda + identifiability files for locating band/recovery outputs
line('='*20,'file listing')
for sub in ['eda','identifiability','standards']:
    d=os.path.join(OUT,sub)
    if os.path.isdir(d):
        line(sub, sorted(os.listdir(d)))
