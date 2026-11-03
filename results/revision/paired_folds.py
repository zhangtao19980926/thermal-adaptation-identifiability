# -*- coding: utf-8 -*-
"""Paired fold-level contrasts between candidate windows (answers R1.4)."""
import pandas as pd, numpy as np
from scipy import stats

P = r'D:\主文件夹\文章返修\_rev\ctcd_fold_metrics.csv'
d = pd.read_csv(P)
for st in ['ALL', 'NV', 'AC', 'Heated']:
    g = d[d.stratum == st]
    if g.empty:
        continue
    piv_m = g.pivot_table(index='fold', columns='config', values='mse')
    piv_l = g.pivot_table(index='fold', columns='config', values='logloss')
    print('=====', st, 'folds', len(piv_m))
    print('  MSE by fold:')
    print(piv_m.round(4).to_string())
    base = piv_m['base']
    for cfg in ['+m7', '+m15', '+m30']:
        diff = piv_m[cfg] - base
        t, p = stats.ttest_rel(piv_m[cfg], base)
        print('   %-4s - base : mean %+.4f  SD %.4f  t=%.2f p=%.3f' %
              (cfg, diff.mean(), diff.std(ddof=1), t, p))
    for a, b in [('+m15', '+m7'), ('+m15', '+m30')]:
        diff = piv_m[a] - piv_m[b]
        t, p = stats.ttest_rel(piv_m[a], piv_m[b])
        print('   %s - %s : mean %+.4f SD %.4f  t=%.2f p=%.3f' % (a, b, diff.mean(), diff.std(ddof=1), t, p))
    print('  logloss by fold:')
    print(piv_l.round(4).to_string())
    for cfg in ['+m7', '+m15', '+m30']:
        diff = piv_l[cfg] - piv_l['base']
        t, p = stats.ttest_rel(piv_l[cfg], piv_l['base'])
        print('   %-4s - base : mean %+.5f SD %.5f t=%.2f p=%.3f' %
              (cfg, diff.mean(), diff.std(ddof=1), t, p))
