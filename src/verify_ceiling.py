# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
from scipy import stats
OUT=r'D:\thermal-adaptation-series\p1\outputs'
def p(*s): print(' '.join(str(x) for x in s))

ev=pd.read_csv(os.path.join(OUT,'eda','dorn_event_tau.csv'))
for crit,lab in [(ev.tau_min>299.5,'tau>299.5'),(ev.tau_min>=299.0,'tau>=299'),(np.isclose(ev.tau_min,300.0,atol=0.5),'|tau-300|<0.5')]:
    sub=ev[crit]
    p(lab,'total=',len(sub),'warm=',int((sub.direction=='warm').sum()),'cool=',int((sub.direction=='cool').sum()))
conv=ev[~np.isclose(ev.tau_min,300.0,atol=0.5)]
p('converged n=',len(conv),'warm=',int((conv.direction=='warm').sum()),'cool=',int((conv.direction=='cool').sum()))
edge=ev[(ev.tau_min>=298)&(ev.tau_min<299.5)]
p('edge 298-299.5 events:'); print(edge[['user','direction','tau_min','r2']].to_string())

het=pd.read_csv(os.path.join(OUT,'eda','dorn_tau_heterogeneity.csv'))
for c in ['tau','tau_warm','tau_cool']:
    r=stats.spearmanr(het[c],het['female'],nan_policy='omit')
    p('Spearman %s ~ female: rho=%.3f p=%.4f'%(c,r.statistic,r.pvalue))
p('occupant tau_warm med=%.2f tau_cool med=%.2f'%(het.tau_warm.median(),het.tau_cool.median()))
p('columns:', list(het.columns))
