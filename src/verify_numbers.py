# -*- coding: utf-8 -*-
import json, glob, os
import numpy as np
import pandas as pd
from scipy import stats

OUT = r'D:\thermal-adaptation-series\p1\outputs'
pd.set_option('display.width', 200)

def line(*s):
    print(' '.join(str(x) for x in s))

# ---------- 1. DORN occupant heterogeneity ----------
line('='*30, 'DORN occupant heterogeneity (n=20)')
het = pd.read_csv(os.path.join(OUT,'eda','dorn_tau_heterogeneity.csv'))
line('users=%d  sum n_ev=%d' % (len(het), het['n_ev'].sum()))
rho_age, p_age = stats.spearmanr(het['tau'], het['Age_n'])
rho_sex, p_sex = stats.spearmanr(het['tau'], het['female'])
line('Spearman tau~Age  rho=%.3f p=%.4f' % (rho_age, p_age))
line('Spearman tau~female rho=%.3f p=%.4f' % (rho_sex, p_sex))
line('occupant tau_warm median=%.2f  tau_cool median=%.2f' % (het['tau_warm'].median(), het['tau_cool'].median()))
line('occupant gain median=%.3f mean=%.3f' % (het['gain'].median(), het['gain'].mean()))
line('occupant tau median=%.2f IQR=%.2f-%.2f' % (het['tau'].median(), het['tau'].quantile(.25), het['tau'].quantile(.75)))

# ---------- 2. DORN event-level fast constants ----------
for cand in ['eda/dorn_event_tau.csv','identifiability/dorn_event_tau.csv']:
    p = os.path.join(OUT,cand)
    if os.path.exists(p):
        line('='*30, 'event tau file:', cand)
        ev = pd.read_csv(p)
        line('cols=%s' % list(ev.columns))
        line('n rows=%d' % len(ev))
        # detect direction column
        dcol = [c for c in ev.columns if c.lower() in ('direction','dir','warm','sign','event_type','type')]
        line('direction candidates=%s' % dcol)
        tcol = [c for c in ev.columns if c.lower().startswith('tau')]
        line('tau cols=%s' % tcol)
        # generic numeric summaries
        for c in ev.columns:
            if pd.api.types.is_numeric_dtype(ev[c]):
                s = ev[c].dropna()
                if len(s)>0:
                    line('  %s: med=%.3f IQR=%.3f-%.3f mean=%.3f' % (c, s.median(), s.quantile(.25), s.quantile(.75), s.mean()))
        # if direction present
        for dc in dcol:
            line('value counts %s:' % dc, ev[dc].value_counts().to_dict())
        break

# ---------- 3. counterfactual ----------
line('='*30, 'In-Gauge counterfactual')
cf = pd.read_csv(os.path.join(OUT,'standards','ingauge_counterfactual.csv'))
line('days=%d' % len(cf))
line('Tout min=%.2f max=%.2f mean=%.3f  daily range=%.2f' % (cf.Tout.min(), cf.Tout.max(), cf.Tout.mean(), cf.Tout.max()-cf.Tout.min()))
line('Tout SD=%.3f' % cf.Tout.std())
for c in ['Trm7','Trm30','Trm_tau21']:
    line('%s range=%.2f SD=%.3f' % (c, cf[c].max()-cf[c].min(), cf[c].std()))
b7 = cf['Tcomf7']-cf['Tcomf_tau21']
b30 = cf['Tcomf30']-cf['Tcomf_tau21']
line('bias 7d-vs-tau21: mean=%.4f integrated=%.2f' % (b7.mean(), b7.sum()))
line('bias 30d-vs-tau21: mean=%.4f integrated=%.2f' % (b30.mean(), b30.sum()))
line('cool_frac mean=%.4f%% max=%.4f%%' % (cf.cool_frac.mean()*100, cf.cool_frac.max()*100))
# CDH attempt: cooling degree hours vs comfort, weighted by occupied fraction proxy
for c in ['Tcomf7','Tcomf30','Tcomf_tau21']:
    cdh = (np.maximum(cf['Tin_occ']-cf[c],0)*24).sum()
    line('CDH(24h,Tin_occ-%s)=%.1f' % (c, cdh))
    cdhf = (np.maximum(cf['Tin_occ']-cf[c],0)*cf['cool_frac']*24).sum()
    line('CDH(cool_frac weighted,%s)=%.1f' % (c, cdhf))

# ---------- 4. DORN daily seasonal amplitude ----------
line('='*30, 'DORN daily amplitude')
for cand in ['eda/dorn_daily_tenv.csv','identifiability/dorn_daily_tenv.csv']:
    p=os.path.join(OUT,cand)
    if os.path.exists(p):
        dd=pd.read_csv(p)
        line('file',cand,'cols',list(dd.columns),'rows',len(dd))
        numcols=[c for c in dd.columns if pd.api.types.is_numeric_dtype(dd[c])]
        for c in numcols:
            s=dd[c].dropna()
            if s.std()>0.1:
                rm=s.rolling(30,min_periods=1).mean()
                line('  %s daily range=%.2f SD=%.3f | 30d-mean range=%.2f SD=%.3f' % (c, s.max()-s.min(), s.std(), rm.max()-rm.min(), rm.std()))
        break

# ---------- 5. identifiability synthetic ceiling ----------
line('='*30, 'ident map profiles')
p=os.path.join(OUT,'identifiability','ident_map_profiles.json')
if os.path.exists(p):
    j=json.load(open(p,encoding='utf-8'))
    line('type=%s' % type(j))
    if isinstance(j,dict):
        line('top keys=%s' % list(j.keys())[:40])
        # print compact summary of each leaf
        def walk(o,path=''):
            if isinstance(o,dict):
                keys=set(o.keys())
                if {'N','T'} & keys or 'signal' in keys or 'regime' in keys:
                    summary={k:o[k] for k in o if not isinstance(o[k],(list,dict))}
                    line(path, summary)
                for k,v in o.items():
                    if isinstance(v,(dict,list)): walk(v,path+'/'+str(k))
            elif isinstance(o,list):
                for i,v in enumerate(o):
                    if isinstance(v,dict): walk(v,path+'[%d]'%i)
        walk(j)
    elif isinstance(j,list):
        line('len=%d'%len(j))
        for rec in j:
            if isinstance(rec,dict):
                line({k:rec[k] for k in rec if not isinstance(rec[k],(list,dict))})

for fn in ['smoke_summary.json','ident_map.json']:
    p=os.path.join(OUT,'identifiability',fn)
    if os.path.exists(p):
        line('='*20, fn)
        j=json.load(open(p,encoding='utf-8'))
        line(json.dumps(j, ensure_ascii=False)[:3000])

for fn in ['cellI_mse.csv','cellII_nll.csv','profile_nll.csv','ident_map.csv']:
    p=os.path.join(OUT,'identifiability',fn)
    if os.path.exists(p):
        line('='*20, fn)
        d=pd.read_csv(p)
        line(d.to_string()[:3000])

# ---------- 6. DB II adaptive slopes recompute ----------
line('='*30, 'DB II adaptive slope recompute')
sl=pd.read_csv(os.path.join(OUT,'standards','db2_adaptive_slope.csv'))
sl['band']=sl['kband'].str[0]
sl['bandname']=sl['kband'].str[2:]
line('raw city rows=%d' % len(sl))
phys = sl[sl['Tneutral']>0].copy()  # drop Bauchi -50.7
line('after physical filter (Tn>0): %d' % len(phys))
for b in ['A','B','C','D']:
    sub=phys[phys['band']==b]
    r=stats.linregress(sub['out_mon'], sub['Tneutral'])
    line('band %s %s: n=%d slope=%.4f p=%.4f r2=%.4f' % (b, sub['bandname'].iloc[0], len(sub), r.slope, r.pvalue, r.rvalue**2))
# try stricter: cities with neutral within plausible 15-32 and out_mon range
line('-- with |Tn| in [15,32] filter --')
f=phys[(phys['Tneutral']>=15)&(phys['Tneutral']<=32)]
for b in ['A','B','C','D']:
    sub=f[f['band']==b]
    if len(sub)>=3:
        r=stats.linregress(sub['out_mon'], sub['Tneutral'])
        line('band %s: n=%d slope=%.4f p=%.4f r2=%.4f' % (b, len(sub), r.slope, r.pvalue, r.rvalue**2))

# ---------- 7. abstract word count ----------
line('='*30, 'abstract word count')
from docx import Document
doc=Document(r'D:\BE投稿1\manuscript.docx')
paras=[p.text for p in doc.paragraphs]
for i,t in enumerate(paras):
    if t.strip().lower().startswith('abstract') or (i<15 and len(t)>200):
        body=t.replace('Abstract','').replace('ABSTRACT','').strip()
        line('para idx=%d words=%d' % (i, len(body.split())))
        break
# also the longest of first 12 paragraphs
for i,t in enumerate(paras[:12]):
    if len(t.split())>100:
        line('  candidate para %d words=%d' % (i,len(t.split())))
