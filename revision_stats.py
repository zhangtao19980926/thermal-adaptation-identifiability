# -*- coding: utf-8 -*-
"""Recompute every statistic added during the revision, from the CSV results in this repo.

Outputs results/revision/revision_stats.json and prints a summary. All numbers cited in
the response letter (censoring, KM median, clustered bootstrap, step-magnitude split,
trait p-values, paired fold contrasts, LRT, tau_s sensitivity) are reproduced here.
Run:  python revision_stats.py   (numpy/pandas/scipy only; no GPU needed)
"""
import os, json
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = {}


def km_median(times, censor_at):
    """Kaplan-Meier median with right-censoring at censor_at."""
    obs = np.asarray(times, float)
    cens = obs > censor_at
    t = np.where(cens, censor_at, obs)
    order = np.argsort(t)
    t, cens = t[order], cens[order]
    n = len(t)
    S = 1.0
    i = 0
    median = None
    while i < n:
        j = i
        while j < n and t[j] == t[i]:
            j += 1
        d = int((~cens[i:j]).sum())
        S *= (1 - d / (n - i)) if d else 1.0
        if median is None and S <= 0.5:
            median = float(t[i])
        i = j
    return median, float(S)


# ---------------- 1. DORN fast constant: censoring, step magnitude ----------------
ev = pd.read_csv(os.path.join(HERE, "results", "eda", "dorn_event_tau.csv"))
tau = ev.tau_min.values
cens = {"n_events": int(len(ev)),
        "n_users": int(ev.user.nunique()),
        "median_min": round(float(np.median(tau)), 2),
        "iqr": [round(float(np.quantile(tau, .25)), 2), round(float(np.quantile(tau, .75)), 2)],
        "n_at_300min_ceiling": int((tau >= 299.99).sum()),
        "n_beyond_40min_window": int((tau > 40).sum()),
        "pct_beyond_40min_window": round(100 * float((tau > 40).mean()), 1),
        "warm_n": int((ev.direction == "warm").sum()),
        "cool_n": int((ev.direction == "cool").sum()),
        "warm_median": round(float(ev[ev.direction == "warm"].tau_min.median()), 2),
        "cool_median": round(float(ev[ev.direction == "cool"].tau_min.median()), 2),
        "r2_median": round(float(ev.r2.median()), 2),
        "gain_median": round(float(ev.gain.median()), 2)}
km, s40 = km_median(tau, 40.0)
cens["kaplan_meier_median_min"] = round(km, 2)
cens["kaplan_meier_S40"] = round(s40, 3)
sub = tau[tau <= 40]
cens["restricted_median_min"] = round(float(np.median(sub)), 2)
cens["restricted_n"] = int(len(sub))
# clustered (occupant) bootstrap CI of the pooled median
rng = np.random.default_rng(0)
users = ev.user.unique()
meds = []
by_user = {u: ev[ev.user == u].tau_min.values for u in users}
for _ in range(2000):
    pick = rng.choice(users, size=len(users), replace=True)
    meds.append(np.median(np.concatenate([by_user[u] for u in pick])))
cens["clustered_bootstrap_95ci"] = [round(float(np.percentile(meds, 2.5)), 1),
                                    round(float(np.percentile(meds, 97.5)), 1)]
# step magnitude
ev["absdT"] = ev.dTair.abs()
small, big = ev[ev.absdT < 3].tau_min, ev[ev.absdT >= 3].tau_min
mw = stats.mannwhitneyu(small, big)
cens["step_magnitude"] = {"small_n": int(len(small)), "small_median": round(float(small.median()), 1),
                          "big_n": int(len(big)), "big_median": round(float(big.median()), 1),
                          "mannwhitney_p": float(mw.pvalue)}
OUT["fast_constant_censoring"] = cens

# ---------------- 2. Trait correlations (n = 20) ----------------
h = pd.read_csv(os.path.join(HERE, "results", "eda", "dorn_tau_heterogeneity.csv"))
traits = {}
for c in ["BMI", "Age_n", "female", "coldhand", "exercise", "years_SG"]:
    subd = h[["tau", c]].dropna()
    if len(subd) < 5:
        continue
    rho, p = stats.spearmanr(subd.tau, subd[c])
    traits[c] = {"n": int(len(subd)), "rho": round(float(rho), 3), "p": round(float(p), 3)}
traits["age_BMI_corr"] = {"rho": round(float(stats.spearmanr(h.Age_n, h.BMI)[0]), 3),
                          "p": round(float(stats.spearmanr(h.Age_n, h.BMI)[1]), 3)}
# OLS joint adjustment
X = np.column_stack([np.ones(len(h)), h.Age_n, h.BMI, h.female])
y = h.tau.values
b, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
resid = y - X @ b
s2 = resid @ resid / (len(y) - X.shape[1])
cov = s2 * np.linalg.inv(X.T @ X)
se = np.sqrt(np.diag(cov))
t_age = b[1] / se[1]
traits["age_joint_OLS"] = {"b": round(float(b[1]), 3), "se": round(float(se[1]), 3),
                           "p": round(float(2 * (1 - stats.t.cdf(abs(t_age), len(y) - 4))), 3)}
OUT["trait_correlations"] = traits

# ---------------- 3. Paired fold contrasts (CTCD) ----------------
fold = pd.read_csv(os.path.join(HERE, "results", "revision", "ctcd_fold_metrics.csv"))
piv = fold[fold.stratum == "ALL"].pivot_table(index="fold", columns="config", values="mse")
folds_stats = {}
for cfg in ["+m7", "+m15", "+m30"]:
    diff = piv[cfg] - piv["base"]
    t, p = stats.ttest_rel(piv[cfg], piv["base"])
    folds_stats["ALL_%s_minus_base" % cfg] = {
        "mean": round(float(diff.mean()), 4), "sd": round(float(diff.std(ddof=1)), 4),
        "t": round(float(t), 2), "p": round(float(p), 2)}
ph = fold[fold.stratum == "Heated"].pivot_table(index="fold", columns="config", values="mse")
diffh = ph["+m15"] - ph["base"]
th, phh = stats.ttest_rel(ph["+m15"], ph["base"])
folds_stats["Heated_m15_minus_base"] = {
    "mean": round(float(diffh.mean()), 2), "sd": round(float(diffh.std(ddof=1)), 2),
    "t": round(float(th), 2), "p": round(float(phh), 2),
    "per_fold": [round(float(v), 3) for v in diffh.values]}
for a, b in [("+m15", "+m7"), ("+m15", "+m30")]:
    d = piv[a] - piv[b]
    t, p = stats.ttest_rel(piv[a], piv[b])
    folds_stats["%s_minus_%s" % (a, b)] = {"mean": round(float(d.mean()), 5),
                                           "sd": round(float(d.std(ddof=1)), 5),
                                           "t": round(float(t), 2), "p": round(float(p), 2)}
rep = pd.read_csv(os.path.join(HERE, "results", "revision", "ctcd_repeated_cv.csv"))
pivr = rep.pivot_table(index="rep", columns="config", values="mse")
dr = pivr["+m15"] - pivr["base"]
folds_stats["repeated_partitions_m15_minus_base"] = {
    "mean": round(float(dr.mean()), 4), "sd": round(float(dr.std(ddof=1)), 4),
    "min": round(float(dr.min()), 4), "max": round(float(dr.max()), 4),
    "n_negative": int((dr < 0).sum()), "n_partitions": int(len(dr))}
OUT["ctcd_paired_folds"] = folds_stats

# ---------------- 4. Likelihood-ratio test (injection, Table 2) ----------------
inj = pd.read_csv(os.path.join(HERE, "results", "identifiability", "realband_injection.csv"))
lrt = {}
for _, r in inj.iterrows():
    n_votes = r["N"] * (150 if r["N"] == 40 else 120)
    lrt["%s_N%d" % (r["band"], r["N"])] = {
        "n_votes": int(n_votes),
        "dNLL_vs_fixed30": round(float(r["dNLL_dual_vs_fixed30"]), 4),
        "LRT_statistic": round(float(2 * r["dNLL_dual_vs_fixed30"] * n_votes), 0)}
OUT["likelihood_ratio_tests"] = lrt

# ---------------- 5. tau_s sensitivity (1800 epochs) ----------------
sens = pd.read_csv(os.path.join(HERE, "results", "revision", "tau_sensitivity_1800ep.csv"))
ae = (sens.recovered_ts - sens.true_ts).abs()
OUT["tau_sensitivity_1800ep"] = {
    "true_days": [float(x) for x in sens.true_ts],
    "recovered_days": [round(float(x), 2) for x in sens.recovered_ts],
    "mean_abs_err_d": round(float(ae.mean()), 2),
    "max_abs_err_d": round(float(ae.max()), 2)}

with open(os.path.join(HERE, "results", "revision", "revision_stats.json"), "w") as f:
    json.dump(OUT, f, indent=1)
print(json.dumps(OUT, indent=1)[:6000])
