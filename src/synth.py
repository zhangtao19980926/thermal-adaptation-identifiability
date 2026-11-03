# -*- coding: utf-8 -*-
"""
Synthetic data generator for the thermal-adaptation memory-kernel identifiability experiment.

Ground-truth data-generating process (DGP), continuous time in DAYS:

    Each subject i has two hidden adaptation states (fast physiological acclimation,
    slow behavioural/psychological adaptation), each a first-order low-pass filter of
    the operative-temperature deviation from the individual neutral temperature:

        da_f/dt = (x(t) - a_f) / tau_f_i
        da_s/dt = (x(t) - a_s) / tau_s_i

    Latent thermal sensation at vote time:

        z(t) = kappa0 * x(t) - kappa_f * a_f(t) - kappa_s * a_s(t) + b_i + eps
        TSV in {-3..+3} via ordered-logit cutpoints (eps ~ standard logistic).

    Temperature exposure x(t) = T_amb(t) - Tn_i, with T_amb built from seasonal cycle +
    diurnal cycle + OU weather fluctuations + heat-wave pulses.

Votes are EMA-like: irregular, daytime only, spanning the study window.
Everything is vectorised over subjects on a dense integration grid (analytic exponential
steps for linear/OU dynamics), so thousands of subjects are cheap.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------
# Exposure trajectory
# ----------------------------------------------------------------------------
def ambient_temperature(t: np.ndarray, rng: np.random.Generator,
                        seasonal_amp: float = 3.5, seasonal_mean: float = 20.0,
                        ou_sigma: float = 1.2, ou_tau: float = 4.0,
                        diurnal_amp: float = 0.8,
                        n_heatwaves: int = 2, hw_amp: float = 3.0,
                        hw_dur: float = 8.0) -> np.ndarray:
    """Continuous ambient temperature on grid t (days). Shared regional weather."""
    T = t[-1]
    # seasonal (annual) cycle, phase so study starts in late spring / warming half
    seasonal = seasonal_mean + seasonal_amp * np.sin(2 * np.pi * (t + 30.0) / 365.0)
    # diurnal cycle
    diurnal = diurnal_amp * np.sin(2 * np.pi * t - np.pi / 2.0)
    # OU weather process (analytic exponential step on the grid)
    dt = np.diff(t)
    ou = np.zeros_like(t)
    for k in range(len(dt)):
        d = dt[k]
        e = rng.normal() * ou_sigma * np.sqrt(1.0 - np.exp(-2.0 * d / ou_tau))
        ou[k + 1] = ou[k] * np.exp(-d / ou_tau) + e
    # heat-wave pulses at random onset times
    hw = np.zeros_like(t)
    for _ in range(n_heatwaves):
        onset = rng.uniform(15.0, max(16.0, T - hw_dur - 15.0))
        # smooth trapezoid: 1-day ramps
        ramp = np.clip((t - onset) / 1.0, 0.0, 1.0) * np.clip((onset + hw_dur - t) / 1.0, 0.0, 1.0)
        hw += hw_amp * ramp
    return seasonal + diurnal + ou + hw


def multisine_exposure(t: np.ndarray, rng: np.random.Generator,
                       periods=(0.5, 1, 2, 4, 8, 16, 32, 64, 128),
                       amp: float = 0.9, noise_sigma: float = 0.3,
                       noise_tau: float = 0.5, mean_temp: float = 22.0) -> np.ndarray:
    """Persistently exciting multi-frequency exposure (identifiability sanity signal)."""
    base = np.zeros_like(t)
    for P in periods:
        base += amp * np.sin(2 * np.pi * t / P + rng.uniform(0, 2 * np.pi))
    dt = np.diff(t)
    ou = np.zeros_like(t)
    for k in range(len(dt)):
        d = dt[k]
        e = rng.normal() * noise_sigma * np.sqrt(1.0 - np.exp(-2.0 * d / noise_tau))
        ou[k + 1] = ou[k] * np.exp(-d / noise_tau) + e
    return mean_temp + base + ou


def events_exposure(t: np.ndarray, rng: np.random.Generator, mean_temp: float = 22.0,
                    dur=(6.0, 14.0), amp=(2.5, 5.0), gap=(8.0, 20.0),
                    ou_sigma: float = 0.8, ou_tau: float = 3.0,
                    diurnal_amp: float = 0.8) -> np.ndarray:
    """Repeated cold/heat step events (heat-waves, cold snaps, travel, AC switches).

    Step relaxations directly encode the adaptation time constants (each post-event
    return segment behaves as 1-exp(-t/tau)), making this a much stronger excitation
    than smooth seasonal drift.
    """
    base = np.zeros_like(t)
    tt = 5.0
    while tt < t[-1]:
        dv = rng.uniform(*dur)
        av = rng.uniform(*amp) * rng.choice([-1.0, 1.0])
        base[(t >= tt) & (t < tt + dv)] += av
        tt += dv + rng.uniform(*gap)
    dt = np.diff(t)
    ou = np.zeros_like(t)
    for k in range(len(dt)):
        d = dt[k]
        e = rng.normal() * ou_sigma * np.sqrt(1.0 - np.exp(-2.0 * d / ou_tau))
        ou[k + 1] = ou[k] * np.exp(-d / ou_tau) + e
    diurnal = diurnal_amp * np.sin(2 * np.pi * t - np.pi / 2)
    return mean_temp + base + ou + diurnal


def step_relax_exposure(t: np.ndarray, rng: np.random.Generator, t_step: float = 10.0,
                        step_amp: float = 4.0, mean_temp: float = 22.0,
                        ou_sigma: float = 0.6, ou_tau: float = 3.0,
                        diurnal_amp: float = 0.8) -> np.ndarray:
    """A single persistent step (relocation / climate-zone move / season switch).

    The post-step relaxation is a clean double-exponential whose early and late limbs
    separate tau_fast and tau_slow — the strongest possible excitation for identifying
    two time constants. Sign is assigned PER SUBJECT by the caller via the returned
    common base (here a warm step; cold/warm balance done by shifting subjects).
    """
    base = np.where(t >= t_step, step_amp, 0.0)
    dt = np.diff(t)
    ou = np.zeros_like(t)
    for k in range(len(dt)):
        d = dt[k]
        e = rng.normal() * ou_sigma * np.sqrt(1.0 - np.exp(-2.0 * d / ou_tau))
        ou[k + 1] = ou[k] * np.exp(-d / ou_tau) + e
    diurnal = diurnal_amp * np.sin(2 * np.pi * t - np.pi / 2)
    return mean_temp + base + ou + diurnal


# ----------------------------------------------------------------------------
# Core dataset simulator
# ----------------------------------------------------------------------------
def simulate_dataset(
    n_subjects: int = 30,
    n_votes: int = 120,
    study_days: float = 180.0,
    tau_fast_pop: float = 2.0,          # days
    tau_slow_pop: float = 21.0,        # days
    sigma_tau_fast: float = 0.25,      # log-scale between-subject SD
    sigma_tau_slow: float = 0.35,
    kappa0: float = 1.0,
    kappa_fast: float = 0.45,
    kappa_slow: float = 0.35,
    sigma_b: float = 1.0,              # subject neutral temperature SD (deg C)
    report_bias_sd: float = 0.25,      # residual per-subject reporting bias SD
    cutpoints: np.ndarray | None = None,
    miss_cov: float = 0.0,             # probability that ambient covariate at a vote is missing
    regular: bool = False,             # True = fixed daily time; False = irregular EMA
    signal: str = "weather",           # weather | multisine | mixed
    seed: int = 0,
    return_states: bool = False,
):
    """Return a long-format vote dataframe (and optionally dense state arrays)."""
    rng = np.random.default_rng(seed)
    if cutpoints is None:
        cutpoints = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5])

    # dense integration grid (~1.4 h steps)
    dt = 0.06
    t = np.arange(0.0, study_days + dt, dt)
    if signal == "multisine":
        T_amb = multisine_exposure(t, rng)
    elif signal == "events":
        T_amb = events_exposure(t, rng)
    elif signal == "step":
        T_amb = step_relax_exposure(t, rng)
    elif signal == "mixed":
        T_amb = ambient_temperature(t, rng) + 0.75 * (multisine_exposure(t, rng) - 22.0)
    else:
        T_amb = ambient_temperature(t, rng)

    # subject-level parameters
    tau_f = np.exp(np.log(tau_fast_pop) + rng.normal(0, sigma_tau_fast, n_subjects))
    tau_s = np.exp(np.log(tau_slow_pop) + rng.normal(0, sigma_tau_slow, n_subjects))
    Tn = rng.normal(22.0, sigma_b, n_subjects)         # absolute neutral temperature (deg C)
    b = rng.normal(0.0, report_bias_sd, n_subjects)    # residual reporting bias

    # integrate states for every subject on the grid (vectorised)
    x = T_amb[None, :] - Tn[:, None]                  # n_subj x n_grid
    af = np.zeros_like(x)
    as_ = np.zeros_like(x)
    for k in range(len(t) - 1):
        d = t[k + 1] - t[k]
        ef = 1.0 - np.exp(-d / tau_f)
        es = 1.0 - np.exp(-d / tau_s)
        af[:, k + 1] = af[:, k] + (x[:, k] - af[:, k]) * ef
        as_[:, k + 1] = as_[:, k] + (x[:, k] - as_[:, k]) * es

    # sample vote times per subject
    rows = []
    for i in range(n_subjects):
        if regular:
            # one vote per day at a fixed afternoon time
            days = np.arange(2.0, study_days, max(1.0, study_days / n_votes))[:n_votes]
            tv = days + 0.625                            # 15:00
        else:
            # irregular EMA: random days (with replacement), daytime uniform
            days = rng.uniform(2.0, study_days, n_votes)
            tod = rng.uniform(8.0 / 24.0, 20.0 / 24.0, n_votes)
            tv = np.sort(days + tod)
        idx = np.searchsorted(t, tv)
        idx = np.clip(idx, 1, len(t) - 1)
        # linear interpolation helper
        def interp(arr):
            t0, t1 = t[idx - 1], t[idx]
            w = (tv - t0) / (t1 - t0)
            return arr[i, idx - 1] * (1 - w) + arr[i, idx] * w
        xv = interp(x)
        afv = interp(af)
        asv = interp(as_)
        z = kappa0 * xv - kappa_fast * afv - kappa_slow * asv + b[i]
        # ordered logit
        u = z + rng.logistic(size=n_votes)
        tsv = np.digitize(u, cutpoints) - 3             # -3..+3
        cov_missing = rng.random(n_votes) < miss_cov
        rows.append(pd.DataFrame({
            "subject": i,
            "t_day": tv,
            "x_obs": np.where(cov_missing, np.nan, xv),
            "x_true": xv,
            "z_latent": z,
            "TSV": tsv,
            "tau_f_true": tau_f[i],
            "tau_s_true": tau_s[i],
            "Tn": Tn[i],
        }))
    votes = pd.concat(rows, ignore_index=True)
    meta = dict(t=t, T_amb=T_amb, tau_f=tau_f, tau_s=tau_s, Tn=Tn, b=b,
                cutpoints=cutpoints)
    if return_states:
        meta.update(x=x, af=af, as_=as_)
    return votes, meta
