# -*- coding: utf-8 -*-
"""Audit DORN longitudinal study: EMA votes, subjects, weather, indoor sensors."""
import os, zipfile, pickle, io, sys
import pandas as pd, numpy as np

BASE = r"D:\thermal-adaptation-series\p1\data\raw\dorn\dorn-longitudinal-tc-study-main"
DATA = os.path.join(BASE, "data")
OUT = r"D:\thermal-adaptation-series\p1\outputs\eda"
os.makedirs(OUT, exist_ok=True)
pd.set_option("display.max_columns", 60); pd.set_option("display.width", 240)

def load_pkl_zip(name):
    p = os.path.join(DATA, name)
    with zipfile.ZipFile(p) as z:
        inner = [n for n in z.namelist() if n.endswith(".pkl")][0]
        with z.open(inner) as f:
            return pickle.load(f)

print("=" * 80, "\n1) df_cozie_env.csv (EMA + matched environment)")
env = pd.read_csv(os.path.join(DATA, "df_cozie_env.csv"))
print("shape:", env.shape)
print("columns:\n ", "\n  ".join(env.columns))
print(env.head(3).to_string())
# find id / time / vote / temperature columns
for key in ("id", "user", "time", "date", "thermal", "vote", "tsv", "comfort", "temp"):
    hits = [c for c in env.columns if key in c.lower()]
    if hits:
        print(f"  cols containing '{key}':", hits)

print("\n" + "=" * 80, "\n2) subject info")
try:
    info = load_pkl_zip("df_info_subjects.pkl.zip")
    print(type(info))
    if isinstance(info, pd.DataFrame):
        print("shape:", info.shape); print(info.to_string())
    else:
        print(info)
except Exception as e:
    print("info fail:", repr(e)[:200])

print("\n" + "=" * 80, "\n3) cozie votes pickle")
try:
    cz = load_pkl_zip("df_cozie.pkl.zip")
    print(type(cz), getattr(cz, "shape", ""))
    if isinstance(cz, pd.DataFrame):
        print("columns:", list(cz.columns)); print(cz.head(3).to_string())
except Exception as e:
    print("cozie fail:", repr(e)[:200])

print("\n" + "=" * 80, "\n4) weather")
try:
    w = load_pkl_zip("df_weather_sg.pkl.zip")
    print(type(w), getattr(w, "shape", ""))
    if isinstance(w, pd.DataFrame):
        print("columns:", list(w.columns)); print(w.head(3).to_string()); print(w.tail(2).to_string())
except Exception as e:
    print("weather fail:", repr(e)[:200])
