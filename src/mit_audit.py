# -*- coding: utf-8 -*-
import pandas as pd, numpy as np, os
D = r"D:\thermal-adaptation-series\p1\data\raw\mit"
print("#" * 70, "\n# participants.csv")
pp = pd.read_csv(os.path.join(D, "participants.csv"))
print("shape", pp.shape, "\ncols:", list(pp.columns))
print(pp.to_string(max_colwidth=22))

print("#" * 70, "\n# consolidated_atlas.csv")
df = pd.read_csv(os.path.join(D, "consolidated_atlas.csv"), low_memory=False)
print("shape", df.shape)
print("cols:", list(df.columns))
print("\nhead:")
print(df.head(3).to_string(max_colwidth=18))
# categorical / id / time columns
for c in df.columns:
    lc = c.lower()
    if any(k in lc for k in ["city", "country", "location", "site", "participant", "subject", "user", "id", "season", "comfort", "thermal", "vote", "sensation", "preference", "date", "time", "timestamp"]):
        try:
            nun = df[c].nunique(dropna=True)
            if nun <= 30:
                print(f"  [{c}] nunique={nun} values={df[c].dropna().unique()[:30].tolist()}")
            else:
                print(f"  [{c}] nunique={nun} sample={df[c].dropna().unique()[:6].tolist()}")
        except Exception as e:
            print("  ", c, "err", repr(e)[:50])
# parse any datetime col and monthly span
for c in df.columns:
    lc = c.lower()
    if any(k in lc for k in ["date", "time", "timestamp"]):
        t = pd.to_datetime(df[c], format="mixed", errors="coerce", utc=True)
        if t.notna().mean() > 0.3:
            print(f"  TIME {c}: {t.min()} -> {t.max()} | months={sorted(t.dt.month.dropna().unique().tolist())}")
