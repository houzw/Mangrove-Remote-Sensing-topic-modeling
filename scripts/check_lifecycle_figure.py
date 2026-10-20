"""What does Fig. 11 actually plot, and is the "majority in the growth quadrant" claim true?

Reproduces the producer's own computation (combined_viz.plot_topic_lifecycle): Kendall's tau and
mean yearly proportion per topic, class = Stable if p >= 0.10 else Hot (tau>0) / Cold (tau<0), and
the two reference lines (tau = 0 vertical, median mean-proportion horizontal).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from src.combined_viz import _compute_yearly_topic_proportions
from src.config import get_config

P_THRESHOLD = 0.10

config = get_config()
yearly_prop, assigned = _compute_yearly_topic_proportions(config)
years = np.array([int(y) for y in yearly_prop.columns])

rows = []
for topic_id, row in yearly_prop.iterrows():
    vals = row.values.astype(float)
    tau, p = kendalltau(years, vals)
    rows.append({"Topic": int(topic_id), "tau": float(tau), "p": float(p),
                 "mean": float(vals.mean()), "active": int((vals > 0).sum())})

df = pd.DataFrame(rows)
df["Class"] = np.where(df["p"] >= P_THRESHOLD, "Stable",
                       np.where(df["tau"] > 0, "Hot", "Cold"))
med = df["mean"].median()

print(f"topics: {len(df)}   years: {years.min()}-{years.max()}   median mean-proportion: {med:.4f}")
print(df["Class"].value_counts().to_string())
print(f"significant (p < {P_THRESHOLD}): {int((df['p'] < P_THRESHOLD).sum())}")

df["above_median"] = df["mean"] > med
df["rising"] = df["tau"] > 0
print("\nquadrants on the plotted axes (x = Kendall tau, y = mean yearly proportion):")
print(f"  rising  & above median (text calls this 'growth'): {int((df.rising & df.above_median).sum())}")
print(f"  rising  & below median                          : {int((df.rising & ~df.above_median).sum())}")
print(f"  falling & above median                          : {int((~df.rising & df.above_median).sum())}")
print(f"  falling & below median                          : {int((~df.rising & ~df.above_median).sum())}")
n_growth = int((df.rising & df.above_median).sum())
print(f"\n=> 'majority in the growth quadrant' would need > {len(df)//2 + 1}; actual {n_growth} "
      f"({n_growth/len(df):.0%})")
print("\ntopics with the largest |tau|:")
print(df.reindex(df.tau.abs().sort_values(ascending=False).index).head(6)[
    ["Topic", "tau", "mean", "active", "Class"]].to_string(index=False))
