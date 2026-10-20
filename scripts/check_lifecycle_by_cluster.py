"""Fig. 11 classification by cluster, and its sensitivity to the p threshold.

Answers two review questions:
  * Cluster B collapses from 14 % to 1 % of assigned publications (Fig. 9) — are its 13 topics
    really classified as rising in the lifecycle figure?
  * the figure declares p < 0.10, which is loose; how many topics stay significantly rising at
    p < 0.05 and p < 0.01?
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from src.combined_viz import _compute_yearly_topic_proportions
from src.config import get_config

config = get_config()
yearly_prop, _ = _compute_yearly_topic_proportions(config)
yearly_prop = yearly_prop[[c for c in yearly_prop.columns if int(c) < 2026]]
years = np.array([int(y) for y in yearly_prop.columns])

rows = []
for tid, row in yearly_prop.iterrows():
    tau, p = kendalltau(years, row.values.astype(float))
    rows.append({"Topic": int(tid), "tau": float(tau), "p": float(p)})
df = pd.DataFrame(rows)

clusters = pd.read_csv("data/processed/topic_clusters.csv", encoding="utf-8-sig")
name_of = dict(zip(clusters["Cluster"], clusters["Cluster_Name"]))
df = df.merge(clusters[["Topic", "Cluster"]], on="Topic", how="left")

print(f"years used: {years.min()}-{years.max()}   topics: {len(df)}")
for thr in (0.10, 0.05, 0.01):
    sig = df["p"] < thr
    up = sig & (df["tau"] > 0)
    down = sig & (df["tau"] < 0)
    print(f"  p < {thr:<5}: significant {int(sig.sum()):>2} | rising {int(up.sum()):>2} | "
          f"falling {int(down.sum()):>2} | not significant {int((~sig).sum()):>2}")

print("\nper cluster (p < 0.10 rule, as in the figure):")
print(f"  {'cluster':<10}{'topics':>7}{'rising':>8}{'falling':>9}{'stable':>8}   "
      f"median tau")
for letter in sorted(df["Cluster"].unique()):
    sub = df[df["Cluster"] == letter]
    sig = sub["p"] < 0.10
    print(f"  {letter:<10}{len(sub):>7}{int((sig & (sub['tau'] > 0)).sum()):>8}"
          f"{int((sig & (sub['tau'] < 0)).sum()):>9}{int((~sig).sum()):>8}   "
          f"{sub['tau'].median():+.3f}")

b = df[df["Cluster"] == "B"].sort_values("tau")
print(f"\nCluster B ({len(b)} topics), sorted by tau:")
for _, r in b.iterrows():
    cls = "rising" if (r["p"] < 0.10 and r["tau"] > 0) else (
        "falling" if (r["p"] < 0.10 and r["tau"] < 0) else "stable")
    print(f"   T{int(r['Topic']):<3} tau={r['tau']:+.3f}  p={r['p']:.4f}  {cls}")
print(f"\nB rising at p<0.05: {int(((b['p'] < 0.05) & (b['tau'] > 0)).sum())} of {len(b)}")
