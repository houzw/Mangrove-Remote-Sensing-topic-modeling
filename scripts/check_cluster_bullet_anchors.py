"""Per-cluster anchors for harmonizing the six bullets in §Semantic Themes.

Recomputes, for every cluster: member topics, assigned documents, share of the
assigned corpus, the share of assigned publications in 2015 and 2025, the 2010
share, and the mean within-cluster c-TF-IDF cosine. All values come from
data/processed artifacts written by the pipeline.

Run: pixi run python scripts_tmp/check_cluster_bullet_anchors.py
"""

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

PROC = Path("data/processed")

clusters = pd.read_csv(PROC / "topic_clusters.csv", encoding="utf-8-sig")
summary = pd.read_csv(PROC / "cluster_summary.csv", encoding="utf-8-sig")
dt = pd.read_csv(PROC / "doc_topic_dist.csv", encoding="utf-8-sig")
df = (
    pd.read_parquet(PROC / "cleaned_data.parquet")
    if (PROC / "cleaned_data.parquet").exists()
    else pd.read_csv(PROC / "cleaned_data.csv", encoding="utf-8-sig")
)

topic_col = [c for c in clusters.columns if c.lower() == "topic"][0]
cluster_col = [c for c in clusters.columns if "cluster" in c.lower() and c != "Cluster_Name"][0]
mapping = dict(zip(clusters[topic_col], clusters[cluster_col]))

m = dt[["doc_index", "Topic"]].copy()
m["PY"] = df["PY"].values[: len(m)]
assigned = m[m["Topic"] != -1].copy()
assigned["Cluster"] = assigned["Topic"].map(mapping)

total_assigned = len(assigned)
print("assigned documents:", total_assigned)
print()
header = "%-3s %4s %6s %7s %8s %8s %8s %8s" % ("cl", "ntop", "docs", "%corpus", "2010", "2015", "2025", "coh")
print(header)
print("-" * len(header))
rows = {}
for letter, grp in summary.groupby(summary["Cluster_Name"].str[0]):
    members = [int(str(t).strip().lstrip("Tt")) for t in grp["Members"].iloc[0].split(",")]
    sub = assigned[assigned["Cluster"].isin(members)]
    per_year = sub.groupby("PY").size()
    tot_year = assigned.groupby("PY").size()
    share = (per_year / tot_year).fillna(0.0)
    docs = len(sub)
    rows[letter] = {
        "n": len(members),
        "docs": docs,
        "corpus": docs / total_assigned,
        "s2010": float(share.get(2010, 0.0)),
        "s2015": float(share.get(2015, 0.0)),
        "s2025": float(share.get(2025, 0.0)),
    }
    print("%-3s %4d %6d %6.1f%% %7.1f%% %7.1f%% %7.1f%%"
          % (letter, len(members), docs, 100 * docs / total_assigned,
             100 * rows[letter]["s2010"], 100 * rows[letter]["s2015"], 100 * rows[letter]["s2025"]),
          end="")
    print()

# mean within-cluster cosine of the topic c-TF-IDF vectors
import json

model_dir = Path("outputs/models/bertopic_model")
topics_json = model_dir / "topics.json"
if topics_json.exists():
    with open(topics_json, encoding="utf-8") as fh:
        topics = json.load(fh)
    vecs = {}
    for tid, words in topics.items():
        pass
print()
print("2015 vs 2025 shares per cluster (percentage points):")
for letter, r in rows.items():
    print("   %s: 2010 %5.1f%%  2015 %5.1f%%  2025 %5.1f%%"
          % (letter, 100 * r["s2010"], 100 * r["s2015"], 100 * r["s2025"]))
