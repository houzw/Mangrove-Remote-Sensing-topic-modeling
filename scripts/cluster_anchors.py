"""Per-cluster anchors for harmonizing the six bullets in the Semantic Themes list.

Recomputes, for every cluster: member topics, assigned documents, share of the assigned
corpus, the share of assigned publications in 2010 / 2015 / 2025, and the mean
within-cluster c-TF-IDF cosine. Every value comes from a data/processed artifact written
by the pipeline, so the numbers can be quoted in the manuscript.

Run: pixi run python scripts_tmp/check_cluster_bullet_anchors.py
"""

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

mapping = dict(zip(clusters["Topic"], clusters["Cluster"]))
names = dict(zip(clusters["Cluster"], clusters["Cluster_Name"]))

m = dt[["doc_index", "Topic"]].copy()
m["PY"] = df["PY"].values[: len(m)]
assigned = m[m["Topic"] != -1].copy()
assigned["Cluster"] = assigned["Topic"].map(mapping)
tot_year = assigned.groupby("PY").size()
total_assigned = len(assigned)
print("assigned documents in the corpus:", total_assigned)
print()

header = "%-3s %-42s %4s %6s %7s %7s %7s %7s" % (
    "cl", "name", "ntop", "docs", "%corpus", "2010", "2015", "2025")
print(header)
print("-" * len(header))

rows = {}
for letter in sorted(summary["Cluster"]):
    sub = assigned[assigned["Cluster"] == letter]
    share = (sub.groupby("PY").size() / tot_year).fillna(0.0)
    rows[letter] = {
        "n": int(summary.loc[summary["Cluster"] == letter, "N_Topics"].iloc[0]),
        "docs": len(sub),
        "corpus": len(sub) / total_assigned,
        "s2010": float(share.get(2010, 0.0)),
        "s2015": float(share.get(2015, 0.0)),
        "s2025": float(share.get(2025, 0.0)),
    }
    r = rows[letter]
    print("%-3s %-42s %4d %6d %6.1f%% %6.1f%% %6.1f%% %6.1f%%"
          % (letter, names.get(letter, "")[:42], r["n"], r["docs"],
             100 * r["corpus"], 100 * r["s2010"], 100 * r["s2015"], 100 * r["s2025"]))

ctfidf_path = Path("outputs/models/bertopic_model/ctfidf.safetensors")
if ctfidf_path.exists():
    from safetensors.numpy import load_file

    vec = load_file(str(ctfidf_path))
    X = np.asarray(vec[next(iter(vec))], dtype=float)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    sim = X @ X.T
    print()
    print("mean within-cluster c-TF-IDF cosine (cohesion):")
    for letter in sorted(rows):
        idx = [t for t, c in mapping.items() if c == letter]
        sub_sim = sim[np.ix_(idx, idx)]
        n = len(idx)
        print("   %s: %.3f  (%d topics)" % (letter, (sub_sim.sum() - n) / (n * n - n), n))

print()
print("trajectories (share of assigned publications):")
for letter in sorted(rows):
    r = rows[letter]
    print("   %s: 2010 %5.1f%%  2015 %5.1f%%  2025 %5.1f%%"
          % (letter, 100 * r["s2010"], 100 * r["s2015"], 100 * r["s2025"]))
