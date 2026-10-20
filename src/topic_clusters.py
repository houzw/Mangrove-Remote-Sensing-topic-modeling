"""Canonical six-cluster aggregation of the 72 BERTopic topics (manuscript Table 1).

Definition
----------
Cluster membership is the **Ward cut at k = 6** of the 72 topic vectors in the
fitted model's class-based TF-IDF (c-TF-IDF) space.  The topic vectors are
$L_2$-normalized and Ward's minimum-variance criterion is applied to their
Euclidean distances (``scipy`` Ward expects Euclidean distances); for unit-norm
vectors the Euclidean distance is $\\sqrt{2(1-\\cos)}$, a monotone transform of
the cosine distance, so this is the usual cosine-space reading made explicit.
All three defensible variants -- Ward on ``1 - cos``, on the Euclidean distances
of the $L_2$-normalized rows, and on their squares -- were checked to give the
**identical** six-cluster partition, so the membership below is not an artifact
of the distance convention.

k = 6 is not selected by silhouette -- silhouette is uniformly low across k
(maximum +0.06 at k = 2), i.e. the topic vectors carry no strong cluster
structure at any k.  It is selected by the linkage-height gap: the widest gap in
the upper dendrogram (+0.18, between the 6- and 5-cluster solutions) is the
largest below the top-level two-cluster split, and it separates exactly six
groups.

The membership below is *frozen*: it was read off the production model
(``outputs/models/bertopic_model``, fitted 2026-08-17) and is re-verified against
that model on every run.  If the topic model changes, this module raises rather
than silently renaming clusters.

Letters are ordered by descending member count (ties broken by descending 2025
share).  Shares are shares of the **assigned** publications (outlier topic -1
excluded), matching the convention of ``combined_viz._compute_yearly_topic_proportions``
and of the manuscript's Table 1.

Usage:
    pixi run python -m src.topic_clusters            # verify + write outputs
    pixi run python -m src.topic_clusters --check     # verify only
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.sparse import csr_matrix
from scipy.spatial.distance import squareform
from safetensors import safe_open

from src.config import get_config

logger = logging.getLogger(__name__)

K = 6
BASE_YEAR = 2025  # latest complete year

# ---------------------------------------------------------------------------
# Frozen canonical definition
# ---------------------------------------------------------------------------
CLUSTERS: list[dict] = [
    {
        "letter": "A",
        "name": "Extent, Deforestation and Shoreline Change",
        "members": [3, 7, 9, 10, 11, 13, 14, 16, 18, 25, 29, 38, 44, 50, 53, 54, 58, 63],
        "keywords": "mangrove forest cover; mangrove deforestation; shoreline changes",
    },
    {
        "letter": "B",
        "name": "Classical Spectral and SAR Methods",
        "members": [12, 22, 36, 37, 46, 47, 48, 51, 52, 55, 69, 70, 71],
        "keywords": "hyperspectral data; SAR backscatter; spectral angle mapper",
    },
    {
        "letter": "C",
        "name": "Mapping for Blue Carbon and Conservation",
        "members": [0, 2, 4, 5, 17, 30, 32, 39, 42, 49, 57, 65],
        "keywords": "blue carbon ecosystems; coastal resilience; mangrove restoration",
    },
    {
        "letter": "D",
        "name": "Biogeomorphology and Sea-Level Response",
        "members": [6, 15, 20, 23, 26, 27, 28, 40, 43, 45, 62, 66],
        "keywords": "estuarine; sea level rise; mangrove sediments",
    },
    {
        "letter": "E",
        "name": "Canopy Structure and Carbon Stocks",
        "members": [1, 19, 21, 24, 31, 33, 34, 64, 67],
        "keywords": "mangrove aboveground biomass; mangrove canopy; mangrove carbon stocks",
    },
    {
        "letter": "F",
        "name": "Deep Learning and Species Mapping",
        "members": [8, 35, 41, 56, 59, 60, 61, 68],
        "keywords": "semantic segmentation; mangrove species classification; deep learning",
    },
]


# ---------------------------------------------------------------------------
# Model loading (c-TF-IDF only -- avoids loading the SciBERT encoder)
# ---------------------------------------------------------------------------
def load_ctfidf(model_dir: Path) -> tuple[np.ndarray, list[int], dict[int, list]]:
    """Return (topic_rows, valid_topic_ids, topic_representations).

    Row order follows the representation key order, exactly as
    ``topic_modeling`` and ``meta_topic_dendrogram`` index ``model.c_tf_idf_``.
    """
    safet = model_dir / "ctfidf.safetensors"
    state_path = model_dir / "topics.json"
    if not safet.exists() or not state_path.exists():
        raise FileNotFoundError(f"Saved c-TF-IDF state not found under {model_dir}")

    with safe_open(str(safet), framework="pt") as f:
        mat = csr_matrix(
            (
                f.get_tensor("data").numpy(),
                f.get_tensor("indices").numpy(),
                f.get_tensor("indptr").numpy(),
            ),
            shape=tuple(f.get_tensor("shape").numpy().tolist()),
        )

    state = json.loads(state_path.read_text(encoding="utf-8"))
    reps_raw = state["topic_representations"]
    keys = [int(k) for k in reps_raw.keys()]
    valid = sorted(t for t in keys if t != -1)
    tid_to_row = {tid: i for i, tid in enumerate(keys)}
    rows = np.asarray([mat[tid_to_row[t]].toarray()[0] for t in valid])
    reps = {int(k): v for k, v in reps_raw.items() if int(k) != -1}
    return rows, valid, reps


def topic_distance_matrix(rows: np.ndarray) -> np.ndarray:
    """Euclidean distances between the L2-normalized c-TF-IDF topic vectors.

    For unit-norm rows ``||a - b|| = sqrt(2 (1 - cos(a, b)))``, so this is the
    cosine dissimilarity of the topic space expressed as a Euclidean distance --
    the metric Ward's minimum-variance criterion assumes.
    """
    unit = rows / np.linalg.norm(rows, axis=1, keepdims=True)
    cos = np.clip(unit @ unit.T, -1.0, 1.0)
    dist = np.sqrt(np.maximum(2.0 * (1.0 - cos), 0.0))
    np.fill_diagonal(dist, 0.0)
    return dist


def ward_partition(rows: np.ndarray, valid: list[int], k: int = K) -> dict[int, int]:
    """Ward cut of the topic space into ``k`` groups."""
    dist = topic_distance_matrix(rows)
    z = linkage(squareform(dist, checks=False), method="ward", optimal_ordering=True)
    labels = fcluster(z, t=k, criterion="maxclust")
    return {tid: int(lab) for tid, lab in zip(valid, labels)}, z, dist


def k_diagnostics(z: np.ndarray, dist: np.ndarray, labels: np.ndarray) -> dict:
    """Silhouette scan and linkage-height gaps behind the k = 6 choice."""
    # Imported here, not at module scope: sklearn pulls in scipy.stats, which is a
    # heavyweight diagnostics-only dependency for this one scan.
    from sklearn.metrics import silhouette_score

    heights = z[:, 2]
    silhouette = {}
    for k in range(2, 13):
        lab = fcluster(z, t=k, criterion="maxclust")
        if len(set(lab)) < 2:
            continue
        silhouette[k] = float(silhouette_score(dist, lab, metric="precomputed"))
    gaps = sorted(
        ((float(heights[i] - heights[i - 1]), i + 1, 72 - (i + 1)) for i in range(1, len(heights))),
        reverse=True,
    )[:4]
    return {
        "silhouette": silhouette,
        "silhouette_argmax_k": max(silhouette, key=silhouette.get),
        "largest_gaps": gaps,
        # h[j] is the height of merge j+1, which leaves 71 - j clusters, so the
        # merge that produces K clusters is at index 71 - K.
        "height_6_clusters": float(heights[71 - K]),
    }


def verify_against_canonical(partition: dict[int, int]) -> None:
    """Raise unless the computed partition equals the frozen membership."""
    computed = {frozenset(t for t, l in partition.items() if l == lab)
                for lab in set(partition.values())}
    canonical = {frozenset(c["members"]) for c in CLUSTERS}
    if computed != canonical:
        only_computed = [sorted(s) for s in computed - canonical]
        only_canonical = [sorted(s) for s in canonical - computed]
        raise AssertionError(
            "Ward k=6 partition no longer matches the frozen cluster definition.\n"
            f"  computed-but-not-canonical: {only_computed}\n"
            f"  canonical-but-not-computed: {only_canonical}\n"
            "Re-derive the membership and update CLUSTERS (and the manuscript) "
            "before regenerating any figure or table."
        )


# ---------------------------------------------------------------------------
# Shares
# ---------------------------------------------------------------------------
def yearly_topic_shares(config) -> pd.DataFrame:
    """Topic x year share of *assigned* publications (outlier -1 excluded)."""
    proc = config.paths.processed_dir
    doc_topic_path = proc / config.topic_modeling.output.doc_topic_dist
    doc = pd.read_csv(doc_topic_path, encoding="utf-8-sig", usecols=["Topic", "PY"])
    doc = doc[doc["Topic"] != -1]
    counts = doc.groupby(["PY", "Topic"]).size().rename("n").reset_index()
    totals = counts.groupby("PY")["n"].sum().rename("year_total")
    counts = counts.merge(totals, on="PY")
    counts["share"] = counts["n"] / counts["year_total"]
    return counts.pivot_table(index="Topic", columns="PY", values="share", fill_value=0.0)


def cluster_frame(config, partition: dict[int, int], shares: pd.DataFrame) -> pd.DataFrame:
    """Per-topic table: cluster letter/name, docs, share in the base year."""
    proc = config.paths.processed_dir
    doc_topic_path = proc / config.topic_modeling.output.doc_topic_dist
    doc = pd.read_csv(doc_topic_path, encoding="utf-8-sig", usecols=["Topic"])
    doc = doc[doc["Topic"] != -1]
    docs = doc.groupby("Topic").size()

    letter_of = {}
    name_of = {}
    for c in CLUSTERS:
        for t in c["members"]:
            letter_of[t] = c["letter"]
            name_of[t] = c["name"]

    rows = []
    for t in sorted(partition):
        have = t in shares.index
        rows.append(
            {
                "Topic": t,
                "Cluster": letter_of[t],
                "Cluster_Name": name_of[t],
                "Docs": int(docs.get(t, 0)),
                f"Share_{BASE_YEAR}": float(shares.loc[t, BASE_YEAR]) if have and BASE_YEAR in shares.columns else 0.0,
                "Share_2026_H1": float(shares.loc[t, 2026]) if have and 2026 in shares.columns else 0.0,
            }
        )
    return pd.DataFrame(rows)


def cluster_summary(frame: pd.DataFrame, reps: dict[int, list]) -> pd.DataFrame:
    """One row per cluster: members, doc total, aggregated top keywords, shares."""
    out = []
    for c in CLUSTERS:
        sub = frame[frame["Cluster"] == c["letter"]]
        words: dict[str, float] = {}
        for t in c["members"]:
            for rank, item in enumerate(reps.get(t, [])[:20]):
                term = item[0] if isinstance(item, (list, tuple)) else str(item)
                weight = float(item[1]) if isinstance(item, (list, tuple)) and len(item) > 1 else 1.0 / (rank + 1)
                words[term] = words.get(term, 0.0) + weight
        top8 = [w for w, _ in sorted(words.items(), key=lambda kv: -kv[1])[:8]]
        # The model vocabulary is lower-case, while the manuscript prints acronyms
        # such as "SAR" in upper case; match case-insensitively so display
        # capitalisation cannot masquerade as a missing keyword.
        vocab = {w.lower() for w in words}
        frozen = [k.strip() for k in c["keywords"].split(";")]
        missing = [k for k in frozen if k.lower() not in vocab]
        out.append(
            {
                "Cluster": c["letter"],
                "Cluster_Name": c["name"],
                "N_Topics": len(c["members"]),
                "Members": ", ".join(f"T{t}" for t in sorted(c["members"])),
                f"Share_{BASE_YEAR}": float(sub[f"Share_{BASE_YEAR}"].sum()),
                "Top8_keywords_by_weight": "; ".join(top8),
                "Table_keywords": c["keywords"],
                "Table_keywords_missing": "; ".join(missing),
            }
        )
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run(check_only: bool = False) -> pd.DataFrame:
    config = get_config()
    model_dir = config.paths.models_dir / config.topic_modeling.output.model_name

    rows, valid, reps = load_ctfidf(model_dir)
    partition, z, dist = ward_partition(rows, valid)
    labels = np.asarray([partition[t] for t in valid])

    diag = k_diagnostics(z, dist, labels)
    print("REPORT 1 -- cluster model provenance")
    print(f"  model                : {model_dir}")
    print(f"  valid topics         : {len(valid)} (Topic {valid[0]}--{valid[-1]})")
    print(f"  space                : L2-normalized c-TF-IDF rows, Euclidean distance "
          f"(monotone transform of cosine), Ward linkage")
    print(f"  cut                  : k={K} -> {len(set(partition.values()))} clusters, "
          f"{sum(1 for _ in partition)} topics assigned")
    sil = ", ".join(f"k={k}:{v:+.4f}" for k, v in diag["silhouette"].items())
    print(f"  silhouette scan      : {sil}")
    print(f"  best silhouette at   : k={diag['silhouette_argmax_k']} "
          f"({diag['silhouette'][diag['silhouette_argmax_k']]:+.4f}) -- k=6 is NOT silhouette-selected")
    print("  largest linkage-height gaps (gap, merge, clusters after merge):")
    for gap, merge, nclu in diag["largest_gaps"]:
        print(f"      +{gap:.4f}  merge {merge:3d}  -> {nclu} clusters")

    verify_against_canonical(partition)
    print("  canonical membership : VERIFIED against the frozen definition")

    shares = yearly_topic_shares(config)
    frame = cluster_frame(config, partition, shares)
    summary = cluster_summary(frame, reps)

    print("\nREPORT 2 -- six clusters (share of assigned publications)")
    print(f"  {'':2s} {'n':>3s} {'share 2025':>11s} {'2026 H1':>9s}  name")
    total = 0.0
    for _, r in summary.iterrows():
        sub = frame[frame["Cluster"] == r["Cluster"]]
        s26 = float(sub["Share_2026_H1"].sum())
        total += r[f"Share_{BASE_YEAR}"]
        print(f"  {r['Cluster']:2s} {r['N_Topics']:3d} {r[f'Share_{BASE_YEAR}']:11.4f} {s26:9.4f}  {r['Cluster_Name']}")
    print(f"  topics assigned = {int(summary['N_Topics'].sum())} (of {len(valid)}); "
          f"shares sum to {total:.4f}")

    bad = summary[summary["Table_keywords_missing"] != ""]
    print("\nREPORT 3 -- frozen table keywords vs computed aggregate")
    if bad.empty:
        print("  all table keywords appear in the c-TF-IDF aggregate of their cluster")
    else:
        for _, r in bad.iterrows():
            print(f"  WARNING cluster {r['Cluster']}: not found -> {r['Table_keywords_missing']}")
    for _, r in summary.iterrows():
        print(f"  {r['Cluster']}: {r['Top8_keywords_by_weight']}")

    if check_only:
        return summary

    out_dir = config.paths.processed_dir
    frame.to_csv(out_dir / "topic_clusters.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out_dir / "cluster_summary.csv", index=False, encoding="utf-8-sig")
    print(f"\nwrote {out_dir / 'topic_clusters.csv'}")
    print(f"wrote {out_dir / 'cluster_summary.csv'}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify and materialise the six-cluster definition")
    parser.add_argument("--check", action="store_true", help="verify only; write nothing")
    args = parser.parse_args()

    from src.utils import setup_logging

    config = get_config()
    setup_logging(config.system.log_level, config.system.log_file)
    run(check_only=args.check)


if __name__ == "__main__":
    main()
