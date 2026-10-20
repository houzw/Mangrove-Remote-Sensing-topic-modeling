"""Six-cluster dendrogram of the 72 BERTopic topics.

Draws the same tree that defines the manuscript's cluster taxonomy
(``src.topic_clusters``: Ward's minimum-variance criterion on the Euclidean
distances between $L_2$-normalized c-TF-IDF topic vectors), colours the branches
by the six frozen clusters, marks the $k=6$ cut, and labels every leaf with its
topic id and final label.

Replaces ``meta_topic_dendrogram.py`` as the manuscript figure source: that
script finds its own cut and its leaf labels collapse at 72 topics.

Usage:
    pixi run python -m src.cluster_dendrogram
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

from src.config import get_config
from src.topic_clusters import CLUSTERS, K, load_ctfidf, topic_distance_matrix

logger = logging.getLogger(__name__)

PALETTE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00"]
ABOVE_CUT = "#9E9E9E"
CUT_DISTANCE = 1.50  # inside the (1.459, 1.638) interval that yields k = 6


def _leaf_labels(config) -> dict[int, str]:
    """Leaf text: topic id plus the final (author-verified) topic label."""
    info = pd.read_csv(config.paths.processed_dir / "topics_info.csv")
    labels: dict[int, str] = {}
    for _, r in info.iterrows():
        tid = int(r["Topic"])
        if tid == -1:
            continue
        short = r.get("Short_Label")
        if pd.isna(short) or not str(short).strip():
            short = r.get("Name", f"topic {tid}")
        labels[tid] = f"T{tid} {str(short).strip()}"
    return labels


def build_tree(config):
    model_dir = config.paths.models_dir / config.topic_modeling.output.model_name
    rows, valid, _ = load_ctfidf(model_dir)
    dist = topic_distance_matrix(rows)
    z = linkage(squareform(dist, checks=False), method="ward", optimal_ordering=True)
    return z, valid


def plot_cluster_dendrogram(config, output_path: Path) -> Path:
    z, valid = build_tree(config)
    labels_map = _leaf_labels(config)
    labels = [labels_map.get(t, f"T{t}") for t in valid]

    cluster_of = {t: i for i, c in enumerate(CLUSTERS) for t in c["members"]}
    leaf_cluster = np.array([cluster_of[t] for t in valid])

    n = len(valid)
    # For each merge node (index n..2n-2) collect its descendant leaves so the
    # branch can be coloured by the frozen cluster it belongs to.
    members: dict[int, set[int]] = {i: {i} for i in range(n)}
    node_colour: dict[int, str] = {}
    for row in range(z.shape[0]):
        a, b = int(z[row, 0]), int(z[row, 1])
        node = n + row
        group = members.pop(a) | members.pop(b)
        members[node] = group
        clusters_in_node = {int(leaf_cluster[i]) for i in group}
        node_colour[node] = PALETTE[next(iter(clusters_in_node))] if len(clusters_in_node) == 1 else ABOVE_CUT

    def link_colour(node_id: int) -> str:
        return node_colour.get(int(node_id), ABOVE_CUT)

    plt.rcParams["font.family"] = config.visualization.font_family
    fig, ax = plt.subplots(figsize=(8.6, 13.0), dpi=300)
    dendrogram(
        z,
        orientation="right",
        labels=labels,
        leaf_font_size=4.6,
        color_threshold=0,
        link_color_func=link_colour,
        ax=ax,
    )

    ax.axvline(CUT_DISTANCE, color="#333333", linestyle="--", linewidth=0.9, alpha=0.85)
    ax.annotate(
        f"cut: $k$ = {K}",
        xy=(CUT_DISTANCE, 1.0), xycoords=("data", "axes fraction"),
        xytext=(0, 3), textcoords="offset points",
        ha="center", va="bottom", fontsize=7.5, color="#333333",
    )
    ax.set_xlabel("Euclidean distance between $L_2$-normalized c-TF-IDF vectors "
                  "(Ward linkage)", fontsize=9)
    ax.set_ylabel("")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    handles = [
        plt.Line2D([0], [0], color=PALETTE[i], lw=3,
                   label=f"{c['letter']}: {c['name']} ({len(c['members'])} topics)")
        for i, c in enumerate(CLUSTERS)
    ]
    ax.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.045), ncol=2,
        fontsize=7.5, framealpha=0.92, edgecolor="#CCCCCC",
        title=f"Six clusters, {n} topics", title_fontsize=8,
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    logger.info("Six-cluster dendrogram saved: %s", output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Six-cluster dendrogram of the 72 topics")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    from src.utils import setup_logging

    config = get_config(args.config)
    setup_logging(config.system.log_level, config.system.log_file)

    out = Path(args.output) if args.output else (
        config.paths.figures_dir / "cluster_dendrogram.png"
    )
    z, valid = build_tree(config)
    print(f"leaves={len(valid)}  merges={z.shape[0]}  "
          f"cut at {CUT_DISTANCE} -> {K} clusters")
    plot_cluster_dendrogram(config, out)


if __name__ == "__main__":
    main()
