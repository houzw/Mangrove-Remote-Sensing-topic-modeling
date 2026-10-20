"""Draw BERTopic dendrogram with cluster labels on coloured branches.

Uses scipy's native dendrogram colouring (~9 clusters, matching what users
see in the original ``topic_hierarchy.png``) and overlays concise cluster
keyword labels on the main branches.  Individual leaf labels are hidden to
keep the figure clean at publication scale.

Output: ``topic_hierarchy_labeled.png`` — the original hierarchy is NOT touched.

Usage:
    python src/meta_topic_dendrogram.py              # default: 9 clusters
    python src/meta_topic_dendrogram.py --n-clusters 10
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Distance matrix
# ---------------------------------------------------------------------------


def _build_condensed_dist(model, valid_topics: List[int]) -> np.ndarray:
    """Cosine distance (condensed) from c-TF-IDF topic vectors."""
    ctfidf = model.c_tf_idf_.toarray() if hasattr(model.c_tf_idf_, "toarray") else model.c_tf_idf_
    all_keys = list(model.topic_representations_.keys())
    tid_to_row = {tid: all_keys.index(tid) for tid in valid_topics}
    topic_rows = np.array([ctfidf[tid_to_row[tid]] for tid in valid_topics])

    cos_sim = cosine_similarity(topic_rows)
    cos_dist = 1.0 - cos_sim
    np.fill_diagonal(cos_dist, 0.0)
    return squareform(cos_dist, checks=False)


# ---------------------------------------------------------------------------
# Cluster assignment via distance threshold
# ---------------------------------------------------------------------------


def _assign_clusters_by_threshold(
    Z: np.ndarray,
    valid_topics: List[int],
    n_clusters: int,
) -> Tuple[Dict[int, int], float]:
    """Find a distance threshold that yields *n_clusters* and return assignments.

    Returns ``(topic_id → cluster_id, threshold)``.
    """
    max_d = max(Z[:, 2])
    # Binary search for the threshold
    lo, hi = 0.0, max_d
    for _ in range(60):
        mid = (lo + hi) / 2.0
        clusters = fcluster(Z, t=mid, criterion="distance")
        n = len(set(clusters))
        if n == n_clusters:
            break
        elif n < n_clusters:
            hi = mid
        else:
            lo = mid
    else:
        # Didn't hit exactly; pick the closest
        mid = (lo + hi) / 2.0
        clusters = fcluster(Z, t=mid, criterion="distance")
        logger.info("Using threshold %.4f → %d clusters (target %d)", mid, len(set(clusters)), n_clusters)

    mapping: Dict[int, int] = {}
    for i, tid in enumerate(valid_topics):
        mapping[tid] = int(clusters[i])
    return mapping, mid


# ---------------------------------------------------------------------------
# Cluster labels
# ---------------------------------------------------------------------------


def _build_cluster_labels(
    model,
    topic_to_cluster: Dict[int, int],
    top_n_words: int = 6,
) -> Dict[int, str]:
    """Aggregate top words from member topics to label each cluster."""
    members: Dict[int, List[int]] = defaultdict(list)
    for tid, cid in topic_to_cluster.items():
        members[cid].append(tid)

    labels: Dict[int, str] = {}
    for cid, tids in sorted(members.items()):
        words: List[str] = []
        for tid in tids:
            if tid in model.topic_representations_:
                words.extend(w for w, _ in model.topic_representations_[tid][:top_n_words])
        top4 = [w for w, _ in Counter(words).most_common(4)]
        labels[cid] = " · ".join(top4)
    return labels


# ---------------------------------------------------------------------------
# Main drawing function
# ---------------------------------------------------------------------------


def draw_labeled_dendrogram(
    model,
    output_path: Path,
    n_clusters: int = 9,
    dpi: int = 300,
    figsize: Tuple[float, float] = (28, 16),
) -> Path:
    """Draw a clean dendrogram with cluster labels on coloured branches.

    - Uses scipy's native dendrogram colouring cut to *n_clusters*.
    - Hides individual topic leaf labels (too dense at 72 topics).
    - Places one keyword label per cluster on its main branch.
    - Adds a right-side legend with full cluster descriptions.
    """
    import matplotlib.pyplot as plt

    # ---- 1. Data -----------------------------------------------------------
    valid_topics = sorted([t for t in model.topic_representations_.keys() if t != -1])
    n = len(valid_topics)
    condensed = _build_condensed_dist(model, valid_topics)
    Z = linkage(condensed, method="ward", optimal_ordering=True)
    max_dist = max(Z[:, 2])

    # ---- 2. Cluster assignment ---------------------------------------------
    topic_to_cluster, threshold = _assign_clusters_by_threshold(Z, valid_topics, n_clusters)
    cluster_labels = _build_cluster_labels(model, topic_to_cluster)

    cluster_ids = sorted(set(topic_to_cluster.values()), key=lambda c: -sum(1 for v in topic_to_cluster.values() if v == c))

    # Colour map for clusters
    if len(cluster_ids) <= 10:
        base_cmap = plt.cm.tab10
    elif len(cluster_ids) <= 20:
        base_cmap = plt.cm.tab20
    else:
        base_cmap = plt.cm.gist_ncar
    cluster_color = {cid: base_cmap(i / max(len(cluster_ids), 1)) for i, cid in enumerate(cluster_ids)}

    # ---- 3. Dendrogram (no_plot) to get structure --------------------------
    ddata = dendrogram(
        Z,
        no_plot=True,
        color_threshold=threshold,
        above_threshold_color="#888888",
    )

    leaves_list = ddata["leaves"]       # leaf ordering → indices into valid_topics
    icoord = ddata["icoord"]
    dcoord = ddata["dcoord"]
    ivl = ddata["ivl"]                  # leaf labels in dendrogram order

    # scipy's color_list tells us the color of each link segment below the threshold
    color_list = ddata.get("color_list", [])
    leaves_color_list = ddata.get("leaves_color_list", [])

    # ---- 4. Build descendant leaf sets for each internal node ---------------
    descendant_leaves: Dict[int, Set[int]] = {i: {i} for i in range(n)}
    for i, row in enumerate(Z):
        node_id = n + i
        left = int(row[0])
        right = int(row[1])
        descendant_leaves[node_id] = descendant_leaves[left] | descendant_leaves[right]

    # ---- 5. Map each link segment → dominant cluster -----------------------
    # icoord/dcoord[k] corresponds to internal node n+k
    link_cluster: Dict[int, int] = {}
    for k in range(len(icoord)):
        node_id = n + k
        desc = descendant_leaves[node_id]
        clusters_under = {topic_to_cluster[valid_topics[leaf_idx]] for leaf_idx in desc}
        if len(clusters_under) == 1:
            link_cluster[k] = clusters_under.pop()

    # ---- 6. Draw -----------------------------------------------------------
    fig = plt.figure(figsize=figsize)

    # Main dendrogram axis
    gs = fig.add_gridspec(1, 2, width_ratios=[3.5, 1], wspace=0.02)
    ax = fig.add_subplot(gs[0, 0])

    # --- 6a. Background: light grey for everything ---
    for k in range(len(icoord)):
        ax.plot(icoord[k], dcoord[k], color="#d5d5d5", linewidth=1.2, zorder=1)

    # --- 6b. Foreground: coloured links for cluster-pure branches ---
    for k in range(len(icoord)):
        cid = link_cluster.get(k)
        if cid is not None:
            color = cluster_color[cid]
            ax.plot(icoord[k], dcoord[k], color=color, linewidth=2.8, alpha=0.9,
                    zorder=2, solid_capstyle="round")

    # --- 6c. Leaf ticks (tiny, no text labels) ---
    for leaf_pos in range(n):
        ax.plot(leaf_pos, 0, marker="|", color="#999999", markersize=3, zorder=1)

    # --- 6d. Cluster labels on branches -------------------------------------
    # For each cluster, find the LCA node and place a label there
    leaf_pos_of_idx: Dict[int, int] = {idx: pos for pos, idx in enumerate(leaves_list)}

    annotations: List[Tuple[float, float, str, int]] = []  # (x, y_h, label, cid)

    for cid in cluster_ids:
        member_leaves = {
            leaf_idx
            for tid, c in topic_to_cluster.items()
            if c == cid
            for leaf_idx, vtid in enumerate(valid_topics)
            if vtid == tid
        }
        if len(member_leaves) < 1:
            continue

        # Find LCA
        lca_height = 0.0
        for i, row in enumerate(Z):
            node_id = n + i
            if member_leaves.issubset(descendant_leaves[node_id]):
                lca_height = row[2]
                break

        positions = sorted(leaf_pos_of_idx[lf] for lf in member_leaves)
        x_mid = (positions[0] + positions[-1]) / 2.0
        annotations.append((x_mid, lca_height, cid))

    # Sort by height (low to high) so higher labels don't overlap lower ones
    annotations.sort(key=lambda a: a[1])

    # De-duplicate overlapping x-positions
    placed_labels: List[Tuple[float, float, str, int]] = []
    for x_mid, y_h, cid in annotations:
        label = cluster_labels.get(cid, f"C{cid}")
        # Shift up if overlaps with a nearby label
        final_y = y_h + max_dist * 0.025
        for px, py, _, _ in placed_labels:
            if abs(x_mid - px) < 5.0:
                if final_y < py + max_dist * 0.055:
                    final_y = py + max_dist * 0.055
        placed_labels.append((x_mid, final_y, label, cid))

    for x_mid, label_y, label, cid in placed_labels:
        color = cluster_color[cid]
        # Dashed leader line
        ax.plot([x_mid, x_mid], [label_y - max_dist * 0.02, label_y],
                color=color, linewidth=1.2, alpha=0.6, linestyle="--", zorder=3)
        # Label box
        ax.text(
            x_mid, label_y, label,
            ha="center", va="bottom",
            fontsize=8.5, fontweight="bold",
            color="white",
            bbox=dict(
                boxstyle="round,pad=0.35",
                facecolor=color,
                edgecolor="none",
                alpha=0.85,
            ),
            zorder=5,
        )

    # --- 6e. Axis styling ---------------------------------------------------
    ax.set_ylabel("Cosine Distance", fontsize=13, fontweight="bold")
    ax.set_title(
        f"BERTopic Topic Hierarchy — {len(valid_topics)} Topics × {len(cluster_ids)} Clusters\n"
        f"(Ward linkage on c‑TF‑IDF centroid distances, cut at {threshold:.3f})",
        fontsize=14, fontweight="bold", pad=14,
    )
    ax.set_xticks([])
    ax.set_xlim(-2, n + 1)
    ax.set_ylim(-max_dist * 0.04, max_dist * 1.22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="y", labelsize=9)
    ax.yaxis.grid(True, linestyle=":", alpha=0.3)

    # --- 6f. Right-side legend panel ----------------------------------------
    ax_legend = fig.add_subplot(gs[0, 1])
    ax_legend.axis("off")

    topic_info = model.get_topic_info()

    y_pos = 0.98
    for cid in cluster_ids:
        color = cluster_color[cid]
        member_tids = sorted([tid for tid, c in topic_to_cluster.items() if c == cid])
        n_docs = sum(
            int(topic_info[topic_info["Topic"] == tid]["Count"].values[0])
            for tid in member_tids
            if tid in topic_info["Topic"].values
        )
        label = cluster_labels.get(cid, f"Cluster {cid}")
        # Truncate label for legend
        words = label.split(" · ")
        line1 = " · ".join(words[:2])
        line2 = " · ".join(words[2:4]) if len(words) > 2 else ""

        ax_legend.text(
            0.02, y_pos,
            f"C{cid}",
            fontsize=9.5, fontweight="bold", color=color,
            va="top", transform=ax_legend.transAxes,
        )
        ax_legend.text(
            0.18, y_pos,
            f"{line1}",
            fontsize=8.2, color="#333333",
            va="top", transform=ax_legend.transAxes,
        )
        if line2:
            ax_legend.text(
                0.18, y_pos - 0.038,
                f"{line2}",
                fontsize=8.2, color="#666666",
                va="top", transform=ax_legend.transAxes,
            )
        ax_legend.text(
            0.18, y_pos - (0.048 if line2 else 0.028),
            f"{len(member_tids)} topics · {n_docs} docs",
            fontsize=7.2, color="#999999",
            va="top", transform=ax_legend.transAxes,
        )
        y_pos -= 0.10

    ax_legend.set_title("Cluster Keywords", fontsize=11, fontweight="bold", pad=8)

    # ---- 7. Save -----------------------------------------------------------
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Labeled dendrogram saved: %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="BERTopic dendrogram with cluster labels")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--n-clusters", type=int, default=9,
                        help="Number of colour clusters to cut (default: 9)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path (default: outputs/figures/topic_hierarchy_labeled.png)")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        from src.config import get_config
        config = get_config(args.config)
        default_out = config.paths.figures_dir / "topic_hierarchy_labeled.png"
    except Exception:
        default_out = Path("outputs/figures/topic_hierarchy_labeled.png")

    output = Path(args.output) if args.output else default_out

    from bertopic import BERTopic

    model_path = Path("outputs/models/bertopic_model")
    if not model_path.exists():
        logger.error("Model not found at %s", model_path)
        sys.exit(1)

    logger.info("Loading model ...")
    model = BERTopic.load(str(model_path))

    draw_labeled_dendrogram(model, output_path=output, n_clusters=args.n_clusters, dpi=args.dpi)


if __name__ == "__main__":
    main()
