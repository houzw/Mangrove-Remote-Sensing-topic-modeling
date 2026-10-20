"""Step 4c: Topic Similarity Network — network graph of inter-topic c-TF-IDF similarity.

Usage:
    python src/topic_network.py              # uses default config.yaml
    python src/topic_network.py --config path/to/config.yaml
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.config import Config, get_config
from src.utils import Timer, ensure_dir

logger = logging.getLogger(__name__)


# =============================================================================
# Similarity computation
# =============================================================================


def load_model(config: Config) -> "BERTopic":  # noqa: F821
    """Load the fitted BERTopic model."""
    from bertopic import BERTopic

    model_path = config.paths.models_dir / config.topic_modeling.output.model_name
    if not model_path.exists():
        raise FileNotFoundError(f"BERTopic model not found: {model_path}. Run topic modeling first.")

    with Timer("Loading BERTopic model"):
        model = BERTopic.load(model_path)
    return model


def compute_topic_similarity(
    model: "BERTopic",  # noqa: F821
    metric: str = "cosine",
    exclude_outlier: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute pairwise similarity between topics based on c-TF-IDF vectors.

    Args:
        model: Fitted BERTopic model.
        metric: Similarity metric (only ``"cosine"`` supported currently).
        exclude_outlier: If True, drop topic -1 from the matrix.

    Returns:
        ``(topic_info_df, similarity_matrix_df)`` where the matrix index/columns
        are topic IDs.
    """
    from sklearn.metrics.pairwise import cosine_similarity

    topic_info = model.get_topic_info()
    c_tf_idf = model.c_tf_idf_

    if c_tf_idf is None:
        raise ValueError("Model has no c-TF-IDF matrix — cannot compute topic similarity.")

    # Convert to dense if needed
    if hasattr(c_tf_idf, "toarray"):
        c_tf_idf = c_tf_idf.toarray()

    # BERTopic internal order: topic_info rows correspond to c_tf_idf rows
    # Topic -1 is usually the first row; we keep the mapping via topic_info["Topic"]
    topic_ids = topic_info["Topic"].tolist()

    sim_matrix = cosine_similarity(c_tf_idf)
    sim_df = pd.DataFrame(sim_matrix, index=topic_ids, columns=topic_ids)

    if exclude_outlier and -1 in sim_df.index:
        sim_df = sim_df.drop(index=-1, columns=-1)
        topic_info = topic_info[topic_info["Topic"] != -1].copy()

    logger.info("Computed %dx%d topic similarity matrix (%s)", len(sim_df), len(sim_df), metric)
    return topic_info, sim_df


def build_edge_list(
    sim_df: pd.DataFrame,
    topic_info: pd.DataFrame,
    threshold: float = 0.15,
    top_n: int = 20,
) -> pd.DataFrame:
    """Build an edge list from the similarity matrix.

    Args:
        sim_df: Topic × topic similarity DataFrame.
        topic_info: Topic info DataFrame (for names and counts).
        threshold: Minimum similarity for an edge.
        top_n: Keep only the top-N topics by document count.

    Returns:
        Edge list DataFrame with columns ``source, target, similarity``.
    """
    # Filter to top-N topics
    top_topics = topic_info[topic_info["Topic"] != -1].nlargest(top_n, "Count")["Topic"].tolist()
    sim_df = sim_df.loc[top_topics, top_topics]

    records = []
    for i, src in enumerate(sim_df.index):
        for j, tgt in enumerate(sim_df.columns):
            if i < j:  # upper triangle only
                sim = sim_df.loc[src, tgt]
                if sim >= threshold:
                    records.append({"source": int(src), "target": int(tgt), "similarity": float(sim)})

    edges = pd.DataFrame(records)
    if not edges.empty:
        edges = edges.sort_values("similarity", ascending=False).reset_index(drop=True)

    logger.info("Edge list: %d edges (threshold=%.2f, top_n=%d)", len(edges), threshold, top_n)
    return edges


# =============================================================================
# Network visualization
# =============================================================================


def plot_topic_network(
    edges: pd.DataFrame,
    topic_info: pd.DataFrame,
    sim_df: pd.DataFrame,
    config: Config,
    output_path: Path,
) -> None:
    """Interactive force-directed network of topic similarities (Plotly).

    Args:
        edges: Edge list DataFrame.
        topic_info: Topic info for node metadata.
        sim_df: Similarity matrix for reference.
        config: Validated configuration.
        output_path: Where to save the HTML.
    """
    if edges.empty:
        logger.warning("No edges above threshold — skipping network plot")
        return

    # Node list
    nodes = sorted(set(edges["source"].tolist() + edges["target"].tolist()))
    n_nodes = len(nodes)

    # Layout: simple circular layout with refinement via spring forces
    angles = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
    pos = {node: (np.cos(a), np.sin(a)) for node, a in zip(nodes, angles)}

    # Refine with a few iterations of spring force (simplified)
    for _ in range(50):
        for node in nodes:
            fx, fy = 0.0, 0.0
            for other in nodes:
                if node == other:
                    continue
                dx = pos[node][0] - pos[other][0]
                dy = pos[node][1] - pos[other][1]
                dist = np.hypot(dx, dy) + 1e-6
                # Repulsion
                force = 0.01 / (dist ** 2)
                fx += force * dx / dist
                fy += force * dy / dist

            # Attraction from edges
            for _, edge in edges.iterrows():
                if node in (edge["source"], edge["target"]):
                    other = edge["target"] if node == edge["source"] else edge["source"]
                    dx = pos[other][0] - pos[node][0]
                    dy = pos[other][1] - pos[node][1]
                    dist = np.hypot(dx, dy) + 1e-6
                    force = 0.02 * edge["similarity"] * dist
                    fx += force * dx / dist
                    fy += force * dy / dist

            x, y = pos[node]
            pos[node] = (x + fx, y + fy)

    # Build node info
    node_info = topic_info.set_index("Topic").loc[nodes]
    node_labels = [f"Topic {tid}: {_truncate_name(node_info.loc[tid, 'Name'])}" for tid in nodes]
    node_sizes = node_info["Count"].values
    node_sizes = 15 + 40 * (node_sizes / node_sizes.max()) if node_sizes.max() > 0 else np.full_like(node_sizes, 20)

    # Edge traces
    edge_x, edge_y = [], []
    edge_text = []
    for _, row in edges.iterrows():
        x0, y0 = pos[row["source"]]
        x1, y1 = pos[row["target"]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_text.append(f"Similarity: {row['similarity']:.3f}")

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(color="rgba(150,150,150,0.4)", width=1.5),
        hoverinfo="none",
        showlegend=False,
    )

    # Node trace
    node_x = [pos[n][0] for n in nodes]
    node_y = [pos[n][1] for n in nodes]
    colors = px.colors.qualitative.Bold[:n_nodes] if n_nodes <= 10 else px.colors.sample_colorscale("viridis", n_nodes)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=[str(n) for n in nodes],
        textposition="top center",
        textfont=dict(size=10, color="#333"),
        marker=dict(
            size=node_sizes,
            color=colors[:n_nodes],
            line=dict(width=1, color="white"),
        ),
        hovertemplate="%{text}: %{customdata}<extra></extra>",
        customdata=node_labels,
        showlegend=False,
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title={
                "text": "Topic Similarity Network",
                "x": 0.5,
                "xanchor": "center",
                "font": {"size": 16},
            },
            template="plotly_white",
            hovermode="closest",
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=700,
            width=900,
            margin=dict(l=20, r=20, t=60, b=20),
        ),
    )

    ensure_dir(output_path.parent)
    fig.write_html(str(output_path))
    logger.info("Topic network plot saved: %s", output_path)


def _truncate_name(name: str, max_words: int = 4) -> str:
    """Shorten a BERTopic name for display."""
    if not isinstance(name, str):
        name = str(name)
    if "_" in name:
        first_sep = name.index("_")
        maybe_id = name[:first_sep]
        if maybe_id.lstrip("-").isdigit():
            name = name[first_sep + 1:]
    name = name.replace("_", " ")
    words = name.split()
    truncated = " ".join(words[:max_words])
    if len(words) > max_words:
        truncated += "..."
    return truncated


# =============================================================================
# Main entry point
# =============================================================================


def run_topic_network(config: Config) -> None:
    """Run the complete topic network analysis.

    Args:
        config: Validated configuration.
    """
    tn_cfg = config.topic_network

    # 1. Load model
    with Timer("Loading model"):
        model = load_model(config)

    # 2. Compute similarity
    with Timer("Computing topic similarities"):
        topic_info, sim_df = compute_topic_similarity(
            model,
            metric=tn_cfg.similarity_metric,
            exclude_outlier=True,
        )

    # 3. Build edge list
    with Timer("Building edge list"):
        edges = build_edge_list(
            sim_df,
            topic_info,
            threshold=tn_cfg.edge_threshold,
            top_n=tn_cfg.top_n_topics,
        )

    # 4. Save edge CSV
    csv_path = config.paths.processed_dir / tn_cfg.output.network_csv
    ensure_dir(csv_path.parent)
    edges.to_csv(csv_path, index=False, encoding="utf-8-sig")
    logger.info("Network edges saved: %s", csv_path)

    # 5. Plot network
    with Timer("Generating network plot"):
        html_path = config.paths.figures_dir / tn_cfg.output.network_html
        plot_topic_network(edges, topic_info, sim_df, config, html_path)

    logger.info("Topic network analysis complete.")


def main() -> None:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Step 4c: Topic Similarity Network")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    config = get_config(args.config)
    from src.utils import setup_logging

    setup_logging(config.system.log_level, config.system.log_file)

    try:
        run_topic_network(config)
    except Exception:
        logger.exception("Topic network analysis failed")
        raise


if __name__ == "__main__":
    main()
