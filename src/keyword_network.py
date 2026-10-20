"""Step 4d: Keyword Co-occurrence Network — VOSviewer-style network from DE field.

Usage:
    python src/keyword_network.py              # uses default config.yaml
    python src/keyword_network.py --config path/to/config.yaml
"""

from __future__ import annotations

import itertools
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import plotly.graph_objects as go

from src.config import Config, get_config
from src.utils import Timer, ensure_dir

logger = logging.getLogger(__name__)


# =============================================================================
# Data loading
# =============================================================================


def load_keywords(config: Config, max_docs: int = 50000) -> List[List[str]]:
    """Load DE (author keywords) from cleaned data.

    Args:
        config: Validated configuration.
        max_docs: Limit number of documents for performance.

    Returns:
        List of keyword lists, one per document.
    """
    proc = config.paths.processed_dir
    cleaned_pq = proc / config.preprocessing.output_file.replace(".csv", ".parquet")
    if cleaned_pq.exists():
        df = pd.read_parquet(cleaned_pq)
    else:
        cleaned_csv = proc / config.preprocessing.output_file
        if not cleaned_csv.exists():
            raise FileNotFoundError(f"Cleaned data not found: {cleaned_csv}")
        df = pd.read_csv(cleaned_csv, encoding="utf-8-sig")

    if "DE" not in df.columns:
        raise ValueError("Cleaned data missing 'DE' column — cannot build keyword network.")

    docs = []
    for de in df["DE"].fillna("").astype(str).head(max_docs):
        if not de.strip():
            continue
        # Split by semicolon or comma
        if ";" in de:
            kw = [k.strip().lower() for k in de.split(";")]
        else:
            kw = [k.strip().lower() for k in de.split(",")]
        kw = [k for k in kw if len(k) > 1]
        if len(kw) >= 2:
            docs.append(kw)

    logger.info("Loaded %d documents with usable keywords", len(docs))
    return docs


# =============================================================================
# Co-occurrence matrix
# =============================================================================


def build_cooccurrence(
    docs: List[List[str]],
    min_freq: int = 5,
    max_nodes: int = 80,
) -> Tuple[Dict[str, int], Dict[Tuple[str, str], int]]:
    """Build keyword frequency and pairwise co-occurrence counts.

    Args:
        docs: List of keyword lists per document.
        min_freq: Minimum document frequency for a keyword to be included.
        max_nodes: Maximum number of nodes in the network.

    Returns:
        ``(freq_dict, cooc_dict)`` where cooc_dict maps sorted keyword pairs
        to their co-occurrence count.
    """
    freq = Counter()
    cooc = Counter()

    for kw_list in docs:
        unique = sorted(set(kw_list))
        for w in unique:
            freq[w] += 1
        for a, b in itertools.combinations(unique, 2):
            key = (a, b) if a < b else (b, a)
            cooc[key] += 1

    # Filter by min_freq
    freq = {w: c for w, c in freq.items() if c >= min_freq}
    logger.info("Keywords with freq >= %d: %d", min_freq, len(freq))

    # Keep top max_nodes by frequency
    top_words = [w for w, _ in Counter(freq).most_common(max_nodes)]
    top_set = set(top_words)

    # Filter co-occurrences to top words only
    cooc = {
        pair: cnt for pair, cnt in cooc.items()
        if pair[0] in top_set and pair[1] in top_set and cnt >= 2
    }

    freq = {w: freq[w] for w in top_words}
    logger.info("Network: %d nodes, %d edges", len(freq), len(cooc))
    return freq, cooc


# =============================================================================
# Network visualization
# =============================================================================


def plot_keyword_network(
    freq: Dict[str, int],
    cooc: Dict[Tuple[str, str], int],
    config: Config,
    output_path: Path,
    edge_threshold: int = 3,
) -> None:
    """Interactive Plotly network of keyword co-occurrences.

    Node size = document frequency. Edge width = co-occurrence count.
    Colors = community clusters detected via Louvain method (if networkx available).

    Args:
        freq: Keyword frequency dict.
        cooc: Co-occurrence count dict.
        config: Validated configuration.
        output_path: Where to save the HTML.
        edge_threshold: Minimum co-occurrence count to draw an edge.
    """
    try:
        import networkx as nx
        has_nx = True
    except ImportError:
        has_nx = False
        logger.warning("networkx not installed — skipping community detection")

    nodes = list(freq.keys())

    # Build NetworkX graph for layout + clustering
    G = nx.Graph()
    for w, c in freq.items():
        G.add_node(w, weight=c)
    for (a, b), c in cooc.items():
        if c >= edge_threshold:
            G.add_edge(a, b, weight=c)

    # Remove isolated nodes
    isolated = [n for n in G.nodes() if G.degree(n) == 0]
    G.remove_nodes_from(isolated)
    nodes = list(G.nodes())
    freq = {n: freq[n] for n in nodes}

    if len(nodes) == 0:
        logger.warning("No nodes left after filtering — skipping network plot")
        return

    # Community detection
    clusters = {}
    if has_nx:
        try:
            communities = nx.community.louvain_communities(G, seed=42)
            for cid, comm in enumerate(communities):
                for node in comm:
                    clusters[node] = cid
        except Exception:
            pass

    if not clusters:
        clusters = {n: 0 for n in nodes}

    # Layout — spring
    pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)

    # Color palette for clusters
    n_clusters = len(set(clusters.values()))
    if n_clusters <= 10:
        palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                   "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    else:
        import matplotlib.pyplot as plt
        cmap = plt.cm.tab20
        palette = [cmap(i / n_clusters) for i in range(n_clusters)]

    # Edge traces
    edge_x, edge_y, edge_width, edge_text = [], [], [], []
    for (a, b), c in cooc.items():
        if c < edge_threshold:
            continue
        if a not in pos or b not in pos:
            continue
        x0, y0 = pos[a]
        x1, y1 = pos[b]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_width.append(c)
        edge_text.append(f"{a} — {b}<br>co-occurrence: {c}")

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(
            color="rgba(150,150,150,0.35)",
            width=1,
        ),
        hoverinfo="none",
        showlegend=False,
    )

    # Node traces (one per cluster for legend)
    node_traces = []
    for cid in sorted(set(clusters.values())):
        cluster_nodes = [n for n in nodes if clusters.get(n, 0) == cid]
        x = [pos[n][0] for n in cluster_nodes]
        y = [pos[n][1] for n in cluster_nodes]
        sizes = [10 + 35 * (freq[n] / max(freq.values())) for n in cluster_nodes]
        color = palette[cid % len(palette)]

        node_traces.append(go.Scatter(
            x=x,
            y=y,
            mode="markers+text",
            text=cluster_nodes,
            textposition="top center",
            textfont=dict(size=8, color="#333"),
            marker=dict(
                size=sizes,
                color=color,
                line=dict(width=1, color="white"),
                opacity=0.9,
            ),
            name=f"Cluster {cid + 1}",
            hovertemplate="%{text}<br>freq: %{customdata}<extra></extra>",
            customdata=[freq[n] for n in cluster_nodes],
        ))

    fig = go.Figure(
        data=[edge_trace] + node_traces,
        layout=go.Layout(
            title={
                "text": "Keyword Co-occurrence Network",
                "x": 0.5,
                "xanchor": "center",
                "font": {"size": 16},
            },
            template="plotly_white",
            hovermode="closest",
            showlegend=True,
            legend={"title": "Clusters", "itemsizing": "constant"},
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=800,
            width=1000,
            margin=dict(l=20, r=20, t=60, b=20),
        ),
    )

    ensure_dir(output_path.parent)
    fig.write_html(str(output_path))
    logger.info("Keyword network plot saved: %s", output_path)


# =============================================================================
# Main entry point
# =============================================================================


def run_keyword_network(config: Config) -> None:
    """Run the complete keyword network analysis.

    Args:
        config: Validated configuration.
    """
    # 1. Load keywords
    with Timer("Loading keywords"):
        docs = load_keywords(config)

    if len(docs) == 0:
        logger.warning("No usable keyword data — skipping keyword network")
        return

    # 2. Build co-occurrence
    with Timer("Building co-occurrence matrix"):
        freq, cooc = build_cooccurrence(docs, min_freq=5, max_nodes=80)

    # 3. Plot
    with Timer("Generating keyword network"):
        html_path = config.paths.figures_dir / "keyword_network.html"
        plot_keyword_network(freq, cooc, config, html_path, edge_threshold=3)

    logger.info("Keyword network analysis complete.")


def main() -> None:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Step 4d: Keyword Co-occurrence Network")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    config = get_config(args.config)
    from src.utils import setup_logging

    setup_logging(config.system.log_level, config.system.log_file)

    try:
        run_keyword_network(config)
    except Exception:
        logger.exception("Keyword network analysis failed")
        raise


if __name__ == "__main__":
    main()
