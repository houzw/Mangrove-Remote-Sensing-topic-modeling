"""Step 5c: Topic Evolution Sankey Diagram — flow-based topic evolution visualization.

Shows how topic "mass" flows across consecutive time periods using a Sankey
(alluvial) diagram. Link widths represent the similarity-weighted proportion
of each topic that persists or transforms into other topics over time.

Algorithm
---------
1. Load topic proportions per period (from topic_evolution_proportions.csv).
2. Load c-TF-IDF cosine similarity matrix from the fitted BERTopic model.
3. For each adjacent period pair, distribute each source topic's proportion
   to target-period topics via temperature-scaled softmax over similarities.
4. Build a Plotly ``go.Sankey`` with columns = time periods, nodes = topics,
   and links = similarity-weighted flow.

Usage:
    python src/topic_sankey.py              # uses default config.yaml
    python src/topic_sankey.py --config path/to/config.yaml
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.config import Config, get_config
from src.utils import Timer, ensure_dir

logger = logging.getLogger(__name__)


# =============================================================================
# Categorical palette — fixed per-topic colour slots (dataviz reference)
# =============================================================================

TOPIC_COLOURS_HEX = [
    "#2a78d6",  #  0  blue
    "#eb6834",  #  1  orange
    "#1baf7a",  #  2  aqua
    "#eda100",  #  3  yellow
    "#e87ba4",  #  4  magenta
    "#008300",  #  5  green
    "#4a3aa7",  #  6  violet
    "#b5541f",  #  7  brown
    "#5f9edb",  #  8  light-blue
    "#cc5e4b",  #  9  coral
    "#3ba88c",  # 10  teal
    "#c9a02b",  # 11  gold
    "#cf5e8a",  # 12  rose
    "#6ba86b",  # 13  sage
    "#7b6fb5",  # 14  lavender
    "#8c564b",  # 15  maroon
]


def _topic_colour(topic_id: int) -> str:
    """Fixed colour for a topic ID — identity, never rank."""
    return TOPIC_COLOURS_HEX[topic_id % len(TOPIC_COLOURS_HEX)]


def _hex_to_rgba(hex_c: str, alpha: float = 0.45) -> str:
    """``#rrggbb`` → ``rgba(r,g,b,alpha)``."""
    h = hex_c.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# =============================================================================
# Data loading
# =============================================================================


def load_topic_proportions(config: Config) -> pd.DataFrame:
    """Load the topic × period proportion matrix.

    Returns:
        DataFrame with topic IDs as row index and period labels as columns.
    """
    csv_path = config.paths.processed_dir / "topic_evolution_proportions.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Topic proportions not found: {csv_path}. "
            "Run topic evolution (--step evolution) first."
        )
    prop_df = pd.read_csv(csv_path, encoding="utf-8-sig", index_col=0)
    # Ensure integer topic IDs
    prop_df.index = prop_df.index.astype(int)
    logger.info(
        "Loaded topic proportions: %d topics × %d periods",
        len(prop_df), len(prop_df.columns),
    )
    return prop_df


def load_topic_labels(config: Config) -> Dict[int, str]:
    """Load compact topic labels from *topics_info.csv*.

    Prefers ``Short_Label``; falls back to a cleaned ``Name`` field.
    Returns ``{topic_id: label}`` (outlier -1 excluded).
    """
    tinfo_csv = config.paths.processed_dir / config.topic_modeling.output.topics_info
    labels: Dict[int, str] = {}
    if not tinfo_csv.exists():
        logger.warning("topics_info.csv not found — using numeric labels only")
        return labels

    tinfo = pd.read_csv(tinfo_csv, encoding="utf-8-sig")
    for _, row in tinfo.iterrows():
        tid = int(row.get("Topic", -1))
        if tid == -1:
            continue
        short = row.get("Short_Label", None)
        if pd.notna(short) and isinstance(short, str) and short.strip():
            labels[tid] = short.strip()
        else:
            name = str(row.get("Name", f"Topic {tid}"))
            # Strip numeric prefix "42_topic_name" → "topic name"
            if "_" in name:
                maybe_prefix = name.split("_", 1)[0]
                if maybe_prefix.lstrip("-").isdigit():
                    name = name.split("_", 1)[1]
            name = name.replace("_", " ")
            labels[tid] = name[:50]
    logger.info("Loaded %d topic labels", len(labels))
    return labels


def load_similarity_matrix(config: Config) -> pd.DataFrame:
    """Load or compute the topic × topic c-TF-IDF cosine similarity matrix.

    Returns:
        Square DataFrame, index/columns = topic IDs (excluding -1).
    """
    # Read the saved c-TF-IDF state rather than loading BERTopic: the row order is the
    # same (see src.topic_clusters.load_ctfidf) and no SciBERT stack is needed for a
    # matrix that is already on disk.
    from src.topic_clusters import load_ctfidf

    model_path = config.paths.models_dir / config.topic_modeling.output.model_name
    if not model_path.exists():
        raise FileNotFoundError(
            f"BERTopic model not found: {model_path}. Run topic modeling first."
        )

    with Timer("Reading c-TF-IDF state for similarity matrix"):
        rows, topic_ids, _ = load_ctfidf(model_path)

    dense = np.asarray(rows)
    unit = dense / np.linalg.norm(dense, axis=1, keepdims=True)
    sim = np.clip(unit @ unit.T, -1.0, 1.0)
    sim_df = pd.DataFrame(sim, index=topic_ids, columns=topic_ids)

    if -1 in sim_df.index:
        sim_df = sim_df.drop(index=-1, columns=-1)

    logger.info("Similarity matrix: %d × %d topics", len(sim_df), len(sim_df))
    return sim_df


# =============================================================================
# Flow computation
# =============================================================================


def _softmax(x: np.ndarray, temperature: float = 0.3) -> np.ndarray:
    """Temperature-scaled softmax over a 1-D array.

    Lower *temperature* → sharper distribution (flows concentrate on the
    single most-similar topic).  Higher → flatter (more mixing).
    """
    x = np.asarray(x, dtype=float)
    x = np.clip(x - np.max(x), -500, 500)  # guard against overflow
    exp_x = np.exp(x / temperature)
    s = exp_x.sum()
    return exp_x / s if s > 0 else np.ones_like(exp_x) / len(exp_x)


def compute_sankey_flows(
    prop_df: pd.DataFrame,
    sim_df: pd.DataFrame,
    topic_labels: Dict[int, str],
    *,
    temperature: float = 0.3,
    min_flow: float = 0.002,
    top_n_topics: int = 12,
) -> dict:
    """Compute nodes & links for the Sankey diagram.

    Parameters
    ----------
    prop_df:
        Topic × Period proportion matrix (index = int topic ID).
    sim_df:
        Topic × Topic c-TF-IDF cosine similarity DataFrame.
    topic_labels:
        ``{topic_id: short_label}`` map (for rich hover text).
    temperature:
        Softmax temperature.  0.2–0.4 works well; lower = sharper persistence.
    min_flow:
        Drop links whose value is below this threshold (de-clutters).
    top_n_topics:
        Limit to the *top_n_topics* topics ranked by mean proportion across
        all periods.

    Returns
    -------
    dict with keys:
        - **node_label** (*List[str]*): display label for each node
        - **node_x** (*List[float]*): column coordinate (0→1)
        - **node_y** (*List[float]*): vertical coordinate (computed)
        - **node_color** (*List[str]*): fill colour per node
        - **node_customdata** (*List[str]*): rich hover text per node
        - **link_source** (*List[int]*): source node indices
        - **link_target** (*List[int]*): target node indices
        - **link_value** (*List[float]*): flow magnitudes
        - **link_color** (*List[str]*): rgba colour per link
        - **link_customdata** (*List[str]*): hover text per link
        - **period_labels** (*List[str]*): column headers
        - **n_periods** (*int*): number of time periods
    """
    periods = list(prop_df.columns)
    n_periods = len(periods)

    # --- restrict to top-N topics by mean proportion --------------------------
    prop_mean = prop_df.mean(axis=1).sort_values(ascending=False)
    selected_topics = set(prop_mean.head(top_n_topics).index.tolist())

    # --- determine which topics appear in each period (above a tiny floor) ----
    period_topics: Dict[str, List[int]] = {}
    for period in periods:
        present = prop_df[period]
        tids = sorted(
            tid for tid in present[present > 0.001].index
            if tid in selected_topics
        )
        period_topics[period] = tids
        logger.info(
            "  %s: %d topics present", period, len(tids),
        )

    # --- compute total flow per node (for vertical ordering) ------------------
    # Pre-compute all links so we know each node's total throughput.
    raw_links: List[Tuple[int, int, float, int, int]] = []
    # (src_node_id, tgt_node_id, flow, period_idx, src_topic_id)

    for pi in range(n_periods - 1):
        p_src = periods[pi]
        p_tgt = periods[pi + 1]
        src_topics = period_topics[p_src]
        tgt_topics = period_topics[p_tgt]

        for src_tid in src_topics:
            supply = float(prop_df.loc[src_tid, p_src])
            if supply < 1e-6:
                continue

            # similarities to all target topics
            sims = np.array([
                float(sim_df.loc[src_tid, tgt_tid])
                if src_tid in sim_df.index and tgt_tid in sim_df.columns
                else 0.0
                for tgt_tid in tgt_topics
            ])
            if sims.sum() < 1e-8:
                continue

            dist = _softmax(sims, temperature=temperature)
            for tgt_tid, weight in zip(tgt_topics, dist):
                flow = supply * float(weight)
                if flow < min_flow:
                    continue
                raw_links.append((src_tid, tgt_tid, flow, pi))

    # --- assign node indices --------------------------------------------------
    # One node per (topic_id, period) pair that actually carries flow.
    node_set: Dict[Tuple[int, int], int] = {}  # (period_idx, topic_id) → node_idx
    node_topic: List[int] = []                  # topic_id per node
    node_period: List[int] = []                 # period_idx per node

    def _get_node(pi: int, tid: int) -> int:
        key = (pi, tid)
        if key not in node_set:
            node_set[key] = len(node_topic)
            node_topic.append(tid)
            node_period.append(pi)
        return node_set[key]

    links_src: List[int] = []
    links_tgt: List[int] = []
    links_val: List[float] = []
    links_topic: List[int] = []  # source topic — drives link colour

    for src_tid, tgt_tid, flow, pi in raw_links:
        src_idx = _get_node(pi, src_tid)
        tgt_idx = _get_node(pi + 1, tgt_tid)
        links_src.append(src_idx)
        links_tgt.append(tgt_idx)
        links_val.append(flow)
        links_topic.append(src_tid)

    n_nodes = len(node_topic)

    # --- compute node throughput for vertical ordering ------------------------
    node_flow = np.zeros(n_nodes, dtype=float)
    for s, v in zip(links_src, links_val):
        node_flow[s] += v
    for t, v in zip(links_tgt, links_val):
        node_flow[t] += v

    # --- vertical positioning: order topics within each column by total flow --
    node_y = np.zeros(n_nodes, dtype=float)

    for pi in range(n_periods):
        # Nodes in this column
        col_nodes = [idx for idx in range(n_nodes) if node_period[idx] == pi]
        if not col_nodes:
            continue
        # Sort by topic's global rank (mean proportion order), then by flow
        topic_rank = {tid: r for r, tid in enumerate(prop_mean.index)}
        col_nodes.sort(key=lambda idx: (
            topic_rank.get(node_topic[idx], 999),
            -node_flow[idx],
        ))

        # Stack nodes vertically with gaps
        total = node_flow[col_nodes].sum()
        if total < 1e-8:
            total = 1.0
        gap = 0.015
        y = 0.0
        for idx in col_nodes:
            h = max(node_flow[idx] / total, 0.008)  # min visible height
            node_y[idx] = y + h / 2
            y += h + gap

    # --- normalize y to [0.03, 0.97] — prevents top/bottom clipping from gaps ---
    y_min, y_max = node_y.min(), node_y.max()
    if y_max > y_min:
        y_span = y_max - y_min
        node_y = 0.03 + (node_y - y_min) / y_span * 0.94

    # --- column x-positions (evenly spaced) -----------------------------------
    node_x = np.array([
        pi / max(n_periods - 1, 1) for pi in node_period
    ], dtype=float)

    # --- node labels ----------------------------------------------------------
    node_label: List[str] = []
    node_cdata: List[str] = []  # hover customdata
    for tid, pi in zip(node_topic, node_period):
        period_name = periods[pi]
        short = topic_labels.get(tid, "")
        if short:
            full_label = f"T{tid}: {short}"
        else:
            full_label = f"Topic {tid}"

        # Node text is the topic id only. Anything longer is clipped by the node box,
        # which ellipsised every label in the 2026-09 render; the manuscript caption
        # points to Supplementary Table S1 for the names, and the full label stays in
        # the hover customdata.
        disp = f"T{tid}"

        node_label.append(disp)
        node_cdata.append(
            f"<b>{full_label}</b><br>Period: {period_name}<br>"
            f"Proportion: {prop_df.loc[tid, period_name]:.2%}"
            if tid in prop_df.index else
            f"<b>{full_label}</b><br>Period: {period_name}"
        )

    # --- node colours ---------------------------------------------------------
    # One palette slot per displayed topic.  `_topic_colour` cycles a 16-entry list by
    # topic id, which made T52 and T36 (both 4 mod 16) the same magenta; sorting the
    # displayed topics keeps the mapping deterministic and collision-free up to 16.
    palette_of = {
        tid: TOPIC_COLOURS_HEX[i % len(TOPIC_COLOURS_HEX)]
        for i, tid in enumerate(sorted(set(node_topic)))
    }
    node_color = [palette_of[tid] for tid in node_topic]

    # --- link colours ---------------------------------------------------------
    link_color = [
        _hex_to_rgba(palette_of.get(tid, _topic_colour(tid)), alpha=0.40)
        for tid in links_topic
    ]

    # --- link hover text ------------------------------------------------------
    link_cdata: List[str] = []
    for i in range(len(links_src)):
        s_tid = links_topic[i]
        t_tid = node_topic[links_tgt[i]]
        src_label = topic_labels.get(s_tid, f"T{s_tid}")
        tgt_label = topic_labels.get(t_tid, f"T{t_tid}")
        sim_val = (
            sim_df.loc[s_tid, t_tid]
            if s_tid in sim_df.index and t_tid in sim_df.columns
            else 0.0
        )
        link_cdata.append(
            f"<b>{src_label}</b> → <b>{tgt_label}</b><br>"
            f"Flow: {links_val[i]:.3f}  |  Similarity: {sim_val:.3f}"
        )

    logger.info(
        "Sankey: %d nodes, %d links across %d periods",
        n_nodes, len(links_val), n_periods,
    )

    return {
        "node_label": node_label,
        "node_x": node_x.tolist(),
        "node_y": node_y.tolist(),
        "node_color": node_color,
        "node_customdata": node_cdata,
        "link_source": links_src,
        "link_target": links_tgt,
        "link_value": links_val,
        "link_color": link_color,
        "link_customdata": link_cdata,
        "period_labels": periods,
        "n_periods": n_periods,
    }


# =============================================================================
# Plotly Sankey figure
# =============================================================================


def build_sankey_figure(
    flow_data: dict,
    config: Config,
) -> go.Figure:
    """Build a publication-quality Plotly Sankey diagram.

    Parameters
    ----------
    flow_data:
        Dict returned by :func:`compute_sankey_flows`.
    config:
        Validated configuration (used for figure metadata).

    Returns
    -------
    A ``plotly.graph_objects.Figure`` ready for ``.write_html()`` or ``.show()``.
    """
    periods = flow_data["period_labels"]
    n_periods = flow_data["n_periods"]

    # --- period annotation positions ------------------------------------------
    # Place a title above each column
    period_annotations = []
    for pi, period_name in enumerate(periods):
        x = pi / max(n_periods - 1, 1)
        period_annotations.append(dict(
            x=x,
            y=1.01,
            xref="paper",
            yref="paper",
            text=f"<b>{period_name}</b>",
            showarrow=False,
            font=dict(size=17, color="#0b0b0b"),
            xanchor="center",
        ))

    sankey = go.Sankey(
        arrangement="snap",
        domain=dict(
            y=[0.02, 0.98],    # extra buffer — prevents top/bottom node clipping
        ),
        node=dict(
            pad=15,             # vertical gap between nodes within a column
            thickness=16,       # narrower nodes → more room for labels
            line=dict(color="white", width=1.0),
            label=flow_data["node_label"],
            x=flow_data["node_x"],
            y=flow_data["node_y"],
            color=flow_data["node_color"],
            customdata=flow_data["node_customdata"],
            hovertemplate="%{customdata}<extra></extra>",
        ),
        link=dict(
            source=flow_data["link_source"],
            target=flow_data["link_target"],
            value=flow_data["link_value"],
            color=flow_data["link_color"],
            customdata=flow_data["link_customdata"],
            hovertemplate="%{customdata}<extra></extra>",
        ),
    )

    fig = go.Figure(
        data=[sankey],
        layout=go.Layout(
            title=dict(
                text=(
                    "Topic Evolution in Mangrove Remote Sensing Research<br>"
                    "<sup>Similarity-weighted flow across time periods</sup>"
                ),
                x=0.5,
                xanchor="center",
                font=dict(size=18, color="#0b0b0b"),
            ),
            # Node labels inherit this font; 18 px prints at ~5.6 pt once the 3000 px
            # canvas is scaled to \columnwidth (Fig. 6 uses 5.5 pt).
            font=dict(family="DejaVu Sans, Arial, sans-serif", size=18, color="#0b0b0b"),
            template="plotly_white",
            height=1800,
            width=1500,          # wider canvas — prevents label clipping
            margin=dict(l=120, r=120, t=80, b=100),  # generous margins; extra bottom for Sankey nodes
            annotations=period_annotations,
            hovermode="closest",
        ),
    )

    return fig


# =============================================================================
# Main entry point
# =============================================================================


def run_topic_sankey(config: Config) -> None:
    """Run the topic evolution Sankey analysis end-to-end.

    1. Load topic proportions + labels + similarity matrix.
    2. Compute bipartite flows across consecutive periods.
    3. Build and save the interactive Sankey diagram.
    """
    sk_cfg = config.topic_sankey

    # 1. Load data
    with Timer("Loading topic proportions"):
        prop_df = load_topic_proportions(config)

    with Timer("Loading topic labels"):
        topic_labels = load_topic_labels(config)

    with Timer("Loading similarity matrix"):
        sim_df = load_similarity_matrix(config)

    # 2. Compute flows
    with Timer("Computing Sankey flows"):
        flow_data = compute_sankey_flows(
            prop_df,
            sim_df,
            topic_labels,
            temperature=sk_cfg.temperature,
            min_flow=sk_cfg.min_flow,
            top_n_topics=sk_cfg.top_n_topics,
        )

    if len(flow_data["link_value"]) == 0:
        logger.warning("No flows above threshold — Sankey would be empty; skipping.")
        return

    # 3. Build & save figure
    with Timer("Building & saving Sankey figure"):
        fig = build_sankey_figure(flow_data, config)
        output_path = config.paths.figures_dir / sk_cfg.output_file
        ensure_dir(output_path.parent)
        fig.write_html(str(output_path))
        logger.info("Topic Sankey diagram saved: %s", output_path)

    # Also save a static PNG version
    try:
        png_path = output_path.with_suffix(".png")
        fig.write_image(str(png_path), width=1500, height=900, scale=2)
        logger.info("Static PNG saved: %s", png_path)
    except Exception as exc:
        logger.warning("Could not save static PNG (kaleido may be missing): %s", exc)

    logger.info("Topic Sankey analysis complete.")


def main() -> None:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Step 5c: Topic Evolution Sankey Diagram",
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to config.yaml",
    )
    args = parser.parse_args()

    config = get_config(args.config)
    from src.utils import setup_logging

    setup_logging(config.system.log_level, config.system.log_file)

    try:
        run_topic_sankey(config)
    except Exception:
        logger.exception("Topic Sankey analysis failed")
        raise


if __name__ == "__main__":
    main()
