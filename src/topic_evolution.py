"""Step 5: Topic Evolution — temporal topic proportion analysis with
interactive river/streamgraph plot.

Usage:
    python src/topic_evolution.py              # uses default config.yaml
    python src/topic_evolution.py --config path/to/config.yaml
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from src.config import Config, get_config
from src.utils import Timer, ensure_dir

logger = logging.getLogger(__name__)


# =============================================================================
# Data merging
# =============================================================================


def merge_topic_doc(config: Config) -> pd.DataFrame:
    """Merge document-topic assignments with cleaned data (year, text).

    Args:
        config: Validated configuration.

    Returns:
        DataFrame with columns: ``doc_index, Topic, PY, text``.

    Raises:
        FileNotFoundError: If required input files are missing.
    """
    proc = config.paths.processed_dir

    # Load cleaned data (prefer parquet)
    cleaned_csv = proc / config.preprocessing.output_file
    cleaned_pq = cleaned_csv.with_suffix(".parquet")
    if cleaned_pq.exists():
        df = pd.read_parquet(cleaned_pq)
    elif cleaned_csv.exists():
        df = pd.read_csv(cleaned_csv, encoding="utf-8-sig")
    else:
        raise FileNotFoundError(f"Cleaned data not found: {cleaned_csv}")

    # Load doc-topic distribution
    doc_topic_csv = proc / config.topic_modeling.output.doc_topic_dist
    if not doc_topic_csv.exists():
        raise FileNotFoundError(f"Document-topic distribution not found: {doc_topic_csv}. Run topic modeling first.")

    doc_topic = pd.read_csv(doc_topic_csv, encoding="utf-8-sig")

    # Merge on index
    merged = doc_topic.copy()
    if "PY" not in merged.columns and "PY" in df.columns:
        merged["PY"] = df["PY"].values[:len(merged)]

    # Drop outliers for evolution analysis
    n_before = len(merged)
    merged = merged[merged["Topic"] != -1].copy()
    if len(merged) < n_before:
        logger.info("Dropped %d outlier (-1) documents for evolution analysis", n_before - len(merged))

    logger.info("Merged data: %d documents, %d unique topics", len(merged), merged["Topic"].nunique())
    return merged


# =============================================================================
# Time slicing
# =============================================================================


def auto_time_periods(df: pd.DataFrame, n_periods: int = 4) -> Tuple[List[List[Optional[int]]], List[str]]:
    """Auto-compute time periods by splitting the year range into *n_periods* equal bins.

    Args:
        df: Merged DataFrame with ``PY`` column.
        n_periods: Number of periods to generate.

    Returns:
        ``(periods, labels)`` — periods are ``[[start, end], ...]`` with None for
        open boundaries; labels are human-readable.
    """
    py_min = int(df["PY"].min())
    py_max = int(df["PY"].max())

    # Create roughly equal-sized bins
    bin_edges = np.linspace(py_min, py_max + 1, n_periods + 1, dtype=int)

    periods = []
    labels = []
    for i in range(len(bin_edges) - 1):
        start = int(bin_edges[i])
        end = int(bin_edges[i + 1] - 1)
        periods.append([start, end])
        labels.append(f"{start}–{end}")

    # Fix last period to include py_max
    periods[-1][1] = py_max
    labels[-1] = f"{periods[-1][0]}–{py_max}"

    logger.info("Auto-computed %d time periods:", n_periods)
    for (s, e), lbl in zip(periods, labels):
        n = len(df[(df["PY"] >= s) & (df["PY"] <= e)])
        logger.info("  %s: %d documents", lbl, n)

    return periods, labels


def create_time_slices(
    df: pd.DataFrame,
    periods: List[List[Optional[int]]],
    labels: List[str],
) -> Dict[str, pd.DataFrame]:
    """Slice the merged DataFrame into time periods.

    Args:
        df: Merged DataFrame with ``PY`` column.
        periods: ``[[start, end], ...]``, where None means open interval.
        labels: Human-readable names for each period.

    Returns:
        ``{label: sliced_dataframe}``.
    """
    slices: Dict[str, pd.DataFrame] = {}

    for (start, end), label in zip(periods, labels):
        mask = pd.Series(True, index=df.index)
        if start is not None:
            mask &= df["PY"] >= start
        if end is not None:
            mask &= df["PY"] <= end
        sliced = df[mask].copy()
        slices[label] = sliced
        logger.info("  %s: %d docs, %d topics", label, len(sliced), sliced["Topic"].nunique())

    return slices


# =============================================================================
# Topic proportions
# =============================================================================


def compute_topic_proportions(
    slices: Dict[str, pd.DataFrame],
    min_share: float = 0.01,
    top_n_topics: int = 12,
) -> pd.DataFrame:
    """Compute topic proportions across time periods.

    Args:
        slices: ``{period_label: dataframe}``.
        min_share: Minimum proportion for a topic to be included.
        top_n_topics: Keep at most this many topics, ranked by overall frequency.

    Returns:
        DataFrame: rows=topics, columns=time_periods, values=proportion.
        Index is the topic ID (as string); extra column ``topic_label`` contains
        the top-3 keywords if available.
    """
    all_periods = list(slices.keys())

    # Build topic × period count matrix
    records = []
    for period, s_df in slices.items():
        counts = s_df["Topic"].value_counts()
        total = counts.sum()
        for topic_id, count in counts.items():
            records.append({
                "Topic": int(topic_id),
                "Period": period,
                "Count": count,
                "Proportion": count / total if total > 0 else 0,
            })

    prop_df = pd.DataFrame(records)
    if prop_df.empty:
        logger.warning("No topic data — all time slices are empty")
        return prop_df

    # Pivot: topics × periods
    pivot = prop_df.pivot_table(
        index="Topic",
        columns="Period",
        values="Proportion",
        fill_value=0.0,
    )

    # Ensure all periods are columns (in order)
    pivot = pivot.reindex(columns=all_periods, fill_value=0.0)

    # Keep only top N topics by mean proportion
    pivot["_mean"] = pivot.mean(axis=1)
    pivot = pivot.sort_values("_mean", ascending=False)
    pivot = pivot.drop(columns=["_mean"])

    # Filter by min_share: keep topics that reach min_share in at least one period
    pivot = pivot[(pivot > min_share).any(axis=1)]

    # Cap at top_n_topics
    if len(pivot) > top_n_topics:
        pivot = pivot.head(top_n_topics)

    logger.info("Topic proportion matrix: %d topics × %d periods", len(pivot), len(all_periods))
    return pivot


# =============================================================================
# River plot (Plotly streamgraph)
# =============================================================================


def plot_river(
    prop_df: pd.DataFrame,
    config: Config,
    output_path: Path,
) -> None:
    """Draw an interactive stacked area chart (river/streamgraph) showing topic evolution.

    Args:
        prop_df: Topic proportion matrix (rows=topics, columns=periods).
        config: Validated configuration.
        output_path: Where to save the HTML file.
    """
    if prop_df.empty:
        logger.warning("Empty proportion DataFrame — skipping river plot")
        return

    # Build topic labels: "Topic N: keyword1_keyword2_keyword3"
    topic_labels = {}
    for tid in prop_df.index:
        topic_labels[tid] = f"Topic {tid}"

    # Colormap
    n_topics = len(prop_df)
    colors = px.colors.qualitative.Bold[:n_topics] if n_topics <= 10 else px.colors.sample_colorscale(
        "viridis", n_topics
    )

    fig = go.Figure()

    for idx, (topic_id, row) in enumerate(prop_df.iterrows()):
        fig.add_trace(go.Scatter(
            x=list(prop_df.columns),
            y=row.values,
            mode="lines",
            stackgroup="one",
            groupnorm="fraction",
            name=topic_labels.get(topic_id, f"Topic {topic_id}"),
            line=dict(width=0.5, color=colors[idx % len(colors)]),
            hovertemplate=(
                f"<b>{topic_labels.get(topic_id, f'Topic {topic_id}')}</b><br>"
                f"Period: %{{x}}<br>"
                f"Share: %{{y:.1%}}<br>"
                "<extra></extra>"
            ),
        ))

    fig.update_layout(
        title={
            "text": "Topic Evolution in Mangrove Remote Sensing Research",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 16},
        },
        xaxis={
            "title": "Time Period",
            "tickangle": -30,
        },
        yaxis={
            "title": "Topic Proportion",
            "tickformat": ".0%",
            "range": [0, 1],
        },
        legend={
            "title": "Topics",
            "orientation": "v",
            "yanchor": "top",
            "y": 0.99,
            "xanchor": "left",
            "x": 1.01,
        },
        template="plotly_white",
        hovermode="x unified",
        height=600,
        width=1000,
    )

    ensure_dir(output_path.parent)
    fig.write_html(str(output_path))
    logger.info("Topic evolution river plot saved: %s", output_path)


# =============================================================================
# Bump chart (ranking evolution)
# =============================================================================


def plot_bump_chart(
    prop_df: pd.DataFrame,
    config: Config,
    output_path: Path,
    top_n: int = 12,
) -> None:
    """Interactive bump chart showing topic *rank* evolution over time.

    Adapted from An et al. (2026) Fig. 5. Each topic's rank in each period
    is computed from its proportion; lines connect the same topic across
    periods, making rising / falling trends immediately visible.

    Args:
        prop_df: Topic proportion matrix (rows=topics, columns=periods).
        config: Validated configuration.
        output_path: Where to save the HTML file.
        top_n: Number of topics to display (ranked by mean proportion).
    """
    if prop_df.empty or len(prop_df.columns) < 2:
        logger.warning("Insufficient data for bump chart — skipping")
        return

    # Keep top-N topics by mean proportion
    df = prop_df.copy()
    df["_mean"] = df.mean(axis=1)
    df = df.sort_values("_mean", ascending=False).head(top_n)
    df = df.drop(columns=["_mean"])

    # Compute rank per period (1 = highest proportion)
    ranks = df.rank(axis=0, method="min", ascending=False).astype(int)

    # Build labels from topics_info if available — prefer Short_Label
    topic_labels = {tid: f"Topic {tid}" for tid in ranks.index}
    topics_csv = config.paths.processed_dir / config.topic_modeling.output.topics_info
    if topics_csv.exists():
        try:
            tinfo = pd.read_csv(topics_csv, encoding="utf-8-sig")
            for _, row in tinfo.iterrows():
                tid = int(row.get("Topic", -1))
                if tid not in topic_labels:
                    continue
                # Prefer curated short label
                short = row.get("Short_Label", None)
                if pd.notna(short) and isinstance(short, str) and short.strip():
                    topic_labels[tid] = f"T{tid}: {short.strip()}"
                elif "Name" in row:
                    name = str(row["Name"])
                    if "_" in name:
                        name = name.split("_", 1)[1]
                    topic_labels[tid] = f"T{tid}: {name[:30]}"
        except Exception:
            pass

    periods = list(ranks.columns)
    n_topics = len(ranks)
    colors = px.colors.qualitative.Bold[:n_topics] if n_topics <= 10 else px.colors.sample_colorscale(
        "viridis", n_topics
    )

    fig = go.Figure()

    for idx, topic_id in enumerate(ranks.index):
        y_vals = [ranks.loc[topic_id, p] for p in periods]
        label = topic_labels.get(topic_id, f"Topic {topic_id}")
        color = colors[idx % len(colors)]

        # Rank values are inverted for display (rank 1 at top)
        fig.add_trace(go.Scatter(
            x=periods,
            y=y_vals,
            mode="lines+markers+text",
            name=label,
            line=dict(width=2.5, color=color),
            marker=dict(size=10, color=color, line=dict(width=1, color="white")),
            text=[str(r) for r in y_vals],
            textposition="top center",
            textfont=dict(size=9, color=color),
            hovertemplate=f"<b>{label}</b><br>Period: %{{x}}<br>Rank: %{{y}}<extra></extra>",
        ))

    fig.update_layout(
        title={
            "text": "Topic Ranking Evolution Over Time",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 16},
        },
        xaxis={"title": "Time Period", "tickangle": -30, "side": "bottom"},
        yaxis={
            "title": "Rank",
            "autorange": "reversed",  # Rank 1 at top
            "tickmode": "linear",
            "dtick": 1,
        },
        template="plotly_white",
        hovermode="x unified",
        legend={
            "title": "Topics",
            "orientation": "v",
            "yanchor": "top",
            "y": 0.99,
            "xanchor": "left",
            "x": 1.01,
        },
        height=650,
        width=1000,
    )

    ensure_dir(output_path.parent)
    fig.write_html(str(output_path))
    logger.info("Topic bump chart saved: %s", output_path)


# =============================================================================
# Main entry point
# =============================================================================


def run_topic_evolution(config: Config) -> None:
    """Run the complete topic evolution analysis.

    Args:
        config: Validated configuration.
    """
    te_cfg = config.topic_evolution

    # 1. Merge data
    with Timer("Merging topic assignments with metadata"):
        merged = merge_topic_doc(config)

    # 2. Determine time periods
    if te_cfg.time_periods and te_cfg.period_labels:
        periods = te_cfg.time_periods
        labels = te_cfg.period_labels
        logger.info("Using %d configured time periods", len(periods))
    else:
        logger.info("Auto-computing time periods (config has no explicit periods)")
        periods, labels = auto_time_periods(merged)

    # 3. Slice
    with Timer("Creating time slices"):
        slices = create_time_slices(merged, periods, labels)

    # 4. Compute proportions
    with Timer("Computing topic proportions"):
        prop_df = compute_topic_proportions(
            slices,
            min_share=te_cfg.min_topic_share,
        )

    # 5. Plot river
    with Timer("Generating topic evolution plot"):
        output_path = config.paths.figures_dir / te_cfg.output_file
        plot_river(prop_df, config, output_path)

    # 6. Plot bump chart
    with Timer("Generating bump chart"):
        bump_path = config.paths.figures_dir / "topic_bump_chart.html"
        plot_bump_chart(prop_df, config, bump_path, top_n=12)

    # Also save the proportion matrix as CSV for further analysis
    csv_path = config.paths.processed_dir / "topic_evolution_proportions.csv"
    prop_df.to_csv(csv_path, encoding="utf-8-sig")
    logger.info("Topic proportion matrix saved: %s", csv_path)

    logger.info("Topic evolution analysis complete.")


def main() -> None:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Step 5: Topic Evolution Analysis")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    config = get_config(args.config)
    from src.utils import setup_logging

    setup_logging(config.system.log_level, config.system.log_file)

    try:
        run_topic_evolution(config)
    except Exception:
        logger.exception("Topic evolution analysis failed")
        raise


if __name__ == "__main__":
    main()
