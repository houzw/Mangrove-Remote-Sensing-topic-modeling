"""Step 4b: Topic Dynamics — BERTopic temporal topic modeling with keyword evolution.

Usage:
    python src/topic_dynamics.py              # uses default config.yaml
    python src/topic_dynamics.py --config path/to/config.yaml
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.config import Config, get_config
from src.utils import Timer, ensure_dir

logger = logging.getLogger(__name__)


# =============================================================================
# Data loading
# =============================================================================


def load_model_and_data(config: Config) -> Tuple["BERTopic", pd.DataFrame, List[str], List[int]]:  # noqa: F821
    """Load fitted BERTopic model, cleaned data, documents, and years.

    Args:
        config: Validated configuration.

    Returns:
        ``(model, dataframe, docs, years)``.
    """
    from bertopic import BERTopic

    model_path = config.paths.models_dir / config.topic_modeling.output.model_name
    if not model_path.exists():
        raise FileNotFoundError(f"BERTopic model not found: {model_path}. Run topic modeling first.")

    with Timer("Loading BERTopic model"):
        model = BERTopic.load(model_path)
    logger.info("Model loaded: %d topics", len(model.get_topic_info()) - 1)

    # Load cleaned data
    csv_path = config.paths.processed_dir / config.preprocessing.output_file
    pq_path = csv_path.with_suffix(".parquet")
    if pq_path.exists():
        df = pd.read_parquet(pq_path)
    elif csv_path.exists():
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
    else:
        raise FileNotFoundError(f"Cleaned data not found: {csv_path}")

    docs = df["text"].fillna("").astype(str).tolist()
    docs = [d.strip() for d in docs if d.strip()]

    if "PY" not in df.columns:
        raise ValueError("Cleaned data missing 'PY' (year) column — required for dynamics.")

    years = df["PY"].astype(int).tolist()

    # Align lengths (model was fitted on docs after filtering empties)
    n_model = len(model.topics_)
    if len(docs) != n_model:
        logger.warning("Document count mismatch: data=%d, model=%d — truncating to model length", len(docs), n_model)
        docs = docs[:n_model]
        years = years[:n_model]

    return model, df, docs, years


# =============================================================================
# Temporal binning
# =============================================================================


def compute_time_bins(
    years: List[int], n_bins: Optional[int] = None,
) -> Tuple[List[int], List[str], List[str]]:
    """Assign each year to a temporal bin.

    Args:
        years: List of publication years.
        n_bins: Number of bins; if None, auto-compute (~8 bins or yearly if range small).

    Returns:
        ``(bin_indices, bin_labels, bin_timestamps)``:
          - *bin_indices*: list of bin indices aligned with *years*
          - *bin_labels*: display labels like "1972–1980"
          - *bin_timestamps*: parseable datetime strings (midpoint year) for BERTopic
    """
    y_min, y_max = min(years), max(years)
    year_range = y_max - y_min + 1

    if n_bins is None:
        n_bins = min(8, year_range)

    n_bins = max(2, min(n_bins, year_range))

    # Use pandas cut for equal-frequency-ish bins
    year_series = pd.Series(years)
    bins = pd.cut(year_series, bins=n_bins, include_lowest=True)
    bin_edges = [int(interval.left) for interval in bins.cat.categories] + [int(bins.cat.categories[-1].right) + 1]

    # Build labels like "1972–1980" and timestamp midpoints like "1976"
    labels = []
    timestamps = []
    for i in range(len(bin_edges) - 1):
        start = bin_edges[i]
        end = bin_edges[i + 1] - 1
        mid = (start + end) // 2
        labels.append(f"{start}–{end}")
        timestamps.append(str(mid))

    # Map each year to bin index
    bin_indices = pd.cut(year_series, bins=bin_edges, labels=False, include_lowest=True).tolist()
    bin_indices = [int(b) if not pd.isna(b) else 0 for b in bin_indices]

    logger.info("Time bins (%d): %s", len(labels), labels)
    return bin_indices, labels, timestamps


# =============================================================================
# BERTopic topics_over_time
# =============================================================================


def run_topics_over_time(
    model: "BERTopic",  # noqa: F821
    docs: List[str],
    years: List[int],
    n_bins: Optional[int] = None,
    evolution_mode: str = "relative",
) -> pd.DataFrame:
    """Run BERTopic's native topics_over_time and return a tidy DataFrame.

    Args:
        model: Fitted BERTopic model.
        docs: Document texts (same order as model fitting).
        years: Publication years aligned with docs.
        n_bins: Number of temporal bins (None = auto).
        evolution_mode: ``"relative"`` for proportions, ``"absolute"`` for counts.

    Returns:
        Tidy DataFrame with columns ``Topic, Timestamp, Frequency, Words``.
    """
    # Prepare timestamps: use midpoint years (parseable as datetime) for BERTopic,
    # keeping bin_labels for display in visualizations.
    bin_indices, bin_labels, bin_timestamps = compute_time_bins(years, n_bins=n_bins)
    timestamps = [bin_timestamps[i] for i in bin_indices]

    logger.info("Running BERTopic topics_over_time with %d bins ...", len(bin_labels))
    with Timer("BERTopic topics_over_time"):
        topics_over_time = model.topics_over_time(
            docs,
            timestamps,
            global_tuning=True,
            evolution_tuning=True,
        )

    # topics_over_time DataFrame columns: Topic, Words, Timestamp, Frequency
    logger.info("Topics over time: %d rows", len(topics_over_time))

    # Normalize frequencies if relative mode
    if evolution_mode == "relative":
        # Compute total per timestamp
        totals = topics_over_time.groupby("Timestamp")["Frequency"].sum().to_dict()
        topics_over_time["Proportion"] = topics_over_time.apply(
            lambda row: row["Frequency"] / totals[row["Timestamp"]] if totals[row["Timestamp"]] > 0 else 0,
            axis=1,
        )

    return topics_over_time


# =============================================================================
# Visualizations
# =============================================================================


def _theil_sen_trend(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Compute Theil-Sen slope and intercept.

    Args:
        x: Independent variable (e.g., time index).
        y: Dependent variable (e.g., topic proportion).

    Returns:
        ``(slope, intercept)``.
    """
    # scipy.stats is imported here: its extension DLLs do not load in this environment,
    # so a module-level import would break every caller.
    from scipy.stats import theilslopes

    try:
        res = theilslopes(y, x, 0.95)
        return float(res.slope), float(res.intercept)
    except Exception:
        # Fallback to simple linear regression
        slope, intercept = np.polyfit(x, y, 1)
        return float(slope), float(intercept)


def plot_topic_dynamics(
    topics_over_time: pd.DataFrame,
    config: Config,
    output_path: Path,
    evolution_mode: str = "relative",
) -> None:
    """Interactive line chart of topic proportions over time with Theil-Sen trends.

    Hu et al. (2025) Fig. 7 style: dashed lines = annual popularity,
    solid lines = Theil-Sen estimated trend.

    Args:
        topics_over_time: Output from ``run_topics_over_time``.
        config: Validated configuration.
        output_path: Path to save HTML.
        evolution_mode: ``"relative"`` or ``"absolute"``.
    """
    if topics_over_time.empty:
        logger.warning("Empty topics_over_time — skipping dynamics plot")
        return

    # Ensure timestamp order
    topics_over_time = topics_over_time.copy()
    topics_over_time["_sort"] = topics_over_time["Timestamp"].apply(lambda x: int(str(x).split("-")[0]))
    topics_over_time = topics_over_time.sort_values(["_sort", "Topic"])

    value_col = "Proportion" if evolution_mode == "relative" and "Proportion" in topics_over_time.columns else "Frequency"
    y_title = "Topic Proportion" if value_col == "Proportion" else "Number of Documents"
    y_format = ".0%" if value_col == "Proportion" else ".0f"

    # Exclude outlier topic -1
    df_plot = topics_over_time[topics_over_time["Topic"] != -1].copy()

    # Build traces
    fig = go.Figure()
    n_topics = df_plot["Topic"].nunique()
    colors = px.colors.qualitative.Bold[:n_topics] if n_topics <= 10 else px.colors.sample_colorscale("viridis", n_topics)

    # Precompute Theil-Sen trends
    trends: Dict[int, Tuple[float, float]] = {}
    for topic_id in sorted(df_plot["Topic"].unique()):
        sub = df_plot[df_plot["Topic"] == topic_id].sort_values("_sort")
        x = np.arange(len(sub))
        y = sub[value_col].values.astype(float)
        if len(x) >= 3:
            slope, intercept = _theil_sen_trend(x, y)
            trends[topic_id] = (slope, intercept)

    for idx, topic_id in enumerate(sorted(df_plot["Topic"].unique())):
        sub = df_plot[df_plot["Topic"] == topic_id].sort_values("_sort")
        words = sub["Words"].iloc[0] if "Words" in sub.columns and len(sub) > 0 else ""
        label = f"Topic {topic_id}: {words[:50]}..."
        color = colors[idx % len(colors)]

        timestamps = sub["Timestamp"].tolist()
        values = sub[value_col].tolist()

        # Dashed line: raw values
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=values,
            mode="lines+markers",
            name=label,
            line=dict(width=1.5, color=color, dash="dash"),
            marker=dict(size=6, color=color),
            hovertemplate=f"<b>{label}</b><br>Period: %{{x}}<br>{y_title}: %{{y:.3f}}<extra></extra>",
            legendgroup=f"topic_{topic_id}",
            showlegend=True,
        ))

        # Solid line: Theil-Sen trend
        if topic_id in trends:
            slope, intercept = trends[topic_id]
            x_arr = np.arange(len(timestamps))
            trend_y = slope * x_arr + intercept
            # Clip to non-negative
            trend_y = np.clip(trend_y, 0, None)

            fig.add_trace(go.Scatter(
                x=timestamps,
                y=trend_y,
                mode="lines",
                name=f"{label} (trend)",
                line=dict(width=2.5, color=color),
                hovertemplate=f"<b>{label} (Theil-Sen trend)</b><br>Slope: {slope:.4f}<extra></extra>",
                legendgroup=f"topic_{topic_id}",
                showlegend=False,
            ))

    fig.update_layout(
        title={
            "text": "Dynamic Topic Evolution with Theil-Sen Trends (Hu et al. 2025 Style)",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 16},
        },
        xaxis={"title": "Time Period", "tickangle": -30},
        yaxis={"title": y_title, "tickformat": y_format},
        template="plotly_white",
        hovermode="x unified",
        legend={"title": "Topics", "orientation": "v", "yanchor": "top", "y": 0.99, "xanchor": "left", "x": 1.01},
        height=650,
        width=1100,
    )

    ensure_dir(output_path.parent)
    fig.write_html(str(output_path))
    logger.info("Topic dynamics plot saved: %s", output_path)


def plot_keyword_evolution(
    topics_over_time: pd.DataFrame,
    config: Config,
    output_path: Path,
    top_n_words: int = 10,
) -> None:
    """Interactive heatmap showing how top keywords of selected topics evolve.

    Args:
        topics_over_time: Output from ``run_topics_over_time``.
        config: Validated configuration.
        output_path: Path to save HTML.
        top_n_words: Number of top words to show.
    """
    if topics_over_time.empty or "Words" not in topics_over_time.columns:
        logger.warning("No Words column in topics_over_time — skipping keyword evolution")
        return

    # Pick top topics by total frequency
    topic_totals = topics_over_time.groupby("Topic")["Frequency"].sum().sort_values(ascending=False)
    topic_totals = topic_totals[topic_totals.index != -1].head(6)

    if len(topic_totals) == 0:
        logger.warning("No valid topics for keyword evolution")
        return

    fig = go.Figure()
    timestamps = sorted(topics_over_time["Timestamp"].unique(), key=lambda x: int(str(x).split("-")[0]))

    for topic_id in topic_totals.index:
        sub = topics_over_time[topics_over_time["Topic"] == topic_id].sort_values("Timestamp")
        words_per_period = []
        for ts in timestamps:
            row = sub[sub["Timestamp"] == ts]
            if not row.empty and isinstance(row["Words"].values[0], str):
                words = row["Words"].values[0].split(", ")[:top_n_words]
                words_per_period.append("<br>".join(words))
            else:
                words_per_period.append("")

        fig.add_trace(go.Scatter(
            x=timestamps,
            y=[f"Topic {topic_id}"] * len(timestamps),
            mode="text",
            text=words_per_period,
            textposition="middle center",
            hoverinfo="skip",
        ))

    fig.update_layout(
        title={
            "text": "Keyword Evolution Across Time Periods",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 16},
        },
        xaxis={"title": "Time Period", "tickangle": -30, "side": "bottom"},
        yaxis={"title": "", "tickfont": {"size": 12}},
        template="plotly_white",
        height=400,
        width=1000,
        margin=dict(l=100, r=20, t=60, b=60),
    )

    ensure_dir(output_path.parent)
    fig.write_html(str(output_path))
    logger.info("Keyword evolution plot saved: %s", output_path)


# =============================================================================
# Main entry point
# =============================================================================


def run_topic_dynamics(config: Config) -> None:
    """Run the complete topic dynamics analysis.

    Args:
        config: Validated configuration.
    """
    td_cfg = config.topic_dynamics

    # 1. Load model + data
    with Timer("Loading model and data"):
        model, df, docs, years = load_model_and_data(config)

    # 2. Compute topics over time
    with Timer("Computing topics over time"):
        topics_over_time = run_topics_over_time(
            model,
            docs,
            years,
            n_bins=td_cfg.n_bins,
            evolution_mode=td_cfg.evolution_mode,
        )

    # 3. Save CSV
    csv_path = config.paths.processed_dir / td_cfg.output.dynamics_csv
    ensure_dir(csv_path.parent)
    topics_over_time.to_csv(csv_path, index=False, encoding="utf-8-sig")
    logger.info("Topic dynamics data saved: %s", csv_path)

    # 4. Plot dynamics
    with Timer("Generating dynamics plot"):
        dynamics_html = config.paths.figures_dir / td_cfg.output.dynamics_html
        plot_topic_dynamics(topics_over_time, config, dynamics_html, evolution_mode=td_cfg.evolution_mode)

    # 5. Plot keyword evolution
    with Timer("Generating keyword evolution plot"):
        kw_html = config.paths.figures_dir / td_cfg.output.keywords_evolution_html
        plot_keyword_evolution(topics_over_time, config, kw_html, top_n_words=td_cfg.top_n_words)

    logger.info("Topic dynamics analysis complete.")


def main() -> None:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Step 4b: Topic Dynamics Analysis")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    config = get_config(args.config)
    from src.utils import setup_logging

    setup_logging(config.system.log_level, config.system.log_file)

    try:
        run_topic_dynamics(config)
    except Exception:
        logger.exception("Topic dynamics analysis failed")
        raise


if __name__ == "__main__":
    main()
