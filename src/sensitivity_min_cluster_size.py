"""HDBSCAN min_cluster_size sensitivity analysis.

Systematically varies min_cluster_size across a range of values while holding
all other parameters (embedding, UMAP, min_samples, vectorizer) constant.
Reports: n_topics, outlier%, C_V coherence, NPMI, topic diversity,
topic distinctiveness, and topic size distribution.

Usage:
    pixi run python -m src.sensitivity_min_cluster_size
    pixi run python -m src.sensitivity_min_cluster_size --values 5,8,10,15,20
"""

from __future__ import annotations

import json
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from hdbscan import HDBSCAN

from src.config import Config, get_config
from src.topic_modeling import (
    evaluate_topic_model,
    fit_model,
    load_texts,
)
from src.utils import Timer, ensure_dir

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers to override min_cluster_size in an otherwise identical BERTopic build
# ---------------------------------------------------------------------------

def _build_model_for_mcs(
    config: Config,
    min_cluster_size: int,
    device: str = "auto",
) -> "BERTopic":  # noqa: F821
    """Build a BERTopic model identical to the config one but with a different
    ``min_cluster_size``.  Everything else (UMAP, vectorizer, representation,
    min_topic_size) stays the same.
    """
    from bertopic import BERTopic
    from bertopic.representation import KeyBERTInspired
    from bertopic.vectorizers import ClassTfidfTransformer
    from sentence_transformers import SentenceTransformer
    from umap import UMAP

    # --- Embedding (shared across runs — most expensive part) ---
    logger.debug("Loading embedding model: %s", config.topic_modeling.embedding_model)
    embedding_model = SentenceTransformer(
        config.topic_modeling.embedding_model,
        device=device,
    )

    # --- UMAP ---
    umap_model = UMAP(
        n_neighbors=config.umap.n_neighbors,
        n_components=config.umap.n_components,
        min_dist=config.umap.min_dist,
        metric=config.umap.metric,
        random_state=config.umap.random_state,
        n_jobs=1,
    )

    # --- HDBSCAN (the parameter being varied) ---
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=config.hdbscan.min_samples,
        metric=config.hdbscan.metric,
        prediction_data=True,
    )

    # --- Vectorizer ---
    cfg = config.topic_modeling
    if cfg.vectorizer.backend == "count":
        from sklearn.feature_extraction.text import CountVectorizer
        vectorizer_model = CountVectorizer(
            stop_words="english",
            ngram_range=(1, 3),
            max_df=0.85,
            min_df=5,
        )
    else:
        try:
            import spacy
            spacy.load(cfg.spacy_model)
            spacy_ok = True
        except (ImportError, OSError, Exception):
            spacy_ok = False

        if spacy_ok:
            from bertopic.vectorizers import KeyphraseCountVectorizer
            vectorizer_model = KeyphraseCountVectorizer(
                spacy_pipeline=cfg.spacy_model,
                pos_pattern=cfg.vectorizer.pos_pattern,
            )
        else:
            from sklearn.feature_extraction.text import CountVectorizer
            vectorizer_model = CountVectorizer(
                stop_words="english",
                ngram_range=(1, 3),
                max_df=0.85,
                min_df=5,
            )

    # --- Representation ---
    representation_model = {
        "Main": KeyBERTInspired(
            top_n_words=cfg.representation.top_n_words,
            nr_repr_docs=cfg.representation.nr_repr_docs,
        ),
    }

    # --- Assemble ---
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        model = BERTopic(
            embedding_model=embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            representation_model=representation_model,
            ctfidf_model=ClassTfidfTransformer(reduce_frequent_words=True),
            min_topic_size=cfg.min_topic_size,
            calculate_probabilities=True,
            verbose=True,
        )

    return model



# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_sensitivity(
    config: Config,
    mcs_values: List[int],
    force_cpu: bool = False,
) -> pd.DataFrame:
    """Run sensitivity analysis over a range of min_cluster_size values.

    Args:
        config: Validated configuration.
        mcs_values: List of min_cluster_size values to test.
        force_cpu: If True, force CPU.

    Returns:
        DataFrame with one row per ``min_cluster_size``.
    """
    # Load data once
    with Timer("Loading documents"):
        df_meta, docs = load_texts(config)
    logger.info("Corpus: %d documents", len(docs))

    device = "cpu" if force_cpu else "auto"
    rows: List[dict] = []

    for i, mcs in enumerate(mcs_values):
        logger.info("=" * 72)
        logger.info("[%d/%d] min_cluster_size = %d", i + 1, len(mcs_values), mcs)
        logger.info("=" * 72)

        # Build
        with Timer(f"Build model (mcs={mcs})"):
            model = _build_model_for_mcs(config, min_cluster_size=mcs, device=device)

        # Fit
        with Timer(f"Fit model (mcs={mcs})"):
            model, topics, probs = fit_model(model, docs)

        # Topic-level info
        topic_info = model.get_topic_info()
        n_total = len(topics)
        n_outliers = int(np.sum(np.array(topics) == -1))
        outlier_pct = 100.0 * n_outliers / n_total
        n_valid_topics = len(topic_info[topic_info["Topic"] != -1])

        # Distribution
        sizes = topic_info[topic_info["Topic"] != -1]["Count"].values
        if len(sizes) > 0:
            mean_size = float(np.mean(sizes))
            median_size = float(np.median(sizes))
            min_size = int(np.min(sizes))
            max_size = int(np.max(sizes))
            std_size = float(np.std(sizes))
        else:
            mean_size = median_size = std_size = 0.0
            min_size = max_size = 0

        # Comprehensive evaluation (C_V, NPMI, diversity, distinctiveness, Gini, noise)
        eval_results = {}
        c_v = float("nan")
        npmi = float("nan")
        diversity = float("nan")
        mean_cos_sim = float("nan")
        high_sim_frac = float("nan")
        gini = 0.0
        try:
            eval_results = evaluate_topic_model(model, docs, topics, config)
            c_v = float(eval_results.get("coherence", {}).get("c_v_global", float("nan")))
            npmi = float(eval_results.get("coherence", {}).get("npmi_global", float("nan")))
            diversity = float(eval_results.get("coherence", {}).get("diversity", float("nan")))
            mean_cos_sim = float(eval_results.get("distinctiveness", {}).get("mean_pairwise_similarity", float("nan")))
            high_sim_frac = float(eval_results.get("distinctiveness", {}).get("fraction_high_similarity", float("nan")))
            gini = float(eval_results.get("distribution", {}).get("gini_coefficient", 0.0))
        except Exception:
            logger.warning("Evaluation failed for mcs=%d", mcs, exc_info=True)

        row = {
            "min_cluster_size": mcs,
            "n_topics": n_valid_topics,
            "n_outliers": n_outliers,
            "outlier_pct": round(outlier_pct, 1),
            "mean_topic_size": round(mean_size, 1),
            "median_topic_size": round(median_size, 1),
            "std_topic_size": round(std_size, 1),
            "min_topic_size_actual": min_size,
            "max_topic_size": max_size,
            "c_v_coherence": round(c_v, 4),
            "npmi_coherence": round(npmi, 4),
            "topic_diversity": round(diversity, 4),
            "mean_cos_sim": round(mean_cos_sim, 4) if not np.isnan(mean_cos_sim) else float("nan"),
            "high_sim_frac": round(high_sim_frac, 4) if not np.isnan(high_sim_frac) else float("nan"),
            "gini": round(gini, 4),
            "n_docs": n_total,
        }
        rows.append(row)

        logger.info(
            "Result: n_topics=%d  outlier=%.1f%%  C_V=%.4f  NPMI=%.4f  diversity=%.3f  mean_cos=%.4f",
            n_valid_topics, outlier_pct, c_v, npmi, diversity, mean_cos_sim,
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_sensitivity(df: pd.DataFrame, output_path: Path, dpi: int = 300) -> None:
    """Generate a 4-panel sensitivity summary figure."""
    import matplotlib.pyplot as plt

    mcs = df["min_cluster_size"].values
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))

    # Colour palette
    color_outlier = "#E74C3C"
    color_topics = "#2C3E50"
    color_cv = "#27AE60"
    color_diversity = "#8E44AD"
    color_cos = "#E67E22"

    # --- Panel A: outlier% + n_topics (dual axis) ---
    ax1 = axes[0, 0]
    ax1b = ax1.twinx()
    line1 = ax1.plot(mcs, df["outlier_pct"], "o-", color=color_outlier, linewidth=2, markersize=7, zorder=5)
    line2 = ax1b.plot(mcs, df["n_topics"], "s--", color=color_topics, linewidth=2, markersize=7, zorder=4)
    ax1.set_xlabel("min_cluster_size", fontsize=11)
    ax1.set_ylabel("Outlier %", color=color_outlier, fontsize=11)
    ax1b.set_ylabel("Number of Topics", color=color_topics, fontsize=11)
    ax1.tick_params(axis="y", labelcolor=color_outlier)
    ax1b.tick_params(axis="y", labelcolor=color_topics)
    ax1.set_title("A. Outlier Rate & Topic Count", fontsize=12, fontweight="bold", loc="left")
    ax1.set_xticks(mcs)
    ax1.grid(True, linestyle=":", alpha=0.4)
    # Adjust: move legend inside axes to avoid overlap
    # inset: place at top-left corner of ax1
    lines = line1 + line2
    labels = ["Outlier %", "N Topics"]
    ax1.legend(lines, labels, loc="upper left", fontsize=9)

    # --- Panel B: C_V coherence ---
    ax2 = axes[0, 1]
    ax2.plot(mcs, df["c_v_coherence"], "o-", color=color_cv, linewidth=2, markersize=7)
    ax2.set_xlabel("min_cluster_size", fontsize=11)
    ax2.set_ylabel("C_V Coherence", color=color_cv, fontsize=11)
    ax2.set_title("B. Topic Coherence (C_V)", fontsize=12, fontweight="bold", loc="left")
    ax2.set_xticks(mcs)
    ax2.grid(True, linestyle=":", alpha=0.4)

    # --- Panel C: topic diversity & distinctiveness ---
    ax3 = axes[1, 0]
    ax3b = ax3.twinx()
    line3 = ax3.plot(mcs, df["topic_diversity"], "o-", color=color_diversity, linewidth=2, markersize=7, zorder=5)
    line4 = ax3b.plot(mcs, df["mean_cos_sim"], "o-", color=color_cos, linewidth=2, markersize=7, zorder=4)
    ax3.set_xlabel("min_cluster_size", fontsize=11)
    ax3.set_ylabel("Topic Diversity", color=color_diversity, fontsize=11)
    ax3b.set_ylabel("Mean c-TF-IDF Cosine Sim", color=color_cos, fontsize=11)
    ax3.tick_params(axis="y", labelcolor=color_diversity)
    ax3b.tick_params(axis="y", labelcolor=color_cos)
    ax3.set_title("C. Diversity & Distinctiveness", fontsize=12, fontweight="bold", loc="left")
    ax3.set_xticks(mcs)
    ax3.grid(True, linestyle=":", alpha=0.4)
    # Adjust: move legend inside axes
    ax3.legend(line3 + line4, ["Diversity", "Mean Cos Sim"], loc="lower left", fontsize=9)

    # --- Panel D: topic size distribution (box-like with percentiles) ---
    ax4 = axes[1, 1]
    ax4.fill_between(
        mcs,
        df["min_topic_size_actual"],
        df["max_topic_size"],
        alpha=0.15,
        color="#3498DB",
        label="Min–Max range",
    )
    ax4.plot(mcs, df["mean_topic_size"], "o-", color="#2980B9", linewidth=2, markersize=7, label="Mean size")
    ax4.plot(mcs, df["median_topic_size"], "s--", color="#1A5276", linewidth=2, markersize=7, label="Median size")
    ax4.set_xlabel("min_cluster_size", fontsize=11)
    ax4.set_ylabel("Documents per Topic", fontsize=11)
    ax4.set_title("D. Topic Size Distribution", fontsize=12, fontweight="bold", loc="left")
    ax4.set_xticks(mcs)
    ax4.grid(True, linestyle=":", alpha=0.4)
    # Adjust: move legend inside axes
    ax4.legend(loc="upper left", fontsize=9)

    fig.suptitle(
        f"HDBSCAN min_cluster_size Sensitivity Analysis\n({len(df)} values, {df['n_docs'].iloc[0]} documents)",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Sensitivity figure saved: %s", output_path)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_summary(df: pd.DataFrame) -> None:
    """Pretty-print the summary table."""
    header = (
        f"{'mcs':>5s}  {'topics':>6s}  {'outlier%':>8s}  "
        f"{'C_V':>8s}  {'NPMI':>8s}  {'diversity':>9s}  "
        f"{'mean_cos':>8s}  {'mean_size':>9s}  {'gini':>7s}"
    )
    sep = "-" * len(header)
    print("\n" + sep)
    print(header)
    print(sep)
    for _, row in df.iterrows():
        print(
            f"{int(row['min_cluster_size']):5d}  {int(row['n_topics']):6d}  "
            f"{row['outlier_pct']:7.1f}%  {row['c_v_coherence']:8.4f}  "
            f"{row['npmi_coherence']:8.4f}  {row['topic_diversity']:9.4f}  "
            f"{row['mean_cos_sim']:8.4f}  {row['mean_topic_size']:9.1f}  "
            f"{row['gini']:7.4f}"
        )
    print(sep)

    # Recommendation
    outlier_ok = df[df["outlier_pct"] < 35]
    if not outlier_ok.empty:
        best_balance = outlier_ok.loc[outlier_ok["c_v_coherence"].idxmax()]
    else:
        best_balance = df.loc[df["c_v_coherence"].idxmax()]

    print(f"\nRecommended min_cluster_size: {int(best_balance['min_cluster_size'])}")
    print("  (highest C_V among configurations with outlier rate < 35%)")
    print(f"  Topics: {int(best_balance['n_topics'])},  "
          f"Outlier: {best_balance['outlier_pct']:.1f}%,  "
          f"C_V: {best_balance['c_v_coherence']:.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="HDBSCAN min_cluster_size sensitivity analysis",
    )
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument(
        "--values", type=str, default="5,8,10,12,15,20",
        help="Comma-separated min_cluster_size values (default: 5,8,10,12,15,20)",
    )
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    args = parser.parse_args()

    config = get_config(args.config)
    from src.utils import setup_logging

    setup_logging(config.system.log_level, config.system.log_file)

    mcs_values = [int(v.strip()) for v in args.values.split(",")]

    logger.info("Sensitivity analysis: min_cluster_size ∈ %s", mcs_values)
    logger.info("All other parameters held constant (UMAP, vectorizer, min_samples=1, etc.)")

    # Run
    df = run_sensitivity(config, mcs_values, force_cpu=args.cpu)

    # Save
    out_dir = config.paths.processed_dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"sensitivity_min_cluster_size_{timestamp}.csv"
    ensure_dir(out_dir)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    logger.info("Results saved: %s", csv_path)

    # Save latest as stable name (overwrites)
    stable_path = out_dir / "sensitivity_min_cluster_size.csv"
    df.to_csv(stable_path, index=False, encoding="utf-8-sig")

    # Save JSON for manuscript integration
    json_path = out_dir / "sensitivity_min_cluster_size.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(df.to_dict(orient="records"), f, indent=2, ensure_ascii=False)
    logger.info("JSON saved: %s", json_path)

    # Print summary
    print_summary(df)

    # Plot
    fig_path = config.paths.figures_dir / "sensitivity_min_cluster_size.png"
    plot_sensitivity(df, fig_path, dpi=config.visualization.dpi)

    logger.info("Sensitivity analysis complete.")


if __name__ == "__main__":
    main()
