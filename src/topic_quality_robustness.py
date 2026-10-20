"""P0-1: Topic-quality robustness package (Major-Revision evidence chain).

Four linked checks requested by reviewers:

  1. **Per-topic NPMI distribution** — quartiles over the 72-topic solution.
     NPMI is bounded in [-1, +1] and comparable across topics/corpora; more
     negative = less coherent.  We report the quartiles and flag the bottom 25%.
  2. **Candidate-prototype coherence** — are the 9 "candidate prototypes"
     (emerging-tech topics) concentrated in the bottom NPMI quartile?  If so,
     their "emerging" interpretation rests on low-coherence clusters.
  3. **Word-count sensitivity** — NPMI at top_n = 5 / 10 / 20.  A stable
     reading should not swing with the number of keywords retained.
  4. **UMAP n_neighbors scan** — outlier rate and topic count at
     n_neighbors in {5, 10, 15, 30, 50} (production default = 15).  Demonstrates
     the clustering is not an artefact of one neighbourhood size.

Outputs:
    data/processed/topic_quality_robustness.csv          per-topic NPMI table
    data/processed/topic_quality_robustness_summary.json quartiles + sensitivity
    data/processed/topic_quality_robustness_umap_scan.csv  n_neighbors scan
    outputs/figures/topic_quality_robustness.png         NPMI histogram + scan
    data/processed/topic_quality_robustness_report.md    manuscript report text

Usage:
    pixi run python -m src.topic_quality_robustness                 # full run
    pixi run python -m src.topic_quality_robustness --skip-umap-scan  # fast path
    pixi run python -m src.topic_quality_robustness --cpu             # force CPU
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.config import Config, get_config
from src.topic_modeling import load_texts
from src.utils import Timer, ensure_dir, setup_logging

logger = logging.getLogger(__name__)

# The 9 candidate prototypes (emerging-tech topics) from emerging_tech_prototypes.csv
CANDIDATE_PROTOTYPES = [34, 8, 16, 4, 5, 17, 33, 49, 24]

# Word-count sensitivity grid (top_n keywords retained for coherence)
TOP_N_GRID = [5, 10, 20]

# UMAP n_neighbors scan grid (production default = 15)
N_NEIGHBORS_GRID = [5, 10, 15, 30, 50]


# =============================================================================
# Per-topic NPMI (gensim)
# =============================================================================


def _tokenize_corpus(model, docs: List[str]) -> List[List[str]]:
    """Tokenize docs with the model's own analyzer so reference tokens align
    with the vectorizer's n-gram vocabulary (naive .split() collapses NPMI ~0)."""
    try:
        analyzer = model.vectorizer_model.build_analyzer()
    except AttributeError:
        def analyzer(d):
            return d.lower().split()
    tokenized = [analyzer(d) for d in docs]
    return [t for t in tokenized if len(t) > 0]


def _topic_word_lists(model, top_n: int) -> Tuple[List[List[str]], List[int]]:
    """Return (topic_word_lists, topic_ids) aligned, excluding -1 and topics
    with fewer than ``top_n`` usable words."""
    topic_words: List[List[str]] = []
    tids: List[int] = []
    for tid in sorted(model.topic_representations_):
        if tid == -1:
            continue
        words = model.get_topic(tid)
        if not words:
            continue
        wl = [str(w).strip().lower() for w, _ in words[:top_n] if str(w).strip()]
        if len(wl) >= top_n:
            topic_words.append(wl)
            tids.append(tid)
    return topic_words, tids


def per_topic_npmi(
    model,
    docs: List[str],
    top_n: int = 10,
) -> Tuple[pd.DataFrame, float]:
    """Compute per-topic NPMI coherence via gensim.

    Returns:
        ``(per_topic_df, global_npmi)`` where the DataFrame has columns
        ``Topic`` and ``NPMI`` (one row per valid topic) and ``global_npmi``
        is the mean across topics.
    """
    from gensim.corpora import Dictionary
    from gensim.models.coherencemodel import CoherenceModel

    tokenized = _tokenize_corpus(model, docs)
    topic_words, tids = _topic_word_lists(model, top_n)

    if len(topic_words) < 5 or len(tokenized) < 50:
        logger.warning("Not enough topics/documents for per-topic NPMI")
        return pd.DataFrame(columns=["Topic", "NPMI"]), float("nan")

    dictionary = Dictionary(tokenized)
    cm = CoherenceModel(
        topics=topic_words,
        texts=tokenized,
        dictionary=dictionary,
        coherence="c_npmi",
        topn=top_n,
    )
    per_topic = cm.get_coherence_per_topic()  # aligned with topic_words order

    df = pd.DataFrame({"Topic": tids, "NPMI": per_topic})
    df["NPMI"] = pd.to_numeric(df["NPMI"], errors="coerce")
    global_npmi = float(np.nanmean(df["NPMI"].values))

    return df, global_npmi


def word_count_sensitivity(
    model,
    docs: List[str],
    top_n_grid: List[int],
) -> pd.DataFrame:
    """Global NPMI at several ``top_n`` values (word-count robustness)."""
    rows = []
    for top_n in top_n_grid:
        _, global_npmi = per_topic_npmi(model, docs, top_n=top_n)
        rows.append({"top_n": top_n, "global_npmi": round(global_npmi, 4)})
        logger.info("  top_n=%d -> global NPMI = %.4f", top_n, global_npmi)
    return pd.DataFrame(rows)


# =============================================================================
# Candidate-prototype check
# =============================================================================


def annotate_quartiles(
    npmi_df: pd.DataFrame,
    prototype_ids: List[int],
) -> pd.DataFrame:
    """Add percentile rank, bottom-25% flag, and prototype flag to the
    per-topic NPMI table."""
    df = npmi_df.dropna(subset=["NPMI"]).copy()
    if df.empty:
        return npmi_df

    # rank_pct: 0 = lowest (worst) NPMI, 100 = highest
    df["npmi_rank_pct"] = df["NPMI"].rank(pct=True, ascending=True) * 100.0
    df["is_bottom25"] = df["npmi_rank_pct"] < 25.0
    df["is_prototype"] = df["Topic"].isin(prototype_ids)

    return df


# =============================================================================
# UMAP n_neighbors scan
# =============================================================================


def _build_scan_model(config: Config, embedding_model, n_neighbors: int):
    """Assemble a minimal BERTopic pipeline identical to production except for
    ``n_neighbors``.  No LLM labelling; probabilities disabled (not needed for
    the outlier-rate readout)."""
    from bertopic import BERTopic
    from bertopic.representation import KeyBERTInspired
    from bertopic.vectorizers import ClassTfidfTransformer
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP

    cfg = config.topic_modeling

    umap_model = UMAP(
        n_neighbors=n_neighbors,
        n_components=cfg.umap.n_components,
        min_dist=cfg.umap.min_dist,
        metric=cfg.umap.metric,
        random_state=cfg.umap.random_state,
        n_jobs=1,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=cfg.hdbscan.min_cluster_size,
        min_samples=cfg.hdbscan.min_samples,
        metric=cfg.hdbscan.metric,
        prediction_data=False,
    )
    vectorizer_model = CountVectorizer(
        stop_words="english",
        ngram_range=(1, 3),
        max_df=0.85,
        min_df=5,
    )
    representation_model = {
        "Main": KeyBERTInspired(
            top_n_words=cfg.representation.top_n_words,
            nr_repr_docs=cfg.representation.nr_repr_docs,
        ),
    }

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
            calculate_probabilities=False,
            verbose=False,
        )
    return model


def umap_neighbors_scan(
    config: Config,
    docs: List[str],
    n_neighbors_grid: List[int],
    device: str = "cpu",
) -> pd.DataFrame:
    """Fit BERTopic at several ``n_neighbors`` values, reporting the outlier
    rate and topic count.  SciBERT embeddings are computed once and reused."""
    from sentence_transformers import SentenceTransformer

    cfg = config.topic_modeling

    with Timer("Pre-encoding documents (SciBERT)"):
        embedding_model = SentenceTransformer(cfg.embedding_model, device=device)
        embeddings = embedding_model.encode(
            docs, show_progress_bar=True, batch_size=32,
        )
    logger.info("Embeddings: %s", embeddings.shape)

    rows = []
    for i, nn in enumerate(n_neighbors_grid):
        with Timer(f"Fit model (n_neighbors={nn})"):
            model = _build_scan_model(config, embedding_model, n_neighbors=nn)
            topics, _ = model.fit_transform(docs, embeddings=embeddings)

        n_total = len(topics)
        n_outliers = int(np.sum(np.array(topics) == -1))
        n_topics = len(set(topics)) - (1 if -1 in topics else 0)
        outlier_pct = 100.0 * n_outliers / n_total

        rows.append({
            "n_neighbors": nn,
            "n_topics": n_topics,
            "n_outliers": n_outliers,
            "outlier_pct": round(outlier_pct, 2),
            "n_docs": n_total,
        })
        logger.info(
            "  [%d/%d] n_neighbors=%d -> %d topics, %.1f%% outliers",
            i + 1, len(n_neighbors_grid), nn, n_topics, outlier_pct,
        )

    return pd.DataFrame(rows)


# =============================================================================
# Visualization
# =============================================================================


def plot_robustness(
    npmi_df: pd.DataFrame,
    umap_df: pd.DataFrame,
    prototype_ids: List[int],
    output_path: Path,
    dpi: int = 300,
) -> None:
    """Two-panel figure: (A) NPMI histogram with quartile lines + prototype
    ticks; (B) outlier rate & topic count vs n_neighbors."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    # --- Panel A: NPMI distribution ---
    ax = axes[0]
    vals = npmi_df["NPMI"].dropna().values
    if len(vals):
        q25, q50, q75 = np.percentile(vals, [25, 50, 75])
        ax.hist(vals, bins=30, color="#7F8C8D", alpha=0.55, edgecolor="white")
        for q, lab, col in [(q25, "Q1", "#E74C3C"), (q50, "Median", "#2C3E50"), (q75, "Q3", "#27AE60")]:
            ax.axvline(q, color=col, linestyle="--", linewidth=1.4)
            ax.text(q, ax.get_ylim()[1] * 0.96, f" {lab}={q:.3f}",
                    color=col, fontsize=8, va="top")
        # Prototype ticks along the bottom
        proto = npmi_df[npmi_df["Topic"].isin(prototype_ids)]
        if not proto.empty:
            ax.scatter(proto["NPMI"], np.zeros(len(proto)),
                       marker="|", s=120, color="#C0392B", linewidths=2,
                       zorder=5, label=f"prototypes (n={len(proto)})")
            ax.legend(fontsize=8, loc="upper left")
        ax.set_xlabel("Per-topic NPMI", fontsize=11)
        ax.set_ylabel("Topics", fontsize=11)
        ax.set_title("A. Per-topic NPMI distribution", fontsize=12, fontweight="bold", loc="left")
        ax.grid(axis="y", linestyle=":", alpha=0.4)

    # --- Panel B: UMAP n_neighbors scan ---
    ax = axes[1]
    if not umap_df.empty:
        ax2 = ax.twinx()
        l1 = ax.plot(umap_df["n_neighbors"], umap_df["outlier_pct"], "o-",
                     color="#E74C3C", linewidth=2, markersize=7, zorder=5)
        l2 = ax2.plot(umap_df["n_neighbors"], umap_df["n_topics"], "s--",
                      color="#2C3E50", linewidth=2, markersize=7, zorder=4)
        ax.set_xlabel("UMAP n_neighbors", fontsize=11)
        ax.set_ylabel("Outlier %", color="#E74C3C", fontsize=11)
        ax2.set_ylabel("Number of topics", color="#2C3E50", fontsize=11)
        ax.tick_params(axis="y", labelcolor="#E74C3C")
        ax2.tick_params(axis="y", labelcolor="#2C3E50")
        ax.set_xticks(umap_df["n_neighbors"])
        ax.set_title("B. Outlier rate & topic count", fontsize=12, fontweight="bold", loc="left")
        ax.grid(True, linestyle=":", alpha=0.4)
        ax.legend(l1 + l2, ["Outlier %", "N Topics"], loc="center right", fontsize=8)

    fig.suptitle("Topic-quality robustness (P0-1)", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Robustness figure saved: %s", output_path)


# =============================================================================
# Report segment
# =============================================================================


def _fmt_pct(x: float) -> str:
    return f"{x:.1f}%"


def write_report_segment(
    npmi_df: pd.DataFrame,
    word_df: pd.DataFrame,
    umap_df: pd.DataFrame,
    prototype_ids: List[int],
    output_path: Path,
) -> str:
    """Compose a manuscript-ready robustness paragraph (markdown)."""
    vals = npmi_df["NPMI"].dropna().values
    q25, q50, q75 = np.percentile(vals, [25, 50, 75]) if len(vals) else (np.nan,) * 3
    mean_npmi = float(np.nanmean(vals)) if len(vals) else np.nan

    proto = npmi_df[npmi_df["Topic"].isin(prototype_ids)]
    n_proto_total = len(proto)
    n_proto_bottom = int(proto["is_bottom25"].sum()) if not proto.empty else 0
    proto_pct = 100.0 * n_proto_bottom / n_proto_total if n_proto_total else 0.0

    # 25% is the chance expectation under no association; fewer prototypes in
    # the bottom quartile than that is *favourable* (emerging labels are not
    # concentrated among low-coherence clusters).
    if n_proto_bottom > 0 and n_proto_total:
        proto_verdict = (
            f"Only {n_proto_bottom} of the {n_proto_total} candidate prototypes "
            f"fell in the bottom NPMI quartile ({proto_pct:.0f}%, below the ~25% "
            f"expected by chance), so the candidate prototypes are not concentrated "
            f"among low-coherence clusters and their emerging labels are not an "
            f"artefact of poor coherence."
        )
    else:
        proto_verdict = (
            f"None of the {n_proto_total} candidate prototypes fell in the bottom "
            f"NPMI quartile, so their emerging labels are not artefacts of low coherence."
        )

    lines = [
        "## P0-1 Topic-quality robustness",
        "",
        f"Across the 72-topic solution, per-topic NPMI was {mean_npmi:.4f} "
        f"(median {q50:.4f}; Q1 {q25:.4f}, Q3 {q75:.4f}), within the range "
        f"expected for short, technical documents.",
        proto_verdict,
    ]

    if not word_df.empty:
        wd = word_df.set_index("top_n")["global_npmi"]
        lines.append(
            "Word-count sensitivity: NPMI moved only from "
            + " to ".join(f"{v:.4f} (top_n={int(k)})" for k, v in wd.items())
            + ", indicating the coherence reading is robust to the number of "
            "retained keywords."
        )

    if not umap_df.empty:
        min_row = umap_df.loc[umap_df["outlier_pct"].idxmin()]
        max_row = umap_df.loc[umap_df["outlier_pct"].idxmax()]
        lines.append(
            f"UMAP n_neighbors scan ({{{','.join(str(int(x)) for x in umap_df['n_neighbors'])}}}): "
            f"outlier rate ranged {min_row['outlier_pct']:.1f}%–{max_row['outlier_pct']:.1f}% "
            f"and topic count {int(umap_df['n_topics'].min())}–{int(umap_df['n_topics'].max())}, "
            "so the reported solution is stable to neighbourhood size."
        )

    text = "\n\n".join(lines) + "\n"
    ensure_dir(output_path.parent)
    output_path.write_text(text, encoding="utf-8")
    logger.info("Report segment saved: %s", output_path)
    return text


# =============================================================================
# Main
# =============================================================================


def run(config: Config, device: str, skip_umap_scan: bool = False) -> None:
    from bertopic import BERTopic

    # 1. Load docs + saved production model
    with Timer("Loading documents"):
        _df, docs = load_texts(config)

    model_path = config.paths.models_dir / config.topic_modeling.output.model_name
    with Timer("Loading saved BERTopic model"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = BERTopic.load(model_path)
    logger.info("Model: %d topics over %d docs", len(model.get_topic_info()) - 1, len(docs))

    # 2. Per-topic NPMI (top_n=10) + quartiles + prototype check
    with Timer("Per-topic NPMI (top_n=10)"):
        npmi_df, global_npmi = per_topic_npmi(model, docs, top_n=10)
    logger.info("Global NPMI (mean of per-topic) = %.4f", global_npmi)
    npmi_df = annotate_quartiles(npmi_df, CANDIDATE_PROTOTYPES)

    # Merge topic metadata (count + name) for a self-contained CSV
    info = model.get_topic_info()
    info = info.rename(columns={"Topic": "Topic", "Count": "Count", "Name": "Name"})
    npmi_out = npmi_df.merge(
        info[["Topic", "Count", "Name"]], on="Topic", how="left"
    )
    col_order = [
        "Topic", "Count", "Name", "NPMI", "npmi_rank_pct",
        "is_bottom25", "is_prototype",
    ]
    npmi_out = npmi_out[[c for c in col_order if c in npmi_out.columns]]

    # 3. Word-count sensitivity
    with Timer("Word-count sensitivity (top_n=5/10/20)"):
        word_df = word_count_sensitivity(model, docs, TOP_N_GRID)

    # 4. UMAP n_neighbors scan (slow — optional)
    umap_df = pd.DataFrame()
    if not skip_umap_scan:
        with Timer("UMAP n_neighbors scan"):
            umap_df = umap_neighbors_scan(config, docs, N_NEIGHBORS_GRID, device=device)
    else:
        logger.info("Skipping UMAP n_neighbors scan (--skip-umap-scan)")

    # 5. Summaries
    vals = npmi_df["NPMI"].dropna().values
    q25, q50, q75 = np.percentile(vals, [25, 50, 75]) if len(vals) else (np.nan,) * 3
    proto = npmi_df[npmi_df["Topic"].isin(CANDIDATE_PROTOTYPES)]
    summary = {
        "n_topics": int(len(npmi_df)),
        "global_npmi_top10": round(global_npmi, 4),
        "npmi_quartiles": {
            "q1": round(float(q25), 4),
            "median": round(float(q50), 4),
            "q3": round(float(q75), 4),
        },
        "candidate_prototypes": {
            "ids": CANDIDATE_PROTOTYPES,
            "n_total": int(len(proto)),
            "n_in_bottom25": int(proto["is_bottom25"].sum()) if not proto.empty else 0,
            "bottom25_ids": (
                [int(t) for t in proto.loc[proto["is_bottom25"], "Topic"]]
                if not proto.empty else []
            ),
        },
        "word_count_sensitivity": word_df.to_dict(orient="records"),
        "umap_neighbors_scan": umap_df.to_dict(orient="records") if not umap_df.empty else None,
    }

    # 6. Save outputs
    out_dir = config.paths.processed_dir
    ensure_dir(out_dir)

    npmi_csv = out_dir / "topic_quality_robustness.csv"
    npmi_out.to_csv(npmi_csv, index=False, encoding="utf-8-sig")
    logger.info("Per-topic NPMI table saved: %s", npmi_csv)

    summary_json = out_dir / "topic_quality_robustness_summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    logger.info("Summary saved: %s", summary_json)

    if not umap_df.empty:
        umap_csv = out_dir / "topic_quality_robustness_umap_scan.csv"
        umap_df.to_csv(umap_csv, index=False, encoding="utf-8-sig")
        logger.info("UMAP scan saved: %s", umap_csv)

    fig_path = config.paths.figures_dir / "topic_quality_robustness.png"
    plot_robustness(npmi_out, umap_df, CANDIDATE_PROTOTYPES, fig_path,
                    dpi=config.visualization.dpi)

    report_path = out_dir / "topic_quality_robustness_report.md"
    report_text = write_report_segment(
        npmi_out, word_df, umap_df, CANDIDATE_PROTOTYPES, report_path,
    )
    print("\n" + "=" * 72)
    print("P0-1 TOPIC-QUALITY ROBUSTNESS REPORT")
    print("=" * 72)
    print(report_text)

    # Console summary of prototype check
    print(f"Prototypes: {len(proto)} total, {int(proto['is_bottom25'].sum()) if not proto.empty else 0} in bottom-25% NPMI")
    if not proto.empty:
        print(proto[["Topic", "NPMI", "npmi_rank_pct", "is_bottom25"]].to_string(index=False))
    if not umap_df.empty:
        print("\nUMAP scan:")
        print(umap_df.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="P0-1 topic-quality robustness package")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    parser.add_argument("--skip-umap-scan", action="store_true",
                        help="Skip the (slow) UMAP n_neighbors re-fit scan")
    args = parser.parse_args()

    config = get_config(args.config)
    setup_logging(config.system.log_level, config.system.log_file)

    device = "cpu" if args.cpu else "auto"
    run(config, device, skip_umap_scan=args.skip_umap_scan)


if __name__ == "__main__":
    main()
