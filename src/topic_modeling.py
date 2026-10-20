"""Step 4: Topic Modeling — BERTopic pipeline with SciBERT embeddings,
KeyphraseVectorizer, UMAP dimensionality reduction, and HDBSCAN clustering.

Usage:
    python src/topic_modeling.py              # uses default config.yaml
    python src/topic_modeling.py --config path/to/config.yaml
    python src/topic_modeling.py --cpu         # force CPU even if GPU available
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.config import Config, get_config
from src.utils import Timer, detect_device, ensure_dir

logger = logging.getLogger(__name__)


# =============================================================================
# Data loading
# =============================================================================


def load_texts(config: Config) -> Tuple[pd.DataFrame, List[str]]:
    """Load cleaned data and extract text documents.

    Args:
        config: Validated configuration.

    Returns:
        ``(full_dataframe, list_of_text_strings)``.

    Raises:
        FileNotFoundError: If cleaned data CSV is missing.
    """
    csv_path = config.paths.processed_dir / config.preprocessing.output_file

    # Prefer parquet if available
    pq_path = csv_path.with_suffix(".parquet")
    if pq_path.exists():
        df = pd.read_parquet(pq_path)
        logger.info("Loaded from parquet: %s", pq_path)
    elif csv_path.exists():
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        logger.info("Loaded from CSV: %s", csv_path)
    else:
        raise FileNotFoundError(
            f"Cleaned data not found at {csv_path} or {pq_path}. Run preprocessing first."
        )

    docs = df["text"].fillna("").astype(str).tolist()
    docs = [d.strip() for d in docs]
    docs = [d for d in docs if len(d) > 0]

    # Basic stats
    doc_lengths = [len(d.split()) for d in docs]
    logger.info("Documents: %d", len(docs))
    logger.info("  Avg length: %.0f words (min=%d, max=%d)", np.mean(doc_lengths), min(doc_lengths), max(doc_lengths))
    logger.info("  Total words: %d", sum(doc_lengths))

    return df, docs


# =============================================================================
# Model construction
# =============================================================================


def _add_llm_labeler(
    cfg: "TopicModelingConfig",  # noqa: F821
    representation_model: dict,
) -> None:
    """Add an LLM-based OpenAI representation model for topic labeling.

    Supports any OpenAI-compatible API endpoint (DeepSeek, OpenAI, Ollama, etc.).
    Picks up the API key from the environment variable named in config.

    Args:
        cfg: Topic modeling configuration.
        representation_model: The representation model dict to mutate.
    """
    import os

    from bertopic.representation import OpenAI

    llm = cfg.llm_labeling

    # --- Resolve API key ---
    api_key = os.environ.get(llm.api_key_env, "")
    if not api_key:
        logger.warning(
            "LLM labeling enabled but %s is not set. Skipping LLM labels.",
            llm.api_key_env,
        )
        return

    # --- Resolve base URL ---
    base_url = llm.base_url
    if base_url is None:
        # Provider defaults
        defaults = {
            "deepseek": "https://api.deepseek.com",
            "openai": "https://api.openai.com/v1",
            "ollama": "http://localhost:11434/v1",
        }
        base_url = defaults.get(llm.provider, "https://api.openai.com/v1")

    # --- Build client ---
    try:
        import openai
    except ImportError:
        logger.warning("openai package not installed. Install with: pip install openai")
        return

    client = openai.OpenAI(api_key=api_key, base_url=base_url)

    # --- Custom prompt for scientific topic labeling ---
    sci_prompt = (
        "You will extract a concise, descriptive topic label from scientific "
        "abstracts about mangrove remote sensing.\n\n"
        "Sample texts from this topic:\n[DOCUMENTS]\n"
        "Keywords: [KEYWORDS]\n\n"
        "Based on the information above, extract a short topic label "
        "(3-5 words, using scientific terminology) in the following format:\n"
        "topic: <topic_label>"
    )

    # Add as a secondary representation model (keeps KeyBERTInspired as Main)
    representation_model["LLM"] = OpenAI(
        client,
        model=llm.model,
        prompt=sci_prompt,
        delay_in_seconds=llm.delay_in_seconds,
        nr_docs=llm.nr_docs,
    )
    logger.info(
        "LLM labeling enabled: provider=%s, model=%s, base_url=%s, key=%s",
        llm.provider, llm.model, base_url, llm.api_key_env,
    )


def build_topic_model(config: Config, device: str = "auto") -> "BERTopic":  # noqa: F821
    """Assemble the BERTopic pipeline from config parameters.

    Args:
        config: Validated configuration.
        device: ``"cuda"``, ``"cpu"``, or ``"auto"`` (detect best).

    Returns:
        Configured BERTopic model (not yet fitted).
    """
    from bertopic import BERTopic
    from bertopic.representation import KeyBERTInspired
    from bertopic.vectorizers import ClassTfidfTransformer
    from hdbscan import HDBSCAN
    from keyphrase_vectorizers import KeyphraseCountVectorizer
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP

    cfg = config.topic_modeling
    sys = config.system

    if device == "auto":
        device = detect_device(prefer_gpu=sys.prefer_gpu)

    # --- Embedding model ---
    logger.info("Loading embedding model: %s [device=%s]", cfg.embedding_model, device)
    embedding_model = SentenceTransformer(cfg.embedding_model, device=device)

    # --- UMAP ---
    logger.info(
        "UMAP: n_neighbors=%d, n_components=%d, min_dist=%.2f, metric=%s",
        cfg.umap.n_neighbors, cfg.umap.n_components, cfg.umap.min_dist, cfg.umap.metric,
    )
    umap_model = UMAP(
        n_neighbors=cfg.umap.n_neighbors,
        n_components=cfg.umap.n_components,
        min_dist=cfg.umap.min_dist,
        metric=cfg.umap.metric,
        random_state=cfg.umap.random_state,
        n_jobs=1,
    )

    # --- HDBSCAN ---
    logger.info(
        "HDBSCAN: min_cluster_size=%d, min_samples=%d, metric=%s",
        cfg.hdbscan.min_cluster_size, cfg.hdbscan.min_samples, cfg.hdbscan.metric,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=cfg.hdbscan.min_cluster_size,
        min_samples=cfg.hdbscan.min_samples,
        metric=cfg.hdbscan.metric,
        prediction_data=True,
    )

    # --- Vectorizer ---
    # Backend is pinned via config. "count" always uses CountVectorizer
    # (reproducible, no scispaCy dependency); "spacy" uses
    # KeyphraseCountVectorizer when the spaCy model is available.
    if cfg.vectorizer.backend == "count":
        logger.info(
            "Vectorizer: CountVectorizer (pinned) stop_words=english ngram_range=(1,3) max_df=0.85 min_df=5"
        )
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
            logger.info("Vectorizer: KeyphraseCountVectorizer with spacy=%s", cfg.spacy_model)
            vectorizer_model = KeyphraseCountVectorizer(
                spacy_pipeline=cfg.spacy_model,
                pos_pattern=cfg.vectorizer.pos_pattern,
            )
        else:
            logger.warning(
                "spaCy model '%s' not available — falling back to CountVectorizer. "
                "Install scispaCy: pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz",
                cfg.spacy_model,
            )
            vectorizer_model = CountVectorizer(
                stop_words="english",
                ngram_range=(1, 3),
                max_df=0.85,
                min_df=5,
            )

    # --- Representation ---
    logger.info("Representation: KeyBERTInspired (top_n=%d, nr_repr_docs=%d)", cfg.representation.top_n_words, cfg.representation.nr_repr_docs)
    representation_model = {
        "Main": KeyBERTInspired(
            top_n_words=cfg.representation.top_n_words,
            nr_repr_docs=cfg.representation.nr_repr_docs,
        ),
    }

    # --- LLM-based topic labeling (optional) ---
    if cfg.llm_labeling.enabled:
        _add_llm_labeler(cfg, representation_model)

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

    logger.info("BERTopic model built successfully")
    return model


# =============================================================================
# Model training
# =============================================================================


def fit_model(
    model: "BERTopic",  # noqa: F821
    docs: List[str],
) -> Tuple["BERTopic", np.ndarray, np.ndarray]:  # noqa: F821
    """Fit the BERTopic model on document texts.

    Args:
        model: Configured (unfitted) BERTopic model.
        docs: List of document texts.

    Returns:
        ``(fitted_model, topics_array, probabilities_array)``.
    """
    logger.info("Fitting BERTopic on %d documents ...", len(docs))
    with Timer("BERTopic fit_transform"):
        topics, probs = model.fit_transform(docs)

    # Summary
    n_topics = len(set(topics)) - (1 if -1 in topics else 0)
    n_outliers = int(np.sum(np.array(topics) == -1))
    outlier_pct = 100.0 * n_outliers / len(topics)

    logger.info("Topic modeling complete:")
    logger.info("  Topics found: %d", n_topics)
    logger.info("  Outliers (-1): %d (%.1f%%)", n_outliers, outlier_pct)
    logger.info("  Avg topic size: %.0f docs", (len(topics) - n_outliers) / max(n_topics, 1))

    return model, topics, probs


# =============================================================================
# Coherence / quality metrics
# =============================================================================


def compute_topic_coherence(
    model: "BERTopic",  # noqa: F821
    docs: List[str],
    topics: np.ndarray,
    top_n: int = 10,
) -> Optional[pd.DataFrame]:
    """Compute topic diversity and (optionally) NPMI coherence.

    Args:
        model: Fitted BERTopic model.
        docs: Document texts.
        topics: Topic assignments.
        top_n: Number of top words per topic.

    Returns:
        DataFrame with per-topic coherence metrics, or None if computation fails.
    """
    logger.info("Computing topic coherence metrics ...")

    topic_info = model.get_topic_info()

    # Topic diversity: fraction of unique words across top-N words
    all_words: List[str] = []
    for topic_id in topic_info["Topic"]:
        if topic_id == -1:
            continue
        words = model.get_topic(topic_id)
        if words:
            all_words.extend([w for w, _ in words[:top_n]])

    unique_words = len(set(all_words))
    total_words = len(all_words)
    diversity = unique_words / total_words if total_words > 0 else 0.0
    logger.info("  Topic diversity: %.3f (%d unique / %d total words)", diversity, unique_words, total_words)

    # Try NPMI coherence via gensim (optional)
    try:
        from gensim.models.coherencemodel import CoherenceModel
        from gensim.corpora import Dictionary

        # Tokenize docs with the model's own analyzer so reference tokens
        # align with the (n-gram) topic words.
        try:
            analyzer = model.vectorizer_model.build_analyzer()
        except AttributeError:
            def analyzer(d):
                return d.lower().split()
        tokenized = [analyzer(d) for d in docs]
        tokenized = [t for t in tokenized if len(t) > 0]

        # Get topic word lists (n-gram words kept intact)
        topic_words = []
        for topic_id in sorted(set(topics)):
            if topic_id == -1:
                continue
            words = model.get_topic(topic_id)
            if words:
                topic_words.append([w for w, _ in words[:top_n]])

        if len(topic_words) > 1 and len(tokenized) > 0:
            dictionary = Dictionary(tokenized)
            cm = CoherenceModel(
                topics=topic_words,
                texts=tokenized,
                dictionary=dictionary,
                coherence="c_v",
                topn=top_n,
            )
            coherence_score = cm.get_coherence()
            logger.info("  C_V coherence: %.4f", coherence_score)
        else:
            coherence_score = float("nan")
            logger.info("  C_V coherence: skipped (not enough topics or documents)")

    except ImportError:
        logger.info("  C_V coherence: skipped (gensim not installed)")
        coherence_score = float("nan")
    except Exception as exc:
        logger.warning("  C_V coherence computation failed: %s", exc)
        coherence_score = float("nan")

    # Build summary DataFrame
    return pd.DataFrame({
        "Topic": topic_info["Topic"],
        "Count": topic_info["Count"],
        "Name": topic_info["Name"],
        "diversity": diversity,
        "c_v_coherence": coherence_score,
    })


# =============================================================================
# Comprehensive topic model evaluation
# =============================================================================


def evaluate_topic_model(
    model: "BERTopic",  # noqa: F821
    docs: List[str],
    topics: "np.ndarray",  # noqa: F821
    config: Config,
) -> dict:
    """Run comprehensive topic model quality evaluation.

    Computes four dimensions:
        1. **Topic Coherence** — C_V + NPMI (gensim), per-topic and global.
        2. **Topic Distinctiveness** — pairwise cosine similarity of c-TF-IDF vectors.
        3. **Topic Distribution** — Gini coefficient, entropy, size distribution stats.
        4. **Noise Analysis** — outlier ratio, characteristics of unassigned documents.

    Args:
        model: Fitted BERTopic model.
        docs: Document texts (same order as fitting).
        topics: Topic assignments array.
        config: Validated configuration.

    Returns:
        Dict with keys: ``coherence``, ``distinctiveness``, ``distribution``, ``noise``.
    """
    import json

    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    logger.info("=" * 60)
    logger.info("Comprehensive Topic Model Evaluation")
    logger.info("=" * 60)

    topic_info = model.get_topic_info()
    n_total = len(docs)
    n_outliers = int(topic_info[topic_info["Topic"] == -1]["Count"].values[0]) if -1 in topic_info["Topic"].values else 0
    n_valid_topics = len(topic_info[topic_info["Topic"] != -1])
    n_assigned = n_total - n_outliers

    results: dict = {}

    # -----------------------------------------------------------------
    # 1. TOPIC COHERENCE (C_V + NPMI via gensim)
    # -----------------------------------------------------------------
    logger.info("--- 1. Topic Coherence ---")
    c_v_global = float("nan")
    c_v_per_topic = {}
    npmi_global = float("nan")
    diversity = float("nan")

    # Topic diversity
    all_words: list[str] = []
    for topic_id in topic_info["Topic"]:
        if topic_id == -1:
            continue
        words = model.get_topic(topic_id)
        if words:
            all_words.extend([w for w, _ in words[:10]])
    unique_words = len(set(all_words))
    total_words = len(all_words)
    diversity = unique_words / total_words if total_words > 0 else 0.0
    logger.info("  Topic diversity: %.4f (%d unique / %d total, top-10 words)", diversity, unique_words, total_words)

    # C_V coherence via gensim
    try:
        from gensim.models.coherencemodel import CoherenceModel
        from gensim.corpora import Dictionary

        # Tokenize the reference corpus with the SAME analyzer the model used
        # (lowercase + stop_words + n-gram range). `.lower().split()` keeps
        # punctuation attached and mismatches the model's stopword/n-gram
        # vocabulary, which collapses NPMI to ~0.
        try:
            analyzer = model.vectorizer_model.build_analyzer()
        except AttributeError:
            def analyzer(d):
                return d.lower().split()
        tokenized = [analyzer(d) for d in docs]
        # Filter empty docs
        tokenized = [t for t in tokenized if len(t) > 0]

        topic_words_list = []
        valid_tids = []
        for _, row in topic_info.iterrows():
            tid = int(row["Topic"])
            if tid == -1:
                continue
            words = model.get_topic(tid)
            if words and len(words) >= 5:
                # Keep n-gram topic words intact — they match the vectorizer's
                # n-gram tokens; splitting them breaks alignment.
                word_list = [str(w).strip().lower() for w, _ in words[:10] if str(w).strip()]
                if len(word_list) >= 5:
                    topic_words_list.append(word_list)
                    valid_tids.append(tid)

        if len(topic_words_list) >= 5 and len(tokenized) >= 50:
            dictionary = Dictionary(tokenized)
            # C_V — use 'c_v' measure
            cm_cv = CoherenceModel(
                topics=topic_words_list, texts=tokenized,
                dictionary=dictionary, coherence="c_v", topn=10,
            )
            c_v_global = float(cm_cv.get_coherence())
            # Per-topic C_V
            try:
                c_v_per_topic_vals = cm_cv.get_coherence_per_topic()
                c_v_per_topic = {
                    str(valid_tids[i]): round(float(v), 4)
                    for i, v in enumerate(c_v_per_topic_vals) if i < len(valid_tids)
                }
            except Exception:
                c_v_per_topic = {}
            logger.info("  C_V coherence (global): %.4f", c_v_global)

            # NPMI
            cm_npmi = CoherenceModel(
                topics=topic_words_list, texts=tokenized,
                dictionary=dictionary, coherence="c_npmi", topn=10,
            )
            npmi_global = float(cm_npmi.get_coherence())
            logger.info("  NPMI coherence (global): %.4f", npmi_global)
        else:
            logger.warning("  Not enough topics/documents for C_V — skipped")
    except Exception as exc:
        logger.warning("  C_V/NPMI computation failed: %s", exc)

    results["coherence"] = {
        "c_v_global": round(c_v_global, 4) if not np.isnan(c_v_global) else None,
        "npmi_global": round(npmi_global, 4) if not np.isnan(npmi_global) else None,
        "diversity": round(diversity, 4),
        "c_v_per_topic": c_v_per_topic,
    }

    # -----------------------------------------------------------------
    # 2. TOPIC DISTINCTIVENESS (c-TF-IDF cosine similarity)
    # -----------------------------------------------------------------
    logger.info("--- 2. Topic Distinctiveness ---")
    ctfidf = model.c_tf_idf_.toarray() if hasattr(model.c_tf_idf_, "toarray") else model.c_tf_idf_
    keys_list = list(model.topic_representations_.keys())
    valid_tids = sorted([t for t in model.topic_representations_.keys() if t != -1])
    topic_rows = np.array([ctfidf[keys_list.index(tid)] for tid in valid_tids])

    cos_sim = cosine_similarity(topic_rows)
    # Mask diagonal
    n = cos_sim.shape[0]
    mask = ~np.eye(n, dtype=bool)
    pairwise_sims = cos_sim[mask]

    mean_sim = float(np.mean(pairwise_sims))
    median_sim = float(np.median(pairwise_sims))
    std_sim = float(np.std(pairwise_sims))
    min_sim = float(np.min(pairwise_sims))
    max_sim = float(np.max(pairwise_sims))

    # Fraction of topic pairs with similarity > 0.5 (potential redundancy)
    high_sim_frac = float(np.mean(pairwise_sims > 0.5))

    logger.info("  Topics evaluated: %d", n)
    logger.info("  Mean pairwise cosine similarity: %.4f", mean_sim)
    logger.info("  Median: %.4f  Std: %.4f  Range: [%.4f, %.4f]", median_sim, std_sim, min_sim, max_sim)
    logger.info("  Fraction with sim > 0.5: %.2f%%", high_sim_frac * 100)

    results["distinctiveness"] = {
        "n_topics": n,
        "mean_pairwise_similarity": round(mean_sim, 4),
        "median_pairwise_similarity": round(median_sim, 4),
        "std_pairwise_similarity": round(std_sim, 4),
        "min_similarity": round(min_sim, 4),
        "max_similarity": round(max_sim, 4),
        "fraction_high_similarity": round(high_sim_frac, 4),
    }

    # -----------------------------------------------------------------
    # 3. TOPIC DISTRIBUTION (size distribution, Gini, entropy)
    # -----------------------------------------------------------------
    logger.info("--- 3. Topic Distribution ---")
    topic_sizes = []
    for _, row in topic_info.iterrows():
        tid = int(row["Topic"])
        if tid == -1:
            continue
        topic_sizes.append(int(row["Count"]))

    sizes = np.array(topic_sizes)
    mean_size = float(np.mean(sizes))
    median_size = float(np.median(sizes))
    min_size = int(np.min(sizes))
    max_size = int(np.max(sizes))
    std_size = float(np.std(sizes))

    # Gini coefficient
    sorted_sizes = np.sort(sizes)
    n_sizes = len(sorted_sizes)
    index = np.arange(1, n_sizes + 1)
    gini = float(2 * np.sum(index * sorted_sizes) / (n_sizes * np.sum(sorted_sizes)) - (n_sizes + 1) / n_sizes)

    # Entropy (normalized)
    proportions = sizes / sizes.sum()
    entropy = float(-np.sum(proportions * np.log(proportions + 1e-12)))
    max_entropy = float(np.log(n_sizes))
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

    logger.info("  Topic sizes: mean=%.1f, median=%.1f, min=%d, max=%d, std=%.1f", mean_size, median_size, min_size, max_size, std_size)
    logger.info("  Gini coefficient: %.4f (0=equal, 1=one-dominant)", gini)
    logger.info("  Normalized entropy: %.4f (1=uniform)", normalized_entropy)

    results["distribution"] = {
        "n_topics": n_sizes,
        "total_assigned": int(n_assigned),
        "mean_size": round(mean_size, 2),
        "median_size": round(median_size, 2),
        "min_size": min_size,
        "max_size": max_size,
        "std_size": round(std_size, 2),
        "gini_coefficient": round(gini, 4),
        "normalized_entropy": round(normalized_entropy, 4),
        "size_skewness": round(float((mean_size - median_size) / std_size if std_size > 0 else 0), 4),
    }

    # -----------------------------------------------------------------
    # 4. NOISE ANALYSIS
    # -----------------------------------------------------------------
    logger.info("--- 4. Noise Analysis ---")
    outlier_ratio = n_outliers / n_total if n_total > 0 else 0.0

    # Compare outlier docs to assigned docs in terms of length
    doc_lengths = np.array([len(d.split()) for d in docs])
    topics_arr = np.asarray(topics, dtype=int)
    outlier_mask = topics_arr == -1
    assigned_mask = topics_arr != -1
    outlier_lengths = doc_lengths[outlier_mask] if outlier_mask.any() else np.array([0])
    assigned_lengths = doc_lengths[assigned_mask] if assigned_mask.any() else np.array([0])

    logger.info("  Outlier ratio: %.2f%% (%d/%d)", outlier_ratio * 100, n_outliers, n_total)
    logger.info("  Outlier doc length: mean=%.0f, median=%.0f", np.mean(outlier_lengths), np.median(outlier_lengths))
    logger.info("  Assigned doc length: mean=%.0f, median=%.0f", np.mean(assigned_lengths), np.median(assigned_lengths))
    logger.info("  Length ratio (outlier/assigned): %.3f", np.mean(outlier_lengths) / np.mean(assigned_lengths))

    results["noise"] = {
        "n_outliers": n_outliers,
        "n_total": n_total,
        "outlier_ratio": round(outlier_ratio, 4),
        "n_valid_topics": n_valid_topics,
        "outlier_mean_length": round(float(np.mean(outlier_lengths)), 1),
        "assigned_mean_length": round(float(np.mean(assigned_lengths)), 1),
    }

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Evaluation Summary:")
    logger.info("  Coherence:   C_V=%.4f, NPMI=%.4f, Diversity=%.3f",
                c_v_global if not np.isnan(c_v_global) else -1,
                npmi_global if not np.isnan(npmi_global) else -1,
                diversity)
    logger.info("  Distinct:    mean_cos=%.4f, high_sim=%.1f%%", mean_sim, high_sim_frac * 100)
    logger.info("  Distribution: Gini=%.3f, entropy=%.3f, sizes [%d–%d]", gini, normalized_entropy, min_size, max_size)
    logger.info("  Noise:       %.1f%% outliers, %d valid topics", outlier_ratio * 100, n_valid_topics)

    # Interpretation guidelines
    if not np.isnan(c_v_global):
        if c_v_global > 0.6:
            logger.info("  [GOOD] C_V coherence >0.6")
        elif c_v_global > 0.4:
            logger.info("  [OK] C_V coherence 0.4-0.6")
        else:
            logger.info("  [LOW] C_V coherence <0.4")
    if mean_sim < 0.3:
        logger.info("  [GOOD] Topic distinctiveness — mean sim <0.3")
    elif mean_sim < 0.5:
        logger.info("  [OK] Topic distinctiveness — mean sim 0.3-0.5")
    else:
        logger.info("  [LOW] Topic distinctiveness — topics may overlap too much")
    if outlier_ratio > 0.4:
        logger.info("  [HIGH] Outlier ratio >40%% — consider adjusting clustering parameters")
    elif outlier_ratio > 0.25:
        logger.info("  [OK] Outlier ratio 25-40%% — expected for heterogeneous corpora")

    logger.info("=" * 60)

    # Save evaluation JSON
    eval_path = config.paths.processed_dir / "topic_evaluation.json"
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    logger.info("Evaluation saved: %s", eval_path)

    return results


# =============================================================================
# Saving results
# =============================================================================


def save_results(
    model: "BERTopic",  # noqa: F821
    topics: np.ndarray,
    probs: np.ndarray,
    df: pd.DataFrame,
    config: Config,
) -> Tuple[Path, Path, Path]:
    """Save model, topic info, and document-topic distribution.

    Args:
        model: Fitted BERTopic model.
        topics: Topic assignment array.
        probs: Topic probability matrix.
        df: Original DataFrame (used to create doc_topic_dist).
        config: Validated configuration.

    Returns:
        ``(model_path, topics_info_path, doc_topic_dist_path)``.
    """
    out = config.topic_modeling.output
    model_dir = config.paths.models_dir
    processed_dir = config.paths.processed_dir

    # --- Save BERTopic model ---
    model_path = model_dir / out.model_name
    with Timer("Saving BERTopic model"):
        try:
            model.save(model_path, serialization="safetensors", save_ctfidf=True, save_embedding_model=False)
        except KeyError as exc:
            logger.warning(
                "BERTopic save with safetensors failed (%s), retrying without ct_fidf ...", exc
            )
            try:
                model.save(model_path, serialization="safetensors", save_ctfidf=False, save_embedding_model=False)
            except Exception:
                logger.warning("Still failed, saving with pytorch serialization ...")
                model.save(model_path, serialization="pytorch", save_ctfidf=False, save_embedding_model=False)
    logger.info("Model saved: %s", model_path)

    # --- Save topics info ---
    topic_info = model.get_topic_info()
    topics_info_path = processed_dir / out.topics_info

    # If LLM labeling was used, extract LLM labels as Short_Label
    if config.topic_modeling.llm_labeling.enabled:
        # BERTopic stores additional representation models as columns named
        # after the dict key (e.g. "LLM") or as "Representation_<key>".
        llm_col = None
        for candidate in ["LLM", "OpenAI", "Representation_LLM", "Representation_OpenAI"]:
            if candidate in topic_info.columns:
                llm_col = candidate
                break
        # Fallback: any column starting with "Representation_" that isn't
        # "Representation" or "Representative_Docs"
        if llm_col is None:
            for col in topic_info.columns:
                if col.startswith("Representation_") and col not in ("Representation", "Representative_Docs"):
                    llm_col = col
                    break

        if llm_col is not None:
            import ast
            short_labels = {}
            for _, row in topic_info.iterrows():
                tid = int(row.get("Topic", -1))
                if tid == -1:
                    continue
                try:
                    rep = row[llm_col]
                    if isinstance(rep, str):
                        rep = ast.literal_eval(rep)
                    if isinstance(rep, list) and len(rep) > 0:
                        label = str(rep[0]).strip().replace("topic: ", "")
                        # Clean up common LLM artifacts
                        label = label.strip('"').strip("'").strip()
                        short_labels[tid] = label
                except (ValueError, SyntaxError, KeyError, TypeError):
                    pass
            topic_info["Short_Label"] = topic_info["Topic"].map(short_labels)
            n_labeled = len(short_labels)
            logger.info("LLM labels extracted from column '%s': %d topics labeled", llm_col, n_labeled)
        else:
            logger.warning("LLM labeling enabled but no LLM representation column found in topic_info")

    topic_info.to_csv(topics_info_path, index=False, encoding="utf-8-sig")
    logger.info("Topics info saved: %s (%d topics)", topics_info_path, len(topic_info))

    # --- Save document-topic distribution ---
    doc_topic_path = processed_dir / out.doc_topic_dist
    doc_topic_df = pd.DataFrame({
        "doc_index": range(len(topics)),
        "Topic": topics,
    })
    # Add topic probabilities if available
    if probs is not None and probs.ndim == 2:
        for i in range(min(probs.shape[1], 5)):
            doc_topic_df[f"prob_topic_{i}"] = probs[:, i]

    # Also add PY (year) and text for downstream convenience
    if "PY" in df.columns:
        doc_topic_df["PY"] = df["PY"].values[:len(topics)]
    if "text" in df.columns:
        doc_topic_df["text"] = df["text"].values[:len(topics)]

    doc_topic_df.to_csv(doc_topic_path, index=False, encoding="utf-8-sig")
    logger.info("Document-topic distribution saved: %s", doc_topic_path)

    # --- Save topic word scores (c-TF-IDF) ---
    topic_word_path = processed_dir / out.topic_word_scores
    with Timer("Saving topic word scores"):
        _save_topic_word_scores(model, topic_word_path, top_n=config.topic_modeling.representation.top_n_words)

    return model_path, topics_info_path, doc_topic_path


# =============================================================================
# Visualizations
# =============================================================================


def _truncate_label(name: str, max_words: int = 6, max_chars: int = 55) -> str:
    """Shorten a topic name for display on the barchart y-axis.

    BERTopic names are long strings like
    ``"0_mangrove area_mangrove landscape_mangrove forest_..."``.
    We strip the topic-ID prefix, replace underscores with spaces, and
    truncate to *max_words* words / *max_chars* characters.
    """
    # Strip leading numeric ID (e.g. "0_" or "-1_")
    label = name
    if "_" in label:
        first_sep = label.index("_")
        maybe_id = label[:first_sep]
        if maybe_id.lstrip("-").isdigit():
            label = label[first_sep + 1:]
    label = label.replace("_", " ")
    words = label.split()
    truncated = " ".join(words[:max_words])
    if len(truncated) > max_chars:
        truncated = truncated[:max_chars - 3] + "..."
    return truncated


def _save_topic_word_scores(
    model: "BERTopic",  # noqa: F821
    output_path: Path,
    top_n: int = 20,
) -> None:
    """Export top words per topic with their c-TF-IDF scores to CSV.

    Args:
        model: Fitted BERTopic model.
        output_path: Where to save the CSV.
        top_n: Number of top words to export per topic.
    """
    records = []
    topic_info = model.get_topic_info()

    for _, row in topic_info.iterrows():
        topic_id = row["Topic"]
        if topic_id == -1:
            continue
        words = model.get_topic(topic_id)
        if not words:
            continue
        for rank, (word, score) in enumerate(words[:top_n], start=1):
            records.append({
                "Topic": int(topic_id),
                "Rank": rank,
                "Word": str(word),
                "c_TFIDF_Score": float(score),
            })

    if records:
        df = pd.DataFrame(records)
        ensure_dir(output_path.parent)
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        logger.info("Topic word scores saved: %s (%d rows)", output_path, len(df))
    else:
        logger.warning("No topic word scores to save")


def _plot_topic_word_scores(
    model: "BERTopic",  # noqa: F821
    output_path: Path,
    top_n: int = 10,
    n_topics: int = 15,
    dpi: int = 300,
) -> None:
    """Plot a grid of horizontal bar charts showing c-TF-IDF word scores per topic.

    Args:
        model: Fitted BERTopic model.
        output_path: Where to save the PNG.
        top_n: Number of top words per topic.
        n_topics: Number of topics to display (excludes outliers).
        dpi: Resolution.
    """
    import matplotlib.pyplot as plt

    topic_info = model.get_topic_info()
    topic_info = topic_info[topic_info["Topic"] != -1].head(n_topics)

    n = len(topic_info)
    if n == 0:
        logger.warning("No topics to plot for word scores")
        return

    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.5, rows * 3.2))
    axes = np.atleast_1d(axes).flatten()

    cmap = plt.cm.viridis

    for idx, (_, row) in enumerate(topic_info.iterrows()):
        ax = axes[idx]
        topic_id = row["Topic"]
        words = model.get_topic(topic_id)
        if not words:
            ax.set_visible(False)
            continue

        words = words[:top_n]
        labels = [w for w, _ in words][::-1]
        scores = [s for _, s in words][::-1]
        colors = cmap(np.linspace(0.2, 0.8, len(labels)))

        ax.barh(range(len(labels)), scores, color=colors, edgecolor="white", height=0.6)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("c-TF-IDF", fontsize=9)
        ax.set_title(f"Topic {topic_id}", fontsize=10, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", linestyle=":", alpha=0.4)

    # Hide unused subplots
    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Topic Word Scores (c-TF-IDF)", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def generate_visualizations(
    model: "BERTopic",  # noqa: F821
    config: Config,
    docs: List[str] | None = None,
) -> List[Path]:
    """Generate and save BERTopic visualizations.

    Args:
        model: Fitted BERTopic model.
        config: Validated configuration.

    Returns:
        List of saved file paths.
    """
    fig_dir = config.paths.figures_dir
    out_cfg = config.topic_modeling.output
    saved: List[Path] = []

    with Timer("Generating topic visualizations"):
        # 1. Interactive topic visualization (HTML)
        try:
            html_path = fig_dir / out_cfg.visualize_html
            fig = model.visualize_topics()
            fig.write_html(str(html_path))
            saved.append(html_path)
            logger.info("  Interactive topics: %s", html_path)
        except Exception as exc:
            logger.warning("  visualize_topics failed: %s", exc)

        # ARCHIVED: topic_barchart
        # try:
        #     barchart_path = fig_dir / "topic_barchart.png"
        #     _plot_topic_barchart(model, barchart_path, top_n=20,
        #                          dpi=config.visualization.dpi)
        #     saved.append(barchart_path)
        #     logger.info("  Barchart: %s", barchart_path)
        # except Exception as exc:
        #     logger.warning("  visualize_barchart failed: %s", exc)

        # 3. Topic hierarchy — multi-level
        n_topics = len(model.get_topic_info()) - 1  # exclude -1
        if n_topics > 5:
            # Compute hierarchical topic tree if docs are available
            hier_df = None
            if docs is not None:
                try:
                    hier_df = model.hierarchical_topics(docs)
                    logger.info("  Hierarchical topic tree: %d merge steps", len(hier_df))
                except Exception as exc:
                    logger.warning("  hierarchical_topics computation failed: %s", exc)

            # 3a. Original BERTopic dendrogram (always generated)
            try:
                hierarchy_path = fig_dir / "topic_hierarchy.png"
                if hier_df is not None:
                    fig = model.visualize_hierarchy(
                        hierarchical_topics=hier_df, custom_labels=True,
                    )
                else:
                    fig = model.visualize_hierarchy(custom_labels=True)
                fig.write_image(str(hierarchy_path), scale=2)
                saved.append(hierarchy_path)
                logger.info("  Hierarchy: %s", hierarchy_path)
            except Exception as exc:
                logger.warning("  visualize_hierarchy failed: %s", exc)

        # 4. Topic similarity heatmap
        if n_topics > 2:
            try:
                heatmap_path = fig_dir / "topic_heatmap.png"
                fig = model.visualize_heatmap(use_ctfidf=True)
                fig.write_image(str(heatmap_path), scale=2)
                saved.append(heatmap_path)
                logger.info("  Heatmap: %s", heatmap_path)
            except Exception as exc:
                logger.warning("  visualize_heatmap failed: %s", exc)

        # 5. Topic word scores (c-TF-IDF) plot
        try:
            word_scores_path = fig_dir / out_cfg.topic_word_scores_plot
            _plot_topic_word_scores(model, word_scores_path, top_n=12,
                                    n_topics=15, dpi=config.visualization.dpi)
            saved.append(word_scores_path)
            logger.info("  Word scores: %s", word_scores_path)
        except Exception as exc:
            logger.warning("  topic_word_scores plot failed: %s", exc)

        # 6. Treemap
        try:
            treemap_path = fig_dir / "topic_treemap.html"
            _plot_treemap(model, treemap_path, top_n=25)
            saved.append(treemap_path)
            logger.info("  Treemap: %s", treemap_path)
        except Exception as exc:
            logger.warning("  treemap failed: %s", exc)

    logger.info("Saved %d visualizations to %s", len(saved), fig_dir)
    return saved


# =============================================================================
# Treemap (Hu et al. 2025 Fig. 6 style)
# =============================================================================


def _plot_treemap(
    model: "BERTopic",  # noqa: F821
    output_path: Path,
    top_n: int = 25,
) -> None:
    """Interactive treemap showing relative popularity of topics.

    Inspired by Hu et al. (2025) Fig. 6. Rectangle area is proportional to
    topic document count; color encodes the same metric.
    """
    info = model.get_topic_info()
    info = info[info["Topic"] != -1].head(top_n).copy()
    info = info.sort_values("Count", ascending=False).reset_index(drop=True)

    if len(info) == 0:
        return

    # Build labels and hover text
    labels = []
    parents = []
    values = []
    hover_texts = []
    for _, row in info.iterrows():
        tid = int(row["Topic"])
        count = int(row["Count"])
        name = _truncate_label(str(row["Name"]), max_words=4, max_chars=30)
        labels.append(f"T{tid}: {name}")
        parents.append("Topics")
        values.append(count)
        # Get top words
        words = model.get_topic(tid)
        word_str = ", ".join([w for w, _ in words[:5]]) if words else ""
        hover_texts.append(f"Topic {tid}<br>Count: {count}<br>Top words: {word_str}")

    # Add root node
    labels.insert(0, "Topics")
    parents.insert(0, "")
    values.insert(0, sum(values[1:]))
    hover_texts.insert(0, f"Total topics: {len(info)}")

    fig = go.Figure(go.Treemap(
        labels=labels,
        parents=parents,
        values=values,
        customdata=hover_texts,
        hovertemplate="%{customdata}<extra></extra>",
        textinfo="label+value+percent parent",
        textfont=dict(size=11),
        marker=dict(
            colorscale="Viridis",
            color=values,
            line=dict(width=1.5, color="white"),
        ),
    ))

    fig.update_layout(
        title={
            "text": "Topic Relative Popularity (Treemap)",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 16},
        },
        template="plotly_white",
        height=650,
        width=950,
        margin=dict(l=20, r=20, t=60, b=20),
    )

    ensure_dir(output_path.parent)
    fig.write_html(str(output_path))
    logger.info("Topic treemap saved: %s", output_path)


# =============================================================================
# Main entry point
# =============================================================================


def run_topic_modeling(config: Config, force_cpu: bool = False) -> None:
    """Run the complete topic modeling pipeline.

    Args:
        config: Validated configuration.
        force_cpu: If True, use CPU even if GPU is available.
    """
    # 1. Load data
    with Timer("Loading documents"):
        df, docs = load_texts(config)

    if len(docs) < config.topic_modeling.hdbscan.min_cluster_size:
        raise ValueError(
            f"Only {len(docs)} documents — need at least "
            f"{config.topic_modeling.hdbscan.min_cluster_size} for HDBSCAN clustering."
        )

    # 2. Build model
    device = "cpu" if force_cpu else "auto"
    model = build_topic_model(config, device=device)

    # 3. Fit
    model, topics, probs = fit_model(model, docs)

    # 4. Compute coherence
    coherence_df = None
    if config.topic_modeling.compute_coherence:
        coherence_df = compute_topic_coherence(model, docs, topics)
        if coherence_df is not None:
            coherence_path = config.paths.processed_dir / "topic_coherence.csv"
            coherence_df.to_csv(coherence_path, index=False, encoding="utf-8-sig")
            logger.info("Coherence metrics saved: %s", coherence_path)

    # 4b. Comprehensive topic model evaluation
    with Timer("Comprehensive model evaluation"):
        _ = evaluate_topic_model(model, docs, topics, config)

    # 5. Save results
    save_results(model, topics, probs, df, config)

    # 6. Generate visualizations
    generate_visualizations(model, config, docs=docs)

    logger.info("Topic modeling step complete.")


def main() -> None:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Step 4: Topic Modeling")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if GPU available")
    args = parser.parse_args()

    config = get_config(args.config)
    from src.utils import setup_logging

    setup_logging(config.system.log_level, config.system.log_file)

    try:
        run_topic_modeling(config, force_cpu=args.cpu)
    except Exception:
        logger.exception("Topic modeling failed")
        raise


if __name__ == "__main__":
    main()
