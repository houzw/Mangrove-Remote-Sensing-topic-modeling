"""BERTopic 词序重排向量器对比实验 — 原始 vs Sorted N-gram.

在同一数据集上分别训练两个 BERTopic 模型（仅 vectorizer 不同），
从六个维度进行系统对比：

  1. 聚类质量     — 主题数量、离群率、主题大小分布
  2. 语义连贯性   — C_V / NPMI 一致性、主题多样性
  3. 主题区分度   — c-TF-IDF 向量 pairwise 余弦相似度
  4. 词序冗余消除 — "X Y" vs "Y X" 颠倒对出现次数
  5. 关键词质量   — 前沿主题 (emerging) 的 c-TF-IDF 关键词对比
  6. 雏形检测     — 两阶段算法在新模型上的检测结果

Usage:
    pixi run python -m src.compare_vectorizers
    pixi run python -m src.compare_vectorizers --skip-training  # 仅对比已有模型
"""

from __future__ import annotations

import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.config import get_config
from src.utils import Timer, ensure_dir, setup_logging

logger = logging.getLogger(__name__)

# =============================================================================
# 模型训练
# =============================================================================


def train_sorted_ngram_model(config, docs, device="cpu") -> "BERTopic":  # noqa: F821
    """使用词序重排向量器训练新的 BERTopic 模型。

    与原始模型共享所有参数 (embedding, UMAP, HDBSCAN, representation)，
    仅将 vectorizer_model 替换为 sorted_ngram_vectorizer。

    Args:
        config: 配置对象。
        docs: 文档文本列表。
        device: 计算设备。

    Returns:
        已拟合的 BERTopic 模型。
    """
    from bertopic import BERTopic
    from bertopic.representation import KeyBERTInspired
    from bertopic.vectorizers import ClassTfidfTransformer
    from hdbscan import HDBSCAN
    from sentence_transformers import SentenceTransformer
    from umap import UMAP

    from src.sorted_ngram_vectorizer import build_sorted_ngram_vectorizer

    cfg = config.topic_modeling

    # --- Embedding model (与原始相同) ---
    logger.info("加载嵌入模型: %s [device=%s]", cfg.embedding_model, device)
    embedding_model = SentenceTransformer(cfg.embedding_model, device=device)

    # --- UMAP (与原始相同) ---
    umap_model = UMAP(
        n_neighbors=cfg.umap.n_neighbors,
        n_components=cfg.umap.n_components,
        min_dist=cfg.umap.min_dist,
        metric=cfg.umap.metric,
        random_state=cfg.umap.random_state,
        n_jobs=1,
    )

    # --- HDBSCAN (与原始相同) ---
    hdbscan_model = HDBSCAN(
        min_cluster_size=cfg.hdbscan.min_cluster_size,
        min_samples=cfg.hdbscan.min_samples,
        metric=cfg.hdbscan.metric,
        prediction_data=True,
    )

    # --- ★ 唯一差异: Sorted N-gram Vectorizer ---
    logger.info("向量器: Sorted N-gram Analyzer (1/2/3-gram, 词序重排 + 领域停用词)")
    vectorizer_model = build_sorted_ngram_vectorizer(
        max_df=0.80,
        min_df=3,
        max_features=8000,
    )

    # --- Representation (与原始相同) ---
    representation_model = {
        "Main": KeyBERTInspired(
            top_n_words=cfg.representation.top_n_words,
            nr_repr_docs=cfg.representation.nr_repr_docs,
        ),
    }

    # --- 组装 ---
    import warnings
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

    logger.info("BERTopic (Sorted N-gram) 模型构建完成。开始拟合 %d 篇文档...", len(docs))
    with Timer("BERTopic fit_transform (Sorted N-gram)"):
        topics, probs = model.fit_transform(docs)

    n_topics = len(set(topics)) - (1 if -1 in topics else 0)
    n_outliers = int(np.sum(np.array(topics) == -1))
    logger.info("拟合完成: %d 个主题, %.1f%% 离群率", n_topics, 100 * n_outliers / len(topics))

    return model, topics, probs


# =============================================================================
# 对比指标计算
# =============================================================================


def _flatten_topic_keywords(model, top_n=20) -> Dict[int, List[str]]:
    """提取每个主题的 top-N 关键词（仅单词，拆开短语）。"""
    result = {}
    for tid in model.topic_representations_:
        if tid == -1:
            continue
        words = []
        for phrase, _ in model.topic_representations_[tid][:top_n]:
            for token in str(phrase).lower().split():
                if token.strip():
                    words.append(token.strip())
        result[tid] = words
    return result


def compare_clustering(
    old_model: "BERTopic",  # noqa: F821
    new_model: "BERTopic",  # noqa: F821
    docs: List[str],
    old_topics: np.ndarray,
    new_topics: np.ndarray,
) -> dict:
    """维度 1: 聚类质量对比。

    Returns:
        Dict with old/new metrics.
    """
    def _metrics(model, topics):
        info = model.get_topic_info()
        n_total = len(topics)
        n_outliers = int(np.sum(topics == -1))
        n_topics = len(info[info["Topic"] != -1])
        outlier_pct = 100 * n_outliers / n_total
        sizes = info[info["Topic"] != -1]["Count"].values
        return {
            "n_topics": n_topics,
            "n_outliers": n_outliers,
            "outlier_pct": round(outlier_pct, 1),
            "mean_topic_size": round(float(np.mean(sizes)), 1) if len(sizes) > 0 else 0,
            "median_topic_size": round(float(np.median(sizes)), 1) if len(sizes) > 0 else 0,
            "min_size": int(np.min(sizes)) if len(sizes) > 0 else 0,
            "max_size": int(np.max(sizes)) if len(sizes) > 0 else 0,
        }

    return {
        "original": _metrics(old_model, old_topics),
        "sorted_ngram": _metrics(new_model, new_topics),
    }


def compare_coherence(
    old_model: "BERTopic",  # noqa: F821
    new_model: "BERTopic",  # noqa: F821
    docs: List[str],
) -> dict:
    """维度 2: 主题连贯性 (C_V + NPMI + Diversity)。"""
    try:
        from gensim.models.coherencemodel import CoherenceModel
        from gensim.corpora import Dictionary
    except ImportError:
        logger.warning("gensim 未安装，跳过连贯性评估")
        return {"error": "gensim not installed"}

    def _compute(model):
        # Tokenize the reference corpus with the model's own analyzer so the
        # reference tokens and n-gram topic words align (mismatch → NPMI ~0).
        try:
            analyzer = model.vectorizer_model.build_analyzer()
        except AttributeError:
            def analyzer(d):
                return d.lower().split()
        tokenized = [analyzer(d) for d in docs if len(d.split()) > 0]
        tokenized = [t for t in tokenized if len(t) > 0]

        topic_words_list = []
        for tid in sorted(model.topic_representations_):
            if tid == -1:
                continue
            words = model.get_topic(tid)
            if not words or len(words) < 5:
                continue
            word_list = [str(w).strip().lower() for w, _ in words[:10] if str(w).strip()]
            if len(word_list) >= 5:
                topic_words_list.append(word_list)

        if len(topic_words_list) < 5:
            return {"c_v": None, "npmi": None, "diversity": None}

        # Diversity
        all_w = [w for tw in topic_words_list for w in tw]
        diversity = len(set(all_w)) / len(all_w) if all_w else 0

        # C_V + NPMI
        dictionary = Dictionary(tokenized)
        c_v = float(CoherenceModel(
            topics=topic_words_list, texts=tokenized,
            dictionary=dictionary, coherence="c_v", topn=10,
        ).get_coherence())
        npmi = float(CoherenceModel(
            topics=topic_words_list, texts=tokenized,
            dictionary=dictionary, coherence="c_npmi", topn=10,
        ).get_coherence())

        return {"c_v": round(c_v, 4), "npmi": round(npmi, 4), "diversity": round(diversity, 4)}

    return {
        "original": _compute(old_model),
        "sorted_ngram": _compute(new_model),
    }


def compare_distinctiveness(
    old_model: "BERTopic",  # noqa: F821
    new_model: "BERTopic",  # noqa: F821
) -> dict:
    """维度 3: 主题区分度 (c-TF-IDF pairwise cosine similarity)。"""

    def _compute(model):
        valid = sorted([t for t in model.topic_representations_ if t != -1])
        ctfidf = model.c_tf_idf_.toarray() if hasattr(model.c_tf_idf_, "toarray") else model.c_tf_idf_
        keys = list(model.topic_representations_)
        rows = np.array([ctfidf[keys.index(t)] for t in valid])
        sim = cosine_similarity(rows)
        n = sim.shape[0]
        mask = ~np.eye(n, dtype=bool)
        pairwise = sim[mask]
        return {
            "n_topics": n,
            "mean_sim": round(float(np.mean(pairwise)), 4),
            "median_sim": round(float(np.median(pairwise)), 4),
            "std_sim": round(float(np.std(pairwise)), 4),
            "pct_high_sim": round(100 * float(np.mean(pairwise > 0.5)), 1),
        }

    return {
        "original": _compute(old_model),
        "sorted_ngram": _compute(new_model),
    }


def compare_word_order_redundancy(
    old_model: "BERTopic",  # noqa: F821
    new_model: "BERTopic",  # noqa: F821
) -> dict:
    """维度 4: 词序冗余 — 统计 "X Y" / "Y X" 颠倒对出现次数。

    对于每个模型，遍历所有主题的 top-20 关键词 (bigram)，
    检测是否存在词序颠倒的等价短语对。
    """

    def _count_flipped_pairs(model, top_n=20):
        """返回 (重复对数, 示例列表)。"""
        bigrams_seen: Dict[frozenset, list] = {}
        for tid in model.topic_representations_:
            if tid == -1:
                continue
            for phrase, _ in model.topic_representations_[tid][:top_n]:
                tokens = str(phrase).lower().split()
                if len(tokens) == 2:
                    key = frozenset(tokens)
                    if key not in bigrams_seen:
                        bigrams_seen[key] = []
                    bigrams_seen[key].append(" ".join(tokens))

        flipped_pairs = []
        for key, variants in bigrams_seen.items():
            unique = list(set(variants))
            if len(unique) >= 2:
                flipped_pairs.append({"key": " | ".join(sorted(key)), "variants": unique})

        return len(flipped_pairs), flipped_pairs

    old_count, old_examples = _count_flipped_pairs(old_model)
    new_count, new_examples = _count_flipped_pairs(new_model)

    return {
        "original": {"flipped_pairs": old_count, "examples": old_examples[:10]},
        "sorted_ngram": {"flipped_pairs": new_count, "examples": new_examples[:10]},
    }


def compare_emerging_keywords(
    old_model: "BERTopic",  # noqa: F821
    new_model: "BERTopic",  # noqa: F821
    emerging_topic_ids: List[int],
) -> dict:
    """维度 5: 前沿雏形主题的关键词质量对比。

    对于在原始模型中检测到的 emerging topics，
    在新模型中找到语义最匹配的主题，并对比其关键词。
    """
    from src.sorted_ngram_vectorizer import sorted_ngram_analyzer

    comparison = []
    for tid_old in emerging_topic_ids:
        old_words = old_model.get_topic(tid_old)
        old_top5 = [(w, round(s, 4)) for w, s in old_words[:5]] if old_words else []

        # 在新模型中找匹配主题: 选关键词 Jaccard 最近的
        old_kw_set = set(w for w, _ in (old_words or [])[:5])
        best_tid = None
        best_score = 0
        for tid_new in new_model.topic_representations_:
            if tid_new == -1:
                continue
            new_words = new_model.get_topic(tid_new)
            if not new_words:
                continue
            new_kw_set = set(w for w, _ in new_words[:5])
            score = len(old_kw_set & new_kw_set) / max(len(old_kw_set | new_kw_set), 1)
            if score > best_score:
                best_score = score
                best_tid = tid_new

        new_words = new_model.get_topic(best_tid) if best_tid is not None else []
        new_top5 = [(w, round(s, 4)) for w, s in new_words[:5]] if new_words else []

        comparison.append({
            "old_topic": tid_old,
            "best_match_new": best_tid,
            "jaccard": round(best_score, 3),
            "old_keywords": old_top5,
            "new_keywords": new_top5,
        })

    return comparison


# =============================================================================
# 报告生成
# =============================================================================


def print_comparison_report(results: dict) -> None:
    """格式化打印六维对比报告。"""
    print("\n" + "=" * 72)
    print("  BERTopic 向量器对比报告: 原始 vs Sorted N-gram")
    print("  控制变量: embedding/UMAP/HDBSCAN/min_topic_size 完全一致")
    print("=" * 72)

    # --- 维度 1: 聚类质量 ---
    cl = results["clustering"]
    print(f"\n{'─' * 60}")
    print("  维度 1: 聚类质量")
    print(f"{'─' * 60}")
    print(f"  {'指标':<28} {'原始':>14} {'Sorted N-gram':>14}")
    print(f"  {'-'*56}")
    for metric in ["n_topics", "n_outliers", "outlier_pct", "mean_topic_size", "median_topic_size", "min_size", "max_size"]:
        old_val = cl["original"][metric]
        new_val = cl["sorted_ngram"][metric]
        if isinstance(old_val, float):
            print(f"  {metric:<28} {old_val:>14.1f} {new_val:>14.1f}")
        else:
            print(f"  {metric:<28} {old_val:>14} {new_val:>14}")

    # --- 维度 2: 连贯性 ---
    if "error" not in results["coherence"]:
        co = results["coherence"]
        print(f"\n{'─' * 60}")
        print("  维度 2: 主题连贯性 (C_V / NPMI / Diversity)")
        print(f"{'─' * 60}")
        print(f"  {'指标':<20} {'原始':>14} {'Sorted N-gram':>14} {'说明':>15}")
        print(f"  {'-'*63}")
        notes = {"c_v": "越高越好 (>0.5)", "npmi": "越高越好 (>0.1)", "diversity": "越高越好"}
        for metric in ["c_v", "npmi", "diversity"]:
            old_val = co["original"].get(metric, "N/A")
            new_val = co["sorted_ngram"].get(metric, "N/A")
            note = notes[metric]
            if isinstance(old_val, (int, float)) and isinstance(new_val, (int, float)):
                delta = new_val - old_val
                marker = " ↑" if delta > 0 else (" ↓" if delta < 0 else " =")
                print(f"  {metric:<20} {old_val:>14.4f} {new_val:>14.4f}{marker:<14} {note}")
            else:
                print(f"  {metric:<20} {str(old_val):>14} {str(new_val):>14}   {note}")

    # --- 维度 3: 区分度 ---
    ds = results["distinctiveness"]
    print(f"\n{'─' * 60}")
    print("  维度 3: 主题区分度 (c-TF-IDF pairwise cosine similarity)")
    print(f"{'─' * 60}")
    print(f"  {'指标':<20} {'原始':>14} {'Sorted N-gram':>14} {'说明':>15}")
    print(f"  {'-'*63}")
    notes = {"n_topics": "主题数", "mean_sim": "越低越好 (<0.3)", "median_sim": "越低越好", "pct_high_sim": "sim>0.5 占比"}
    for metric in ["n_topics", "mean_sim", "median_sim", "pct_high_sim"]:
        old_val = ds["original"][metric]
        new_val = ds["sorted_ngram"][metric]
        note = notes[metric]
        if isinstance(old_val, (int, float)) and isinstance(new_val, (int, float)):
            delta = new_val - old_val
            marker = " ↑" if delta > 0 else (" ↓" if delta < 0 else " =")
            unit = "%" if metric == "pct_high_sim" else ""
            print(f"  {metric:<20} {old_val:>14.4f} {new_val:>14.4f}{marker:<14} {note}")

    # --- 维度 4: 词序冗余 ---
    wr = results["word_order"]
    print(f"\n{'─' * 60}")
    print("  维度 4: 词序颠倒冗余 (Bigram 'X Y' / 'Y X' 对)")
    print(f"{'─' * 60}")
    old_pairs = wr["original"]["flipped_pairs"]
    new_pairs = wr["sorted_ngram"]["flipped_pairs"]
    reduction = (1 - new_pairs / max(old_pairs, 1)) * 100 if old_pairs > 0 else 0
    print(f"  原始模型颠倒对:     {old_pairs}")
    print(f"  Sorted N-gram 颠倒对: {new_pairs}")
    print(f"  冗余消减率:         {reduction:.1f}%")
    if wr["original"]["examples"]:
        print(f"\n  原始模型示例:")
        for ex in wr["original"]["examples"][:5]:
            print(f"    {ex['key']}: {ex['variants']}")
    if wr["sorted_ngram"]["examples"]:
        print(f"\n  Sorted N-gram 残留示例:")
        for ex in wr["sorted_ngram"]["examples"][:5]:
            print(f"    {ex['key']}: {ex['variants']}")

    # --- 维度 5: 前沿关键词 ---
    ek = results.get("emerging_keywords", [])
    if ek:
        print(f"\n{'─' * 60}")
        print("  维度 5: 前沿雏形主题关键词对比 (Top-5 c-TF-IDF)")
        print(f"{'─' * 60}")
        for item in ek[:6]:
            print(f"\n  原始 Topic {item['old_topic']} ↔ 新模型 Topic {item['best_match_new']} (Jaccard={item['jaccard']:.2f})")
            print(f"    原始关键词:   {', '.join(f'{w}({s:.4f})' for w, s in item['old_keywords'])}")
            print(f"    新模型关键词: {', '.join(f'{w}({s:.4f})' for w, s in item['new_keywords'])}")

    # --- 维度 6 (可选): 跑两阶段算法 ---
    print(f"\n{'─' * 60}")
    print("  维度 6: 后续运行 emerging_tech_detection 以对比雏形检测结果")
    print(f"{'─' * 60}")
    print("  (请在新模型保存后运行: pixi run python -m src.emerging_tech_detection)")

    print(f"\n{'═' * 72}")
    print("  对比报告结束。")
    print(f"{'═' * 72}\n")


# =============================================================================
# 主入口
# =============================================================================


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="BERTopic 向量器对比实验 — 原始 vs Sorted N-gram"
    )
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--skip-training", action="store_true",
                        help="跳过训练，仅对比已保存的两组模型")
    parser.add_argument("--cpu", action="store_true", default=True,
                        help="使用 CPU 训练 (默认)")
    args = parser.parse_args()

    config = get_config(args.config)
    setup_logging(config.system.log_level, config.system.log_file)

    # --- 加载数据 ---
    from src.topic_modeling import load_texts

    with Timer("加载文档"):
        df, docs = load_texts(config)

    # --- 加载原始模型 ---
    from bertopic import BERTopic

    old_model_path = config.paths.models_dir / config.topic_modeling.output.model_name
    if not old_model_path.exists():
        logger.error("原始模型不存在: %s。请先运行 topic modeling。", old_model_path)
        sys.exit(1)

    with Timer("加载原始 BERTopic 模型"):
        old_model = BERTopic.load(old_model_path)
    old_topics = np.array(old_model.topics_)
    logger.info("原始模型: %d 主题, %.1f%% 离群率",
                len(set(old_topics)) - 1,
                100 * np.sum(old_topics == -1) / len(old_topics))

    # --- 训练或加载新模型 ---
    new_model_path = config.paths.models_dir / "bertopic_model_sorted_ngram"

    if args.skip_training and new_model_path.exists():
        logger.info("跳过训练，加载已有模型: %s", new_model_path)
        new_model = BERTopic.load(new_model_path)
        new_topics = np.array(new_model.topics_)
    else:
        new_model, new_topics, new_probs = train_sorted_ngram_model(config, docs, device="cpu")

        # 保存新模型
        with Timer("保存 Sorted N-gram 模型"):
            try:
                new_model.save(new_model_path, serialization="safetensors",
                               save_ctfidf=True, save_embedding_model=False)
            except Exception:
                new_model.save(new_model_path, serialization="pytorch",
                               save_ctfidf=False, save_embedding_model=False)
        logger.info("新模型保存至: %s", new_model_path)

        # 保存 topic info
        topic_info = new_model.get_topic_info()
        info_path = config.paths.processed_dir / "topics_info_sorted_ngram.csv"
        topic_info.to_csv(info_path, index=False, encoding="utf-8-sig")
        logger.info("Topic info 保存至: %s", info_path)

    n_new = len(set(new_topics)) - (1 if -1 in new_topics else 0)
    logger.info("新模型 (Sorted N-gram): %d 主题, %.1f%% 离群率",
                n_new, 100 * np.sum(new_topics == -1) / len(new_topics))

    # --- 六维对比 ---
    results = {}

    with Timer("维度 1: 聚类质量"):
        results["clustering"] = compare_clustering(old_model, new_model, docs, old_topics, new_topics)

    with Timer("维度 2: 主题连贯性"):
        results["coherence"] = compare_coherence(old_model, new_model, docs)

    with Timer("维度 3: 主题区分度"):
        results["distinctiveness"] = compare_distinctiveness(old_model, new_model)

    with Timer("维度 4: 词序冗余"):
        results["word_order"] = compare_word_order_redundancy(old_model, new_model)

    # 维度 5: 前沿关键词 (使用原始模型检测到的 emerging topics)
    emerging_ids = [34, 8, 16, 4, 5, 17, 33, 49, 24]
    with Timer("维度 5: 前沿关键词对比"):
        results["emerging_keywords"] = compare_emerging_keywords(old_model, new_model, emerging_ids)

    # --- 保存 JSON ---
    report_path = config.paths.processed_dir / "vectorizer_comparison.json"
    ensure_dir(report_path.parent)

    # 转换 numpy 类型以便 JSON 序列化
    def _serialize(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {str(k): _serialize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_serialize(v) for v in obj]
        return obj

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(_serialize(results), f, indent=2, ensure_ascii=False, default=str)
    logger.info("对比报告 JSON 保存至: %s", report_path)

    # --- 打印报告 ---
    print_comparison_report(results)


if __name__ == "__main__":
    main()
