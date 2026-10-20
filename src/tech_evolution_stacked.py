"""科技感演化堆叠图 — Decadal Evolution of Mangrove Remote Sensing Technology Clusters.

基于真实 BERTopic 模型的 72 个主题，按技术生命周期自动分层为三个梯队：
  - Traditional Baseline（传统基线）: peak_year < 2010，历史占比高，近期衰减
  - Advanced UAV/ML（先进技术）: 2005--2020 年间涌现，近期稳定增长
  - Emerging Prototypes（前沿雏形）: 两阶段突变检测捕获的冷启动主题

Usage:
    pixi run python src/tech_evolution_stacked.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

from src.config import get_config
from src.emerging_tech_detection import load_model_and_topic_data, compute_yearly_topic_counts

logger = logging.getLogger(__name__)

# =============================================================================
# 主题分层分类
# =============================================================================

# 基于 2026-07-29 检测结果的三层分类（由 data-driven 自动确定）
TRADITIONAL_TOPICS = [37, 52, 3, 36, 11, 45, 10, 46, 15, 13]
ADVANCED_TOPICS = [7, 30, 35, 39, 32, 0, 19, 57, 44, 42]
EMERGING_TOPICS = [34, 8, 16, 4, 5, 17, 33, 49, 24]


def aggregate_tier_proportions(
    yearly_props: pd.DataFrame,
    topic_ids: list[int],
) -> pd.Series:
    """将同一技术梯队的多个主题占比聚合为单一时间序列。

    使用求和聚合（sum），因为同一文档可属于不同主题的概率分布，
    但 BERTopic 的硬分配使 topic assignment 互斥，求和合理。

    Args:
        yearly_props: 年份 x 主题 的相对占比矩阵。
        topic_ids: 该梯队的主题 ID 列表。

    Returns:
        年份索引的聚合占比序列。
    """
    available = [t for t in topic_ids if t in yearly_props.columns]
    if not available:
        return pd.Series(0.0, index=yearly_props.index)
    return yearly_props[available].sum(axis=1)


def rolling_window_smooth(
    years: np.ndarray,
    values: np.ndarray,
    window: int = 5,
    min_docs_per_year: np.ndarray | None = None,
    min_docs_threshold: int = 5,
) -> np.ndarray:
    """使用滚动窗口 + 最小文档数阈值预平滑稀疏年份噪声。

    标准文献计量学处理：当年份文档数过少时（如 1972--1989 年每年仅 1 篇），
    单一主题占比会出现 0/1 极值。滚动窗口平均可抑制该噪声，同时保留长期趋势。

    Args:
        years: 年份数组。
        values: 原始占比数组（按年份对齐）。
        window: 滚动窗口大小（默认 5 年，居中）。
        min_docs_per_year: 每年文档计数数组（与 years 对齐），用于阈值过滤。
        min_docs_threshold: 年文档数低于此阈值时，该年权重衰减。

    Returns:
        滚动平滑后的占比数组（与 values 同长度）。
    """
    smoothed = np.full_like(values, np.nan, dtype=float)

    for i in range(len(years)):
        # 定义窗口范围
        half = window // 2
        start = max(0, i - half)
        end = min(len(years), i + half + 1)

        win_vals = values[start:end]
        win_years = years[start:end]

        if min_docs_per_year is not None:
            # 加权平均：文档数低于阈值的年份降权
            win_docs = min_docs_per_year[start:end].astype(float)
            weights = np.where(win_docs < min_docs_threshold,
                               win_docs / min_docs_threshold, 1.0)
            # 确保所有权重之和不为零
            if weights.sum() > 1e-10:
                smoothed[i] = np.average(win_vals, weights=weights)
            else:
                smoothed[i] = np.mean(win_vals)
        else:
            smoothed[i] = np.mean(win_vals)

    return smoothed


def smooth_series(
    years: np.ndarray,
    values: np.ndarray,
    n_points: int = 350,
    k: int = 3,
    pre_window: int = 5,
    min_docs_per_year: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """两阶段平滑：滚动窗口降噪 + 三次样条插值。

    1. 滚动窗口 (默认 5 年) 抑制稀疏年份的 0/1 极值噪声。
    2. 三次样条插值到 n_points 实现视觉平滑。

    Args:
        years: 原始年份数组。
        values: 原始占比数组。
        n_points: 平滑后采样点数。
        k: 样条阶数 (3 = cubic)。
        pre_window: 预平滑滚动窗口大小。
        min_docs_per_year: 每年文档计数（用于加权）。

    Returns:
        (x_smooth, y_smooth) 平滑后的年份与占比。
    """
    # --- Stage 1: 滚动窗口预平滑 ---
    y_pre = rolling_window_smooth(
        years, values, window=pre_window,
        min_docs_per_year=min_docs_per_year,
    )
    # 填充两端 NaN（滚动窗口在边界产生的）
    y_pre = pd.Series(y_pre).interpolate(method="linear", limit_direction="both").values

    # --- Stage 2: 三次样条 ---
    valid_mask = ~np.isnan(y_pre) & (y_pre >= -1e-9)
    x_valid = years[valid_mask]
    y_valid = np.maximum(y_pre[valid_mask], 0.0)  # clip negative to zero

    if len(x_valid) < 4:
        if len(x_valid) < 2:
            x_new = np.linspace(years.min(), years.max(), n_points)
            return x_new, np.full_like(x_new, np.mean(y_valid) if len(y_valid) > 0 else 0.0)
        k_use = 1
    else:
        k_use = min(k, len(x_valid) - 1)

    x_new = np.linspace(x_valid.min(), x_valid.max(), n_points)

    try:
        spl = make_interp_spline(x_valid, y_valid, k=k_use)
        y_new = spl(x_new)
    except Exception:
        y_new = np.interp(x_new, x_valid, y_valid)

    return x_new, np.clip(y_new, 0, None)


def get_tier_keywords(
    model: "BERTopic",  # noqa: F821
    topic_ids: list[int],
    top_n: int = 5,
) -> str:
    """从 BERTopic 模型提取一个技术梯队的代表性关键词。

    对梯队内所有主题的 c-TF-IDF 词频加权求和（以主题文档数为权重），
    取 top-N 关键词作为该梯队的标签。

    Args:
        model: 已拟合的 BERTopic 模型。
        topic_ids: 梯队主题 ID 列表。
        top_n: 返回前 N 个关键词。

    Returns:
        管道符分隔的关键词字符串，如 "Landsat | MSS/TM | Pixel-based | ..."
    """
    from collections import Counter

    topic_info = model.get_topic_info()
    weighted_words: Counter = Counter()

    for tid in topic_ids:
        if tid == -1:
            continue
        words = model.get_topic(tid)
        if not words:
            continue
        # 获取该主题的文档数作为权重
        count_row = topic_info[topic_info["Topic"] == tid]
        weight = int(count_row["Count"].values[0]) if len(count_row) > 0 else 1
        for word, score in words[:10]:
            weighted_words[word] += score * weight

    top_words = [w for w, _ in weighted_words.most_common(top_n)]
    return " | ".join(top_words)


# =============================================================================
# 主绘图函数
# =============================================================================


def plot_tech_evolution_stacked(
    yearly_props: pd.DataFrame,
    yearly_counts: pd.DataFrame,
    model: "BERTopic",  # noqa: F821
    output_path: Path,
) -> None:
    """绘制三层技术梯队演化堆叠面积图。

    视觉设计遵循学术出版物标准：
    - 灰色底噪 → 工业蓝 → 爆发荧光橙 三色渐变
    - 2024--2026 雏形爆发区间以半透明遮罩高亮
    - 2026 年末端自动锚定标注各梯队核心关键词
    - 三次样条平滑，350 采样点保证曲线流畅

    Args:
        yearly_props: 年份 x 主题 相对占比矩阵。
        model: BERTopic 模型 (提取关键词)。
        output_path: 图片保存路径。
    """
    # ------------------------------------------------------------------
    # 1. 聚合三层占比
    # ------------------------------------------------------------------
    trad_series = aggregate_tier_proportions(yearly_props, TRADITIONAL_TOPICS)
    adv_series = aggregate_tier_proportions(yearly_props, ADVANCED_TOPICS)
    emer_series = aggregate_tier_proportions(yearly_props, EMERGING_TOPICS)

    years = np.array(trad_series.index, dtype=float)

    # 输出统计信息
    logger.info("三层聚合统计:")
    logger.info("  Traditional: peak=%.3f (year=%d), 2026=%.3f",
                trad_series.max(), int(trad_series.idxmax()), trad_series.iloc[-1])
    logger.info("  Advanced:    peak=%.3f (year=%d), 2026=%.3f",
                adv_series.max(), int(adv_series.idxmax()), adv_series.iloc[-1])
    logger.info("  Emerging:    peak=%.3f (year=%d), 2026=%.3f",
                emer_series.max(), int(emer_series.idxmax()), emer_series.iloc[-1])

    # ------------------------------------------------------------------
    # 2. 三次样条平滑（含滚动窗口预降噪）
    # ------------------------------------------------------------------
    # 每年文档总数（用于加权滚动窗口，抑制稀疏年份噪声）
    min_docs_per_year = yearly_counts.sum(axis=1).values.astype(float)

    x_smooth, y_trad_s = smooth_series(
        years, trad_series.values, min_docs_per_year=min_docs_per_year,
    )
    _, y_adv_s = smooth_series(
        years, adv_series.values, min_docs_per_year=min_docs_per_year,
    )
    _, y_emer_s = smooth_series(
        years, emer_series.values, min_docs_per_year=min_docs_per_year,
    )

    # 构建堆叠数据（bottom-up: Traditional -> Advanced -> Emerging）
    y_stack = np.vstack([y_trad_s, y_adv_s, y_emer_s])

    # ------------------------------------------------------------------
    # 3. 关键词提取
    # ------------------------------------------------------------------
    kw_trad = get_tier_keywords(model, TRADITIONAL_TOPICS, top_n=4)
    kw_adv = get_tier_keywords(model, ADVANCED_TOPICS, top_n=4)
    kw_emer = get_tier_keywords(model, EMERGING_TOPICS, top_n=4)

    logger.info("关键词 Traditional: %s", kw_trad)
    logger.info("关键词 Advanced:    %s", kw_adv)
    logger.info("关键词 Emerging:    %s", kw_emer)

    # ------------------------------------------------------------------
    # 4. 画布与学术主题色配置
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 7.5), dpi=300)
    ax.set_facecolor("#F9FBFC")
    fig.patch.set_facecolor("#F9FBFC")

    # 色彩矩阵：灰色底噪 -> 工业蓝 -> 爆发荧光橙
    colors = ["#BACAD6", "#5C768D", "#FF4500"]
    alphas = [0.35, 0.65, 0.92]

    stacks = ax.stackplot(
        x_smooth, y_stack,
        colors=colors,
        alpha=alphas,
        edgecolors="white",
        linewidths=0.15,
    )

    # ------------------------------------------------------------------
    # 5. 2026年末端关键词自动锚定标注
    # ------------------------------------------------------------------
    last_idx = -1
    y_pos_trad = y_stack[0, last_idx] / 2.0
    y_pos_adv = y_stack[0, last_idx] + y_stack[1, last_idx] / 2.0
    y_pos_emer = y_stack[0, last_idx] + y_stack[1, last_idx] + y_stack[2, last_idx] / 2.0

    anchors = [y_pos_trad, y_pos_adv, y_pos_emer]
    tier_names = ["Traditional Baseline", "Advanced UAV & ML", "Emerging Prototypes"]
    tier_kw = [kw_trad, kw_adv, kw_emer]

    # 关键词自动换行：每行最多 2 个关键词，形成高窄文本框
    def wrap_keywords(kw_string: str, per_line: int = 2) -> str:
        parts = [p.strip() for p in kw_string.split("|")]
        lines = []
        for i in range(0, len(parts), per_line):
            lines.append(" | ".join(parts[i:i + per_line]))
        return "\n".join(lines)

    for i, (y_pos, name, kw) in enumerate(zip(anchors, tier_names, tier_kw)):
        is_emerging = (i == 2)
        text_color = "#B71C1C" if is_emerging else "#2C3E50"
        box_edge = "#FF4500" if is_emerging else "#95A5A6"
        box_bg = "#FFFFFF" if is_emerging else "#F5F5F5"
        font_w = "bold" if is_emerging else "normal"
        font_sz = 9.5 if is_emerging else 8

        # 关键词每行 2 个 → 高窄布局
        kw_wrapped = wrap_keywords(kw, per_line=2)
        label = f"{name}\n{kw_wrapped}"

        ax.annotate(
            label,
            xy=(2026, y_pos),
            xytext=(2028, y_pos + (0.070 if is_emerging else 0.040)),
            arrowprops=dict(
                arrowstyle="-",
                color=text_color,
                linestyle="-" if is_emerging else ":",
                linewidth=1.4 if is_emerging else 0.8,
            ),
            fontsize=font_sz,
            fontweight=font_w,
            color=text_color,
            va="center",
            bbox=dict(
                boxstyle="round,pad=0.3",
                fc=box_bg,
                ec=box_edge,
                lw=1.2 if is_emerging else 0.6,
                alpha=0.92,
            ),
        )

    # ------------------------------------------------------------------
    # 6. 2023--2026 雏形爆发区间视觉强化（2026年仅半年数据）
    # ------------------------------------------------------------------
    ax.axvspan(2023, 2026, color="#FF4500", alpha=0.05, zorder=0)
    ax.axvline(x=2023, color="#FF4500", linestyle="--", linewidth=1.2, alpha=0.55, zorder=5)

    ylim = ax.get_ylim()
    ax.text(
        2023.3, ylim[1] * 0.88,
        "2023--2026\nBurst Phase\n(2026: H1 only)",
        color="#D84315",
        fontsize=9.5,
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.25",
            fc="#FFF3E0",
            ec="#FF4500",
            lw=0.8,
            alpha=0.85,
        ),
    )

    # 标注关键时间节点
    ax.axvline(x=2008, color="#7F8C8D", linestyle=":", linewidth=0.6, alpha=0.35)
    ax.text(2008.3, ylim[1] * 0.06, "Free Landsat\nArchive (2008)",
            color="#7F8C8D", fontsize=7, alpha=0.7)

    # ------------------------------------------------------------------
    # 7. 坐标轴与学术规范细节
    # ------------------------------------------------------------------
    ax.set_xlim(1970, 2032)  # 右侧仅预留少量空间给窄文本框
    ax.set_xticks(range(1970, 2031, 5))
    ax.set_xticklabels([str(y) for y in range(1970, 2031, 5)], fontsize=8.5)

    ax.set_ylim(0, np.max(y_stack.sum(axis=0)) * 1.08)

    # 隐藏边框
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CCCCCC")
    ax.spines["bottom"].set_color("#CCCCCC")
    ax.grid(axis="y", linestyle=":", alpha=0.45, color="#D5D5D5")

    # 标题与轴标签
    ax.set_title(
        "Decadal Evolution of Technology Clusters in Mangrove Remote Sensing\n"
        "Three-Tier Stacked Model: Traditional -> Advanced UAV/ML -> Emerging Prototypes (2023--2026 Burst)",
        fontsize=13,
        fontweight="bold",
        pad=22,
        color="#1A1A1A",
    )
    ax.set_xlabel("Web of Science Publication Year", fontsize=10.5, labelpad=9, color="#333333")
    ax.set_ylabel("Aggregated Relative Topic Proportion (∑ c-TF-IDF Weight)", fontsize=10.5, labelpad=9, color="#333333")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    # 图例（手动构建，避免 stackplot 的 alpha 叠加问题）
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors[0], alpha=alphas[0], edgecolor="white", linewidth=0.3,
              label=f"Traditional Baseline ({len(TRADITIONAL_TOPICS)} topics)"),
        Patch(facecolor=colors[1], alpha=alphas[1], edgecolor="white", linewidth=0.3,
              label=f"Advanced UAV/ML ({len(ADVANCED_TOPICS)} topics)"),
        Patch(facecolor=colors[2], alpha=alphas[2], edgecolor="white", linewidth=0.3,
              label=f"Emerging Prototypes ({len(EMERGING_TOPICS)} topics, detected via Two-Stage Burst)"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper left",
        fontsize=8.5,
        framealpha=0.85,
        edgecolor="#CCCCCC",
        ncol=1,
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    logger.info("演化堆叠图保存至: %s", output_path)


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="技术演化堆叠图 — 基于真实 BERTopic 数据的三层演化可视化"
    )
    parser.add_argument("--config", type=str, default="config.yaml", help="配置文件路径")
    parser.add_argument("--output", type=str, default=None, help="输出路径 (默认: outputs/figures/tech_evolution_stacked.png)")
    args = parser.parse_args()

    config = get_config(args.config)
    from src.utils import setup_logging
    setup_logging(config.system.log_level, config.system.log_file)

    # 加载模型和数据
    model, doc_topic_df, _ = load_model_and_topic_data(config)

    # 构建年度占比矩阵
    yearly_counts, total_per_year = compute_yearly_topic_counts(doc_topic_df)
    yearly_props = yearly_counts.div(
        yearly_counts.index.map(lambda y: total_per_year.get(int(y), 1)), axis=0
    )

    # 绘制
    output_path = Path(args.output) if args.output else (
        config.paths.figures_dir / "tech_evolution_stacked.png"
    )
    plot_tech_evolution_stacked(yearly_props, yearly_counts, model, output_path)


if __name__ == "__main__":
    main()
