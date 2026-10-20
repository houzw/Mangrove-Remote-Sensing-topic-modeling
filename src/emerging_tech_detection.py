"""Step 5: Candidate Emerging Research Fronts --- Kleinberg two-state automaton.

在 BERTopic 动态主题模型 (Grootendorst 2022) 之上，对每个主题的逐年份额序列
拟合 Kleinberg (2003) 两状态爆发自动机（Binomial 观测形式，状态定义在**份额**
而非绝对频数上），识别 2023--2025 冷启动、2026 H1 仍活跃的候选新兴前沿。

理论框架
--------
1. **DTM 阶段** --- BERTopic ``topics_over_time`` 严格按年 (Annual) 时间片，
   开启 ``global_tuning=True`` + ``evolution_tuning=True``，防止跨越 50+
   年的语义断裂 (Grootendorst, 2022)。

2. **Kleinberg 两状态自动机** --- 对每个主题 t，观测
   ``N[t,y] ~ Binomial(T[y], p_{s[y]})``，``s[y] in {0,1}``（baseline / burst）。
   两个阶段分开：**(a) 率估计**用 regularized hard-EM（E-step = 在当前率下逐点
   把每年指派到更便宜的状态，**不含**转移代价；M-step = Jeffreys Beta(1/2,1/2)
   平滑加权均值 + 分离约束 p1 >= 2 p0 + 10 次随机重启）；**(b) 路径解码**用
   Viterbi 在估计出的 (p0, p1) 与转移代价 gamma 下求最小代价路径。取末端落在
   终年的 "terminal burst"，其 burst strength 为 burst 区间内 burst 状态相对
   baseline 的对数似然比。（这不是 Viterbi training：转移结构只进解码，不进
   率估计，见 ``src/burst_detection_kleinberg.py`` 模块 docstring。）

   序列只含**语料库覆盖年**（T_y > 0）：1972--2025 共 54 个日历年份中 41 个
   有文献，其余 13 年不进序列，故不存在 T_y = 0 的空 bin。

3. **三判据联合过滤**（候选新兴前沿 = 同时满足）：
   - (i)   recency  --- terminal burst 在 2025（末完整年）仍活跃；
   - (ii)  novelty  --- 1972--2020 历史均值份额 < 0.005，且历史窗口 ≥5 个
     语料库覆盖年（实测 36 个）；
   - (iii) liveness --- 2026 H1 文档数 > 0 且份额 ≥ 0.5 × 2025 份额。

*关键设计决策*：
- 状态定义在**份额**上（Binomial 分母 T[y] 吸收字段指数增长），而非绝对频数。
- 2026 为 H1 半年数据，**不进自动机**，仅作 criterion (iii) 的 liveness 确认；
  份额本就是 rate，无需年化（均匀 missingness 下份额不变，见 audit 文档）。
- gamma ∈ {0.5, 1.0, 2.0}、novelty floor ∈ {0.0025, 0.005, 0.01}、recency
  （onset ∈ {2020, 2021, 2022} 且**联动** novelty 窗口末端 = onset - 1；
  min-span ∈ {2, 3}）三组敏感性分析进 Appendix S1（``run_gamma_sensitivity`` /
  ``run_novelty_sensitivity`` / ``run_recency_sensitivity``）。

Usage
-----
.. code-block:: bash

    # 独立运行
    pixi run python -m src.emerging_tech_detection
    pixi run python -m src.emerging_tech_detection --config path/to/config.yaml

    # 通过 pipeline 运行
    pixi run python -m src.run_pipeline --step emerging

References
----------
- Grootendorst, M. (2022). BERTopic: Neural topic modeling with a class-based
  TF-IDF procedure. arXiv:2203.05794.
- Kleinberg, J. (2003). Bursty and hierarchical structure in streams.
  Data Mining and Knowledge Discovery, 7(4), 373--397.
- Taylor, M. & Taylor, A. (2012). The technology life cycle: Conceptualization
  and managerial implications. IJPM, 30(6), 792--812.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.burst_detection_kleinberg import (
    detect_recent_burst,
)
from src.config import Config, get_config
from src.utils import Timer, ensure_dir

logger = logging.getLogger(__name__)

# =============================================================================
# 常量定义
# =============================================================================

# 冷启动历史窗口：1972--2020（criterion ii 的 novelty 判据）。下界取语料库首年
# (1972)，与全文其它"1972--"表述统一；窗口内每年一个 row，历史均值按语料库
# 覆盖年（T_y > 0）求平均，实测 36 个覆盖年（最大支持年数，见 MIN_SUPPORTING_YEARS）。
# 上界 2020 与 MIN_BURST_ONSET(2021) 严格错开，避免爆发年(2021+)的份额
# 污染自己的"历史基线"，从而抬高校准均值、误杀早爆发 1--2 年的主题。
HISTORICAL_START = 1972
HISTORICAL_END = 2020

# Kleinberg 自动机在 1972--2025 的**语料库覆盖年**上拟合（1972--2025 是 54 个
# 日历年份，其中 13 个年份无任何入库文献，共 41 个覆盖年）；2026 H1 排除在外。
# 空年份不进入序列，因此不存在 T_y = 0 的 bin：两态似然都无定义/恒 1 的"免费"
# 年份不存在，burst 区间的"连续"是**所供给年份序列内**的连续。
AUTOMATON_END = 2025

# 2026 H1 确认年：criterion iii 的 liveness 判据（不进自动机）
CONFIRM_YEAR = 2026

# 转移代价 gamma = 1.0（不是 Kleinberg 2003 的默认值——原文双态形式未定义该常数，
# 1.0 是 burst_detection 类实现的常用缺省；本研究报告取值 + 敏感性即可）。
# 敏感性网格进 Appendix S1。
GAMMA = 1.0
GAMMA_GRID = (0.5, 1.0, 2.0)

# criterion (i) recency 的"真·新兴"收紧（用户决策 B，2026-09-08）：
# terminal burst 必须起点 ≥ 2021 且长度 ≥ 2 年——排除十年前就活跃的长尾
# （如 2014--2025）与单年脉冲（如 2025--2025）。
MIN_BURST_ONSET = 2021
MIN_BURST_LENGTH = 2

# criterion (i) 的敏感性网格。**关键联动**：onset 网格的每个点必须配套把
# novelty 窗口末端移到 onset - 1（2020->2019, 2021->2020, 2022->2021），
# 否则 onset 年的爆发份额会混进它自己的历史基线，被测到的是"污染效应"
# 而不是 onset 的效应，表格会被无声做歪。min-span 不涉及窗口重叠，故在
# 生产 onset/window 下扫描。实现见 run_recency_sensitivity 的 kwargs 构造。
ONSET_GRID = (2020, 2021, 2022)
MIN_SPAN_GRID = (2, 3)

# criterion iii："占比不降"的保守判据 p_2026H1 >= H1_FRACTION * p_2025
# （份额本就是 rate，无需年化；0.5 = "至少前一年一半节奏"的直觉下界）
H1_FRACTION = 0.5

# criterion ii：历史期占比上限（冷启动）
COLD_START_THRESHOLD = 0.005
# criterion ii："≥5 个支持年份" = 1972--2020 内语料库存在的年份数（T_y > 0）。
# 精确定义为语料库覆盖年（非"主题非零年"），避免误杀 2020+ 才首次出现的
# 真正冷启动主题。语料库 1972 才起步且早期年份有缺口，故支持年数由数据
# 决定（实测 36 年），远大于门槛——该门槛保证历史均值定义良定而非噪声过滤。
MIN_SUPPORTING_YEARS = 5

# 输出控制
TOP_N_EMERGING = 8             # 输出前 N 个雏形主题（= 候选前沿全集；
                                # 手稿 Fig.~4 的 caption 声明八个面板，
                                # 设为 5 会让 T57/T35/T68 从图里消失）
TOP_N_KEYWORDS = 10            # 每个主题展示的关键词数量


# =============================================================================
# 第一阶段：年度动态主题提取 (Yearly DTM)
# =============================================================================


def load_model_and_topic_data(
    config: Config,
    model_name: str | None = None,
) -> Tuple["BERTopic", pd.DataFrame, pd.DataFrame]:  # noqa: F821
    """加载已训练的 BERTopic 模型、文档-主题分布和清洗数据。

    Args:
        config: 已验证的配置对象。
        model_name: 模型目录名（默认从 config 读取，如 ``"bertopic_model"``）。
                    可传入 ``"bertopic_model_sorted_ngram"`` 使用新模型。

    Returns:
        ``(model, doc_topic_df, cleaned_df)`` 三元组：
          - *model*: 已拟合的 BERTopic 模型
          - *doc_topic_df*: 文档-主题分布 (含 PY 年份列)
          - *cleaned_df*: 清洗后的原始数据 (含 PY、text 列)

    Raises:
        FileNotFoundError: 模型或数据文件缺失。
    """
    from bertopic import BERTopic

    # --- 加载 BERTopic 模型 ---
    _name = model_name or config.topic_modeling.output.model_name
    model_path = config.paths.models_dir / _name
    if not model_path.exists():
        raise FileNotFoundError(
            f"BERTopic 模型不存在: {model_path}。请先运行 topic modeling 步骤。"
        )

    with Timer("加载 BERTopic 模型"):
        model = BERTopic.load(model_path)
    n_topics = len(model.get_topic_info()) - 1  # 排除 outlier (-1)
    logger.info("模型已加载: %d 个主题", n_topics)

    # --- 加载文档-主题分布 ---
    _suffix = f"_{model_name}" if model_name else ""
    doc_topic_filename = f"doc_topic_dist{_suffix}.csv"
    doc_topic_path = config.paths.processed_dir / doc_topic_filename
    if not doc_topic_path.exists():
        # Fallback to default
        doc_topic_path = config.paths.processed_dir / config.topic_modeling.output.doc_topic_dist
    if not doc_topic_path.exists():
        raise FileNotFoundError(f"文档-主题分布不存在: {doc_topic_path}")

    doc_topic_df = pd.read_csv(doc_topic_path, encoding="utf-8-sig")
    logger.info("文档-主题分布: %d 条记录", len(doc_topic_df))

    if "PY" not in doc_topic_df.columns:
        raise ValueError("doc_topic_dist.csv 缺少 'PY' (年份) 列------无法按年切分。")

    # --- 加载清洗数据 (获取文档文本) ---
    csv_path = config.paths.processed_dir / config.preprocessing.output_file
    pq_path = csv_path.with_suffix(".parquet")
    if pq_path.exists():
        cleaned_df = pd.read_parquet(pq_path)
    elif csv_path.exists():
        cleaned_df = pd.read_csv(csv_path, encoding="utf-8-sig")
    else:
        raise FileNotFoundError(f"清洗数据不存在: {csv_path}")

    logger.info("清洗数据: %d 条记录, 年份范围 %d--%d",
                len(cleaned_df), cleaned_df["PY"].min(), cleaned_df["PY"].max())

    return model, doc_topic_df, cleaned_df


def compute_yearly_topic_counts(
    doc_topic_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[int, int]]:
    """从文档-主题分布构建「年份 x 主题」频数矩阵。

    该函数独立于 BERTopic 的 ``topics_over_time``，直接从已有的
    文档级主题分配中按年聚合，实现真正的 **年度** 时间粒度。

    Args:
        doc_topic_df: 文档-主题分布 DataFrame，必须包含 ``Topic`` 和 ``PY`` 列。

    Returns:
        ``(yearly_counts, total_per_year)``：
          - *yearly_counts*: 透视表，index=年份，columns=主题ID，values=文档数
          - *total_per_year*: dict，年份 -> 该年文档总数
    """
    # 过滤有效年份范围
    df = doc_topic_df.copy()
    df["PY"] = pd.to_numeric(df["PY"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["PY", "Topic"])
    df["Topic"] = df["Topic"].astype(int)

    # 按年 x 主题 聚合文档数
    yearly_counts = df.groupby(["PY", "Topic"]).size().unstack(fill_value=0)

    # 计算每年文档总数
    total_per_year = df.groupby("PY").size().to_dict()

    logger.info(
        "年度主题频数矩阵: %d 年 x %d 主题, 年份范围 %d--%d",
        yearly_counts.shape[0],
        yearly_counts.shape[1],
        yearly_counts.index.min(),
        yearly_counts.index.max(),
    )
    return yearly_counts, total_per_year


def run_yearly_topics_over_time(
    model: "BERTopic",  # noqa: F821
    docs: List[str],
    years: List[int],
) -> pd.DataFrame:
    """使用 BERTopic 原生 ``topics_over_time`` 按年度提取动态主题。

    与 ``topic_dynamics.py`` 中基于分箱 (binning) 的实现不同，
    本函数直接使用年份字符串作为时间戳，实现 **逐年** 粒度。

    关键参数：
    - ``global_tuning=True``: 全局调整主题表示，使多年份的主题词保持
      语义一致性，防止因年份切割导致的主题漂移。
    - ``evolution_tuning=True``: 演化调整，比较相邻时间片的 c-TF-IDF
      表示并进行平滑，防止跨越 50+ 年的语义断裂。

    Args:
        model: 已拟合的 BERTopic 模型。
        docs: 文档文本列表（与模型拟合时顺序一致）。
        years: 与 docs 对齐的出版年份列表。

    Returns:
        BERTopic ``topics_over_time`` 输出 DataFrame，含列:
        ``Topic, Words, Timestamp, Frequency``。
    """
    # 以年份中点的日期字符串作为时间戳 (e.g. "2024-01-01")
    # BERTopic 要求时间戳可被 pandas 解析为 datetime
    timestamps = [f"{y}-01-01" for y in years]

    logger.info(
        "运行 BERTopic topics_over_time (年度粒度, global_tuning=True, evolution_tuning=True)..."
    )
    with Timer("BERTopic topics_over_time (yearly)"):
        topics_over_time = model.topics_over_time(
            docs,
            timestamps,
            global_tuning=True,     # 全局主题表示对齐
            evolution_tuning=True,  # 相邻时间片 c-TF-IDF 平滑
        )

    logger.info("年度动态主题: %d 行, %d 个唯一时间戳",
                len(topics_over_time),
                topics_over_time["Timestamp"].nunique())

    return topics_over_time


# =============================================================================
# 第二阶段：两阶段雏形捕捉算法 (Two-Stage Burst Detection)
# =============================================================================


def _compute_yearly_proportions(
    yearly_counts: pd.DataFrame,
    total_per_year: Dict[int, int],
) -> pd.DataFrame:
    """将绝对频数矩阵转换为相对占比矩阵。

    公式: ``Proportion[topic, year] = count[topic, year] / total_docs[year]``

    这是整个算法的关键前提------消除 WOS 历年文献总量指数级增长的混淆效应。
    如果不做归一化，近期年份会因文献总量大而天然拥有更高的绝对频数，
    导致早期稀疏但重要的主题被淹没。

    Args:
        yearly_counts: 年份 x 主题 的绝对频数透视表。
        total_per_year: 年份 -> 该年文献总数 的映射。

    Returns:
        年份 x 主题 的相对占比矩阵 (0--1 之间的浮点数)。
    """
    proportions = yearly_counts.copy().astype(float)
    for year in proportions.index:
        total = total_per_year.get(int(year), 1)
        if total > 0:
            proportions.loc[year] = proportions.loc[year] / total
    return proportions


def detect_emerging_tech_prototypes(
    yearly_counts: pd.DataFrame,
    total_per_year: Dict[int, int],
    df_topics_over_time: Optional[pd.DataFrame] = None,
    gamma: float = GAMMA,
    automaton_end: int = AUTOMATON_END,
    confirm_year: int = CONFIRM_YEAR,
    historical_start: int = HISTORICAL_START,
    historical_end: int = HISTORICAL_END,
    h1_fraction: float = H1_FRACTION,
    cold_start_threshold: float = COLD_START_THRESHOLD,
    min_supporting_years: int = MIN_SUPPORTING_YEARS,
    min_burst_onset: int = MIN_BURST_ONSET,
    min_burst_length: int = MIN_BURST_LENGTH,
    return_all: bool = False,
) -> pd.DataFrame:
    """【核心算法】Kleinberg 两状态自动机 + 三判据，识别候选新兴前沿。

    对每个主题 t：
      1. 在 1972--2025 的**语料库覆盖年**上拟合 Kleinberg 两状态自动机
         (Binomial 观测)：先用 regularized hard-EM 逐点估计 (p0, p1)，再用
         Viterbi 在 (p0, p1, gamma) 下解码最小代价路径，取最近一段 burst
         (末端落在终年的 "terminal burst")。
      2. 三判据联合过滤：
         (i)   recency  — terminal burst 末端 = automaton_end，起点
              ≥ min_burst_onset，且跨度 ≥ min_burst_length 年；
         (ii)  novelty  — 历史均值份额 < cold_start_threshold，且历史窗口
              有 ≥ min_supporting_years 个语料库覆盖年；
         (iii) liveness — confirm_year (2026 H1) 文档数 > 0 且份额 ≥
              h1_fraction × 上一年 (automaton_end) 份额。
      3. 按 burst strength（burst 区间内 burst 状态相对 baseline 的对数似然比）
         降序排列。

    序列仅由 ``proportions.index``（即语料库中实际存在的年份）构成，空年份
    (T_y = 0) 不进序列，故不存在可零成本穿过的空 bin；``Burst_Interval`` 的
    "连续" 因此是**所供给年份序列内**的连续。

    Args:
        yearly_counts: 年份 x 主题 的绝对频数透视表。
        total_per_year: 年份 -> 该年文献总数 映射。
        df_topics_over_time: 可选，BERTopic topics_over_time 输出 (含 Words 列)，
                             用于提取每个主题在确认年的代表性关键词。
        gamma: 状态转移代价 (本研究报告值 1.0；非 Kleinberg 2003 默认值)。
        automaton_end: 自动机末完整年 (默认 2025；2026 H1 不进自动机)。
        confirm_year: 确认年 (默认 2026 H1)。
        historical_start / historical_end: novelty 判据的历史窗口 (默认 1972--2020)。
        h1_fraction: liveness 判据系数 (p_confirm >= h1_fraction * p_prev)。
        cold_start_threshold: 冷启动阈值 (历史均值份额上限)。
        min_supporting_years: 历史窗口最少语料库覆盖年数。
        min_burst_onset: terminal burst 起点年份下限 (真·新兴收紧，默认 2021)。
        min_burst_length: terminal burst 最少跨度的年数 (默认 2)。
        return_all: 若 True，返回所有被评估主题 (含未通过者及 Fail_Reason)。

    Returns:
        DataFrame，按 Burst_Strength 降序排列，列: ``Topic, Baseline_Share,
        Burst_Share, Burst_Strength, Burst_Interval, Active_Terminal,
        Hist_Mean_Share, Hist_Nonzero_Years, Hist_Supporting_Years,
        Novelty_Pass, Count_2026_H1, Share_2026_H1, Share_2025,
        Liveness_2026_H1, Passed_Filter, Fail_Reason, Top_Keywords``。
    """
    proportions = _compute_yearly_proportions(yearly_counts, total_per_year)

    all_years = sorted(int(y) for y in proportions.index)
    complete_years = [y for y in all_years if y <= automaton_end]
    hist_years = [y for y in all_years if historical_start <= y <= historical_end]

    # "支持年份" = 历史窗口内语料库存在的年份（T_y > 0）
    n_supporting = sum(1 for y in hist_years if total_per_year.get(y, 0) > 0)

    T_complete = np.array([total_per_year.get(y, 0) for y in complete_years], dtype=float)
    years_arr = np.array(complete_years, dtype=int)

    logger.info("=" * 60)
    logger.info("【Kleinberg 两状态自动机 + 三判据】")
    logger.info("  自动机窗口: %s (%d 覆盖年)",
                f"{complete_years[0]}--{complete_years[-1]}" if complete_years else "n/a",
                len(complete_years))
    logger.info("  历史窗口  : %d--%d (%d 年, 其中 %d 个语料库覆盖年)",
                historical_start, historical_end, len(hist_years), n_supporting)
    logger.info("  确认年    : %d (H1, 不进自动机)", confirm_year)
    logger.info("  gamma     : %.2f", gamma)
    logger.info("  冷启动阈值: < %.4f", cold_start_threshold)
    logger.info("=" * 60)

    keyword_year = confirm_year

    results = []
    for topic_id in sorted(proportions.columns):
        if topic_id == -1:
            continue

        N_complete = np.array(
            [yearly_counts.loc[y, topic_id] if topic_id in yearly_counts.columns else 0
             for y in complete_years], dtype=float)

        # ---- Kleinberg 自动机 (criterion i: terminal burst) ----
        det = detect_recent_burst(years_arr, N_complete, T_complete, gamma=gamma)
        p0, p1 = det["p0"], det["p1"]
        terminal = det["terminal_burst"]
        strength = det["burst_strength"]
        active = det["active_in_terminal_year"]

        # criterion (i) recency 收紧：起点 ≥ min_burst_onset 且长度 ≥ min_burst_length
        onset_ok = terminal is not None and terminal[0] >= min_burst_onset
        length_ok = terminal is not None and (terminal[1] - terminal[0] + 1) >= min_burst_length
        recency = bool(active and onset_ok and length_ok)

        # ---- criterion (ii): novelty ----
        hist_mean = float(proportions.loc[hist_years, topic_id].mean()) if hist_years else 0.0
        hist_nonzero = int((yearly_counts.loc[hist_years, topic_id] > 0).sum()) if hist_years else 0
        novelty_pass = (hist_mean < cold_start_threshold) and (n_supporting >= min_supporting_years)

        # ---- criterion (iii): liveness in confirm_year (2026 H1) ----
        count_confirm = 0
        share_confirm = float("nan")
        share_ref = float("nan")
        liveness = False
        if confirm_year in all_years:
            count_confirm = int(yearly_counts.loc[confirm_year, topic_id])
            share_confirm = float(proportions.loc[confirm_year, topic_id])
            if automaton_end in all_years:
                share_ref = float(proportions.loc[automaton_end, topic_id])
                liveness = (
                    count_confirm > 0
                    and not np.isnan(share_confirm)
                    and not np.isnan(share_ref)
                    and share_ref > 0
                    and share_confirm >= h1_fraction * share_ref
                )

        passed = bool(recency and novelty_pass and liveness)
        if passed:
            fail_reason = ""
        elif not active:
            fail_reason = "no_terminal_burst"
        elif not onset_ok:
            fail_reason = "burst_too_early"
        elif not length_ok:
            fail_reason = "burst_too_short"
        elif not novelty_pass:
            fail_reason = "cold_start_failed"
        elif not liveness:
            fail_reason = "liveness_2026H1_failed"
        else:
            fail_reason = "unknown"

        # ---- 关键词 ----
        keywords = ""
        if df_topics_over_time is not None and "Words" in df_topics_over_time.columns:
            tot_row = df_topics_over_time[
                (df_topics_over_time["Topic"] == topic_id)
                & (df_topics_over_time["Timestamp"].astype(str).str.startswith(str(keyword_year)))
            ]
            if not tot_row.empty:
                keywords = str(tot_row["Words"].iloc[0])

        results.append(_topic_row(
            topic_id=topic_id,
            p0=p0, p1=p1, strength=strength,
            terminal=terminal, active=active,
            hist_mean=hist_mean, hist_nonzero=hist_nonzero,
            hist_supporting=n_supporting, novelty_pass=novelty_pass,
            count_confirm=count_confirm, share_confirm=share_confirm,
            share_ref=share_ref, liveness=liveness,
            passed=passed, fail_reason=fail_reason, keywords=keywords,
        ))

    if not results:
        logger.warning("[WARN] 未检测到任何候选新兴前沿。")
        return pd.DataFrame()

    df_results = pd.DataFrame(results)
    if not return_all:
        df_results = df_results[df_results["Passed_Filter"]].copy()
    if df_results.empty:
        logger.warning("[WARN] 无主题通过三判据过滤。")
        return pd.DataFrame()
    df_results = df_results.sort_values("Burst_Strength", ascending=False).reset_index(drop=True)

    n_passed = int(df_results["Passed_Filter"].sum()) if return_all else len(df_results)
    logger.info("[OK] 通过三判据的候选新兴前沿: %d 个", n_passed)
    if n_passed:
        for _, row in df_results[df_results["Passed_Filter"]].iterrows():
            logger.info(
                "  Topic %d | p0=%.4f p1=%.4f | strength=%.1f | burst=%s | 2026H1=%s",
                row["Topic"], row["Baseline_Share"], row["Burst_Share"],
                row["Burst_Strength"], row["Burst_Interval"], row["Liveness_2026_H1"],
            )
    return df_results


def _topic_row(
    topic_id: int,
    p0: float,
    p1: float,
    strength: Optional[float],
    terminal: Optional[Tuple[int, int]],
    active: bool,
    hist_mean: float,
    hist_nonzero: int,
    hist_supporting: int,
    novelty_pass: bool,
    count_confirm: int,
    share_confirm: float,
    share_ref: float,
    liveness: bool,
    passed: bool,
    fail_reason: str,
    keywords: str,
) -> Dict:
    """构造单个主题的检测结果行。"""
    return {
        "Topic": int(topic_id),
        "Baseline_Share": round(p0, 6),
        "Burst_Share": round(p1, 6),
        "Burst_Strength": round(strength, 2) if strength is not None else float("nan"),
        "Burst_Interval": f"{terminal[0]}-{terminal[1]}" if terminal else "",
        "Active_Terminal": bool(active),
        "Hist_Mean_Share": round(hist_mean, 6),
        "Hist_Nonzero_Years": int(hist_nonzero),
        "Hist_Supporting_Years": int(hist_supporting),
        "Novelty_Pass": bool(novelty_pass),
        "Count_2026_H1": int(count_confirm),
        "Share_2026_H1": round(share_confirm, 6),
        "Share_2025": round(share_ref, 6),
        "Liveness_2026_H1": bool(liveness),
        "Passed_Filter": bool(passed),
        "Fail_Reason": fail_reason,
        "Top_Keywords": keywords,
    }


# =============================================================================
# 结果输出与行业预测
# =============================================================================


def print_top_emerging_prototypes(
    df_results: pd.DataFrame,
    model: Optional["BERTopic"] = None,  # noqa: F821
    top_n: int = TOP_N_EMERGING,
) -> None:
    """格式化输出前 N 个雏形主题的详细信息。

    输出内容：
    - 主题 ID 与爆发系数
    - 各指标数值 (历史均值、近期均值、斜率)
    - 2026 年 c-TF-IDF 核心关键词
    - 如果提供了 BERTopic 模型，同时输出模型中的 Topic Representation

    Args:
        df_results: ``detect_emerging_tech_prototypes`` 的输出。
        model: 可选，BERTopic 模型 (用于提取 c-TF-IDF 完整关键词列表)。
        top_n: 输出前 N 个主题。
    """
    if df_results.empty:
        print("\n[WARN] 未检测到通过三判据的候选新兴前沿。\n")
        return

    print("\n" + "=" * 72)
    print("   [Microscope] 红树林遥感候选新兴前沿检测报告")
    print("   Candidate Emerging Research Fronts (Kleinberg two-state automaton)")
    print("=" * 72)

    df_display = df_results.head(top_n)

    for rank, (_, row) in enumerate(df_display.iterrows(), start=1):
        topic_id = int(row["Topic"])
        strength = row["Burst_Strength"]
        interval = row.get("Burst_Interval", "")

        print(f"\n{'-' * 60}")
        print(f"  [Rank] Rank #{rank}  |  Topic {topic_id}  |  burst strength W = {strength:.1f}")
        print(f"{'-' * 60}")
        print(f"  baseline share p0        : {row['Baseline_Share']:.6f}")
        print(f"  burst share p1           : {row['Burst_Share']:.6f}")
        print(f"  burst interval           : {interval}")
        print(f"  hist mean share (1972-20): {row['Hist_Mean_Share']:.6f}")
        print(f"  hist nonzero years       : {int(row['Hist_Nonzero_Years'])}")

        c_count = int(row.get("Count_2026_H1", 0))
        c_share = row.get("Share_2026_H1", float("nan"))
        s_2025 = row.get("Share_2025", float("nan"))
        c_str = f"{float(c_share):.6f}" if not pd.isna(c_share) else "N/A"
        live = bool(row.get("Liveness_2026_H1", False))
        print(f"  2026 H1: count={c_count}, share={c_str}, 2025 share="
              f"{f'{float(s_2025):.6f}' if not pd.isna(s_2025) else 'N/A'}, "
              f"live={'是' if live else '否'}")

        keywords_str = str(row.get("Top_Keywords", ""))
        if keywords_str and keywords_str != "nan":
            print(f"  最新年份 c-TF-IDF 关键词: {keywords_str}")

        if model is not None:
            try:
                topic_words = model.get_topic(topic_id)
                if topic_words:
                    top_words_str = ", ".join(
                        f"{w} ({s:.4f})" for w, s in topic_words[:TOP_N_KEYWORDS]
                    )
                    print(f"  模型 Topic Representation: {top_words_str}")
            except Exception:
                pass

    print("=" * 72)
    print(f"  共检测到 {len(df_results)} 个候选新兴前沿，"
          f"以上展示前 {min(top_n, len(df_results))} 个。")
    print("=" * 72 + "\n")


# =============================================================================
# 可视化
# =============================================================================


def plot_burst_index_barchart(
    df_results: pd.DataFrame,
    output_path: Path,
    top_n: int = TOP_N_EMERGING,
) -> None:
    """绘制爆发系数 (Burst Index) 水平条形图。

    X 轴为 Burst Index (对数尺度)，Y 轴为主题标签 (含 top 关键词)。
    颜色映射为 Burst Index 渐变，并用虚线标注"冷启动"和"突变"阈值位置。

    Args:
        df_results: ``detect_emerging_tech_prototypes`` 的输出。
        output_path: 图表保存路径 (PNG)。
        top_n: 展示前 N 个主题。
    """
    import matplotlib.pyplot as plt

    if df_results.empty:
        logger.warning("无数据可绘图 --- 跳过 Burst Strength 条形图")
        return

    df_plot = df_results.head(top_n).iloc[::-1]  # 反转以在水平图中从大到小

    fig, ax = plt.subplots(figsize=(12, 2 + 0.6 * len(df_plot)))

    # 构建 Y 轴标签: "Topic X: keyword1, keyword2, ..."
    y_labels = []
    for _, row in df_plot.iterrows():
        tid = int(row["Topic"])
        kw = str(row.get("Top_Keywords", ""))
        if kw and kw != "nan":
            kw_short = ", ".join(kw.replace("_", " ").split(", ")[:4])
        else:
            kw_short = "---"
        y_labels.append(f"T{tid}: {kw_short}")

    burst_values = df_plot["Burst_Strength"].values
    colors = plt.cm.YlOrRd(burst_values / burst_values.max() * 0.7 + 0.3)

    bars = ax.barh(range(len(df_plot)), burst_values, color=colors, edgecolor="white", height=0.65)

    for i, (bar, val) in enumerate(zip(bars, burst_values)):
        ax.text(bar.get_width() + burst_values.max() * 0.01,
                bar.get_y() + bar.get_height() / 2.0,
                f"{val:.0f}", va="center", fontsize=10, fontweight="bold", color="#333")

    ax.set_yticks(range(len(df_plot)))
    ax.set_yticklabels(y_labels, fontsize=9)
    ax.set_xlabel("Burst strength $W$ (log-likelihood ratio of burst vs. baseline)", fontsize=11)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", linestyle=":", alpha=0.4)

    ax.set_title(
        "Candidate Emerging Research Fronts in Mangrove Remote Sensing\n"
        "Burst strength (Kleinberg two-state automaton, Binomial form) — higher = stronger burst",
        fontsize=12, fontweight="bold", pad=12,
    )

    fig.tight_layout()
    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Burst Strength 条形图保存至: %s", output_path)


def plot_emerging_trajectories(
    yearly_counts: pd.DataFrame,
    total_per_year: Dict[int, int],
    df_results: pd.DataFrame,
    output_path: Path,
    top_n: int = 5,
    topic_labels: Optional[Dict[int, str]] = None,
    redetection: Optional[Dict[int, float]] = None,
) -> None:
    """绘制雏形主题的时间轨迹子图 (Small Multiples)。

    每个子图展示一个主题的逐年占比变化，高亮标注：
    - 灰色背景：历史期 (1972--2020)
    - 青绿色背景：terminal burst 区间
    - 绿色虚线：部分收录的 2026 (不进自动机, 仅作 liveness 确认)
    - 橙色虚线：该主题的历史均值参考线
    - 红色趋势线：2024--2026 的 OLS 拟合

    Args:
        yearly_counts: 年份 x 主题 频数透视表。
        total_per_year: 年份 -> 文献总数。
        df_results: 检测结果 (用于确定展示哪些主题)。
        output_path: 图表保存路径 (PNG)。
        top_n: 展示前 N 个主题轨迹。
    """
    import matplotlib.pyplot as plt

    if df_results.empty:
        logger.warning("无数据可绘图 --- 跳过雏形轨迹图")
        return

    proportions = _compute_yearly_proportions(yearly_counts, total_per_year)
    redetection = redetection or {}
    df_plot = df_results.head(top_n)
    n = len(df_plot)

    if n == 0:
        return

    # Two columns, not three.  The manuscript places this figure at
    # \columnwidth = 6.50 in, so a wider canvas is scaled down harder: at
    # 3 x 5.5 in = 16.5 in the scale is 0.39x and the 7 pt tick labels
    # printed at ~2.8 pt.  At 2 x 4.3 in = 8.6 in the scale is ~0.76x and
    # every label stays above 5 pt.
    cols = 1 if n == 1 else 2
    rows = int(np.ceil(n / cols))

    # 2.30 in per row (not 2.45): the two-line panel titles below cost about a
    # line of height, and the figure must keep printing inside the text block at
    # \columnwidth (see the note above on canvas size).
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.3, rows * 2.30))
    axes = np.atleast_1d(axes).flatten()

    for idx, (_, row) in enumerate(df_plot.iterrows()):
        ax = axes[idx]
        topic_id = int(row["Topic"])
        topic_prop = proportions[topic_id]
        all_years = sorted(topic_prop.index)

        # 全时间序列
        ax.plot(all_years, topic_prop.values, color="#555555", linewidth=0.8,
                alpha=0.7, marker="o", markersize=2)

        # 高亮部分收录的 2026 (不进自动机, 仅 liveness 确认)
        if CONFIRM_YEAR in all_years:
            ax.axvline(CONFIRM_YEAR, color="#2a9d8f", linewidth=1.2, linestyle="--",
                       alpha=0.8, label="2026 (partial year, liveness)")

        # 高亮 Kleinberg burst 区间
        interval = str(row.get("Burst_Interval", ""))
        if interval and "-" in interval:
            lo_s, hi_s = interval.split("-")
            lo, hi = int(lo_s), int(hi_s)
            burst_years = [y for y in all_years if lo <= y <= hi]
            burst_vals = [topic_prop.loc[y] for y in burst_years]
            ax.fill_between(
                [lo - 0.5, hi + 0.5], 0, max(topic_prop.values) * 1.15,
                color="#4ecdc4", alpha=0.12,
            )
            ax.plot(burst_years, burst_vals, color="#e63946", linewidth=2.2,
                    marker="s", markersize=5, zorder=5, label="burst interval")

        # baseline share 参考线
        p0 = row["Baseline_Share"]
        ax.axhline(y=p0, color="orange", linewidth=1, linestyle=":", alpha=0.7,
                   label=f"p0={p0:.4f}")

        # Panel title carries the topic identifier, the burst strength, the bootstrap
        # re-detection rate and the canonical short label, so Fig. 8 reads without
        # Table 5 and carries its own uncertainty: the rate is the share of the 1,000
        # year-stratified replicates in which the front is re-detected.
        label = (topic_labels or {}).get(topic_id, "")
        title = f"T{topic_id} · W = {row['Burst_Strength']:.0f}"
        det = redetection.get(topic_id)
        if det is not None:
            title += f" · re-det. {det:.1f}%"
        if label:
            title += f"\n{label}"
        ax.set_title(title, fontsize=9.0, fontweight="bold", linespacing=1.3)
        ax.set_xlabel("")
        ax.set_ylabel("Proportion", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linestyle=":", alpha=0.3)
        ax.legend(fontsize=7, loc="upper left")

    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    # No figure-level title: it duplicated the caption (fronts, W, method) and cost
    # vertical space, and the caption of Fig. 8 now carries that wording instead.
    fig.tight_layout()
    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("雏形轨迹图保存至: %s", output_path)


def plot_burst_index_radar(
    df_results: pd.DataFrame,
    output_path: Path,
    top_n: int = TOP_N_EMERGING,
) -> None:
    """绘制多雏形主题的多维指标雷达图 (Radar / Spider Chart)。

    每个轴代表一个检测指标 (归一化后)，用于直观对比不同雏形主题的
    多维特征差异。适合放入论文的 Supplementary Materials。

    Args:
        df_results: 检测结果 DataFrame。
        output_path: 图表保存路径 (PNG)。
        top_n: 展示前 N 个主题。
    """
    import matplotlib.pyplot as plt

    if df_results.empty:
        logger.warning("无数据可绘图 --- 跳过雷达图")
        return

    df_plot = df_results.head(min(top_n, 6))  # 最多 6 个主题，避免雷达图过于拥挤

    # 三维特征
    dimensions = ["Cold start\n(1 - Hist mean)", "Burst\nstrength W", "Burst\nshare p1"]
    categories = len(dimensions)

    metrics_raw = pd.DataFrame({
        "Cold start\n(1 - Hist mean)": 1.0 - df_plot["Hist_Mean_Share"] / df_plot["Hist_Mean_Share"].max(),
        "Burst\nstrength W": df_plot["Burst_Strength"] / df_plot["Burst_Strength"].max(),
        "Burst\nshare p1": df_plot["Burst_Share"] / df_plot["Burst_Share"].max(),
    })

    angles = np.linspace(0, 2 * np.pi, categories, endpoint=False).tolist()
    angles += angles[:1]  # 闭合

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(df_plot)))

    for i, (_, row) in enumerate(df_plot.iterrows()):
        topic_id = int(row["Topic"])
        values = metrics_raw.iloc[i].tolist()
        values += values[:1]  # 闭合

        ax.fill(angles, values, alpha=0.08, color=colors[i])
        ax.plot(angles, values, "o-", linewidth=2, color=colors[i],
                label=f"T{topic_id} (W={row['Burst_Strength']:.0f})", markersize=5)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dimensions, fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=7)
    ax.set_title(
        "Multi-Dimensional Profile of Candidate Emerging Research Fronts",
        fontsize=13, fontweight="bold", pad=20,
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.10), fontsize=9)

    fig.tight_layout()
    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("雷达图保存至: %s", output_path)


# =============================================================================
# 敏感性分析：gamma（转移代价）与 novelty floor（冷启动阈值）
# =============================================================================

NOVELTY_GRID = (0.0025, 0.005, 0.01)


def _sensitivity_pass_sets(
    yearly_counts: pd.DataFrame,
    total_per_year: Dict[int, int],
    param: str,
    values,
    kwargs_builder,
) -> Tuple[Dict[str, List[int]], pd.DataFrame]:
    """Run ``detect_emerging_tech_prototypes`` over a parameter grid.

    Returns ``(per_config_passed, long_df)`` where ``per_config_passed`` maps a
    config name to the sorted list of passing topic ids.
    """
    per_config_passed: Dict[str, List[int]] = {}
    frames: List[pd.DataFrame] = []
    for v in values:
        name = f"{param}_{v}"
        kw = kwargs_builder(v)
        df = detect_emerging_tech_prototypes(
            yearly_counts=yearly_counts,
            total_per_year=total_per_year,
            return_all=True,
            **kw,
        )
        passed_ids = sorted(int(t) for t in df.loc[df["Passed_Filter"], "Topic"])
        per_config_passed[name] = passed_ids
        sub = df[["Topic", "Burst_Strength", "Burst_Interval", "Active_Terminal",
                  "Novelty_Pass", "Liveness_2026_H1", "Passed_Filter",
                  "Fail_Reason"]].copy()
        sub.insert(0, "Config", name)
        sub.insert(1, "Param_Value", v)
        frames.append(sub)
    return per_config_passed, pd.concat(frames, ignore_index=True)


def _write_sensitivity_summary(
    config: Config,
    prefix: str,
    per_config_passed: Dict[str, List[int]],
    sens_df: pd.DataFrame,
) -> Tuple[Path, Path, dict]:
    names = list(per_config_passed.keys())
    sets = {n: set(per_config_passed[n]) for n in names}
    union_all = sorted(set().union(*sets.values()))
    stable_core = sorted(set.intersection(*sets.values()))
    jaccard = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ia, ib = sets[a], sets[b]
            jaccard[f"{a}_vs_{b}"] = round(len(ia & ib) / max(1, len(ia | ib)), 3)
    summary = {
        "configs": {n: {"passed": per_config_passed[n]} for n in names},
        "n_passed_per_config": {n: len(per_config_passed[n]) for n in names},
        "stable_core": stable_core,
        "passed_in_any_config": union_all,
        "passed_only_in_subset": [t for t in union_all if t not in stable_core],
        "pairwise_jaccard": jaccard,
    }
    csv_path = config.paths.processed_dir / f"emerging_{prefix}_sensitivity.csv"
    json_path = config.paths.processed_dir / f"emerging_{prefix}_sensitivity_summary.json"
    ensure_dir(csv_path.parent)
    sens_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return csv_path, json_path, summary


def run_gamma_sensitivity(
    config: Config,
    yearly_counts: pd.DataFrame,
    total_per_year: Dict[int, int],
) -> pd.DataFrame:
    """转移代价 gamma ∈ {0.5, 1.0, 2.0} 的敏感性分析（Appendix S1 证据）。"""
    per_config_passed, sens_df = _sensitivity_pass_sets(
        yearly_counts, total_per_year, "gamma", GAMMA_GRID,
        lambda g: {"gamma": g},
    )
    csv_path, json_path, summary = _write_sensitivity_summary(
        config, "gamma", per_config_passed, sens_df)
    logger.info("=" * 60)
    logger.info("【gamma 敏感性分析】")
    for n in per_config_passed:
        logger.info("  %-10s -> %d 个通过: %s",
                    n, len(per_config_passed[n]), per_config_passed[n])
    logger.info("  稳定核心: %s", summary["stable_core"])
    logger.info("  两两 Jaccard: %s", summary["pairwise_jaccard"])
    logger.info("  CSV : %s", csv_path)
    logger.info("  JSON: %s", json_path)
    logger.info("=" * 60)
    return sens_df


def run_novelty_sensitivity(
    config: Config,
    yearly_counts: pd.DataFrame,
    total_per_year: Dict[int, int],
) -> pd.DataFrame:
    """冷启动阈值 ∈ {0.0025, 0.005, 0.01} 的敏感性分析（Appendix S1 证据）。"""
    per_config_passed, sens_df = _sensitivity_pass_sets(
        yearly_counts, total_per_year, "novelty", NOVELTY_GRID,
        lambda f: {"cold_start_threshold": f},
    )
    csv_path, json_path, summary = _write_sensitivity_summary(
        config, "novelty", per_config_passed, sens_df)
    logger.info("=" * 60)
    logger.info("【novelty floor 敏感性分析】")
    for n in per_config_passed:
        logger.info("  %-14s -> %d 个通过: %s",
                    n, len(per_config_passed[n]), per_config_passed[n])
    logger.info("  稳定核心: %s", summary["stable_core"])
    logger.info("  两两 Jaccard: %s", summary["pairwise_jaccard"])
    logger.info("  CSV : %s", csv_path)
    logger.info("  JSON: %s", json_path)
    logger.info("=" * 60)
    return sens_df


def run_recency_sensitivity(
    config: Config,
    yearly_counts: pd.DataFrame,
    total_per_year: Dict[int, int],
) -> pd.DataFrame:
    """recency 判据参数的敏感性分析（Appendix S1 证据）。

    两组扫描：
      * onset      in ``ONSET_GRID``      --- 每个点配套 ``historical_end = onset - 1``，
        以保持主设计的非重叠不变量（novelty 窗口闭合于最早可能 onset 的前一年）；
      * min-span   in ``MIN_SPAN_GRID``   --- 窗口重叠与该参数无关，故固定生产设定
        (onset=2021, window 1972--2020)。

    输出 ``emerging_recency_sensitivity.csv`` / ``_summary.json``；JSON 里的
    ``design_invariant`` 字段把联动规则写明，便于表注披露。
    """
    onset_passed, onset_df = _sensitivity_pass_sets(
        yearly_counts, total_per_year, "onset", ONSET_GRID,
        lambda onset: {"min_burst_onset": int(onset), "historical_end": int(onset) - 1},
    )
    onset_df = onset_df.copy()
    onset_df["Family"] = "onset"
    onset_df["Novelty_Window_End"] = onset_df["Param_Value"].astype(int) - 1

    span_passed, span_df = _sensitivity_pass_sets(
        yearly_counts, total_per_year, "min_span", MIN_SPAN_GRID,
        lambda span: {"min_burst_length": int(span)},
    )
    span_df = span_df.copy()
    span_df["Family"] = "min_span"
    span_df["Novelty_Window_End"] = HISTORICAL_END

    sens_df = pd.concat([onset_df, span_df], ignore_index=True)
    csv_path = config.paths.processed_dir / "emerging_recency_sensitivity.csv"
    json_path = config.paths.processed_dir / "emerging_recency_sensitivity_summary.json"
    ensure_dir(csv_path.parent)
    sens_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    def _family_summary(passed: Dict[str, List[int]]) -> dict:
        names = list(passed.keys())
        sets = {n: set(passed[n]) for n in names}
        jac = {}
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                jac[f"{a}_vs_{b}"] = round(len(sets[a] & sets[b]) / max(1, len(sets[a] | sets[b])), 3)
        return {
            "configs": {n: {"passed": passed[n]} for n in names},
            "n_passed_per_config": {n: len(passed[n]) for n in names},
            "stable_core": sorted(set.intersection(*sets.values())) if sets else [],
            "passed_in_any_config": sorted(set().union(*sets.values())) if sets else [],
            "pairwise_jaccard": jac,
        }

    summary = {
        "design_invariant": (
            "novelty_window_end = onset - 1 for every onset variant, "
            "so the non-overlapping-window design is preserved throughout"
        ),
        "production": {
            "min_burst_onset": MIN_BURST_ONSET,
            "min_burst_length": MIN_BURST_LENGTH,
            "historical_start": HISTORICAL_START,
            "historical_end": HISTORICAL_END,
        },
        "families": {
            "onset": {**_family_summary(onset_passed),
                      "novelty_window_end_per_variant": {str(k): int(k) - 1 for k in ONSET_GRID}},
            "min_span": _family_summary(span_passed),
        },
    }
    all_sets = ([set(onset_passed[k]) for k in onset_passed]
                + [set(span_passed[k]) for k in span_passed])
    summary["passed_under_all_variants"] = sorted(set.intersection(*all_sets))
    summary["passed_under_any_variant"] = sorted(set().union(*all_sets))
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info("=" * 60)
    logger.info("【recency 敏感性分析】")
    for n in onset_passed:
        logger.info("  %-12s (window end %d) -> %d 个通过: %s",
                    n, int(n.split("_")[1]) - 1, len(onset_passed[n]), onset_passed[n])
    for n in span_passed:
        logger.info("  %-12s (window end %d) -> %d 个通过: %s",
                    n, HISTORICAL_END, len(span_passed[n]), span_passed[n])
    logger.info("  全部变体稳定核心: %s", summary["passed_under_all_variants"])
    logger.info("  CSV : %s", csv_path)
    logger.info("  JSON: %s", json_path)
    logger.info("=" * 60)
    return sens_df


# =============================================================================
# 主入口
# =============================================================================


def run_emerging_tech_detection(
    config: Config,
    model_name: str | None = None,
    skip_sensitivity: bool = False,
) -> pd.DataFrame:
    """运行完整的新兴技术雏形检测管线。

    执行流程:
        1. 加载 BERTopic 模型和文档级数据
        2. 按年度聚合主题频数（独立于 BERTopic DTM 分箱）
        3. 运行三判据雏形过滤算法 (novelty 历史期 1972--2020, 自动机窗口
           1972--2025, 2026 H1 仅作 liveness 确认)
        4. 敏感性分析: gamma / novelty floor / recency (onset x novelty 窗口联动,
           min-span), 输出 CSV + JSON
        5. 格式化打印检测结果（含虚拟前沿示例）
        6. 生成可视化图表 (Burst Index 条形图 + 轨迹图 + 雷达图)

    Args:
        config: 已验证的配置对象。
        model_name: 可选，指定模型目录名（如 ``"bertopic_model_sorted_ngram"``）。
        skip_sensitivity: 若 True，跳过窗口敏感性分析。

    Returns:
        检测结果 DataFrame (通过四维过滤的主题)，可能为空。
    """
    # ---- Step 1: 加载模型和数据 ----
    with Timer("加载模型与数据"):
        model, doc_topic_df, cleaned_df = load_model_and_topic_data(config, model_name=model_name)

    # ---- Step 2: 年度频数聚合 ----
    with Timer("按年聚合主题频数"):
        yearly_counts, total_per_year = compute_yearly_topic_counts(doc_topic_df)

    # ---- Step 3: 运行 BERTopic yearly topics_over_time ----
    # 提取文档文本和年份（与模型拟合时对齐）
    docs_list = cleaned_df["text"].fillna("").astype(str).tolist()
    docs_list = [d.strip() for d in docs_list if d.strip()]
    years_list = cleaned_df["PY"].astype(int).tolist()

    # 对齐文档与模型内部索引
    n_model = len(model.topics_)
    if len(docs_list) != n_model:
        logger.warning("文档数 (%d) 与模型主题数 (%d) 不一致，截断至模型长度",
                       len(docs_list), n_model)
        docs_list = docs_list[:n_model]
        years_list = years_list[:n_model]

    with Timer("BERTopic 年度 topics_over_time"):
        df_topics_over_time = run_yearly_topics_over_time(model, docs_list, years_list)

    # ---- Step 4: 两阶段雏形检测 ----
    with Timer("两阶段雏形捕捉算法"):
        df_emerging = detect_emerging_tech_prototypes(
            yearly_counts=yearly_counts,
            total_per_year=total_per_year,
            df_topics_over_time=df_topics_over_time,
        )

    # ---- Step 4b: 敏感性分析 (gamma + novelty + recency, 论文附录证据) ----
    if not skip_sensitivity:
        with Timer("gamma 敏感性分析"):
            run_gamma_sensitivity(config, yearly_counts, total_per_year)
        with Timer("novelty floor 敏感性分析"):
            run_novelty_sensitivity(config, yearly_counts, total_per_year)
        with Timer("recency 敏感性分析"):
            run_recency_sensitivity(config, yearly_counts, total_per_year)

    # ---- Step 5: 保存结果 ----
    _suffix = f"_{model_name}" if model_name else ""
    csv_path = config.paths.processed_dir / f"emerging_tech_prototypes{_suffix}.csv"
    if not df_emerging.empty:
        ensure_dir(csv_path.parent)
        df_emerging.to_csv(csv_path, index=False, encoding="utf-8-sig")
        logger.info("雏形检测结果保存至: %s", csv_path)

    # ---- Step 6: 打印报告 ----
    print_top_emerging_prototypes(df_emerging, model=model, top_n=TOP_N_EMERGING)

    # ---- Step 7: 可视化 ----
    fig_dir = config.paths.figures_dir
    _suffix = f"_{model_name}" if model_name else ""

    with Timer("雏形检测可视化"):
        # 7a. Burst Index 条形图
        plot_burst_index_barchart(
            df_emerging,
            fig_dir / f"emerging_tech_burst_index{_suffix}.png",
            top_n=TOP_N_EMERGING,
        )

        # 7b. 雏形主题时间轨迹
        plot_emerging_trajectories(
            yearly_counts, total_per_year, df_emerging,
            fig_dir / f"emerging_tech_trajectories{_suffix}.png",
            top_n=TOP_N_EMERGING,
        )

        # 7c. 多维雷达图
        if len(df_emerging) >= 2:
            plot_burst_index_radar(
                df_emerging,
                fig_dir / f"emerging_tech_radar{_suffix}.png",
                top_n=TOP_N_EMERGING,
            )

    logger.info("新兴技术雏形检测完成。")
    return df_emerging


def main() -> None:
    """CLI 入口。"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Step 5: 新兴技术雏形检测 (Emerging Tech Prototype Detection)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    pixi run python -m src.emerging_tech_detection
    pixi run python -m src.emerging_tech_detection --config config.yaml
    pixi run python -m src.emerging_tech_detection --cold-start 0.008 --gamma 2.0
    pixi run python -m src.emerging_tech_detection --skip-sensitivity
        """,
    )
    parser.add_argument("--config", type=str, default="config.yaml", help="配置文件路径")
    parser.add_argument("--model-name", type=str, default=None,
                        help="模型目录名 (默认从 config 读取; 如 bertopic_model_sorted_ngram)")
    # 允许从 CLI 覆盖参数（方便灵敏度调参）
    parser.add_argument("--cold-start", type=float, default=COLD_START_THRESHOLD,
                        help=f"冷启动阈值 novelty floor (默认: {COLD_START_THRESHOLD})")
    parser.add_argument("--gamma", type=float, default=GAMMA,
                        help=f"Kleinberg 转移代价 gamma (默认: {GAMMA})")
    parser.add_argument("--skip-sensitivity", action="store_true",
                        help="跳过敏感性分析 (默认运行 gamma + novelty + recency 三组)")
    args = parser.parse_args()

    config = get_config(args.config)
    from src.utils import setup_logging

    setup_logging(config.system.log_level, config.system.log_file)

    try:
        df_emerging = run_emerging_tech_detection(
            config, model_name=args.model_name, skip_sensitivity=args.skip_sensitivity,
        )
        # 如果 CLI 参数与默认不同，重新运行检测并覆盖结果
        if args.cold_start != COLD_START_THRESHOLD or args.gamma != GAMMA:
            logger.info(
                "使用自定义参数重新检测: cold_start=%.4f, gamma=%.2f",
                args.cold_start, args.gamma,
            )
            _, doc_topic_df, _ = load_model_and_topic_data(config)
            yearly_counts, total_per_year = compute_yearly_topic_counts(doc_topic_df)
            df_emerging = detect_emerging_tech_prototypes(
                yearly_counts=yearly_counts,
                total_per_year=total_per_year,
                cold_start_threshold=args.cold_start,
                gamma=args.gamma,
            )
            print_top_emerging_prototypes(df_emerging, top_n=TOP_N_EMERGING)
    except Exception:
        logger.exception("新兴技术雏形检测失败")
        raise


if __name__ == "__main__":
    main()
