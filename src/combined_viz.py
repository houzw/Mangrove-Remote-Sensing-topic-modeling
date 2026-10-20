"""Combined visualizations — cross-cutting bibliometric × topic visualizations.

Provides five publication-quality figures that integrate multiple data dimensions:

======  ======  ===========================================================
Priority  Fig    Description
======  ======  ===========================================================
P0       P0-1   Bubble scatter: mangrove area vs publications vs dominant topic
P0       P0-2   Topic × Year heatmap with hierarchical row clustering
P1       P1-1   Continuous yearly streamgraph (Plotly interactive)
P2       P2-1   Topic lifecycle classification: Hot / Cold / Stable
P2       P2-2   Dual choropleth: publication counts + dominant topic map
======  ======  ===========================================================

Usage:
    python src/combined_viz.py              # uses default config.yaml
    python src/combined_viz.py --config path/to/config.yaml
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from scipy.cluster.hierarchy import dendrogram, linkage

from src.config import Config, get_config
from src.utils import Timer, ensure_dir

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ══════════════════════════════════════════════════════════════════════════════


def _load_area_data(config: Config) -> pd.DataFrame:
    """Load HGMF mangrove area data and return {Country: Area} mapping."""
    area_path = config.paths.external_dir / config.country_analysis.mangrove_area_file
    if not area_path.exists():
        logger.warning("Mangrove area file not found: %s", area_path)
        return pd.DataFrame({"Country": [], "Area": []})
    df = pd.read_csv(area_path)
    df = df.rename(columns={config.country_analysis.area_column: "Area"})
    return df[["Country", "Area"]]


def _load_si_matrix(config: Config) -> pd.DataFrame:
    """Load spatial_topic_si.csv and return the SI matrix (rows=countries, cols=topics)."""
    si_path = config.paths.processed_dir / config.spatial_topic_analysis.output.si_matrix
    if not si_path.exists():
        raise FileNotFoundError(f"SI matrix not found: {si_path}. Run spatial_topic_analysis first.")
    si = pd.read_csv(si_path, encoding="utf-8-sig", index_col=0)
    return si


def _load_topic_labels(config: Config) -> Dict[int, str]:
    """Build {topic_id: short_label} from topics_info.csv."""
    topics_csv = config.paths.processed_dir / config.topic_modeling.output.topics_info
    if not topics_csv.exists():
        return {}
    df = pd.read_csv(topics_csv, encoding="utf-8-sig")
    labels: Dict[int, str] = {}
    for _, row in df.iterrows():
        tid = int(row.get("Topic", -1))
        if tid == -1:
            continue
        short = row.get("Short_Label", None)
        if pd.notna(short) and isinstance(short, str) and short.strip():
            labels[tid] = short.strip()
        else:
            labels[tid] = f"Topic_{tid}"
    return labels


# HGMF country name → processed country name mapping.
# The HGMF dataset uses slightly different names than WoS-extracted countries.
_HGMF_NAME_MAP: Dict[str, str] = {
    "Venzuela": "Venezuela",
    "Congo.DRC": "Democratic Republic of the Congo",
    "Solomon Is.": "Solomon Islands",
    "Trinidad & Tobago": "Trinidad and Tobago",
    "St. Vincent & the Grenadines": "St. Vincent & the Grenadines",
    "Timor Leste": "Timor-Leste",
    "Netherlands Antilles": "Netherlands",
    "Wallis & Futuna": "France",
    "French Guiana": "France",
    "New Caledonia": "France",
    "Mayotte": "France",
    "Guadeloupe": "France",
    "Martinique": "France",
    "American Samoa": "United States",
    "Guam": "United States",
    "Northern Mariana Islands": "United States",
    "Puerto Rico": "Puerto Rico",
    "The Gambia": "Gambia",
    "The Bahamas": "Bahamas",
    "Camerocn": "Cameroon",
}


def _compute_country_pubs(config: Config) -> pd.DataFrame:
    """Compute publication counts per country from cleaned data."""
    cleaned_pq = config.paths.processed_dir / config.preprocessing.output_file.replace(".csv", ".parquet")
    if cleaned_pq.exists():
        df = pd.read_parquet(cleaned_pq)
    else:
        df = pd.read_csv(config.paths.processed_dir / config.preprocessing.output_file, encoding="utf-8-sig")

    # Use config's country name mapping
    mapping = {
        "USA": "United States",
        "England": "United Kingdom",
        "Scotland": "United Kingdom",
        "Wales": "United Kingdom",
        "Peoples R China": "China",
        "Russia": "Russian Federation",
        "South Korea": "Republic of Korea",
        "North Korea": "Dem. Rep. Korea",
        "Vietnam": "Vietnam",
        "Viet Nam": "Vietnam",
        "Iran": "Iran",
        "Tanzania": "Tanzania",
        "Brunei": "Brunei",
        "Ivory Coast": "Cote d'Ivoire",
        "Côte d'Ivoire": "Cote d'Ivoire",
        "Congo": "Congo",
        "Dem Rep Congo": "Democratic Republic of the Congo",
        "Venezuela": "Venezuela",
        "Cameroon": "Cameroon",
        "Myanmar": "Myanmar",
        "Burma": "Myanmar",
        "Papua New Guinea": "Papua New Guinea",
        "Trinidad and Tobago": "Trinidad & Tobago",
        "Gambia": "Gambia",
        "Bahamas": "Bahamas",
        "Taiwan": "China",
        "Syria": "Syria",
        "Laos": "Laos",
    }

    records = []
    for cu_val in df["CU"].dropna():
        countries = [c.strip() for c in str(cu_val).split(";")]
        for raw_name in countries:
            if raw_name and raw_name.lower() != "unknown":
                records.append(mapping.get(raw_name, raw_name))

    pub_series = pd.Series(records).value_counts().reset_index()
    pub_series.columns = ["Country", "Publications"]
    return pub_series


# ══════════════════════════════════════════════════════════════════════════════
# P0-1: Bubble scatter — Area vs Publications vs Dominant Topic
# ══════════════════════════════════════════════════════════════════════════════


def _draw_bidirectional_inset(ax_ins, x_idx, area_vals, pub_vals, bar_w, labeled):
    """Draw shared-baseline bidirectional inset: area ↑, pubs ↓ (normalized)."""
    area_norm = area_vals / area_vals.max()
    pub_norm = pub_vals / pub_vals.max()

    ax_ins.bar(x_idx, area_norm, width=bar_w,
               color="#29cc3f", alpha=0.82, edgecolor="white", linewidth=0.3, zorder=3)
    ax_ins.bar(x_idx, -pub_norm, width=bar_w,
               color="#4472C4", alpha=0.82, edgecolor="white", linewidth=0.3, zorder=3)

    # Value labels at bar tips
    for i, (a, p) in enumerate(zip(area_vals, pub_vals)):
        a_str = f"{a:,.0f}" if a >= 10 else f"{a:.1f}"
        ax_ins.text(i, area_norm[i] + 0.04, a_str, ha="center", va="bottom",
                    fontsize=4.2, color="#29cc3f", fontweight="bold", rotation=90)
        ax_ins.text(i, -pub_norm[i] - 0.04, str(p), ha="center", va="top",
                    fontsize=4.2, color="#4472C4", fontweight="bold", rotation=90)

    ax_ins.axhline(y=0, color="#333333", linewidth=0.8, zorder=2)
    ax_ins.set_ylim(-1.25, 1.35)
    ax_ins.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax_ins.set_yticklabels(["1.0", "0.5", "0", "0.5", "1.0"], fontsize=5)
    ax_ins.set_ylabel("Normalized scale", fontsize=6, color="#555555")
    ax_ins.tick_params(axis="y", labelsize=5, colors="#555555")
    ax_ins.spines["top"].set_visible(False)
    ax_ins.spines["left"].set_visible(False)
    ax_ins.spines["right"].set_visible(False)

    ax_ins.legend(
        handles=[
            Patch(facecolor="#29cc3f", alpha=0.82, edgecolor="white",
                  label=f"Area (km²) — max {area_vals.max():,.0f}"),
            Patch(facecolor="#4472C4", alpha=0.82, edgecolor="white",
                  label=f"Pubs — max {pub_vals.max():.0f}"),
        ],
        loc="lower left", fontsize=4.5, framealpha=0.85, ncol=1,
        borderpad=0.4, labelspacing=0.3,
    )

    ax_ins.set_xticks(list(x_idx))
    ax_ins.set_xticklabels(labeled["Country"].values, rotation=60, ha="right", fontsize=5.5)
    ax_ins.set_title("Labeled Countries  |  Area ↑  ·  Pubs ↓", fontsize=7,
                     fontweight="bold", pad=3)


def _draw_original_inset(ax_ins, x_idx, area_vals, pub_vals, bar_w, labeled):
    """Draw original 3-axis inset: area (log ↑), pubs (inverted ↓), density (dashed)."""
    # Left y-axis: area upward (green, log)
    ax_ins.bar(x_idx, area_vals, width=bar_w,
               color="#29cc3f", alpha=0.80, edgecolor="white", linewidth=0.2)
    ax_ins.set_yscale("log")
    ax_ins.set_ylim(None, max(area_vals) * 2)
    ax_ins.set_ylabel("Area (km², log)", fontsize=6, color="#29cc3f")
    ax_ins.tick_params(axis="y", labelsize=5, colors="#29cc3f")
    ax_ins.spines["left"].set_color("#29cc3f")
    ax_ins.spines["top"].set_visible(False)

    # Right y-axis: publications downward (blue, inverted)
    ax_ins2 = ax_ins.twinx()
    ax_ins2.bar(x_idx, pub_vals, width=bar_w,
                color="#4472C4", alpha=0.80, edgecolor="white", linewidth=0.2)
    ax_ins2.invert_yaxis()
    ax_ins2.set_ylim(max(pub_vals) * 1.2, 0)
    ax_ins2.set_ylabel("Publications", fontsize=6, color="#4472C4")
    ax_ins2.tick_params(axis="y", labelsize=5, colors="#4472C4")
    ax_ins2.spines["right"].set_color("#4472C4")
    ax_ins2.spines["top"].set_visible(False)

    # Third y-axis (offset right): research density dashed line
    density_vals = pub_vals / area_vals
    ax_ins3 = ax_ins.twinx()
    ax_ins3.spines["right"].set_position(("axes", 1.14))
    ax_ins3.plot(x_idx, density_vals, color="#D44528", linestyle="--",
                 linewidth=1.0, marker=".", markersize=3, alpha=0.85, zorder=5)
    ax_ins3.set_ylabel("Pubs / km²", fontsize=6, color="#D44528")
    ax_ins3.tick_params(axis="y", labelsize=5, colors="#D44528")
    ax_ins3.spines["right"].set_color("#D44528")
    ax_ins3.spines["top"].set_visible(False)
    if min(density_vals) > 0:
        ax_ins3.set_yscale("log")
        ax_ins3.set_ylim(min(density_vals) * 0.5, max(density_vals) * 2)

    ax_ins.axhline(y=0, color="#333333", linewidth=0.5)
    ax_ins.set_xticks(list(x_idx))
    ax_ins.set_xticklabels(labeled["Country"].values, rotation=60, ha="right", fontsize=5)
    ax_ins.set_title("Labeled Countries: Area (km²) ↑  /  Pubs ↓  /  Density ––", fontsize=7,
                     fontweight="bold", pad=3)


def plot_area_pubs_bubble(
    config: Config,
    output_path: Path,
    min_pubs: int = 30,
    inset_style: str = "bidirectional",
) -> None:
    """Bubble scatter: x = mangrove area, y = publications, color = dominant topic.

    Each country is a bubble positioned by its mangrove area and publication
    count, sized by publications, and colored by its dominant (highest-SI)
    research topic.  Reference lines partition the space into quadrants.

    Args:
        config: Validated configuration.
        output_path: Where to save the PNG.
        min_pubs: Minimum publications for a country to be labelled.
        inset_style: "bidirectional" (shared baseline, normalized) or
            "original" (3-axis with log area, inverted pubs, density line).
    """
    vis = config.visualization
    plt.rcParams["font.family"] = vis.font_family

    # --- Load & merge data ---
    area_df = _load_area_data(config)
    if area_df.empty:
        logger.warning("No mangrove area data — skipping bubble plot")
        return

    pub_stats = _compute_country_pubs(config)
    si_matrix = _load_si_matrix(config)
    topic_labels = _load_topic_labels(config)

    # Normalize area country names
    area_df["Country_orig"] = area_df["Country"]
    area_df["Country"] = area_df["Country"].map(_HGMF_NAME_MAP).fillna(area_df["Country"])

    # Aggregate area by country — overseas territories (French Guiana → France,
    # Guam → United States, etc.) produce duplicate country rows after mapping
    area_df = area_df.groupby("Country", as_index=False)["Area"].sum()
    logger.info("  %d unique countries after area aggregation", len(area_df))

    # Merge area + pubs
    merged = pub_stats.merge(area_df, on="Country", how="inner")
    logger.info("Bubble plot: %d countries with both area and pub data", len(merged))

    # Add dominant topic
    dominant = si_matrix.idxmax(axis=1)
    max_si = si_matrix.max(axis=1)
    dominant_df = pd.DataFrame({
        "Country": si_matrix.index,
        "Dominant_Topic": dominant.values.astype(int),
        "Max_SI": max_si.values,
    })
    merged = merged.merge(dominant_df, on="Country", how="left")

    # Filter tiny countries for cleaner plot
    merged = merged[merged["Area"] > 0].copy()

    if len(merged) == 0:
        logger.warning("No countries after merging — skipping bubble plot")
        return

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(14, 10))
    fig.subplots_adjust(left=0.08, right=0.94, top=0.94, bottom=0.15)

    # Color by dominant topic
    all_topics = sorted(merged["Dominant_Topic"].dropna().unique())
    n_topics_display = len(all_topics)
    if n_topics_display <= 10:
        cmap = plt.cm.tab10
    elif n_topics_display <= 20:
        cmap = plt.cm.tab20
    else:
        cmap = plt.cm.gist_ncar

    topic_color_map = {
        int(tid): cmap(i / max(n_topics_display, 1))
        for i, tid in enumerate(all_topics)
    }
    # NaN (no dominant topic / SI<=1) → grey
    topic_color_map[-1] = np.array([0.75, 0.75, 0.75, 0.6])

    # Size mapping
    pub_min, pub_max = merged["Publications"].min(), merged["Publications"].max()
    size_min, size_max = 60, 900

    for _, row in merged.iterrows():
        tid = int(row["Dominant_Topic"]) if pd.notna(row["Dominant_Topic"]) else -1
        color = topic_color_map.get(tid, topic_color_map[-1])
        size = size_min + (size_max - size_min) * (
            (row["Publications"] - pub_min) / (pub_max - pub_min)
        ) ** 0.7

        ax.scatter(
            row["Area"], row["Publications"],
            s=size, c=[color], edgecolors="white", linewidth=1.0,
            alpha=0.85, zorder=3,
        )

        # Label countries with >= min_pubs
        if row["Publications"] >= min_pubs:
            offset_y = 5 + size * 0.015
            ax.annotate(
                row["Country"],
                (row["Area"], row["Publications"]),
                textcoords="offset points",
                xytext=(0, offset_y),
                fontsize=7.5,
                fontweight="bold",
                ha="center",
                alpha=0.85,
            )

    # Log scale for area (mangrove areas span 4 orders of magnitude)
    ax.set_xscale("log")
    ax.set_xlabel(f"Mangrove Area ({config.country_analysis.area_unit}) — log scale", fontsize=11)
    ax.set_ylabel("Number of Publications", fontsize=11)

    # Median reference lines
    median_area = merged["Area"].median()
    median_pubs = merged["Publications"].median()
    ax.axhline(y=median_pubs, color="#888888", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.axvline(x=median_area, color="#888888", linewidth=0.8, linestyle="--", alpha=0.6)
    # Annotate median values on axes
    ax.annotate(f"{median_area:,.0f}",
                xy=(median_area, 0), xycoords="data",
                xytext=(0, -24), textcoords="offset points",
                fontsize=7.5, color="#888888", ha="center", va="top",
                arrowprops=dict(arrowstyle="-", color="#cccccc", lw=0.6))
    ax.annotate(f"{median_pubs:.0f}",
                xy=(0, median_pubs), xycoords=ax.get_yaxis_transform(),
                xytext=(-8, 0), textcoords="offset points",
                fontsize=7.5, color="#888888", ha="right", va="center",
                arrowprops=dict(arrowstyle="-", color="#cccccc", lw=0.6))

    # Quadrant labels — axes-fraction corners, out of data-point way
    # y=0.02 is ≈12 pubs, just above the median_pubs=9 line (at ~1.5% of y range)
    q_labels = [
        dict(text="Small Area,\nHigh Pubs\n(Tech-Driven)",     x=0.05, y=0.93, ha="left",   va="top"),
        dict(text="High Area,\nHigh Pubs\n(Research Leaders)",  x=0.97, y=0.93, ha="right",  va="top"),
        dict(text="High Area,\nLow Pubs\n(Understudied)",       x=0.97, y=0.005, ha="right",  va="bottom"),
        dict(text="Small Area,\nLow Pubs",                      x=0.05, y=0.005, ha="left",   va="bottom"),
    ]
    for kw in q_labels:
        ax.text(kw["x"], kw["y"], kw["text"],
                transform=ax.transAxes, fontsize=8, color="0.3",
                ha=kw["ha"], va=kw["va"], alpha=0.7, style="italic")

    # Inset: country-level area vs pubs comparison
    # Position is in figure-fraction coordinates (fig size 14×10 in, 300 dpi → 4200×3000 px).
    # ── Manual adjustment guide ───────────────────────────────────────────
    # X = main_pos.x0 + x_offset      │  ↑ x_offset → move RIGHT  (~1 px ≈ 0.00024)
    # Y = main_pos.y1 - inset_h - gap │  ↓ gap      → move DOWN   (~1 px ≈ 0.00033)
    # Quick rebuild: pixi run python -m src.combined_viz --step bubble
    # ──────────────────────────────────────────────────────────────────────
    main_pos = ax.get_position()
    inset_w, inset_h = 0.40, 0.24
    ax_ins = fig.add_axes([
        main_pos.x0 + 0.055, main_pos.y1 - inset_h - 0.2,
        inset_w, inset_h,
    ])
    labeled = merged[merged["Publications"] >= min_pubs].sort_values("Area", ascending=False)
    n_labeled = len(labeled)
    x_idx = np.arange(n_labeled)
    area_vals = labeled["Area"].values  # km² (HGMF data is already in km²)
    pub_vals = labeled["Publications"].values
    bar_w = 0.55

    if inset_style == "bidirectional":
        _draw_bidirectional_inset(ax_ins, x_idx, area_vals, pub_vals, bar_w, labeled)
    else:
        _draw_original_inset(ax_ins, x_idx, area_vals, pub_vals, bar_w, labeled)

    # Legend for dominant topics
    legend_patches = []
    for tid in all_topics[:15]:  # show top 15 to keep legend manageable
        label = topic_labels.get(int(tid), f"T{int(tid)}")
        legend_patches.append(
            Patch(facecolor=topic_color_map.get(int(tid), topic_color_map[-1]),
                  edgecolor="white", linewidth=0.5, label=label)
        )
    if len(all_topics) > 15:
        legend_patches.append(Patch(facecolor="none", edgecolor="none", label=f"... +{len(all_topics) - 15} more"))

    ax.legend(
        handles=legend_patches,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.09),
        ncol=min(5, len(legend_patches)),
        fontsize=7,
        framealpha=0.9,
        title="Dominant Topic (highest SI)",
        title_fontsize=8,
    )

    ax.set_title(
        "Mangrove Remote Sensing: Mangrove Area vs Publication Output\n"
        "Bubble size = publications; Color = dominant research topic",
        fontsize=11,
        fontweight="bold",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle=":", alpha=0.3)

    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=vis.dpi, format=vis.figure_format)
    plt.close(fig)
    logger.info("P0-1 Bubble scatter saved: %s", output_path)


# ══════════════════════════════════════════════════════════════════════════════
# P0-1c: Combined bubble scatter + polar mirror (replaces bar chart inset)
# ══════════════════════════════════════════════════════════════════════════════


def plot_area_pubs_bubble_polar(
    config: Config,
    output_path: Path,
    min_pubs: int = 30,
    polar_min_pubs: int = 10,
    polar_max_countries: int = 30,
) -> None:
    """Bubble scatter with polar mirror inset in the "Small Area, High Pubs" quadrant.

    The polar mirror replaces the quadrant label — mangrove area extends inward
    (green), publication count outward (blue).

    Args:
        config: Validated configuration.
        output_path: Where to save the PNG.
        min_pubs: Minimum publications for country labelling in bubble panel.
        polar_min_pubs: Minimum publications for country inclusion in polar inset.
        polar_max_countries: Max countries to show in the polar chart.
    """
    vis = config.visualization
    plt.rcParams["font.family"] = vis.font_family

    # --- Load & merge data (shared) ---
    area_df = _load_area_data(config)
    if area_df.empty:
        logger.warning("No mangrove area data — skipping bubble+polar plot")
        return

    pub_stats = _compute_country_pubs(config)
    si_matrix = _load_si_matrix(config)
    topic_labels = _load_topic_labels(config)

    # Normalize area country names
    area_df["Country_orig"] = area_df["Country"]
    area_df["Country"] = area_df["Country"].map(_HGMF_NAME_MAP).fillna(area_df["Country"])
    area_df = area_df.groupby("Country", as_index=False)["Area"].sum()

    # Merge area + pubs
    merged = pub_stats.merge(area_df, on="Country", how="inner")
    logger.info("Bubble+polar: %d countries with both area and pub data", len(merged))

    # Add dominant topic
    dominant = si_matrix.idxmax(axis=1)
    max_si = si_matrix.max(axis=1)
    dominant_df = pd.DataFrame({
        "Country": si_matrix.index,
        "Dominant_Topic": dominant.values.astype(int),
        "Max_SI": max_si.values,
    })
    merged = merged.merge(dominant_df, on="Country", how="left")
    merged = merged[merged["Area"] > 0].copy()

    if len(merged) == 0:
        logger.warning("No countries after merging — skipping bubble+polar plot")
        return

    # =========================================================================
    # Bubble scatter (main panel)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(16, 12))
    fig.subplots_adjust(left=0.08, right=0.94, top=0.94, bottom=0.15)

    # Color by dominant topic
    all_topics = sorted(merged["Dominant_Topic"].dropna().unique())
    n_topics_display = len(all_topics)
    if n_topics_display <= 10:
        cmap = plt.cm.tab10
    elif n_topics_display <= 20:
        cmap = plt.cm.tab20
    else:
        cmap = plt.cm.gist_ncar

    topic_color_map = {
        int(tid): cmap(i / max(n_topics_display, 1))
        for i, tid in enumerate(all_topics)
    }
    topic_color_map[-1] = np.array([0.75, 0.75, 0.75, 0.6])

    pub_min, pub_max = merged["Publications"].min(), merged["Publications"].max()
    # Size — figsize 16×12 vs original 14×10 → bubbles need ×1.44 to match
    # visual weight + 20% user-requested enlargement.
    #   size_min/max ↑ → all bubbles bigger
    #   exponent 0.7  ↑ → large/small contrast stronger (1.0 = linear)
    size_min, size_max = 86, 1300

    for _, row in merged.iterrows():
        tid = int(row["Dominant_Topic"]) if pd.notna(row["Dominant_Topic"]) else -1
        color = topic_color_map.get(tid, topic_color_map[-1])
        size = size_min + (size_max - size_min) * (
            (row["Publications"] - pub_min) / (pub_max - pub_min)
        ) ** 0.7

        ax.scatter(
            row["Area"], row["Publications"],
            s=size, c=[color], edgecolors="white", linewidth=1.0,
            alpha=0.85, zorder=3,
        )

        if row["Publications"] >= min_pubs:
            offset_y = 5 + size * 0.015
            ax.annotate(
                row["Country"],
                (row["Area"], row["Publications"]),
                textcoords="offset points",
                xytext=(0, offset_y),
                fontsize=7.5, fontweight="bold",
                ha="center", alpha=0.85,
            )

    ax.set_xscale("log")
    ax.set_xlabel(f"Mangrove Area ({config.country_analysis.area_unit}) — log scale", fontsize=11)
    ax.set_ylabel("Number of Publications", fontsize=11)

    median_area = merged["Area"].median()
    median_pubs = merged["Publications"].median()
    ax.axhline(y=median_pubs, color="#888888", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.axvline(x=median_area, color="#888888", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.annotate(f"{median_area:,.0f}",
                xy=(median_area, 0), xycoords="data",
                xytext=(0, -24), textcoords="offset points",
                fontsize=7.5, color="#888888", ha="center", va="top",
                arrowprops=dict(arrowstyle="-", color="#cccccc", lw=0.6))
    ax.annotate(f"{median_pubs:.0f}",
                xy=(0, median_pubs), xycoords=ax.get_yaxis_transform(),
                xytext=(-8, 0), textcoords="offset points",
                fontsize=7.5, color="#888888", ha="right", va="center",
                arrowprops=dict(arrowstyle="-", color="#cccccc", lw=0.6))

    # Quadrant labels
    ax.text(0.97, 0.93, "High Area,\nHigh Pubs\n(Research Leaders)",
            transform=ax.transAxes, fontsize=8, color="0.3",
            ha="right", va="top", alpha=0.7, style="italic")
    ax.text(0.97, 0.005, "High Area,\nLow Pubs\n(Understudied)",
            transform=ax.transAxes, fontsize=8, color="0.3",
            ha="right", va="bottom", alpha=0.7, style="italic")
    ax.text(0.05, 0.005, "Small Area,\nLow Pubs",
            transform=ax.transAxes, fontsize=8, color="0.3",
            ha="left", va="bottom", alpha=0.7, style="italic")
    ax.text(0.05, 0.93, "Small Area,\nHigh Pubs",
            transform=ax.transAxes, fontsize=8, color="0.3",
            ha="left", va="top", alpha=0.7, style="italic")

    ax.set_title(
        "Mangrove Remote Sensing: Mangrove Area vs Publication Output\n"
        "Bubble size = publications; Color = dominant research topic",
        fontsize=11, fontweight="bold",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle=":", alpha=0.3)

    # =========================================================================
    # Polar mirror inset — top-left "Small Area, High Pubs" quadrant.
    # =========================================================================
    # ── Layout geometry (all values in figure-fraction [0,1]) ─────────────────
    # Figure = 16×12 in @ 300 dpi → 4800×3600 px.
    # Circle needs w/h ≈ fig_h/fig_w = 12/16 = 0.75 (so displayed circle ≈ square px).
    #
    #   ┌──────────────────────────────────────────────────────────┐ ← top=0.94
    #   │  ┌──────────────────────┐                                │
    #   │  │                      │     Bubble scatter             │
    #   │  │   Polar mirror       │     (main axes,               │
    #   │  │   (inset axes)       │      x=[0.08, 0.94])          │
    #   │  │                      │                                │
    #   │  └──────────────────────┘                                │
    #   │                                                          │
    #   │  ← legend at bottom (bbox_to_anchor y=-0.09) →          │
    #   └──────────────────────────────────────────────────────────┘ ← bottom=0.15
    #
    # Quick rebuild: pixi run python -m src.combined_viz --step bubble_polar
    # ──────────────────────────────────────────────────────────────────────────
    # Size — keep w/h ≈ 0.75 for a circle; increase both together to scale up.
    polar_w, polar_h = 0.33, 0.44   # ≈ 1584×1584 px
    # Position — (x, y) = bottom-left corner of the polar axes rect.
    #   x ↑ → move RIGHT  │  x ↓ → move LEFT
    #   y ↑ → move UP     │  y ↓ → move DOWN
    #   Must satisfy: x + w ≤ ~0.94  and  y + h ≤ 0.94  (fits within figure).
    polar_x = 0.15                   # left edge
    polar_y = 0.38                   # bottom edge (top edge = 0.48 + 0.44 = 0.92)
    # ═════════════════════════════════════════════════════════════════════════
    ax_polar = fig.add_axes([polar_x, polar_y, polar_w, polar_h], projection="polar")
    # Filter data for polar
    polar_data = merged[
        (merged["Publications"] >= polar_min_pubs) & (merged["Area"] >= 1.0)
    ].sort_values("Area", ascending=False).reset_index(drop=True)

    if len(polar_data) > polar_max_countries:
        polar_data = polar_data.head(polar_max_countries).reset_index(drop=True)

    n_polar = len(polar_data)
    logger.info("Polar inset: %d countries (min_pubs=%d)", n_polar, polar_min_pubs)

    angles = np.linspace(0, 2 * np.pi, n_polar, endpoint=False)
    area_vals = polar_data["Area"].values.astype(float)
    pubs_vals = polar_data["Publications"].values.astype(float)
    polar_countries = polar_data["Country"].values

    # ── Internal polar geometry (radial data units) ─────────────────────────
    # These control the visual "thickness" of bars inside the polar circle.
    #   base_r     ↑ → ring radius ↑, bars shift outward (more whitespace centre)
    #   max_pub_h  ↑ → publication bars extend further outward
    #   max_area_h ↑ → area bars extend further inward
    #   bar_width  multiplier ↑ → sectors wider (0.72 = 72% of available angle)
    # Scale these proportionally when changing polar_w/polar_h.
    base_r = 52.0
    max_pub_h = 36.0
    max_area_h = 30.0
    r_pubs = base_r + (pubs_vals / pubs_vals.max() * max_pub_h)
    r_area = base_r - (area_vals / area_vals.max() * max_area_h)
    bar_width = (2 * np.pi / n_polar) * 0.72

    ax_polar.bar(angles, r_pubs - base_r, width=bar_width, bottom=base_r,
                 color="#2B5C8F", alpha=0.88, edgecolor="none")
    ax_polar.bar(angles, base_r - r_area, width=bar_width, bottom=r_area,
                 color="#2E8B57", alpha=0.88, edgecolor="none")
    ax_polar.plot(np.linspace(0, 2 * np.pi, 200), [base_r] * 200,
                  color="white", linewidth=1.2, zorder=5)

    # Pre-compute top-10 by publication count (match standalone polar chart)
    top10_pub_indices = set(np.argsort(pubs_vals)[-10:])

    # Country labels — radial orientation, reading outward from centre.
    # Top-10 by pubs get bold blue + "(N)" count + area annotation inside.
    #   top10_pub_indices: change [-10:] to show more/fewer highlighted countries.
    #   fs: font size (↑ bigger, ↓ smaller). Scale with polar_w/polar_h.
    #   text_buffer: distance from bar tip to label (radial units).
    for i, (angle, country) in enumerate(zip(angles, polar_countries)):
        rot = np.rad2deg(angle)
        if angle < np.pi:
            ha, rotation = "left", -rot + 90
        else:
            ha, rotation = "right", -rot - 90

        pub_val = int(pubs_vals[i])
        area_val = area_vals[i]
        in_top10_pub = i in top10_pub_indices

        # Pub-side label: top-10 gets bold blue + "(N)" count
        if in_top10_pub:
            label = f"{country} ({pub_val})"
        else:
            label = country
        if len(label) > 26:
            label = label[:24] + ".."

        fs = 7.0 if in_top10_pub else 5.5
        fw = "bold" if in_top10_pub else "normal"
        fc = "#2B5C8F" if in_top10_pub else "#666666"
        text_buffer = 2.5 if in_top10_pub else 1.8
        label_r = max(r_pubs[i], base_r) + text_buffer

        ax_polar.text(
            angle, label_r, label,
            ha=ha, va="bottom", rotation=rotation, rotation_mode="anchor",
            fontsize=fs, color=fc, fontweight=fw,
        )

        # Top-10: annotate mangrove area at bar midpoint (avoids overlap with bar tip)
        if in_top10_pub:
            area_mid = r_area[i] - 15  # midpoint of the area bar
            ax_polar.text(
                angle, area_mid, f"{area_val:.2f}",
                ha=ha, va="center", rotation=rotation, rotation_mode="anchor",
                fontsize=6.0, color="#2E8B57", fontweight="bold",
            )

    ax_polar.set_theta_zero_location("N")   # largest country at 12 o'clock
    ax_polar.set_theta_direction(-1)        # clockwise
    ax_polar.set_rticks([])
    ax_polar.set_thetagrids([])
    ax_polar.spines["polar"].set_visible(False)

    # Legend at title position (replaces title) — outside top of polar axes.
    #   loc="lower center" + bbox_to_anchor=(0.5, 1.10) places it above the circle.
    #   y ↑ → legend moves UP (further from polar edge)
    #   y ↓ → legend moves DOWN (into the circle)
    polar_legend = [
        Patch(facecolor="#2B5C8F", alpha=0.85,
              label=f"Publications (≥{polar_min_pubs})"),
        Patch(facecolor="#2E8B57", alpha=0.85,
              label="Mangrove Area (≥1 km²)"),
    ]
    ax_polar.legend(
        handles=polar_legend,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=2, frameon=True, facecolor="white",
        fontsize=7, edgecolor="#dddddd",
    )
    # No title — legend serves as the label for the polar inset.

    # Dominant topic legend (bottom)
    legend_patches = []
    for tid in all_topics[:15]:
        label = topic_labels.get(int(tid), f"T{int(tid)}")
        legend_patches.append(
            Patch(facecolor=topic_color_map.get(int(tid), topic_color_map[-1]),
                  edgecolor="white", linewidth=0.5, label=label)
        )
    if len(all_topics) > 15:
        legend_patches.append(
            Patch(facecolor="none", edgecolor="none",
                  label=f"... +{len(all_topics) - 15} more"))

    ax.legend(
        handles=legend_patches,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.09),
        ncol=min(5, len(legend_patches)),
        fontsize=7, framealpha=0.9,
        title="Dominant Topic (highest SI)",
        title_fontsize=8,
    )

    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=vis.dpi, format=vis.figure_format)
    plt.close(fig)
    logger.info("P0-1c Bubble + polar mirror inset saved: %s", output_path)


# ══════════════════════════════════════════════════════════════════════════════
# P0-2: Topic × Year heatmap
# ══════════════════════════════════════════════════════════════════════════════


def _compute_yearly_topic_proportions(config: Config) -> Tuple[pd.DataFrame, pd.Series]:
    """Compute topic proportions per year from doc-topic assignments.

    Returns:
        ``(proportions, assigned_per_year)``. ``proportions`` has rows=topics,
        columns=years and values that sum to one within each year across the
        assigned topics; ``assigned_per_year`` is the number of assigned
        documents in each year, i.e. the denominator of those proportions
        (documents carrying topic label ``-1`` are outliers and excluded).
    """
    proc = config.paths.processed_dir

    # Load cleaned data for years
    cleaned_pq = proc / config.preprocessing.output_file.replace(".csv", ".parquet")
    if cleaned_pq.exists():
        df = pd.read_parquet(cleaned_pq)
    else:
        df = pd.read_csv(proc / config.preprocessing.output_file, encoding="utf-8-sig")

    # Load doc-topic assignments
    doc_topic = pd.read_csv(proc / config.topic_modeling.output.doc_topic_dist, encoding="utf-8-sig")

    # Merge
    merged = doc_topic[["doc_index", "Topic"]].copy()
    merged["PY"] = df["PY"].values[:len(merged)]
    merged = merged[merged["Topic"] != -1].copy()

    # Yearly proportions per topic
    yearly_counts = merged.groupby(["PY", "Topic"]).size().reset_index(name="Count")
    yearly_totals = merged.groupby("PY").size().to_dict()
    yearly_counts["Proportion"] = yearly_counts.apply(
        lambda r: r["Count"] / yearly_totals[r["PY"]], axis=1
    )

    pivot = yearly_counts.pivot_table(
        index="Topic", columns="PY", values="Proportion", fill_value=0.0
    )
    # Sort columns (years) and rows (topics)
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)
    pivot = pivot.reindex(sorted(pivot.index))

    return pivot, pd.Series(yearly_totals, dtype=float).sort_index()


# Final printed width of the figure. The CAS single-column class sets
# \columnwidth = 468.33 pt = 6.50 in, so the heatmap is designed at exactly that
# width and placed at scale ~1.0: its 6.5 pt labels print at ~6.5 pt. (The old
# 14-in-wide design was shrunk to 0.46x, which turned 6.5 pt tick labels into
# ~3 pt and truncated the longest row label at the canvas edge.)
_PRINT_WIDTH_IN = 6.50

# Latest complete publication year. The WoS retrieval (2026-07-15) leaves 2026 a
# seven-month partial year, which the manuscript excludes from primary analyses;
# the same convention is hard-coded as `fit_end_year = 2025` in
# src/bibliometric_analysis.py and src/logistic_growth_ci.py.
LAST_COMPLETE_YEAR = 2025


def _prepare_topic_year_matrix(
    config: Config,
    top_n_topics: int = 35,
    min_years_active: int = 3,
    last_complete_year: int = LAST_COMPLETE_YEAR,
    ensure_latest_top_n: int = 5,
    clip_percentile: float = 95.0,
) -> dict:
    """Row/column selection for the topic-year heatmap.

    Rows are the ``top_n_topics`` topics with the largest cumulative share over
    the complete years, plus any of the ``ensure_latest_top_n`` leaders of the
    latest complete year that the cumulative cut omitted (marked ``*``), in Ward
    order of their yearly-share profiles. Columns are the years that actually
    contain an assigned document -- no marker columns, so the axis looks like the
    pre-2026-09-24 figure; the years that are absent are listed in the caption,
    which is the only place the omission is recorded.

    Returns:
        Dict with ``frame`` (topics x data years), ``display`` (the same matrix),
        the year metadata, labels and the colour ceiling. Returns ``None`` when
        there is nothing to draw.
    """
    yearly_prop, assigned_counts = _compute_yearly_topic_proportions(config)
    if yearly_prop.empty:
        logger.warning("No yearly topic data — skipping heatmap")
        return None

    # Complete years only, keyed on the years that actually contain an assigned
    # document. The axis is categorical (one slot per data year) so cells stay
    # wide, but every run of missing calendar years gets its own grey slot that
    # names the span it replaces, so no omission is silent (the old figure
    # simply placed 1972 next to 1985).
    years_present = sorted(int(y) for y in yearly_prop.columns if int(y) <= last_complete_year)
    if not years_present:
        logger.warning("No years <= %d — skipping heatmap", last_complete_year)
        return None
    yearly_prop = yearly_prop.reindex(columns=years_present, fill_value=0.0)
    assigned_counts = assigned_counts.reindex(years_present, fill_value=0.0)

    # Filter topics: exclude topics that are non-zero in fewer than
    # min_years_active years, then take the top-N by cumulative share.
    active_years = (yearly_prop > 0).sum(axis=1)
    yearly_prop = yearly_prop[active_years >= min_years_active]
    if len(yearly_prop) < 2:
        logger.warning("Too few topics for heatmap — skipping")
        return None

    totals = yearly_prop.sum(axis=1).sort_values(ascending=False)
    selected = list(totals.index[:top_n_topics])
    latest_year = years_present[-1]
    latest_leaders = list(
        yearly_prop[latest_year].sort_values(ascending=False).head(ensure_latest_top_n).index
    )
    added = [t for t in latest_leaders if t not in selected]
    frame = yearly_prop.loc[selected + added]

    # Hierarchical clustering of rows (topics) based on temporal profile
    row_linkage = linkage(frame.values, method="ward", metric="euclidean")
    row_order = dendrogram(row_linkage, no_plot=True)["leaves"]
    frame = frame.iloc[row_order]

    # Column slots: the data years only, one per year, equally spaced -- the
    # pre-fix figure's axis. The runs of missing calendar years are NOT marked in
    # the figure (author decision, 2026-09-24: no grey break bands); they are
    # named in the caption instead, together with the fact that spacing between
    # adjacent columns is therefore not to scale.
    slots: list[tuple[str, object]] = [("year", year) for year in years_present]
    slot_labels = [str(value) for _, value in slots]
    # The year labels are drawn at 45 degrees (the pre-fix figure's angle), which
    # needs ~13 pt of pitch; a column is only ~5 pt wide, so every third slot
    # carries a label. The first and the LAST year are always labelled (author
    # decision, 2026-09-24): if the last year lands within two slots of the
    # previous label, that one is dropped instead of leaving the axis unresolved.
    labelled = set(range(0, len(slots), 3))
    last_slot = len(slots) - 1
    if last_slot not in labelled:
        previous = max(i for i in labelled if i < last_slot)
        if last_slot - previous < 3:
            labelled.discard(previous)
        labelled.add(last_slot)
    column_labels = [
        slot_labels[i] if i in labelled else "" for i in range(len(slots))
    ]
    missing_spans: list[str] = []
    for i, year in enumerate(years_present):
        if i and year - years_present[i - 1] > 1:
            missing = list(range(years_present[i - 1] + 1, year))
            missing_spans.append(
                f"{missing[0]}\u2013{missing[-1]}" if len(missing) > 1 else f"{missing[0]}"
            )

    # Display matrix; every column carries data now, so no mask is needed.
    display = frame.values.astype(float).copy()
    year_position: dict[int, int] = {int(year): i for i, year in enumerate(years_present)}

    # Topic labels: topic id + short name, so rows stay distinguishable (T10 and
    # T69 share the short name "Remote sensing of mangrove ecosystems") and
    # traceable to Table 1 / Supplementary Table S1. One line each, as in the
    # pre-fix figure (author decision, 2026-09-24: no wrapping).
    topic_labels = _load_topic_labels(config)
    row_labels = [
        ("*" if int(tid) in added else "")
        + f"T{int(tid)} {topic_labels.get(int(tid), '')}".strip()
        for tid in frame.index
    ]

    positive = display[display > 0]
    data_max = float(display.max())
    vmax = float(min(data_max, np.percentile(positive, clip_percentile))) if positive.size else 0.1

    return {
        "frame": frame,
        "display": display,
        "slots": slots,
        "slot_labels": slot_labels,
        "column_labels": column_labels,
        "missing_spans": missing_spans,
        "year_position": year_position,
        "years": [int(value) for kind, value in slots if kind == "year"],
        "assigned_counts": assigned_counts,
        "selected": selected,
        "added": added,
        "latest_year": latest_year,
        "n_topics": len(frame),
        "row_labels": row_labels,
        "data_max": data_max,
        "vmax": max(vmax, 1e-6),
    }


def plot_topic_year_heatmap(
    config: Config,
    output_path: Path,
    top_n_topics: int = 35,
    min_years_active: int = 3,
    last_complete_year: int = LAST_COMPLETE_YEAR,
    ensure_latest_top_n: int = 5,
    clip_percentile: float = 95.0,
) -> None:
    """Topic x year heatmap of the leading topics, designed at printed size.

    Rows = topics, one line per label; columns = the years that contain an
    assigned document, in the pre-2026-09-24 axis form; cell colour = that topic's
    share of the year's assigned documents.

    Rows are the ``top_n_topics`` topics with the largest cumulative share,
    plus any of the ``ensure_latest_top_n`` leaders of the latest complete year
    that the cumulative cut left out; those extra rows are marked ``*`` so the
    figure always shows the topics the surrounding text names. Row order is a
    Ward clustering of the yearly-share profiles.

    Args:
        config: Validated configuration.
        output_path: Where to save the PNG.
        top_n_topics: Rows kept by cumulative share over the complete years.
        min_years_active: Minimum number of years with a non-zero share.
        last_complete_year: Last year drawn (2026 is a partial year).
        ensure_latest_top_n: Leaders of the latest complete year forced into
            the row set when the cumulative ranking omits them.
        clip_percentile: Percentile of the non-zero cells used as the colour
            ceiling; the top colourbar tick is labelled ``>=`` that value.
    """
    vis = config.visualization
    plt.rcParams["font.family"] = vis.font_family

    prep = _prepare_topic_year_matrix(
        config,
        top_n_topics=top_n_topics,
        min_years_active=min_years_active,
        last_complete_year=last_complete_year,
        ensure_latest_top_n=ensure_latest_top_n,
        clip_percentile=clip_percentile,
    )
    if prep is None:
        return

    frame = prep["frame"]
    display = prep["display"]
    slots = prep["slots"]
    column_labels = prep["column_labels"]
    year_position = prep["year_position"]
    years = prep["years"]
    assigned_counts = prep["assigned_counts"]
    selected = prep["selected"]
    added = prep["added"]
    latest_year = prep["latest_year"]
    n_topics = prep["n_topics"]
    n_years = len(slots)
    row_labels = prep["row_labels"]
    vmax = prep["vmax"]

    # Figure layout with marginal bars, designed at final printed size (see
    # _PRINT_WIDTH_IN). Single-line labels need only ~9 pt of row pitch and there
    # is no axis title, so the figure is compact and the white space below the
    # rotated year labels stays small.
    fig_width = _PRINT_WIDTH_IN
    fig_height = 0.132 * n_topics + 0.80
    fig = plt.figure(figsize=(fig_width, fig_height))

    from matplotlib.gridspec import GridSpec

    # 3-column layout: heatmap area | right marginal | colorbar strip
    # Colorbar in its own column won't shrink ax_main — ax_top & ax_main stay
    # equal-width. The two right columns are kept narrow (1.7 / 0.7) so the
    # matrix, which is the informative part, takes as much of the page as the
    # 6.5-in column allows.
    gs = GridSpec(
        2, 3,
        width_ratios=[n_years, 1.7, 0.7],
        height_ratios=[3.0, n_topics],
        wspace=0.02, hspace=0.02,
    )

    ax_main = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)
    ax_cbar = fig.add_subplot(gs[1, 2])

    # Hide top-right corner (covers cols 1-2 in row 0)
    ax_corner = fig.add_subplot(gs[0, 1:])
    ax_corner.set_visible(False)

    # Main heatmap — colorbar in dedicated far-right axis. The colour ceiling is
    # the 95th percentile of the non-zero cells (single-document years reach a
    # share of 1.0 and would otherwise flatten the whole map), so the top
    # colourbar tick is labelled ">= ceiling" instead of the raw maximum.
    data = frame.values
    sns.heatmap(
        display,
        ax=ax_main,
        cmap="YlOrRd",
        xticklabels=column_labels,
        yticklabels=row_labels,
        cbar_ax=ax_cbar,
        linewidths=0.2,
        linecolor="#f0f0f0",
        vmin=0,
        vmax=vmax,
    )
    cbar = ax_main.collections[0].colorbar
    colorbar_ticks = list(np.linspace(0.0, vmax, 6))
    cbar.set_ticks(colorbar_ticks)
    cbar.set_ticklabels([f"{t:.2f}" for t in colorbar_ticks[:-1]] + [f"\u2265{vmax:.2f}"])
    cbar.set_label("Topic share", fontsize=7.5)
    cbar.ax.tick_params(labelsize=6)
    # No axis title: the 45-degree year labels (first and last always shown) and
    # the caption carry the span and the omissions, and dropping the title keeps
    # the white space below the axis small (author decision, 2026-09-24).
    ax_main.set_xlabel("")
    ax_main.set_ylabel("")
    # Years at 45 degrees, as in the pre-fix figure; every third column carries a
    # label (see _prepare_topic_year_matrix) and the axis title gives the span.
    ax_main.tick_params(axis="x", labelsize=6.0, rotation=45)
    ax_main.tick_params(axis="y", labelsize=5.5)

    # Top marginal: yearly publication count, with the assigned-document count
    # (the denominator of the cell shares) overlaid so the two are not confused.
    cleaned_pq = config.paths.processed_dir / config.preprocessing.output_file.replace(".csv", ".parquet")
    if cleaned_pq.exists():
        cleaned_df = pd.read_parquet(cleaned_pq)
    else:
        cleaned_df = pd.read_csv(config.paths.processed_dir / config.preprocessing.output_file, encoding="utf-8-sig")

    yearly_pubs = cleaned_df.groupby("PY").size().reindex(years, fill_value=0)
    assigned_series = assigned_counts.reindex(years, fill_value=0.0)
    bar_x = [year_position[y] + 0.5 for y in years]
    ax_top.bar(bar_x, yearly_pubs.values, color="#4472C4", edgecolor="white",
               width=0.7, alpha=0.8, label="Publications")
    ax_top.step(bar_x, assigned_series.values, where="mid", color="#C00000",
                linewidth=1.1, label="Topic-assigned")
    ax_top.set_ylabel("Per year", fontsize=6.5)
    # Headroom so the legend cannot sit on top of the tallest (2025) bar.
    ax_top.set_ylim(0, float(max(yearly_pubs.max(), assigned_series.max())) * 1.35)
    ax_top.tick_params(axis="x", labelbottom=False)
    ax_top.tick_params(axis="y", labelsize=5.5)
    ax_top.set_yticks([0, float(yearly_pubs.max())])
    ax_top.legend(fontsize=5.0, frameon=False, loc="upper left", ncol=2,
                  handlelength=1.2, columnspacing=0.8, borderaxespad=0.15)
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)

    # Right marginal: total topic share. Its ticks and label go on top so they
    # cannot collide with the rotated year labels of the main axis.
    topic_totals = frame.sum(axis=1).values
    ax_right.barh(range(n_topics), topic_totals, color="#70AD47", edgecolor="white",
                  height=0.8, alpha=0.8)
    ax_right.xaxis.set_label_position("top")
    ax_right.xaxis.tick_top()
    ax_right.set_xlabel("Σ share", fontsize=6.5, labelpad=1.5)
    ax_right.tick_params(axis="y", left=False, labelleft=False)
    ax_right.tick_params(axis="x", labelsize=5.5)
    ax_right.spines["top"].set_visible(False)
    ax_right.spines["right"].set_visible(False)
    ax_right.invert_yaxis()

    # Reserve exactly enough room for the row labels, measured from the rendered
    # text, so no label is cut off at the canvas edge.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    label_w_px = max(
        (t.get_window_extent(renderer=renderer).width for t in ax_main.get_yticklabels()),
        default=0.0,
    )
    # 0.20 in of slack: the measured label width does not include the tick mark
    # and its pad (~7 pt), which sit between the label and the axes edge.
    left = min(0.52, (label_w_px + 0.20 * fig.dpi) / (fig_width * fig.dpi))
    # bottom leaves just room for the rotated year labels (no axis title below
    # them any more); right is pushed out to 0.90 so the matrix gets the width,
    # leaving just enough for the colourbar tick labels.
    fig.subplots_adjust(left=left, right=0.90, top=0.985, bottom=0.075)
    fig.canvas.draw()

    # Shorten colorbar to ~60% of heatmap height, centered vertically
    cbar_pos = ax_cbar.get_position()
    cbar_h = cbar_pos.height * 0.6
    cbar_y0 = cbar_pos.y0 + (cbar_pos.height - cbar_h) / 2
    ax_cbar.set_position([cbar_pos.x0, cbar_y0, cbar_pos.width, cbar_h])

    heatmap_box = ax_main.get_position()
    logger.info(
        "P0-2 layout: matrix %.0f%% of figure width, row-label strip %.0f%%, "
        "right marginals+colorbar %.0f%%",
        100 * heatmap_box.width,
        100 * heatmap_box.x0,
        100 * (1 - heatmap_box.x1),
    )

    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=vis.dpi, format=vis.figure_format)
    plt.close(fig)
    logger.info(
        "P0-2 Topic x year heatmap saved: %s | %d rows (%d by cumulative share + %d "
        "%d-year leaders %s), %d data years %d-%d (omitted spans, caption only: %s), "
        "colour ceiling %.3f (p%.0f of non-zero cells), %d cells above it",
        output_path,
        n_topics,
        len(selected),
        len(added),
        latest_year,
        [int(t) for t in added],
        len(years),
        years[0],
        years[-1],
        prep["missing_spans"],
        vmax,
        clip_percentile,
        int((data > vmax).sum()),
    )


# ══════════════════════════════════════════════════════════════════════════════
# P1-1: Continuous yearly streamgraph (Plotly)
# ══════════════════════════════════════════════════════════════════════════════


def plot_yearly_streamgraph(
    config: Config,
    output_path: Path,
    top_n_topics: int = 20,
    min_yearly_share: float = 0.002,
) -> None:
    """Interactive yearly streamgraph showing smooth topic proportion evolution.

    Uses continuous yearly data (not discrete periods), with rolling-average
    smoothing.  Topics are stacked as a fraction-normalized streamgraph.

    Args:
        config: Validated configuration.
        output_path: Where to save the HTML file.
        top_n_topics: Maximum number of topics to display.
        min_yearly_share: Filter topics that never exceed this share.
    """
    import plotly.graph_objects as go
    import plotly.express as px

    # _compute_yearly_topic_proportions returns (shares, assigned_counts) since it was
    # extended for the Fig. 6 marginals; these two call sites were never updated and
    # raised AttributeError, so the streamgraph and lifecycle figures were stale.
    yearly_prop, _assigned_counts = _compute_yearly_topic_proportions(config)

    if yearly_prop.empty:
        logger.warning("No yearly topic data — skipping streamgraph")
        return

    # Filter to top topics by total proportion
    yearly_prop["_total"] = yearly_prop.sum(axis=1)
    yearly_prop = yearly_prop.sort_values("_total", ascending=False)
    # Also filter by min yearly share
    yearly_prop = yearly_prop[(yearly_prop.drop(columns=["_total"]) > min_yearly_share).any(axis=1)]
    yearly_prop = yearly_prop.head(top_n_topics)
    yearly_prop = yearly_prop.drop(columns=["_total"])

    # Apply 3-year rolling average for smoothing
    years = yearly_prop.columns.tolist()
    smoothed = yearly_prop.T.rolling(window=3, center=True, min_periods=1).mean().T
    smoothed = smoothed.clip(lower=0)

    # Re-normalize to fractions after smoothing
    col_sums = smoothed.sum(axis=0)
    smoothed = smoothed.div(col_sums, axis=1).fillna(0)

    # Build labels
    topic_labels = _load_topic_labels(config)
    n_topics = len(smoothed)
    colors = px.colors.qualitative.Bold[:n_topics] if n_topics <= 10 else px.colors.sample_colorscale(
        "viridis", n_topics
    )

    fig = go.Figure()

    for idx, (topic_id, row) in enumerate(smoothed.iterrows()):
        tid = int(topic_id)
        label = topic_labels.get(tid, f"T{tid}")
        color = colors[idx % len(colors)]

        fig.add_trace(go.Scatter(
            x=years,
            y=row.values,
            mode="lines",
            stackgroup="one",
            groupnorm="fraction",
            name=label,
            line=dict(width=0.3, color=color),
            hovertemplate=(
                f"<b>{label}</b><br>"
                f"Year: %{{x}}<br>"
                f"Share: %{{y:.2%}}<br>"
                "<extra></extra>"
            ),
        ))

    fig.update_layout(
        title={
            "text": "Continuous Yearly Topic Evolution in Mangrove Remote Sensing",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 16},
        },
        xaxis={
            "title": "Year",
            "tickangle": -30,
            "tickmode": "linear",
            "dtick": 5,
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
        height=650,
        width=1100,
    )

    ensure_dir(output_path.parent)
    fig.write_html(str(output_path))

    # Also save a static PNG for publication
    png_path = output_path.with_suffix(".png")
    try:
        fig.write_image(str(png_path), width=1100, height=650, scale=2)
        logger.info("P1-1 Yearly streamgraph PNG saved: %s", png_path)
    except Exception as exc:
        logger.warning("Could not save streamgraph PNG (kaleido may be missing): %s", exc)

    logger.info("P1-1 Yearly streamgraph HTML saved: %s", output_path)


# ══════════════════════════════════════════════════════════════════════════════
# P2-1: Topic lifecycle classification
# ══════════════════════════════════════════════════════════════════════════════


def _theil_sen_slope(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """Theil-Sen slope, intercept, and p-value approximation.

    Returns:
        (slope, intercept, p_value_approx).
        p_value is approximated via Kendall's tau test.
    """
    # scipy.stats is imported inside the function: its extension DLLs do not load in
    # this environment, so a module-level import would break every caller.
    from scipy.stats import kendalltau, theilslopes

    try:
        res = theilslopes(y, x, 0.95)
        slope = float(res.slope)
        intercept = float(res.intercept)
    except Exception:
        slope, intercept = np.polyfit(x, y, 1)
        slope, intercept = float(slope), float(intercept)

    # Kendall's tau for significance
    try:
        tau, p_value = kendalltau(x, y)
    except Exception:
        p_value = 0.5

    return slope, intercept, float(p_value)


def plot_topic_lifecycle(
    config: Config,
    output_path: Path,
    p_threshold: float = 0.10,
    max_show: int = 35,
) -> None:
    """A+B combined lifecycle visualization.

    Panel A — Bubble scatter of Kendall's $\tau$ against the mean yearly share: topics ranked by Kendall's τ.
             τ > 0 → Hot (rising), τ < 0 → Cold (declining).
    Panel B — Faceted line charts: temporal trajectories grouped by class,
             with thin individual lines + bold class average.

    Args:
        config: Validated configuration.
        output_path: Where to save the PNG.
        p_threshold: p-value threshold for significance.
        max_show: Max Hot+Cold topics to display in panel A.
    """
    vis = config.visualization
    plt.rcParams["font.family"] = vis.font_family

    yearly_prop, _assigned_counts = _compute_yearly_topic_proportions(config)
    # The partially indexed 2026 (records to 15 July 2026) is excluded from every primary
    # analysis, and a rank correlation is exactly the kind of statistic a partial year
    # distorts, so drop that column before computing tau (it changes two topics' class).
    yearly_prop = yearly_prop[[c for c in yearly_prop.columns if int(c) < 2026]]
    if yearly_prop.empty:
        logger.warning("No yearly topic data — skipping lifecycle plot")
        return

    topic_labels = _load_topic_labels(config)
    years_arr = np.array([int(y) for y in yearly_prop.columns])

    # Compute Kendall's τ for each topic (rank correlation, robust to zeros)
    from scipy.stats import kendalltau as _kt
    records = []
    for topic_id, row in yearly_prop.iterrows():
        tid = int(topic_id)
        y_vals = row.values.astype(float)
        mean_prop = float(np.mean(y_vals))
        tau, p_value = _kt(years_arr, y_vals)
        # Also compute Theil-Sen for reference
        slope, _, _ = _theil_sen_slope(years_arr, y_vals)
        records.append({
            "Topic": tid,
            "Mean_Proportion": mean_prop,
            "Tau": tau,
            "P_Value": float(p_value),
            "Slope": slope,
            "Label": topic_labels.get(tid, f"T{tid}"),
        })

    # Compute active years for each topic
    for rec in records:
        tid = rec["Topic"]
        if tid in yearly_prop.index:
            rec["Active_Years"] = int((yearly_prop.loc[tid].values > 0).sum())
        else:
            rec["Active_Years"] = 0

    df = pd.DataFrame(records)

    def classify(row):
        if row["P_Value"] >= p_threshold:
            return "Stable"
        if row["Tau"] > 0:
            return "Hot"
        return "Cold"

    df["Class"] = df.apply(classify, axis=1)
    counts = df["Class"].value_counts()
    hot_n = counts.get("Hot", 0)
    cold_n = counts.get("Cold", 0)
    stable_n = counts.get("Stable", 0)
    logger.info("Lifecycle classification: Hot=%d, Cold=%d, Stable=%d", hot_n, cold_n, stable_n)

    color_map = {"Hot": "#d73027", "Cold": "#4575b4", "Stable": "#888888"}

    # ── Figure layout ──────────────────────────────────────────────────────
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(18, 13))
    gs = GridSpec(2, 1, height_ratios=[1.0, 1.0], hspace=0.22,
                  top=0.94, bottom=0.06, left=0.10, right=0.94)

    # ═══════════════════════════════════════════════════════════════════════
    # Panel A — Bubble scatter: τ vs Mean Proportion
    # ═══════════════════════════════════════════════════════════════════════
    ax_a = fig.add_subplot(gs[0])

    # Size = active years, alpha = significance
    min_s, max_s = 25, 300
    active = df["Active_Years"].values.astype(float)
    arange = active.max() - active.min()
    sizes = min_s + (max_s - min_s) * (active - active.min()) / (arange if arange > 0 else 1.0)

    # Alpha: significant topics more opaque
    sig_mask = df["P_Value"] < p_threshold
    alphas = np.where(sig_mask, 0.85, 0.35)

    # Plot by class for correct z-order
    for cls_name in ["Stable", "Cold", "Hot"]:
        sub_mask = df["Class"] == cls_name
        if not sub_mask.any():
            continue
        sub = df[sub_mask]
        ax_a.scatter(
            sub["Tau"], sub["Mean_Proportion"],
            s=sizes[sub_mask], c=color_map[cls_name],
            alpha=alphas[sub_mask],
            edgecolors="white", linewidth=0.4,
            zorder=3 if cls_name != "Stable" else 1,
            label=f"{cls_name} ({len(sub)})",
        )

    # Reference lines
    ax_a.axvline(x=0, color="#333333", linewidth=1.0, linestyle="-", alpha=0.5)
    median_prop = df["Mean_Proportion"].median()
    ax_a.axhline(y=median_prop, color="#999999", linewidth=0.7, linestyle=":", alpha=0.4)

    # Label extremes: top 6 by |τ| + top 3 by mean proportion
    df["_abs_tau"] = df["Tau"].abs()
    to_label_tau = df.nlargest(6, "_abs_tau")
    to_label_prop = df.nlargest(3, "Mean_Proportion")
    to_label = pd.concat([to_label_tau, to_label_prop]).drop_duplicates("Topic")
    df.drop(columns=["_abs_tau"], inplace=True)

    for _, row in to_label.iterrows():
        offset_y = 6 if row["Tau"] > 0 else -10
        ax_a.annotate(
            row["Label"],
            (row["Tau"], row["Mean_Proportion"]),
            textcoords="offset points",
            xytext=(8, offset_y),
            fontsize=7, fontweight="bold",
            color=color_map[row["Class"]],
            ha="left",
            arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.4),
        )

    ax_a.set_yscale("log")
    from matplotlib.ticker import LogLocator, NullFormatter
    ax_a.yaxis.set_major_locator(LogLocator(base=10.0, numticks=6))
    ax_a.yaxis.set_minor_formatter(NullFormatter())

    ax_a.set_xlabel("Kendall's τ (trend direction & strength)", fontsize=10)
    ax_a.set_ylabel("Mean Topic Proportion (log scale)", fontsize=10)
    ax_a.set_xlim(-1.05, 1.05)
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)
    ax_a.grid(True, linestyle=":", alpha=0.2)
    ax_a.legend(fontsize=8, loc="upper left", framealpha=0.85,
                scatterpoints=1, handletextpad=0.5)

    # Size legend
    size_legend_years = [3, 10, 25, 40]
    for n_y in size_legend_years:
        s = min_s + (max_s - min_s) * (n_y - active.min()) / (arange if arange > 0 else 1.0)
        ax_a.scatter([], [], s=s, c="#555555", alpha=0.5, edgecolors="white", linewidth=0.3,
                     label=f"{n_y} yr")
    leg2 = ax_a.legend(fontsize=7, loc="lower right", framealpha=0.7, title="Active years",
                       title_fontsize=7, ncol=2, handletextpad=0.3)
    ax_a.add_artist(leg2)

    ax_a.set_title(
        f"Panel A — Topic τ vs Prevalence (p < {p_threshold} = significant)",
        fontsize=12, fontweight="bold", loc="left",
    )

    # ═══════════════════════════════════════════════════════════════════════
    # Panel B — Faceted temporal trajectories
    # ═══════════════════════════════════════════════════════════════════════
    gs_b = gs[1].subgridspec(1, 3, wspace=0.06)

    for i, (cls_name, cls_label) in enumerate([
        ("Hot", "Hot: rising"), ("Cold", "Cold: declining"), ("Stable", "Stable: no trend"),
    ]):
        ax_b = fig.add_subplot(gs_b[i])
        sub = df[df["Class"] == cls_name]
        cls_topics = sub["Topic"].tolist()
        cls_color = color_map[cls_name]

        if cls_topics:
            all_vals = []
            for tid in cls_topics:
                if tid in yearly_prop.index:
                    y_vals = yearly_prop.loc[tid].values.astype(float)
                    # Normalize to [0, 1] for comparable trajectories
                    y_max = y_vals.max()
                    y_norm = y_vals / y_max if y_max > 0 else y_vals
                    ax_b.plot(years_arr, y_norm, color=cls_color, alpha=0.18,
                              linewidth=0.5)
                    all_vals.append(y_norm)

            # Bold class average
            if all_vals:
                avg = np.mean(all_vals, axis=0)
                ax_b.plot(years_arr, avg, color=cls_color, linewidth=2.2,
                          alpha=0.95, label="Class average")

        # Counts are singular-aware; the panel-level label is added once for the whole row
        # after the loop, in panel A's style.
        n_cls = len(cls_topics)
        plural = "topic" if n_cls == 1 else "topics"
        ax_b.set_title(f"{cls_label}\n({n_cls} {plural})", fontsize=11,
                       color=cls_color, fontweight="bold")
        ax_b.set_xlabel("Year", fontsize=8)
        if i == 0:
            ax_b.set_ylabel("Normalized Proportion", fontsize=9)
        ax_b.spines["top"].set_visible(False)
        ax_b.spines["right"].set_visible(False)
        ax_b.tick_params(labelsize=7)
        ax_b.grid(True, linestyle=":", alpha=0.25)
        ax_b.set_xlim(years_arr.min(), years_arr.max())

    # Panel B row label: same style and left alignment as panel A's title, sitting above
    # the facet titles (panel A's x-label is centred, so the left edge is clear).
    pos_b = gs[1].get_position(fig)
    fig.text(pos_b.x0, pos_b.y1 + 0.028, "Panel B — Yearly share trajectories by class",
             fontsize=12, fontweight="bold", ha="left", va="bottom", color="black")

    # No figure-level title: the caption names the figure, states the p < 0.10 threshold and
    # defines the three classes, so a suptitle would only duplicate it and cost height.

    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=vis.dpi, format=vis.figure_format, bbox_inches="tight")
    plt.close(fig)
    logger.info("P2-1 Topic lifecycle plot saved: %s", output_path)


# ══════════════════════════════════════════════════════════════════════════════
# Main entry point
# ══════════════════════════════════════════════════════════════════════════════


def run_combined_viz(config: Config) -> None:
    """Run all six combined visualizations.

    Args:
        config: Validated configuration.
    """
    paths = config.paths

    # P0-1: Bubble scatter (bidirectional inset — default)
    with Timer("P0-1: Area vs Pubs bubble scatter"):
        plot_area_pubs_bubble(config, paths.figures_dir / "country_area_pubs_bubble.png",
                              inset_style="bidirectional")
    # P0-1 backup: Bubble scatter with original 3-axis inset
    with Timer("P0-1-orig: Area vs Pubs bubble scatter (original inset backup)"):
        plot_area_pubs_bubble(config, paths.figures_dir / "country_area_pubs_bubble_original.png",
                              inset_style="original")

    # P0-1c: Bubble scatter + polar mirror (combined, replaces bar chart inset)
    with Timer("P0-1c: Bubble + polar mirror combined"):
        plot_area_pubs_bubble_polar(config, paths.figures_dir / "country_area_pubs_bubble_polar.png")

    # P0-2: Topic × Year heatmap
    with Timer("P0-2: Topic × Year heatmap"):
        plot_topic_year_heatmap(config, paths.figures_dir / "topic_year_heatmap.png")

    # P1-1: Yearly streamgraph
    with Timer("P1-1: Continuous yearly streamgraph"):
        plot_yearly_streamgraph(config, paths.figures_dir / "topic_streamgraph_yearly.html")

    # P2-1: Topic lifecycle
    with Timer("P2-1: Topic lifecycle classification"):
        plot_topic_lifecycle(config, paths.figures_dir / "topic_lifecycle.png")

    logger.info("All combined visualizations complete.")


def main() -> None:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Combined bibliometric × topic visualizations (P0-P2)"
    )
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument(
        "--step", type=str, default=None,
        choices=["bubble", "bubble_polar", "heatmap", "streamgraph", "lifecycle"],
        help="Run a single visualization step.",
    )
    args = parser.parse_args()

    config = get_config(args.config)
    from src.utils import setup_logging
    setup_logging(config.system.log_level, config.system.log_file)

    paths = config.paths

    step_map = {
        "bubble": lambda: plot_area_pubs_bubble(config, paths.figures_dir / "country_area_pubs_bubble.png"),
        "bubble_polar": lambda: plot_area_pubs_bubble_polar(config, paths.figures_dir / "country_area_pubs_bubble_polar.png"),
        "heatmap": lambda: plot_topic_year_heatmap(config, paths.figures_dir / "topic_year_heatmap.png"),
        "streamgraph": lambda: plot_yearly_streamgraph(config, paths.figures_dir / "topic_streamgraph_yearly.html"),
        "lifecycle": lambda: plot_topic_lifecycle(config, paths.figures_dir / "topic_lifecycle.png"),
    }

    try:
        if args.step:
            logger.info("Running single step: %s", args.step)
            step_map[args.step]()
        else:
            run_combined_viz(config)
    except Exception:
        logger.exception("Combined visualization step '%s' failed", args.step or "all")
        raise


if __name__ == "__main__":
    main()
