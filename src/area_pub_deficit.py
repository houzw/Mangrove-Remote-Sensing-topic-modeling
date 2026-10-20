"""Area-weighted publication deficit — country-level under/over-research score.

Replaces the median four-quadrant "Understudied" classification with a
continuous, threshold-free ranking that compares a country's share of global
mangrove area against its share of global mangrove-RS research:

    deficit_c = area_share_c - pub_share_c
              = (area_c / Σ area_global) - (pubs_c / Σ pubs_global)

* ``area_c`` — HGMF 2020 mangrove area for country *c* (km²).
* ``pubs_c`` — fractional-counted publications for *c* (weight ``w_{d,c} = 1/m_d``,
  Perianes-Rodríguez, Waltman & Van Eck 2016).

``deficit_c > 0`` → resource share exceeds research share → *under-researched*.
``deficit_c < 0`` → research share exceeds resource share → *over-researched*.

Outputs (standalone — does not overwrite existing figures/CSVs):
    ``data/processed/area_pub_deficit.csv``
    ``outputs/figures/area_pub_deficit_scatter.png``
    ``outputs/figures/area_pub_deficit_quadrant.png``

Usage:
    python -m src.area_pub_deficit              # uses default config.yaml
    python -m src.area_pub_deficit --top-n 20   # label N countries in scatter
"""

from __future__ import annotations

import argparse
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch

from src.config import Config, get_config
from src.utils import Timer, canonical_countries, ensure_dir, setup_logging

logger = logging.getLogger(__name__)


# =============================================================================
# Country-name tables
# =============================================================================

# HGMF name typos/abbreviations → canonical name as produced by preprocessing's
# ``CU`` field (so area and publications join on the same spelling).
_HGMF_NAME_FIX: Dict[str, str] = {
    "Camerocn": "Cameroon",
    "Venzuela": "Venezuela",
    "Solomon Is.": "Solomon Islands",
    "Congo.DRC": "Congo",                # DR Congo's pubs are named 'Congo' in CU
    "Cote d'Ivory": "Cote d'Ivoire",
    "The Bahamas": "Bahamas",
    "The Gambia": "Gambia",
}

# Overseas territories → administering metropole. Applied to BOTH area and
# publication sides so a country's mangroves and research are on equal footing
# (e.g. French Guiana's area and its papers both count toward France).
_TERRITORY_TO_METROPOLE: Dict[str, str] = {
    "French Guiana": "France",
    "Guadeloupe": "France",
    "Martinique": "France",
    "Mayotte": "France",
    "New Caledonia": "France",
    "Wallis & Futuna": "France",
    "Puerto Rico": "United States",
    "American Samoa": "United States",
    "Virgin Is.": "United States",
    "Netherlands Antilles": "Netherlands",
    "Aruba": "Netherlands",
    "British Virgin Is.": "United Kingdom",
    "Cayman Is.": "United Kingdom",
    "Turks & Caicos Is.": "United Kingdom",
}

# Residual false positives in the ``CU`` field.  Person-name leaks ("Jamali"→
# Mali, "Jordan M."→Jordan, "Montenegro J."→Montenegro, "Moldovanu"→Moldova,
# "Lomonaco"→Monaco, "Israel R."→Israel) were eliminated at the source:
# ``_extract_country`` now matches only affiliation text (author-name brackets
# are stripped), so those countries are no longer excluded and genuine
# affiliations (e.g. an "Amman, Jordan" address) count again.  What remains are
# place/state-name collisions *inside* real affiliations that a body-wide
# substring match cannot tell apart from the sovereign country: Niger Delta/
# State (Nigeria), Georgia (US state) institutions, Inner Mongolia (China).
# Dropped so they do not absorb fractional weight from the real countries.
_CU_FALSE_POSITIVES: set[str] = {"Niger", "Georgia", "Mongolia"}

# Taiwan hosts native mangroves but is absent from the HGMF table (folded into
# China); attribute its publications to China to avoid silently dropping them.
_PUB_ONLY_FOLD: Dict[str, str] = {"Taiwan": "China"}


# =============================================================================
# Data loading
# =============================================================================


def _load_cleaned(config: Config) -> pd.DataFrame:
    """Load cleaned data (prefer fast parquet, fall back to CSV)."""
    parquet = config.paths.processed_dir / config.preprocessing.output_file.replace(".csv", ".parquet")
    if parquet.exists():
        return pd.read_parquet(parquet, columns=["CU"])
    csv = config.paths.processed_dir / config.preprocessing.output_file
    return pd.read_csv(csv, encoding="utf-8-sig", usecols=["CU"])


def _load_area(config: Config) -> pd.DataFrame:
    """Load HGMF mangrove area, normalise names, fold territories, aggregate.

    Returns a DataFrame with columns ``Country, Area`` (km²).
    """
    area_path = config.paths.external_dir / config.country_analysis.mangrove_area_file
    if not area_path.exists():
        raise FileNotFoundError(f"Mangrove area file not found: {area_path}")

    area_col = config.country_analysis.area_column
    df = pd.read_csv(area_path)[["Country", area_col]].rename(columns={area_col: "Area"})

    n_raw = len(df)
    # Normalise HGMF names → canonical CU spellings, then fold territories.
    df["Country"] = df["Country"].map(_HGMF_NAME_FIX).fillna(df["Country"])
    df["Country"] = df["Country"].map(_TERRITORY_TO_METROPOLE).fillna(df["Country"])

    df = df.groupby("Country", as_index=False)["Area"].sum()
    df["Area"] = pd.to_numeric(df["Area"], errors="coerce")
    df = df[df["Area"] > 0].reset_index(drop=True)
    logger.info("Loaded mangrove area: %d territories → %d after name fix + folding",
                n_raw, len(df))
    return df


# =============================================================================
# Fractional publication counting
# =============================================================================


def _fractional_country_pubs(df: pd.DataFrame) -> pd.DataFrame:
    """Fractional-count publications per country (w_{d,c} = 1/m_d).

    Also returns the integer (full-count) total per country, used only for the
    overlap report against the legacy median-quadrant "Understudied" list.

    Returns a DataFrame with columns ``Country, Pubs, Pubs_int``.
    """
    frac: Counter = Counter()
    integer: Counter = Counter()
    n_docs = 0

    for cu in df["CU"].dropna():
        # Fold territories into their metropoles AND de-duplicate, so a
        # "China;Taiwan" record counts once for China (m_d = 1, weight 1.0)
        # instead of twice at weight 0.5 (see src.utils.canonical_countries).
        countries = canonical_countries(
            cu,
            fold=(_TERRITORY_TO_METROPOLE, _PUB_ONLY_FOLD),
            drop=_CU_FALSE_POSITIVES,
        )

        if not countries:
            continue
        n_docs += 1
        w = 1.0 / len(countries)
        for c in countries:
            frac[c] += w
            integer[c] += 1

    out = pd.DataFrame({
        "Country": list(frac.keys()),
        "Pubs": [frac[k] for k in frac.keys()],
        "Pubs_int": [integer[k] for k in frac.keys()],
    })
    logger.info("Fractional pubs: %d docs → %d countries (Σ pubs = %.1f)",
                n_docs, len(out), out["Pubs"].sum())
    return out


# =============================================================================
# Deficit computation
# =============================================================================


def compute_deficit(config: Config) -> pd.DataFrame:
    """Compute the area-weighted publication deficit for all merged countries.

    Returns a DataFrame sorted by deficit descending with columns ``Country,
    Area, Pubs, Pubs_int, area_share, pub_share, deficit, rank``.
    """
    area_df = _load_area(config)
    pubs_df = _fractional_country_pubs(_load_cleaned(config))

    merged = pubs_df.merge(area_df, on="Country", how="inner")
    logger.info("Merged: %d countries with both area and publication data", len(merged))

    total_area = merged["Area"].sum()
    total_pubs = merged["Pubs"].sum()

    merged["area_share"] = merged["Area"] / total_area
    merged["pub_share"] = merged["Pubs"] / total_pubs
    merged["deficit"] = merged["area_share"] - merged["pub_share"]

    merged = merged.sort_values("deficit", ascending=False).reset_index(drop=True)
    merged.insert(0, "rank", merged.index + 1)
    return merged


def _to_csv_frame(merged: pd.DataFrame) -> pd.DataFrame:
    """Reshape the merged frame into the requested CSV column order/names."""
    return merged.rename(columns={
        "Country": "country", "Area": "area", "Pubs": "pubs",
    })[["country", "area", "pubs", "area_share", "pub_share", "deficit", "rank"]]


# =============================================================================
# Overlap report (deficit vs legacy median-quadrant "Understudied")
# =============================================================================


def _legacy_understudied(merged: pd.DataFrame) -> List[str]:
    """Reproduce the legacy median four-quadrant "Understudied" list.

    A country is "Understudied" if its area is above the median area AND its
    *integer* publication count is below the median.  Mirrors
    ``combined_viz.plot_area_pubs_bubble`` (full-count pubs, median split).
    """
    med_area = merged["Area"].median()
    med_pubs = merged["Pubs_int"].median()
    under = merged[(merged["Area"] > med_area) & (merged["Pubs_int"] < med_pubs)]
    return under.sort_values("Area", ascending=False)["Country"].tolist()


def log_overlap_report(merged: pd.DataFrame, top_n: int = 14) -> None:
    """Log the top-N countries by |deficit| — the strongest area-vs-pubs contrast.

    Ranking by absolute deficit captures BOTH extremes: over-researched
    method-exporters (China, USA — negative deficit) and under-researched
    mangrove-rich countries (Indonesia, Nigeria — positive deficit). Also logs
    overlap with the legacy median-quadrant "Understudied" list.
    """
    top_contrast = (
        merged.reindex(merged["deficit"].abs().sort_values(ascending=False).index)
        .head(top_n)
    )
    legacy = _legacy_understudied(merged)
    overlap = [c for c in legacy if c in top_contrast["Country"].tolist()]

    logger.info("=" * 60)
    logger.info("Strongest area-vs-pubs contrast (top %d by |deficit|)", top_n)
    logger.info("=" * 60)
    for i, (_, row) in enumerate(top_contrast.iterrows(), 1):
        side = "under" if row["deficit"] > 0 else "over"
        logger.info("  %2d  %-22s %+.6f  (%s-researched)",
                    i, row["Country"], row["deficit"], side)
    logger.info("Legacy 'Understudied' (median quadrant, N=%d):", len(legacy))
    logger.info("  %s", ", ".join(legacy))
    pct = 100 * len(overlap) / len(legacy) if legacy else 0.0
    logger.info("Overlap: %d / %d = %.0f%%", len(overlap), len(legacy), pct)
    logger.info("  %s", ", ".join(overlap))


# =============================================================================
# Visualization: area_share vs pub_share scatter
# =============================================================================


def plot_deficit_scatter(
    merged: pd.DataFrame,
    config: Config,
    output_path: Path,
    top_n: int = 20,
    over_n: int = 10,
) -> None:
    """Scatter of publication share vs area share, coloured by deficit.

    Points above the ``y = x`` diagonal are under-researched (deficit > 0).
    Both the most under-researched (top-``top_n`` by deficit) and the most
    over-researched (bottom-``over_n``, e.g. China, USA) countries are labelled
    in a fixed dark colour so they stay legible against the coloured points.
    """
    vis = config.visualization
    plt.rcParams["font.family"] = vis.font_family

    df = merged.copy()

    # Diverging colormap centred on zero deficit.
    vmax = max(abs(df["deficit"].min()), abs(df["deficit"].max())) or 1e-9
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = plt.cm.RdYlGn_r  # red = positive (under-researched), green = negative
    colors = cmap(norm(df["deficit"].values))

    fig, ax = plt.subplots(figsize=(13, 9))
    ax.set_facecolor("none")   # transparent: inset (below) shows through, points on top
    ax.set_zorder(3)           # composite main axes ABOVE the bar-chart inset

    # All countries (small, low alpha).
    # Dark edge (not white): RdYlGn_r's near-zero band is pale yellow and would
    # vanish on a white background; the outline keeps balanced countries legible.
    ax.scatter(
        df["area_share"], df["pub_share"],
        s=45, c=colors, edgecolors="#333333", linewidth=0.6, alpha=1.0, zorder=3,
    )

    # Diagonal y = x (deficit = 0).
    lims = [max(df["area_share"].min(), df["pub_share"].min()) * 0.7,
            min(df["area_share"].max(), df["pub_share"].max()) * 1.4]
    ax.plot(lims, lims, color="#333333", linewidth=1.2, linestyle="--", zorder=4)
    ax.annotate(
        "deficit = 0", xy=(lims[1], lims[1]), xytext=(-8, -8),
        textcoords="offset points", fontsize=9, color="#333333",
        ha="right", va="top",
    )

    # Label the most under-researched (top-N deficit) and most over-researched
    # (bottom-N deficit, e.g. China, USA). Fixed dark colour keeps labels
    # legible against the diverging-coloured points.
    labeled = pd.concat([df.head(top_n), df.tail(over_n)])
    for _, row in labeled.iterrows():
        ax.annotate(
            row["Country"],
            (row["area_share"], row["pub_share"]),
            textcoords="offset points",
            xytext=(6, 5),
            fontsize=6.75,
            fontweight="bold",
            color="#111111",
            alpha=0.95,
            zorder=6,
        )

    # Region shading labels (axes-fraction, not data points).
    ax.text(0.98, 0.02, "Under-researched\n(deficit > 0)",
            transform=ax.transAxes, fontsize=9, color="#b2182b",
            ha="right", va="bottom", style="italic")
    ax.text(0.38, 0.98, "Over-researched\n(deficit < 0)",
            transform=ax.transAxes, fontsize=9, color="#1a7837",
            ha="left", va="top", style="italic")

    # Log-log axes (shares span several orders of magnitude).
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Mangrove area share (log scale)", fontsize=11)
    ax.set_ylabel("Publication share (log scale)", fontsize=11)
    ax.set_title(
        "Mangrove-Area-Weighted Publication Deficit\n"
        "area share vs publication share (fractional counting); diagonal = balanced",
        fontsize=12, fontweight="bold",
    )
    ax.grid(True, linestyle=":", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Colorbar for deficit.
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Publication deficit (area share − pub share)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    # Polar mirror inset (replaces the grouped bar chart) — same strong-contrast
    # top-14 countries by |deficit|, sorted by area descending for the radial
    # (clockwise-from-12-o'clock) order.
    polar_data = (
        merged.reindex(merged["deficit"].abs().sort_values(ascending=False).index)
        .head(14)
        .sort_values("Area", ascending=False)
        .reset_index(drop=True)
    )
    _plot_polar_mirror_inset(fig, ax, merged, dpi=vis.dpi, data=polar_data,
                             position="upper-left", polar_w=0.2) # 大小（圆宽）↑ 变大，↓ 变小（高按 fig_w/fig_h 自动算，保持正圆）

    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=vis.dpi, format=vis.figure_format, bbox_inches="tight")
    plt.close(fig)
    logger.info("Deficit scatter saved: %s (%d countries labelled)", output_path, len(labeled))


def _draw_area_pubs_grouped_inset(
    fig: plt.Figure,
    ax: plt.Axes,
    merged: pd.DataFrame,
    n_bars: int = 14,
) -> None:
    """Grouped bar inset (upper-middle-left) inside the deficit scatter: mangrove
    area (green) and integer publications (blue), both bars upward, for the
    top-``n_bars`` countries by |deficit| (strongest area-vs-pubs contrast).

    Countries are sorted left→right from under-researched (tall green, short
    blue) to over-researched (tall blue, short green), so the panel spans the
    full contrast spectrum — Indonesia at the far left, China at the far right.
    """
    # ── Layout geometry (axes-fraction [0,1], relative to the main axes) ────
    # The inset sits in the upper-left, ABOVE Singapore's label (bottom 0.68 >
    # Singapore label top ~0.64) and shifted right 10 px (left 0.03 → 0.0335,
    # 10 px @300 dpi = 0.0035 axes-frac). The legend is placed BELOW the inset
    # (bbox_to_anchor y=-0.10). The inset stays composited BELOW the main axes
    # (zorder 2 < 3) with a transparent main background so any labelled point
    # still under it (e.g. Japan) draws on top of the bars.
    #
    #   ┌─────────────────────────────┐ ← ax.y1 (top)
    #   │ ┌───────────┐               │
    #   │ │  bars     │   scatter     │
    #   │ ├───────────┤               │
    #   │ │ legend ↓  │               │
    #   │ └───────────┘               │
    #   │   Singapore / Japan         │
    #   └─────────────────────────────┘ ← ax.y0 (bottom)
    #
    # Quick rebuild: pixi run python -m src.area_pub_deficit
    # ────────────────────────────────────────────────────────────────────────
    ax_ins = ax.inset_axes([0.0335, 0.68, 0.36, 0.28])  # [left, bottom, w, h]
    ax_ins.set_zorder(2)   # below main axes (zorder 3) → points render on top
    #   left ↑ → right  │  bottom ↑ → up  │  w/h ↑ → bigger inset

    data = (
        merged.reindex(merged["deficit"].abs().sort_values(ascending=False).index)
        .head(n_bars)
        .sort_values("deficit", ascending=False)  # under-researched → over-researched (left→right)
        .reset_index(drop=True)
    )
    n = len(data)
    x_idx = np.arange(n)
    area_vals = data["Area"].values.astype(float)
    pubs_vals = data["Pubs_int"].values.astype(float)
    area_norm = area_vals / area_vals.max()
    pub_norm = pubs_vals / pubs_vals.max()
    bar_w = 0.36

    ax_ins.bar(x_idx - bar_w / 2, area_norm, width=bar_w,
               color="#29cc3f", alpha=0.82, edgecolor="white", linewidth=0.3, zorder=3)
    ax_ins.bar(x_idx + bar_w / 2, pub_norm, width=bar_w,
               color="#4472C4", alpha=0.82, edgecolor="white", linewidth=0.3, zorder=3)

    ax_ins.set_ylim(0, 1.25)
    ax_ins.set_yticks([0, 0.5, 1.0])
    ax_ins.set_yticklabels(["0", "0.5", "1.0"], fontsize=5.5)
    ax_ins.set_ylabel("Normalized", fontsize=6.5, color="#555555")
    ax_ins.tick_params(axis="y", labelsize=5.5, colors="#555555")
    ax_ins.spines["top"].set_visible(False)
    ax_ins.spines["right"].set_visible(False)
    ax_ins.spines["left"].set_visible(False)

    _short = {"United States": "USA", "United Kingdom": "UK"}
    x_labels = [_short.get(c, c) for c in data["Country"].values]
    ax_ins.set_xticks(list(x_idx))
    ax_ins.set_xticklabels(x_labels, rotation=60, ha="right", fontsize=5.5)
    ax_ins.tick_params(axis="x", labelsize=5.5)

    ax_ins.legend(
        handles=[
            Patch(facecolor="#29cc3f", alpha=0.82, edgecolor="white",
                  label=f"Area (km²) — max {area_vals.max():,.0f}"),
            Patch(facecolor="#4472C4", alpha=0.82, edgecolor="white",
                  label=f"Pubs — max {pubs_vals.max():.0f}"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.10),   # below the inset (legend ↓)
        ncol=2,
        fontsize=5.5,
        frameon=True,
        facecolor="white",
        edgecolor="#dddddd",
        borderpad=0.4,
        labelspacing=0.3,
    )


def _plot_polar_mirror_inset(
    fig: plt.Figure,
    ax: plt.Axes,
    merged: pd.DataFrame,
    dpi: int = 300,
    polar_min_pubs: int = 10,
    polar_max_countries: int = 30,
    data: pd.DataFrame | None = None,
    position: str = "left",
    polar_w: float = 0.28,
) -> None:
    """Polar mirror inset (area inward, publications outward).

    Ported from ``combined_viz.plot_area_pubs_bubble_polar``. Uses integer
    full-count publications (``Pubs_int``) so bar lengths read as raw counts.

    ``data`` (optional) pre-selects the countries (columns Country/Area/Pubs_int);
    when omitted it defaults to the largest-area countries with at least
    ``polar_min_pubs`` publications. ``position`` = "left" (quadrant: left side,
    vertically centred) or "upper-left" (scatter: top-left, replacing the bars).
    """
    # ── Layout geometry (all values in figure-fraction [0,1]) ───────────────
    # Figure = 12×9 in @ vis.dpi dpi. The main axes has already been shrunk by
    # the colorbar; read its live bbox so the inset tracks it.
    # A circle needs its rectangle square in pixels:
    #   polar_h = polar_w * (fig_w_in / fig_h_in).
    #
    #   ┌──────────────────────────────────────────┐ ← fig top
    #   │  title (2 lines)                         │
    #   ├──────────────────────────────────────────┤ ← ax_bbox.y1 (axes top)
    #   │  ┌─────────┐                             │
    #   │  │  polar  │    deficit scatter          │
    #   │  │  mirror │    (main axes)              │
    #   │  └─────────┘                             │
    #   ├──────────────────────────────────────────┤ ← ax_bbox.y0
    #   └──────────────────────────────────────────┘
    #
    # Quick rebuild: pixi run python -m src.area_pub_deficit
    # ────────────────────────────────────────────────────────────────────────
    fig_w_in, fig_h_in = fig.get_size_inches()
    ax_bbox = ax.get_position()                     # Bbox in figure-fraction
    polar_h = polar_w * (fig_w_in / fig_h_in)       # keeps circle square in px
    font_scale = polar_w / 0.28                     # fonts tuned for polar_w=0.28 (quadrant)
    px_x = 1.0 / (fig_w_in * dpi)                   # one rendered pixel, x (fraction)
    if position == "upper-left":
        # Top-left, just inside the main axes (replaces the grouped bar chart).
        polar_x = ax_bbox.x0 + 0.06 # 水平位置，0.06是离散点图左缘的偏移：↑ 右移，↓ 左移 
        polar_y = ax_bbox.y1 - polar_h - 0.01   # -0.01 (was 0.02) moves the mirror up a bit5# 垂直位置，0.02 是离散点图顶缘的偏移：↑ 下移，↓ 上移 
    else:  # "left" — quadrant: left side, vertically centred, nudged right 10 px.
        polar_x = ax_bbox.x0 + 0.04 + 10 * px_x
        polar_cy = ax_bbox.y0 + ax_bbox.height * 0.50
        polar_y = polar_cy - polar_h / 2
    ax_polar = fig.add_axes(
        [polar_x, polar_y, polar_w, polar_h], projection="polar",
    )

    # Data: pre-selected if provided, else largest-area countries with enough pubs.
    if data is not None:
        polar_data = data.reset_index(drop=True)
    else:
        polar_data = merged[
            (merged["Pubs_int"] >= polar_min_pubs) & (merged["Area"] >= 1.0)
        ].sort_values("Area", ascending=False).reset_index(drop=True)
        if len(polar_data) > polar_max_countries:
            polar_data = polar_data.head(polar_max_countries).reset_index(drop=True)
    n_polar = len(polar_data)
    logger.info("Deficit polar inset: %d countries (min_pubs=%d)", n_polar, polar_min_pubs)

    angles = np.linspace(0, 2 * np.pi, n_polar, endpoint=False)
    area_vals = polar_data["Area"].values.astype(float)
    pubs_vals = polar_data["Pubs_int"].values.astype(float)
    polar_countries = polar_data["Country"].values

    # ── Internal polar geometry (radial data units) ─────────────────────────
    # base_r / max_pub_h / max_area_h control the ring radius and bar extent;
    # the bar_width multiplier controls sector width. Scale proportionally with polar_w.
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

    for i, (angle, country) in enumerate(zip(angles, polar_countries)):
        rot = np.rad2deg(angle)
        if angle < np.pi:
            ha, rotation = "left", -rot + 90
        else:
            ha, rotation = "right", -rot - 90

        pub_val = int(pubs_vals[i])
        area_val = area_vals[i]

        # Country name (blue bold) — unified style for all countries.
        name = country if len(country) <= 26 else country[:24] + ".."
        name_r = max(r_pubs[i], base_r) + 2.0
        ax_polar.text(
            angle, name_r, name,
            ha=ha, va="bottom", rotation=rotation, rotation_mode="anchor",
            fontsize=5.5 * font_scale, color="#2B5C8F", fontweight="bold",
        )

        # Publication count (bold blue) — on the outer side of the name. The
        # name's radial extent ≈ 2.5 units/char, so offset by that + a 2-unit gap.
        pub_r = name_r + 2.5 * len(name) + 2.0
        ax_polar.text(
            angle, pub_r, f"({pub_val})",
            ha=ha, va="bottom", rotation=rotation, rotation_mode="anchor",
            fontsize=5.5 * font_scale, color="#2B5C8F", fontweight="bold",
        )

        # Area value — inside the area bar, toward the centre (all countries).
        area_mid = r_area[i] - 22   # further toward centre to avoid occlusion
        ax_polar.text(
            angle, area_mid, f"{area_val:.2f}",
            ha=ha, va="center", rotation=rotation, rotation_mode="anchor",
            fontsize=6.0 * font_scale, color="#2E8B57", fontweight="bold",
        )

    ax_polar.set_theta_zero_location("N")
    ax_polar.set_theta_direction(-1)
    ax_polar.set_rticks([])
    ax_polar.set_thetagrids([])
    ax_polar.spines["polar"].set_visible(False)

    polar_legend = [
        Patch(facecolor="#2B5C8F", alpha=0.85,
              label=f"Pubs {pubs_vals.min():.0f}–{pubs_vals.max():.0f}"),
        Patch(facecolor="#2E8B57", alpha=0.85,
              label=f"Area {area_vals.min():,.0f}–{area_vals.max():,.0f} km²"),
    ]
    # Legend position: to the right of the circle in the scatter; above it in
    # the quadrant.
    if position == "upper-left":
        legend_loc, legend_anchor, legend_ncol = "center left", (1.08, 0.5), 1
    else:
        legend_loc = "lower center"
        legend_anchor = (0.5, 1.06 + 10 / (polar_h * fig_h_in * dpi))
        legend_ncol = 2
    ax_polar.legend(
        handles=polar_legend,
        loc=legend_loc,
        bbox_to_anchor=legend_anchor,
        ncol=legend_ncol, frameon=True, facecolor="white",
        fontsize=7 * font_scale, edgecolor="#dddddd",
    )


def plot_deficit_quadrant(
    merged: pd.DataFrame,
    config: Config,
    output_path: Path,
    top_n: int = 20,
) -> None:
    """Four-quadrant view of the area-weighted publication deficit.

    x = deficit (area_share − pub_share, continuous); y = fractional pubs (log).
    Quadrant lines at deficit = 0 (under/over boundary) and the median pub count
    (high/low pubs boundary). Points are coloured by deficit intensity; the
    top-N countries by |deficit| are labelled.
    """
    vis = config.visualization
    plt.rcParams["font.family"] = vis.font_family

    df = merged.copy()
    med_pubs = df["Pubs"].median()

    vmax = max(abs(df["deficit"].min()), abs(df["deficit"].max())) or 1e-9
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = plt.cm.RdYlGn_r  # red = positive (under-researched), green = negative
    colors = cmap(norm(df["deficit"].values))

    fig, ax = plt.subplots(figsize=(12, 9))

    # Dark edge (not white): RdYlGn_r's near-zero band is pale yellow and would
    # vanish on a white background; the outline keeps balanced countries legible.
    ax.scatter(
        df["deficit"], df["Pubs"],
        s=55, c=colors, edgecolors="#333333", linewidth=0.6, alpha=1.0, zorder=3,
    )

    # Quadrant boundaries: deficit = 0 (vertical) and median pubs (horizontal).
    ax.axvline(0.0, color="#333333", linewidth=1.1, linestyle="--", zorder=2)
    ax.axhline(med_pubs, color="#333333", linewidth=1.1, linestyle="--", zorder=2)

    # Label the top-N countries by |deficit| (captures both extremes).
    # Fixed dark colour keeps labels legible against the diverging-coloured points.
    top = df.reindex(df["deficit"].abs().sort_values(ascending=False).index).head(top_n)
    for _, row in top.iterrows():
        ax.annotate(
            row["Country"],
            (row["deficit"], row["Pubs"]),
            textcoords="offset points",
            xytext=(0, 7),
            ha="center",
            va="bottom",
            fontsize=7.5,
            fontweight="bold",
            color="#111111",
            alpha=0.95,
            zorder=6,
        )

    # Quadrant captions (axes-fraction; left = over-researched, right = under).
    # Top-left caption sits below China; y nudged up ~30 px (0.80 → 0.814).
    ax.text(0.02, 0.814, "Over-researched\nhigh-pubs\n(method-export)",
            transform=ax.transAxes, fontsize=9, color="#1a7837",
            ha="left", va="top", style="italic")
    ax.text(0.98, 0.98, "Under-researched\nhigh-pubs",
            transform=ax.transAxes, fontsize=9, color="#b2182b",
            ha="right", va="top", style="italic")
    ax.text(0.02, 0.02, "Over-researched\nlow-pubs",
            transform=ax.transAxes, fontsize=9, color="#1a7837",
            ha="left", va="bottom", style="italic")
    ax.text(0.98, 0.02, "Under-researched\nlow-pubs\n(conservation gap)",
            transform=ax.transAxes, fontsize=9, color="#b2182b",
            ha="right", va="bottom", style="italic")

    ax.set_yscale("log")
    ax.set_xlabel("Publication deficit (area share − pub share)", fontsize=11)
    ax.set_ylabel("Fractional publications (log scale)", fontsize=11)
    ax.set_title(
        "Four-Quadrant View of the Area-Weighted Publication Deficit\n"
        f"under/over boundary at deficit = 0; pubs boundary at median pubs ≈ {med_pubs:.1f}",
        fontsize=12, fontweight="bold",
    )
    ax.grid(True, linestyle=":", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Publication deficit (area share − pub share)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    _plot_polar_mirror_inset(fig, ax, merged, dpi=vis.dpi)

    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=vis.dpi, format=vis.figure_format, bbox_inches="tight")
    plt.close(fig)
    logger.info("Deficit quadrant saved: %s (%d countries labelled)", output_path, len(top))


# =============================================================================
# Orchestration
# =============================================================================


def run_area_pub_deficit(config: Config, top_n: int = 20, top_n_report: int = 14) -> None:
    """Compute the deficit, write the CSV, plot the scatter, and log the overlap."""
    paths = config.paths

    with Timer("Computing area-weighted publication deficit"):
        merged = compute_deficit(config)

    log_overlap_report(merged, top_n=top_n_report)

    csv_path = paths.processed_dir / "area_pub_deficit.csv"
    csv_frame = _to_csv_frame(merged)
    csv_frame.to_csv(csv_path, index=False, encoding="utf-8-sig", float_format="%.6f")
    logger.info("Saved area_pub_deficit.csv (%d countries)", len(csv_frame))

    fig_path = paths.figures_dir / "area_pub_deficit_scatter.png"
    plot_deficit_scatter(merged, config, fig_path, top_n=top_n)

    quad_path = paths.figures_dir / "area_pub_deficit_quadrant.png"
    plot_deficit_quadrant(merged, config, quad_path, top_n=top_n)

    logger.info("Area-weighted publication deficit complete.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Area-weighted publication deficit (under/over-research score)"
    )
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--top-n", type=int, default=20, help="Countries labelled in the scatter")
    args = parser.parse_args()

    config = get_config(args.config)
    setup_logging(config.system.log_level, config.system.log_file)
    run_area_pub_deficit(config, top_n=args.top_n)


if __name__ == "__main__":
    main()
