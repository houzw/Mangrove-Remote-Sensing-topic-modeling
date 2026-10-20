"""Step 3: Bibliometric Analysis — logistic growth fitting, research phase
classification, and country-level research density choropleth mapping.

Usage:
    python src/bibliometric_analysis.py              # uses default config.yaml
    python src/bibliometric_analysis.py --config path/to/config.yaml
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from scipy.optimize import curve_fit

from src.config import Config, get_config
from src.utils import Timer, canonical_countries, ensure_dir

logger = logging.getLogger(__name__)


# =============================================================================
# 3.1 — Logistic Growth Fitting
# =============================================================================


def logistic_function(t: np.ndarray, L: float, k: float, t0: float) -> np.ndarray:
    """3-parameter logistic growth function.

    ``f(t) = L / (1 + exp(-k * (t - t0)))``

    Args:
        t: Normalised time (years since first data year, i.e. t = PY - PY_min).
        L: Carrying capacity (asymptotic maximum cumulative publications).
        k: Intrinsic growth rate (steepness of the S-curve).
        t0: Inflection point in normalised years (where f(t0) = L/2).

    Returns:
        Predicted cumulative publication counts.
    """
    return L / (1.0 + np.exp(-k * (t - t0)))


def compute_cumulative(df: pd.DataFrame) -> pd.DataFrame:
    """Compute cumulative publication counts by year.

    Args:
        df: Cleaned DataFrame with ``PY`` column.

    Returns:
        DataFrame with columns ``PY``, ``count``, ``cumulative``.
    """
    yearly = df.groupby("PY").size().reset_index(name="count")
    yearly = yearly.sort_values("PY")
    yearly["cumulative"] = yearly["count"].cumsum()
    logger.info("Yearly stats: %d years, %d total publications", len(yearly), yearly["cumulative"].iloc[-1])
    return yearly


def fit_logistic(
    x: np.ndarray,
    y: np.ndarray,
    p0: List[float],
    bounds: Tuple[List[float], List[float]],
    maxfev: int = 10000,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Fit the logistic growth function to (x, y) data.

    Args:
        x: Calendar year values.
        y: Cumulative publication counts.
        p0: Initial guesses ``[L, k, t0]`` (t0 in normalised years).
        bounds: ``([L_lo, k_lo, t0_lo], [L_hi, k_hi, t0_hi])``.
        maxfev: Maximum function evaluations for ``curve_fit``.

    Returns:
        ``(popt, pcov, r_squared, t_offset)`` where ``popt = [L, k, t0]``
        and *t_offset* is ``x.min()`` (added back to recover calendar years).

    Raises:
        RuntimeError: If the fit fails to converge.
    """
    t_offset = float(x.min())
    t_norm = x.astype(float) - t_offset

    logger.info("Fitting logistic curve with p0=%s", p0)
    logger.info("  x range: %.0f – %.0f  (t_norm: %.0f – %.0f),  y range: %.0f – %.0f",
                x.min(), x.max(), t_norm.min(), t_norm.max(), y.min(), y.max())

    try:
        popt, pcov = curve_fit(
            logistic_function,
            t_norm,
            y.astype(float),
            p0=p0,
            bounds=bounds,
            maxfev=maxfev,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Logistic curve fitting failed: {exc}. "
            f"Try adjusting 'logistic_fit.p0' in config.yaml (current: {p0})."
        ) from exc

    # R^2
    y_pred = logistic_function(t_norm, *popt)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    L_fit, k_fit, t0_fit = popt
    t_mid_year = t_offset + t0_fit
    peak_annual = k_fit * L_fit / 4.0

    logger.info("Fit results: L=%.0f, k=%.4f, t0=%.1f (norm) / %.1f (year)",
                L_fit, k_fit, t0_fit, t_mid_year)
    logger.info("  R^2=%.4f, peak_annual=%.0f", r_squared, peak_annual)

    return popt, pcov, r_squared, t_offset


def compute_phase_boundaries(
    popt: np.ndarray,
    t_offset: float,
) -> Dict[str, Dict[str, float]]:
    """Compute 10%, 50%, 90% phase-boundary years from fitted logistic.

    Analytical solution (reference method):
        ``t = t0 - ln((1-y)/y) / k``   →   real year = t + t_offset

    For y = 0.1:  ``t10 = t0 - ln(9) / k``
    For y = 0.5:  ``t50 = t0``
    For y = 0.9:  ``t90 = t0 + ln(9) / k``

    Args:
        popt: Fitted ``[L, k, t0]`` (t0 in normalised years).
        t_offset: ``PY.min()`` — added back to get calendar years.

    Returns:
        Dict with keys ``"10%"``, ``"50%"``, ``"90%"``, each mapping to
        ``{"year": …, "value": …}``.
    """
    L_fit, k_fit, t0_fit = popt
    ln9 = np.log(9.0)

    t10_norm = t0_fit - ln9 / k_fit
    t50_norm = t0_fit
    t90_norm = t0_fit + ln9 / k_fit

    return {
        "10%": {"year": t_offset + t10_norm, "value": L_fit * 0.10},
        "50%": {"year": t_offset + t50_norm, "value": L_fit * 0.50},
        "90%": {"year": t_offset + t90_norm, "value": L_fit * 0.90},
    }


def classify_phases(
    yearly: pd.DataFrame,
    popt: np.ndarray,
    t_offset: float,
    thresholds: Dict[str, List[float]],
    labels: Dict[str, str],
) -> pd.DataFrame:
    """Assign research phases to each year using the fitted logistic curve.

    Args:
        yearly: DataFrame with ``PY`` and ``cumulative`` columns.
        popt: Fitted ``[L, k, t0]``.
        t_offset: ``PY.min()``.
        thresholds: e.g. ``{"embryonic": [0, 0.1], "growth": [0.1, 0.5],
            "mature": [0.5, 0.9], "saturation": [0.9, 1.0]}``.
        labels: Human-readable phase labels.

    Returns:
        *yearly* with added ``y_norm`` and ``phase`` columns.
    """
    L_fit = popt[0]
    yearly = yearly.copy()
    yearly["y_norm"] = yearly["cumulative"] / L_fit

    def _label(norm_val: float) -> str:
        for phase_key, (lo, hi) in thresholds.items():
            if lo < norm_val <= hi or (lo == 0 and norm_val == 0):
                return labels.get(phase_key, phase_key)
        return "Unknown"

    yearly["phase"] = yearly["y_norm"].apply(_label)

    for phase in yearly["phase"].unique():
        yrs = yearly[yearly["phase"] == phase]["PY"]
        logger.info("  %s: %d – %d (%d years)",
                    phase, yrs.min(), yrs.max(), len(yrs))

    return yearly


def _auto_label(ax, rects):
    """Add count labels above bar tops (reference style)."""
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2.0 - 0.08,
                1.02 * height, "%s" % int(height),
                size=10, family="Times New Roman")


def plot_publication_trend(
    yearly: pd.DataFrame,
    popt: np.ndarray,
    r_squared: float,
    t_offset: float,
    config: Config,
    output_path: Path,
    df_cleaned: pd.DataFrame | None = None,
    rank_by: str = "influence",
    show_formula: bool = True,
) -> None:
    """Plot cumulative publications with logistic fit, plus optional journal inset.

    Dual-Y axes (reference: ``logistic_mangrove_uav.py``):
    - Left Y: sky-blue bars with data labels for annual publications.
    - Right Y: blue-star scatter (cumulative) + red dotted logistic curve
      + green diamond future predictions.
    - Grey dashed vertical lines at 10% and 90% phase boundaries.
    - ``annotate`` phase labels inside the plot area.

    When *df_cleaned* is provided, a compact Top-15 journal bar chart is
    inserted on the left side.  *rank_by* controls the ranking metric:
    ``"influence"`` (I_journal = cites/articles) or ``"articles"`` (N_pub).
    *show_formula* toggles the fitted LGM equation annotation.
    """
    cfg = config.logistic_fit
    vis = config.visualization

    L_fit, k_fit, t0_fit = popt
    boundaries = compute_phase_boundaries(popt, t_offset)

    # Convert to normalised years for the reference-style approach
    x_norm = yearly["PY"].values.astype(float) - t_offset
    annual = yearly["count"].values.astype(float)
    cumulative = yearly["cumulative"].values.astype(float)

    plt.rcParams["font.family"] = vis.font_family
    fig, ax1 = plt.subplots(figsize=(16, 6))
    ax2 = ax1.twinx()
    fig.subplots_adjust(left=0.40, right=0.92)

    # ── left Y: annual publications (bars with centred labels) ─────
    bars = ax1.bar(x_norm, annual, color="skyblue",
                   label="Annual Publications")
    for rect in bars:
        height = rect.get_height()
        ax1.text(rect.get_x() + rect.get_width() / 2.0, 1.02 * height,
                 "%s" % int(height), size=8, ha="center",
                 family="Times New Roman")
    ax1.set_ylabel("Annual Publications")
    ax1.set_ylim(0, annual.max() * 1.20)

    # ── right Y: cumulative + logistic fit ─────────────────────────
    plot1, = ax2.plot(x_norm, cumulative, "*",
                      label="Cumulative Publications", color="b")

    t_smooth_norm = np.linspace(0, max(x_norm[-1] + 15, t0_fit * 1.5), 250)
    y_smooth = logistic_function(t_smooth_norm, L_fit, k_fit, t0_fit)
    plot2, = ax2.plot(t_smooth_norm, y_smooth, color="red",
                      linestyle=":", label="Logistic Growth Curve")

    future_norm = np.arange(x_norm[-1] + 1, int(t0_fit * 1.6))
    if len(future_norm) > 0:
        future_pred = logistic_function(future_norm, L_fit, k_fit, t0_fit)
        plot3, = ax2.plot(future_norm, future_pred, "D", color="g",
                          label="Predicted Publications")

    ax2.set_ylabel("Cumulative Publications")

    # ── x-axis: sparse labels, every 5 years ────────────────────────
    x_end = max(int(t_smooth_norm[-1]), int(future_norm[-1]) if len(future_norm) > 0 else int(x_norm[-1]))
    tick_positions = list(range(0, x_end + 2, 5))
    tick_labels = [str(int(p + t_offset)) for p in tick_positions]
    plt.xticks(tick_positions, tick_labels, rotation=45)
    plt.xlabel("Year")

    # ── phase boundaries: 10% and 90% only (3-stage model) ──────────
    t10_norm = boundaries["10%"]["year"] - t_offset
    t90_norm = boundaries["90%"]["year"] - t_offset

    ax2.axvline(x=t10_norm, color="grey", linestyle="--")
    ax2.axvline(x=t90_norm, color="grey", linestyle="--")

    y_annotate = L_fit * 0.25
    ax2.annotate("Infancy Stage", xy=(t10_norm / 2.0, y_annotate),
                 fontsize=9, fontweight="bold", color="grey", ha="center")
    ax2.annotate("Growth Stage", xy=((t10_norm + t90_norm) / 2.0, y_annotate),
                 fontsize=9, fontweight="bold", color="grey", ha="center")
    ax2.annotate("Mature Stage",
                 xy=((t90_norm + t_smooth_norm[-1]) / 2.0, y_annotate),
                 fontsize=9, fontweight="bold", color="grey", ha="center")

    # ── legend ─────────────────────────────────────────────────────
    handles = [plot1, plot2]
    labels_list = ["Cumulative Publications", "Logistic Growth Curve"]
    if len(future_norm) > 0:
        handles.append(plot3)
        labels_list.append("Predicted Publications")
    handles.append(bars)
    labels_list.append("Annual Publications")
    plt.legend(handles=handles, labels=labels_list, loc="upper left")

    # ── fitted LGM formula (Schraven et al. 2021 parametrization) ──
    if show_formula:
        t_mid_year = t_offset + t0_fit
        b_fit = k_fit * t_mid_year
        formula = (
            rf"$y = \frac{{{L_fit:.0f}}}{{1 + e^{{{b_fit:.1f}-{k_fit:.3f}x}}}}$"
        )
        ax2.text(0.99, 0.97, formula, transform=ax2.transAxes, fontsize=12,
                 ha="right", va="top", color="red",
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                           edgecolor="red", linewidth=1.0))

    # ── x-axis range ───────────────────────────────────────────────
    x_max_norm = max(t_smooth_norm[-1], future_norm[-1] + 2 if len(future_norm) > 0 else 0)
    plt.xlim(-1, x_max_norm)
    ax2.set_ylim(0, L_fit * 1.10)

    # Layout main axes first so tight_layout doesn't trip over the inset
    fig.tight_layout()

    # ── left inset: Top-15 journal bar chart ─────────────────────────
    if df_cleaned is not None and "SO" in df_cleaned.columns:
        _JCR: dict[str, str] = {
            "GLOBAL ECOLOGY AND BIOGEOGRAPHY":                 "Global Ecol. Biogeogr.",
            "MICROBIAL ECOLOGY":                               "Microb. Ecol.",
            "PROCEEDINGS OF THE NATIONAL ACADEMY OF SCIENCES "
            "OF THE UNITED STATES OF AMERICA":                 "P. Natl. Acad. Sci. USA",
            "SCIENCE BULLETIN":                                "Sci. Bull.",
            "NATURE":                                          "Nature",
            "PROGRESS IN PHYSICAL GEOGRAPHY-EARTH "
            "AND ENVIRONMENT":                                 "Prog. Phys. Geog.",
            "AQUATIC BOTANY":                                  "Aquat. Bot.",
            "COMPLEX & INTELLIGENT SYSTEMS":                   "Complex Intell. Syst.",
            "NATURE COMMUNICATIONS":                           "Nat. Commun.",
            "AQUACULTURE":                                     "Aquaculture",
            "ENVIRONMENTAL CONSERVATION":                      "Environ. Conserv.",
            "ISPRS JOURNAL OF PHOTOGRAMMETRY "
            "AND REMOTE SENSING":                              "ISPRS J. Photogramm.",
            "BOTANICAL JOURNAL OF THE LINNEAN SOCIETY":        "Bot. J. Linn. Soc.",
            "REMOTE SENSING OF ENVIRONMENT":                   "Remote Sens. Environ.",
            # ── article-count table journals ──
            "REMOTE SENSING":                                  "Remote Sens.",
            "INTERNATIONAL JOURNAL OF REMOTE SENSING":         "Int. J. Remote Sens.",
            "REGIONAL STUDIES IN MARINE SCIENCE":              "Reg. Stud. Mar. Sci.",
            "ESTUARINE COASTAL AND SHELF SCIENCE":             "Estuar. Coast. Shelf Sci.",
            "INTERNATIONAL JOURNAL OF APPLIED EARTH "
            "OBSERVATION AND GEOINFORMATION":                  "Int. J. Appl. Earth Obs. Geoinf.",
            "JOURNAL OF COASTAL RESEARCH":                     "J. Coast. Res.",
            "FORESTS":                                         "Forests",
            "ECOLOGICAL INDICATORS":                           "Ecol. Indic.",
            "OCEAN & COASTAL MANAGEMENT":                      "Ocean Coast. Manag.",
            "IEEE JOURNAL OF SELECTED TOPICS IN APPLIED "
            "EARTH OBSERVATIONS AND REMOTE SENSING":           "IEEE J. Sel. Top. Appl. Earth Obs. Remote Sens.",
            "MARINE POLLUTION BULLETIN":                       "Mar. Pollut. Bull.",
            "ENVIRONMENTAL MONITORING AND ASSESSMENT":         "Environ. Monit. Assess.",
            "JOURNAL OF COASTAL CONSERVATION":                 "J. Coast. Conserv.",
            "SUSTAINABILITY":                                  "Sustainability",
        }
        jstats = df_cleaned.groupby("SO").agg(
            Articles=("SO", "size"), Total_TC=("TC", "sum"),
        )
        jstats["Influence"] = jstats["Total_TC"] / jstats["Articles"].clip(lower=1)

        if rank_by == "articles":
            jstats = jstats.sort_values("Articles", ascending=False).head(15).iloc[::-1]
            j_vals = jstats["Articles"].values.astype(float)
            xlabel = "Number of Publications"
            label_fmt = lambda abbr, v, art, tc: f"{abbr}  ({int(art)})"
        else:
            jstats = jstats.sort_values("Influence", ascending=False).head(15).iloc[::-1]
            j_vals = jstats["Influence"].values
            xlabel = "Influence of Journals"
            label_fmt = lambda abbr, v, art, tc: f"{abbr}  {v:.1f} ({int(tc)}/{int(art)})"

        j_abbrevs = [_JCR.get(n, n[:30]) for n in jstats.index]
        j_art = jstats["Articles"].values
        j_tc = jstats["Total_TC"].values

        # Anchor journal axes to the main axes position after tight_layout
        mpos = ax1.get_position()
        ax_j = fig.add_axes([
            mpos.x0 + 0.045,
            mpos.y0 + mpos.height * 0.35,
            mpos.width * 0.28,
            mpos.height * 0.45,
        ])
        cmap_j = plt.cm.viridis
        colors_j = cmap_j(np.linspace(0.2, 0.8, len(j_abbrevs)))

        ax_j.barh(range(len(j_abbrevs)), j_vals, color=colors_j,
                  edgecolor="white", height=0.7)
        ax_j.set_yticks(range(len(j_abbrevs)))
        ax_j.set_yticklabels([""] * len(j_abbrevs))
        ax_j.invert_yaxis()
        ax_j.set_xlabel(xlabel, fontsize=8)
        ax_j.set_ylabel("Top 15 Journals", fontsize=8, labelpad=6)
        ax_j.set_xlim(0, j_vals.max() * 1.45)
        ax_j.tick_params(axis="x", labelsize=7)
        ax_j.spines["top"].set_visible(False)
        ax_j.spines["right"].set_visible(False)
        ax_j.tick_params(axis="y", length=0)
        ax_j.grid(axis="x", linestyle=":", alpha=0.3)

        for bar, val, abbr, art, tc in zip(
            ax_j.containers[0], j_vals, j_abbrevs, j_art, j_tc,
        ):
            label = label_fmt(abbr, val, art, tc)
            ax_j.text(bar.get_width() + j_vals.max() * 0.01,
                      bar.get_y() + bar.get_height() / 2.0,
                      label, va="center", fontsize=6.5,
                      color="0.15", fontweight="bold")
    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=vis.dpi, format=vis.figure_format, bbox_inches="tight")
    plt.close(fig)
    logger.info("Publication trend plot saved: %s", output_path)


# =============================================================================
# 3.2 — Country Choropleth
# =============================================================================


COUNTRY_NAME_MAPPING_DEFAULT: Dict[str, str] = {
    "USA": "United States of America",
    "England": "United Kingdom",
    "Scotland": "United Kingdom",
    "Wales": "United Kingdom",
    "Peoples R China": "China",
    "Taiwan": "China",
    "Russia": "Russian Federation",
    "South Korea": "Republic of Korea",
    "North Korea": "Dem. Rep. Korea",
    "Vietnam": "Viet Nam",
    "Iran": "Iran (Islamic Republic of)",
    "Tanzania": "United Republic of Tanzania",
    "Syria": "Syrian Arab Republic",
    "Laos": "Lao People's Democratic Republic",
    "Brunei": "Brunei Darussalam",
    "Ivory Coast": "Côte d'Ivoire",
    "Congo": "Republic of the Congo",
    "Dem Rep Congo": "Democratic Republic of the Congo",
}

# Natural Earth sub-entities that should inherit a parent country's data
# (entities with separate geometries but same sovereign country).
# Key = sub-entity ADMIN name, Value = parent ADMIN name (must exist in data).
_SUBENTITY_PARENT: Dict[str, str] = {
    "Taiwan": "China",
    "Hong Kong": "China",
    "Macau": "China",
    "Puerto Rico": "United States of America",
}


def compute_country_stats(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    """Count publications per country, handling multi-country entries.

    WoS ``CU`` field may contain semicolon-separated country lists (e.g.,
    ``"USA;China;Australia"``). Each country in a multi-country record
    receives one count — after the name mapping of ``mapping`` is applied and
    the result is de-duplicated, so a ``"China;Taiwan"`` record counts once for
    China rather than twice (see :func:`src.utils.canonical_countries`).

    Args:
        df: Cleaned DataFrame with ``CU`` column.
        mapping: Country name normalisation dict (WoS name → standard name).

    Returns:
        DataFrame: ``Country, Publications`` sorted descending.
    """
    logger.info("Computing country statistics ...")

    records = []
    for cu_val in df["CU"].dropna():
        records.extend(canonical_countries(cu_val, fold=mapping))

    pub_series = pd.Series(records).value_counts().reset_index()
    pub_series.columns = ["Country", "Publications"]
    logger.info("  %d unique countries, %d total country-assignments", len(pub_series), len(records))
    return pub_series


def load_mangrove_area(path: Path, area_column: str = "sum") -> pd.DataFrame:
    """Load mangrove area CSV from ``data/external/``.

    Expects columns: ``Country`` and the configured area column.

    Args:
        path: Path to the CSV file.
        area_column: Name of the column containing mangrove area values.

    Returns:
        DataFrame with columns ``Country`` and ``Area`` (standardised).

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If required columns are missing.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Mangrove area file not found: {path}. "
            f"Place your mangrove area CSV in data/external/."
        )

    df = pd.read_csv(path)
    required = {"Country", area_column}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Mangrove area file missing columns: {missing}. "
            f"Expected: Country and {area_column}. Found: {list(df.columns)}"
        )
    # Rename to internal standard
    df = df.rename(columns={area_column: "Area"})
    logger.info("Loaded mangrove area data: %d countries (column '%s')", len(df), area_column)
    return df


def build_choropleth_geodataframe(
    pub_stats: pd.DataFrame,
    area_df: pd.DataFrame,
    mapping: Dict[str, str],
    world_map_source: str,
    area_unit: str = "km2",
) -> "gpd.GeoDataFrame":  # noqa: F821
    """Merge publication stats, mangrove area, and world geometry.

    Args:
        pub_stats: Country publication counts.
        area_df: Mangrove area data (columns: Country, Area).
        mapping: Country name mapping for additional normalization.
        world_map_source: Path to world map shapefile.
        area_unit: Unit label for the area values (e.g., "km2").

    Returns:
        GeoDataFrame with columns: ``Country, Publications, Area,
        research_density, geometry``.
    """
    import geopandas as gpd

    logger.info("Building choropleth GeoDataFrame ...")

    # Load world map from configured path
    world_path = Path(world_map_source)
    if not world_path.is_absolute():
        world_path = Path.cwd() / world_path
    if not world_path.exists():
        raise RuntimeError(
            f"World map not found at {world_path}. "
            f"Set world_map_source in config.yaml to a valid shapefile path."
        )
    world = gpd.read_file(world_path)

    logger.info("  World map: %d countries", len(world))

    # Merge publication counts with world map (Natural Earth uses ADMIN column)
    merge_col = "ADMIN" if "ADMIN" in world.columns else "name"
    world = world.merge(pub_stats, left_on=merge_col, right_on="Country", how="left")

    # Normalize area_df country names
    area_df = area_df.copy()
    area_df["Country"] = area_df["Country"].map(mapping).fillna(area_df["Country"])

    # Merge area data
    world = world.merge(area_df, left_on=merge_col, right_on="Country", how="left")

    # Compute research density
    world["Publications"] = world["Publications"].fillna(0).astype(int)
    world["Area"] = pd.to_numeric(world["Area"], errors="coerce")
    world["research_density"] = np.where(
        world["Area"].notna() & (world["Area"] > 0),
        world["Publications"] / world["Area"],
        0.0,
    )

    # --- Propagate parent-country data to sub-entities ---
    # Some Natural Earth geometries (e.g., Taiwan, Hong Kong) have separate
    # rows but belong to the same sovereign country.  Copy the parent's
    # Publications / Area / research_density so they render with the same
    # colour on the choropleth.
    for sub, parent in _SUBENTITY_PARENT.items():
        sub_mask = world[merge_col] == sub
        parent_mask = world[merge_col] == parent
        if sub_mask.any() and parent_mask.any():
            parent_row = world.loc[parent_mask].iloc[0]
            world.loc[sub_mask, "Publications"] = parent_row["Publications"]
            world.loc[sub_mask, "Area"] = parent_row["Area"]
            world.loc[sub_mask, "research_density"] = parent_row["research_density"]
            logger.info("  %s → inherited %s data (pubs=%d, area=%s)",
                        sub, parent,
                        int(parent_row["Publications"]),
                        f"{parent_row['Area']:.1f}" if pd.notna(parent_row["Area"]) else "NaN")

    # Count countries with mangrove data
    n_with_area = (world["Area"] > 0).sum()
    n_with_pubs = (world["Publications"] > 0).sum()
    logger.info("  Countries with mangrove area data: %d", n_with_area)
    logger.info("  Countries with publications: %d", n_with_pubs)

    return world


def plot_choropleth(
    gdf: "gpd.GeoDataFrame", config: Config, output_path: Path, area_unit: str = "km2"  # noqa: F821
) -> None:
    """Plot research density choropleth map.

    Args:
        gdf: GeoDataFrame from ``build_choropleth_geodataframe``.
        config: Full configuration.
        output_path: Where to save the figure.
        area_unit: Unit label for mangrove area (e.g., "km2").
    """
    vis = config.visualization
    plt.rcParams["font.family"] = vis.font_family

    fig, ax = plt.subplots(1, 1, figsize=(16, 9))

    # --- Base layer: all countries with visible boundaries ---
    # Draw the full world outline first so every country boundary is visible,
    # even for nations without mangrove data.
    gdf.plot(
        ax=ax,
        facecolor="whitesmoke",
        edgecolor="#444444",
        linewidth=0.4,
        zorder=0,
    )

    # --- Choropleth layer: countries with mangrove area data ---
    plot_gdf = gdf[gdf["Area"] > 0].copy()

    if len(plot_gdf) == 0:
        logger.warning("No countries with both mangrove area and publications — plotting all countries")
        plot_gdf = gdf.copy()

    density_label = f"Research Density (publications / {area_unit})"
    plot_gdf.plot(
        column="research_density",
        ax=ax,
        legend=True,
        legend_kwds={"label": density_label, "shrink": 0.6},
        cmap=vis.color_palette,
        edgecolor="#222222",
        linewidth=0.5,
        missing_kwds={"color": "lightgrey", "edgecolor": "#444444", "linewidth": 0.4, "label": "No data"},
        zorder=1,
    )

    ax.set_title(
        f"Mangrove Remote Sensing Research Density by Country\n"
        f"(Publications per {area_unit} of Mangrove Area)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_axis_off()

    fig.tight_layout()
    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=vis.dpi, format=vis.figure_format, bbox_inches="tight")
    plt.close(fig)
    logger.info("Choropleth map saved: %s", output_path)


# =============================================================================
# 3.3 — Journal & country bar charts (Hu et al. 2025 style)
# =============================================================================


def plot_top_journals(
    df: pd.DataFrame,
    config: Config,
    output_path: Path,
    top_n: int = 15,
) -> None:
    """Horizontal bar chart of top-N journals by total citations (Hu 2025 Fig. 2b).

    Influence is approximated by total citations per journal divided by
    number of articles (mean citations per article), since we lack external
    impact-factor data.

    Args:
        df: Cleaned DataFrame with ``SO`` (journal) and ``TC`` (citations).
        config: Validated configuration.
        output_path: Where to save the PNG.
        top_n: Number of journals to display.
    """
    if "SO" not in df.columns:
        logger.warning("No 'SO' column — skipping journal ranking plot")
        return

    vis = config.visualization
    plt.rcParams["font.family"] = vis.font_family

    # Compute influence = total citations / article count
    journal_stats = df.groupby("SO").agg(
        Articles=("SO", "size"),
        Total_TC=("TC", "sum"),
    ).reset_index()
    journal_stats["Influence"] = journal_stats["Total_TC"] / journal_stats["Articles"].clip(lower=1)
    journal_stats = journal_stats.sort_values("Influence", ascending=False).head(top_n)
    journal_stats = journal_stats.iloc[::-1]  # largest at top

    if len(journal_stats) == 0:
        logger.warning("No journal data — skipping plot")
        return

    influence = journal_stats["Influence"].values
    articles = journal_stats["Articles"].values
    total_tc = journal_stats["Total_TC"].values

    # JCR-style abbreviations (Title Case)
    _JCR_ABBREV: dict[str, str] = {
        "GLOBAL ECOLOGY AND BIOGEOGRAPHY":                 "Global Ecol. Biogeogr.",
        "MICROBIAL ECOLOGY":                               "Microb. Ecol.",
        "PROCEEDINGS OF THE NATIONAL ACADEMY OF SCIENCES "
        "OF THE UNITED STATES OF AMERICA":                 "P. Natl. Acad. Sci. USA",
        "SCIENCE BULLETIN":                                "Sci. Bull.",
        "NATURE":                                          "Nature",
        "PROGRESS IN PHYSICAL GEOGRAPHY-EARTH "
        "AND ENVIRONMENT":                                 "Prog. Phys. Geog.",
        "AQUATIC BOTANY":                                  "Aquat. Bot.",
        "COMPLEX & INTELLIGENT SYSTEMS":                   "Complex Intell. Syst.",
        "NATURE COMMUNICATIONS":                           "Nat. Commun.",
        "AQUACULTURE":                                     "Aquaculture",
        "ENVIRONMENTAL CONSERVATION":                      "Environ. Conserv.",
        "ISPRS JOURNAL OF PHOTOGRAMMETRY "
        "AND REMOTE SENSING":                              "ISPRS J. Photogramm.",
        "BOTANICAL JOURNAL OF THE LINNEAN SOCIETY":        "Bot. J. Linn. Soc.",
        "REMOTE SENSING OF ENVIRONMENT":                   "Remote Sens. Environ.",
    }

    abbrevs = [_JCR_ABBREV.get(n, n[:30]) for n in journal_stats["SO"]]

    fig, ax = plt.subplots(figsize=(13, 7.5))
    cmap = plt.cm.viridis
    colors = cmap(np.linspace(0.2, 0.8, len(abbrevs)))

    bars = ax.barh(range(len(abbrevs)), influence, color=colors,
                   edgecolor="white", height=0.7)

    # Y-axis: no tick labels; use axis label instead
    ax.set_yticks(range(len(abbrevs)))
    ax.set_yticklabels([""] * len(abbrevs))
    ax.invert_yaxis()
    ax.set_ylabel("Top 15 Journals", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_xlabel("Influence of Journals ($I_{\\mathrm{journal}}$)", fontsize=12)
    ax.set_xlim(0, influence.max() * 1.50)

    # Bar-head annotations: abbreviation + influence + (articles, citations)
    for bar, inf, abbr, art, tc in zip(bars, influence, abbrevs, articles, total_tc):
        label = f"{abbr}  {inf:.1f} (n={int(art)}, {int(tc)} cites)"
        ax.text(bar.get_width() + influence.max() * 0.01,
                bar.get_y() + bar.get_height() / 2.0,
                label, va="center", fontsize=9,
                color="0.15", fontweight="bold")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", linestyle=":", alpha=0.4)

    ax.set_title("Top 15 Journals by Influence Index\n"
                 "$I_{\\mathrm{journal}} = N_{\\mathrm{cite}} \\,/\\, N_{\\mathrm{pub}}$",
                 fontsize=14, fontweight="bold")

    fig.tight_layout()
    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=vis.dpi, format=vis.figure_format, bbox_inches="tight")
    plt.close(fig)
    logger.info("Top journals plot saved: %s", output_path)


def plot_country_bar_chart(
    pub_stats: pd.DataFrame,
    config: Config,
    output_path: Path,
    top_n: int = 20,
) -> None:
    """Horizontal bar chart of top-N countries by publication count (Hu 2025 Fig. 8a).

    Args:
        pub_stats: DataFrame from ``compute_country_stats`` with
            ``Country`` and ``Publications`` columns.
        config: Validated configuration.
        output_path: Where to save the PNG.
        top_n: Number of countries to display.
    """
    vis = config.visualization
    plt.rcParams["font.family"] = vis.font_family

    df = pub_stats.head(top_n).copy()
    df = df.iloc[::-1]  # largest at top

    if len(df) == 0:
        logger.warning("No country data — skipping bar chart")
        return

    labels = df["Country"].values
    counts = df["Publications"].values.astype(int)

    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = plt.cm.viridis
    colors = cmap(np.linspace(0.2, 0.8, len(labels)))

    bars = ax.barh(range(len(labels)), counts, color=colors, edgecolor="white", height=0.7)

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Number of Publications", fontsize=11)
    ax.set_xlim(0, counts.max() * 1.18)

    for bar, val in zip(bars, counts):
        ax.text(bar.get_width() + counts.max() * 0.008,
                bar.get_y() + bar.get_height() / 2.0,
                str(val), va="center", fontsize=8, color="0.3")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", linestyle=":", alpha=0.4)

    ax.set_title("Publication Count by Country",
                 fontsize=13, fontweight="bold")

    fig.tight_layout()
    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=vis.dpi, format=vis.figure_format, bbox_inches="tight")
    plt.close(fig)
    logger.info("Country bar chart saved: %s", output_path)


# =============================================================================
# Main entry point
# =============================================================================


def run_bibliometric(config: Config) -> None:
    """Run both parts of the bibliometric analysis (3.1 + 3.2)."""
    paths = config.paths

    # ------------------------------------------------------------------
    # 3.1 — Logistic growth fitting
    # ------------------------------------------------------------------
    with Timer("Step 3.1 — Logistic Growth Fitting"):
        cleaned_path = paths.processed_dir / config.preprocessing.output_file
        if not cleaned_path.exists():
            raise FileNotFoundError(f"Cleaned data not found: {cleaned_path}. Run preprocessing first.")

        df = pd.read_csv(cleaned_path, encoding="utf-8-sig")
        logger.info("Loaded cleaned data: %d records", len(df))

        yearly = compute_cumulative(df)

        # The final corpus year (2026, retrieved 2026-07-15) is only partially
        # indexed: complete years enter the fit; the full yearly table (which
        # still contains the truncated year) is kept for plotting and phases.
        fit_end_year = 2025
        yearly_fit = yearly[yearly["PY"] <= fit_end_year]

        x = yearly_fit["PY"].values.astype(float)
        y = yearly_fit["cumulative"].values.astype(float)

        # Build bounds from config
        b = config.logistic_fit.bounds
        lower = [b.L[0], b.k[0], b.t0[0]]
        upper = [b.L[1], b.k[1], b.t0[1]]

        popt, _pcov, r_squared, t_offset = fit_logistic(
            x, y,
            p0=config.logistic_fit.p0,
            bounds=(lower, upper),
            maxfev=config.logistic_fit.maxfev,
        )

        yearly = classify_phases(
            yearly, popt, t_offset,
            config.logistic_fit.phase_thresholds,
            config.logistic_fit.phase_labels,
        )

        # Print lifecycle summary (reference style)
        boundaries = compute_phase_boundaries(popt, t_offset)
        logger.info("=== Academic Lifecycle Stages ===")
        logger.info("  Embryonic  (0%%–10%%):  %d – %.1f",
                    int(x.min()), boundaries["10%"]["year"])
        logger.info("  Growth     (10%%–90%%): %.1f – %.1f",
                    boundaries["10%"]["year"], boundaries["90%"]["year"])
        logger.info("  Mature     (90%%–100%%): %.1f –",
                    boundaries["90%"]["year"])

        plot_publication_trend(
            yearly, popt, r_squared, t_offset, config,
            paths.figures_dir / "bibliometric_dynamics_annual_vs_cumulative_with_logistic_fit.png",
            df_cleaned=df, rank_by="influence",
        )
        plot_publication_trend(
            yearly, popt, r_squared, t_offset, config,
            paths.figures_dir / "bibliometric_dynamics_annual_vs_cumulative_with_articles_inset.png",
            df_cleaned=df, rank_by="articles", show_formula=False,
        )

    # ------------------------------------------------------------------
    # 3.2 — Country choropleth
    # ------------------------------------------------------------------
    with Timer("Step 3.2 — Country Choropleth"):
        # Build combined mapping
        mapping = {**COUNTRY_NAME_MAPPING_DEFAULT, **config.country_analysis.country_name_mapping}

        pub_stats = compute_country_stats(df, mapping)

        area_cfg = config.country_analysis
        area_path = paths.external_dir / area_cfg.mangrove_area_file

        if area_path.exists():
            area_df = load_mangrove_area(area_path, area_column=area_cfg.area_column)
        else:
            logger.warning("Mangrove area file not found: %s — plotting publication counts only.", area_path)
            area_df = pd.DataFrame({"Country": pub_stats["Country"], "Area": np.nan})

        gdf = build_choropleth_geodataframe(
            pub_stats, area_df, mapping,
            world_map_source=area_cfg.world_map_source,
            area_unit=area_cfg.area_unit,
        )
        plot_choropleth(gdf, config, paths.figures_dir / "country_map.png", area_unit=area_cfg.area_unit)

    # ------------------------------------------------------------------
    # 3.3 — Journal & country bar charts (Hu et al. 2025 style)
    # ------------------------------------------------------------------
    with Timer("Step 3.3 — Journal & Country Bar Charts"):
        plot_top_journals(df, config, paths.figures_dir / "top_journals.png", top_n=15)
        plot_country_bar_chart(pub_stats, config, paths.figures_dir / "country_publications_bar.png", top_n=20)


def main() -> None:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Step 3: Bibliometric Analysis")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    config = get_config(args.config)
    from src.utils import setup_logging

    setup_logging(config.system.log_level, config.system.log_file)

    try:
        run_bibliometric(config)
        logger.info("Bibliometric analysis complete.")
    except Exception:
        logger.exception("Bibliometric analysis failed")
        raise


if __name__ == "__main__":
    main()
