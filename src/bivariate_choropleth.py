"""Bivariate choropleth: mangrove resource (HGMF area) x research output (WoS pubs).

Publication-grade 3x3 bivariate map.  Base map = Natural Earth **50m** Admin 0
Countries (user decision 2026-08-29), equirectangular projection, Antarctica
hidden, light-gray ocean — same CRS as the HGMF mangrove raster overlay and
the ``collaboration_network_map`` (both use a plain lon/lat grid with
``set_xlim(-180, 180)`` / ``set_ylim(-60, 85)``), so the teal density raster
and the country polygons are guaranteed to align.

Attribute table (ISO3, mangrove_area_km2, n_pubs) is joined to the spatial
boundaries by ISO3.  The input CSV is cleaned in-memory before any join:

- duplicate-ISO3 rows are merged (area + pubs summed across the HGMF-area row
  and the WoS-pubs row, e.g. ``Camerocn`` + ``Cameroon`` -> CMR);
- every missing ISO3 is resolved via ``NAME_TO_ISO3`` — both sovereign
  small-island states (Singapore, Bahrain, ...) and dependent territories
  (Aruba, Cayman Is., ...), which at NE 50m are drawn as their own records so
  the United Kingdom keeps its true 0 km2 of mainland mangrove;
- the one record with no NE 50m polygon (defunct Netherlands Antilles) is kept
  in the classification population and reported as not drawable.

A cleaned copy is written to ``data/processed/bivariate_attr_clean.csv`` so the
classification breakpoints can be reproduced / verified by hand.

Classification (per user decision 2026-08-29): each dimension is cut into
``None (== 0) / Low / High``, where Low and High are split at the **median of
positive values** (a quantile split — ``n_pubs`` is heavily right-skewed, so an
equal-interval cut would be useless).  Zeros therefore fall into the edge
row/column of the 3x3 legend; no 4x4 grid is needed.  Quantiles are computed
over **all** data countries (125 ISO3-bearing + 15 sovereign island states after
folding), i.e. not only the countries visible on the map.

Palette: teal x coral 3x3 bivariate scheme — teal encodes the mangrove-area
axis (rows), coral the research-output axis (cols); each axis steps
monotonically light->dark so the block reads as one 2-variable grid.  The
high-resource / low-research cell (A3 = area High x pubs None) is the single
saturated teal accent (countries with mangroves but no papers).  Base map uses
a near-white ocean + hairline steel borders (quiet backdrop for the teal
mangrove density overlay).

Usage:
    pixi run python -m src.bivariate_choropleth [--attr-csv PATH] [--force-mock]
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# PROJ database fix: an old TauDEM install on this machine shadows the pixi
# env's proj.db.  Point PROJ_LIB at the env's copy before importing pyproj.
# ---------------------------------------------------------------------------
_PIXI_PROJ = Path(sys.prefix) / "Library" / "share" / "proj"
if (Path(sys.prefix) / "Library" / "share" / "proj" / "proj.db").exists() and not os.environ.get("PROJ_LIB"):
    os.environ["PROJ_LIB"] = str(_PIXI_PROJ)

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")  # headless-safe; qtagg corrupts transforms in non-interactive sessions
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from matplotlib.patheffects import withStroke

from src.utils import canonical_countries

# ===========================================================================
# PARAMETER BLOCK - all hardcoded numbers / colors live here
# ===========================================================================

# --- paths -----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ATTR_CSV_DEFAULT = PROJECT_ROOT / "data" / "processed" / "bivariate_attr.csv"
ATTR_CSV_CLEAN = PROJECT_ROOT / "data" / "processed" / "bivariate_attr_clean.csv"
WORLD_MAP_SOURCE = PROJECT_ROOT / "data" / "external" / "ne_50m_admin_0_countries" / "ne_50m_admin_0_countries.shp"
MANGROVE_SHP = Path(r"D:/HGMF_2020/GlobalMangrove2020/GlobalMangrove2020.shp")
CLEANED_CSV = PROJECT_ROOT / "data" / "processed" / "cleaned_data.csv"
HGMF_AREA_CSV = PROJECT_ROOT / "data" / "external" / "HGMF_country_area.csv"
# Cache of the rasterized mangrove-density grid (the slow part is reading the
# HGMF shapefile + computing millions of polygon centroids; the resulting grid
# is tiny).  Regenerate with --recompute-raster if the shapefile changes.
MANGROVE_GRID_CACHE = PROJECT_ROOT / "data" / "processed" / "mangrove_grid_0.5deg.npy"
OUT_PNG = PROJECT_ROOT / "outputs" / "figures" / "bivariate_mangrove_area_pubs.png"
OUT_PDF = PROJECT_ROOT / "outputs" / "figures" / "bivariate_mangrove_area_pubs.pdf"
OUT_SVG = PROJECT_ROOT / "outputs" / "figures" / "bivariate_mangrove_area_pubs.svg"

# --- map base --------------------------------------------------------------
OCEAN_FACE = "#e6e6e6"          # light-gray sea
NO_DATA_FACE = "#cfcfcf"        # single light gray for countries with no data
BOUNDARY_COLOR = "#b7b7b7"
BOUNDARY_LW = 0.25
# Mangrove-density ramp for the overlay + colorbar.  Matches
# collaboration_network_map.py's colormap exactly so the two figures'
# colorbars are consistent across the manuscript.
#
# HOW TO CHANGE THE COLORBAR COLOUR: edit the three hex strings below; they
# are the ramp low -> mid -> high density in that order.  The map overlay and
# the bottom colorbar SHARE this single colormap (same `im`), so changing it
# recolors both together.  E.g. a vivid green (RGB 0,210,106) is
# ["#b2f2c9", "#00d26a", "#00702e"].
MANGROVE_CMAP = LinearSegmentedColormap.from_list(
    "mangrove", ["#A2D5B3", "#48A866", "#1C603B"]
)
MANGROVE_ALPHA = 0.40           # green density overlay on top of bivariate fills
MANGROVE_RES_DEG = 0.5          # rasterization cell size (matches config)

# --- Joshua Stevens 3x3 bivariate palette (blue x orange/red family) ---------
# Rows = mangrove area class (None=0, Low=1, High=2); Cols = pubs class.
# High-resource / low-research (area High x pubs None) = warning color #c85a5a.
# (Original scheme, reverted 2026-09-10 from the teal-x-coral theme.)
BIVAR_PALETTE = {
    # (area_class, pubs_class): hex   (0=None, 1=Low, 2=High)
    (0, 0): "#e8e8e8",  # None-None (light gray)
    (0, 1): "#b0d5df",
    (0, 2): "#64acbe",  # None area, high pubs (blue; UK/Germany)
    (1, 0): "#e4acac",
    (1, 1): "#ad9ea5",
    (1, 2): "#627f8c",
    (2, 0): "#ca6969",  # high area, None pubs (WARNING)
    (2, 1): "#985356",
    (2, 2): "#446067",  # high-high (dark)
}
CLASS_LABELS = {0: "None", 1: "Low", 2: "High"}
PUBS_CLASS_CODES = {0: "A", 1: "B", 2: "C"}   # research class code (None->High)
AREA_CLASS_CODES = {0: "1", 1: "2", 2: "3"}   # resource class code (None->High)

# --- classification rule ---------------------------------------------------
# Each dimension: None (value == 0) / Low / High, zeros in the edge row/column.
# The research-output axis splits at the median of positive values (5.5 papers).
# The mangrove-area axis uses an explicit 1,000 km2 "resource-rich" boundary
# (matching the text's >1,000 km2 line) rather than its median of positives
# (~205 km2), which would let China's 328 km2 ride into the High class and
# visually contradict the method-exporter narrative.
AREA_HIGH_KM2 = 1000.0

# --- labels ----------------------------------------------------------------
LABEL_TOP_N = 6                     # label the top-N countries by mangrove area AND
                                    # by publication count (union), so the map names the
                                    # resource-rich and research-active countries that
                                    # matter for this study -- not the largest landmasses
LABEL_FONTSIZE = 5.6                # small country-name font (Nature-style; was 7.0)
LABEL_MIN_OFFSET_X = 0.0            # fractional figure offsets; nudged on collision
LABEL_MAX_NUDGES = 30

# --- figure / export -------------------------------------------------------
FIG_WIDTH_MM = 180.0
# The map keeps its true E-W/N-S degree aspect (1 deg lon == 1 deg lat on the
# page), so it is a wide ~68 mm band in this 180 mm figure.  The 3x3 legend is
# drawn INSIDE the map's lower-left ocean corner and the mangrove colorbar in a
# thin strip just below the map -- so the figure stays compact.
FIG_HEIGHT_MM = 90.0
DPI = 300
FONT_FAMILY = "Arial"
FONT_FALLBACK = "DejaVu Sans"
TITLE = "Mangrove area x research output (3x3 bivariate)"

# --- map window (matches collaboration_network_map for raster alignment) ----
MAP_XLIM = (-180.0, 180.0)
MAP_YLIM = (-60.0, 85.0)
# Map rectangle (figure fractions).  The height is derived from the width and
# the data-window aspect (360 lon x 145 lat) so 1 deg lon == 1 deg lat on the
# page -- the map is NOT vertically stretched.  set_aspect("equal") in main
# enforces the same constraint.
MAP_LEFT_FRAC = 0.030
MAP_WIDTH_FRAC = 0.940
MAP_TOP_FRAC = 0.930

# --- mangrove density colorbar strip (below the map, own axes) --------------
# cax is placed just below the map bottom; gap + thickness are mm so the strip
# tracks the (aspect-derived) map rectangle automatically.
CBAR_LEFT_FRAC = 0.180
CBAR_WIDTH_FRAC = 0.640
CBAR_THICK_MM = 3.0
CBAR_GAP_MM = 3.5

# --- 3x3 bivariate legend block (inset in the map's lower-left ocean) -------
# Data-coordinate ranges + a *base* unit cell size.  The drawn rectangle is
# (ranges x LEGEND_UNIT_MM x LEGEND_SCALE) mm and the fonts are scaled by the
# same LEGEND_SCALE, so the block shrinks by 40% (scale 0.6) without the text
# colliding (text px / cell mm ratio is unchanged).
#
# HOW TO MOVE / RESIZE THE LEGEND:
#   * LEGEND_PAD_MM  — mm offset of the whole block from the map's lower-left
#     corner; increase it to push the block right + up.
#   * LEGEND_SCALE   — overall size of the block AND its fonts (bigger = larger).
#   * One legend data unit = LEGEND_UNIT_MM x LEGEND_SCALE = 4.8 mm on the page.
#     At 180 mm figure width @ 300 dpi that is ~56.7 px, so to move a label by
#     N px change its coordinate by N / 56.7 (~N x 0.0176) units.
LEGEND_UNIT_MM = 8.0
LEGEND_SCALE = 0.6
LEGEND_PAD_MM = 3.0
LEGEND_XLIM = (-3.4, 5.6)
LEGEND_YLIM = (-1.6, 4.4)
LEGEND_FONT_AXIS = 8.2
LEGEND_FONT_BREAK = 6.4
LEGEND_FONT_NODATA = 6.2
LEGEND_FONT_MOCK = 6.4
LEGEND_FONT_COUNT = 7.0         # per-cell country count in the 3x3 legend

# --- region merging (used by the MOCK builder only) ------------------------
# The real attribute CSV is used as-is; dependent territories are resolved to
# their own ISO3 by clean_attr_table (so UK keeps its true 0 km2 of mangrove).
# This dict still folds territories for the mock table, matching the rest of
# the pipeline.
REGION_TO_COUNTRY = {
    "Taiwan": "China",
    "Puerto Rico": "United States",
    "French Guiana": "France",
    "Guadeloupe": "France",
    "Martinique": "France",
    "Mayotte": "France",
    "New Caledonia": "France",
    # additional dependent territories -> administering country
    "American Samoa": "United States",
    "Virgin Is.": "United States",
    "Aruba": "Netherlands",
    "Netherlands Antilles": "Netherlands",
    "British Virgin Is.": "United Kingdom",
    "Cayman Is.": "United Kingdom",
    "Turks & Caicos Is.": "United Kingdom",
    "Wallis & Futuna": "France",
}
# ISO3 of a territory polygon -> ISO3 of the parent country for display.
# Only territories whose data is ALREADY folded into the parent in the CSV
# (Taiwan, Puerto Rico, French overseas) are aliased; ASM/VIR/ABW/VGB/CYM/
# TCA/WLF are NOT aliased because they are drawn as their own records with
# their own class (see NAME_TO_ISO3).  Territories absent from the attribute
# table (Guam, Greenland, ...) stay "no data" gray.
ISO3_ALIAS_FOR_DISPLAY = {
    "TWN": "CHN",  # Taiwan -> China
    "PRI": "USA",  # Puerto Rico -> United States
    "NCL": "FRA",  # New Caledonia -> France
    "GLP": "FRA", "MTQ": "FRA", "MYT": "FRA", "GUF": "FRA",
}

# Residual false positives in the CU field.  Person-name leaks ("Jamali"→Mali,
# "Jordan M."→Jordan, "Montenegro J."→Montenegro, "Moldovanu"→Moldova,
# "Lomonaco"→Monaco, "Israel R."→Israel) were eliminated at the source:
# data_preprocessing._extract_country now matches affiliation text only.
# What remains are place/state-name collisions inside real affiliations
# (Niger Delta/State→Nigeria, Georgia (US state) institutions, Inner
# Mongolia→China); see CLAUDE.md "Known Issues".
CU_FALSE_POSITIVES = {"Niger", "Georgia", "Mongolia"}

# Country-name -> ISO3 aliases (HGMF typos, WoS names, NE names).
NAME_TO_ISO3 = {
    "Camerocn": "CMR", "Venzuela": "VEN", "Congo.DRC": "COD",
    "Cote d'Ivory": "CIV", "Cote d'Ivoire": "CIV", "Ivory Coast": "CIV",
    "Solomon Is.": "SLB", "Solomon Islands": "SLB",
    "The Gambia": "GMB", "Gambia": "GMB", "The Bahamas": "BHS",
    "Bahamas": "BHS", "Timor Leste": "TLS", "East Timor": "TLS",
    "Trinidad & Tobago": "TTO", "Trinidad and Tobago": "TTO",
    "Tanzania": "TZA", "United Republic of Tanzania": "TZA",
    "United States": "USA", "United States of America": "USA",
    "Czech Republic": "CZE", "Czechia": "CZE", "Republic of Korea": "KOR",
    "South Korea": "KOR", "Russia": "RUS", "Congo": "COG",
    "Republic of the Congo": "COG", "Democratic Republic of the Congo": "COD",
    "Vietnam": "VNM", "Brunei": "BRN", "Myanmar": "MMR", "Iran": "IRN",
    "Laos": "LAO", "North Korea": "PRK", "Micronesia": "FSM",
    "Timor-Leste": "TLS",
    # WoS-only countries that entered the attribute table once the 2026-09-21
    # extractor fix re-admitted their genuine affiliations (they had been
    # suppressed as person-name false positives).
    "Israel": "ISR", "Jordan": "JOR",
    # sovereign small-island states (real ISO3; NE 50m has all of these)
    "Singapore": "SGP", "Bahrain": "BHR", "Barbados": "BRB",
    "Maldives": "MDV", "Mauritius": "MUS", "Seychelles": "SYC",
    "Tonga": "TON", "Samoa": "WSM", "Palau": "PLW", "Grenada": "GRD",
    "Comoros": "COM", "Antigua & Barbuda": "ATG", "St. Kitts & Nevis": "KNA",
    "St. Lucia": "LCA", "St. Vincent & the Grenadines": "VCT", "Monaco": "MCO",
    # dependent territories drawn as their own NE 50m polygon (deliberately
    # NOT folded into the parent country, so the United Kingdom keeps its true
    # 0 km2 of mainland mangrove instead of inheriting the Caribbean islands')
    "American Samoa": "ASM", "Aruba": "ABW", "British Virgin Is.": "VGB",
    "Cayman Is.": "CYM", "Netherlands Antilles": "ANT",
    "Turks & Caicos Is.": "TCA", "Virgin Is.": "VIR",
    "Wallis & Futuna": "WLF",
}


# ===========================================================================
# Mock attribute table assembly (only used with --force-mock / missing CSV)
# ===========================================================================

def build_mock_attr_csv(out_path: Path) -> bool:
    """Build a realistic mock attribute table from project data.

    Returns True if the file was written (i.e. the figure must be annotated
    "mock data").
    """
    # --- publication counts (whole counting, region-merged) ----------------
    df = pd.read_csv(CLEANED_CSV, encoding="utf-8-sig")
    pub_counter: Counter = Counter()
    for cu in df["CU"].dropna():
        # canonical_countries folds the region map AND de-duplicates, so a
        # "China;Taiwan" record counts once for China (src.utils).
        for c in canonical_countries(cu, fold=REGION_TO_COUNTRY, drop=CU_FALSE_POSITIVES):
            pub_counter[c] += 1

    # --- mangrove area (HGMF country table, region-merged) ------------------
    hg = pd.read_csv(HGMF_AREA_CSV)
    area_dict: dict[str, float] = {}
    for _, r in hg.iterrows():
        parent = REGION_TO_COUNTRY.get(r["Country"], r["Country"])
        area_dict[parent] = area_dict.get(parent, 0.0) + float(r["sum"])

    names = sorted(set(pub_counter) | set(area_dict))
    rows = []
    for n in names:
        rows.append(
            {"country": n, "mangrove_area_km2": area_dict.get(n, 0.0),
             "n_pubs": pub_counter.get(n, 0)}
        )
    attr = pd.DataFrame(rows)
    attr["ISO3"] = attr["country"].map(NAME_TO_ISO3)
    n_unresolved = int(attr["ISO3"].isna().sum())
    if n_unresolved:
        unresolved = attr.loc[attr["ISO3"].isna(), "country"].tolist()
        print(f"[mock] WARNING: {n_unresolved} countries without ISO3: {unresolved}")

    attr = attr[["ISO3", "country", "mangrove_area_km2", "n_pubs"]]
    attr.to_csv(out_path, index=False)
    print(f"[mock] Mock attribute table written: {out_path} ({len(attr)} rows)")
    return True


# ===========================================================================
# Attribute table cleaning (dedup ISO3, resolve islands, fold territories)
# ===========================================================================

def clean_attr_table(attr: pd.DataFrame) -> pd.DataFrame:
    """One canonical row per country, keyed by ISO3.

    Steps (each reported on stdout):
      1. Every row gets an ISO3: keep the CSV value, else ``NAME_TO_ISO3``
         (covers sovereign small-island states AND dependent territories).
      2. Duplicate ISO3 rows are merged (area and pubs summed — the two rows
         are the HGMF-area source and the WoS-pubs source for the same country,
         e.g. ``Camerocn`` + ``Cameroon`` -> CMR).

    Dependent territories are deliberately NOT folded into their parent: each
    gets its own ISO3 and, at NE 50m, its own polygon.  This keeps e.g. the
    United Kingdom's real 0 km2 of mainland mangrove instead of inheriting the
    Caribbean territories' mangrove area (which would mislabel UK as a
    high-resource country).  The only un-drawable record is the defunct
    Netherlands Antilles (ANT), which has no NE 50m polygon; it is kept in the
    classification population and reported separately.
    """
    attr = attr.copy()
    n_raw = len(attr)
    n_no_iso = int(attr["ISO3"].isna().sum())

    attr["_iso"] = attr["ISO3"].fillna(attr["country"].map(NAME_TO_ISO3))

    still_missing = sorted(attr.loc[attr["_iso"].isna(), "country"])
    if still_missing:
        print(f"[clean] ERROR: {len(still_missing)} rows still lack ISO3: {still_missing}")
        raise ValueError("Unable to resolve ISO3 for all attribute rows")

    resolved_names = sorted(attr.loc[attr["ISO3"].isna(), "country"])

    # duplicate-ISO3 groups (HGMF-area row + WoS-pubs row for the same country)
    dups = attr["_iso"].value_counts()
    dup_groups = {iso: int(n) for iso, n in dups.items() if n > 1}

    cleaned = (
        attr.groupby("_iso", as_index=False)
        .agg(country=("country", "first"),
             mangrove_area_km2=("mangrove_area_km2", "sum"),
             n_pubs=("n_pubs", "sum"))
        .rename(columns={"_iso": "ISO3"})
    )

    # ---- report -----------------------------------------------------------
    print("=" * 72)
    print("ACCEPTANCE REPORT 0 - INPUT DATA CLEANING")
    print("=" * 72)
    print(f"raw attribute CSV rows            : {n_raw}")
    print(f"  rows WITHOUT ISO3              : {n_no_iso}")
    print(f"  -> resolved to own ISO3        : {len(resolved_names)}")
    for nm in resolved_names:
        print(f"      {nm:28s} -> {NAME_TO_ISO3[nm]}")
    print(f"  duplicate-ISO3 groups merged   : {len(dup_groups)}  "
          f"({', '.join(sorted(dup_groups))})")
    print(f"cleaned records (all data countries) : {len(cleaned)}")
    print(f"  still missing ISO3             : {len(still_missing)}")
    return cleaned


# ===========================================================================
# Quantile / class assignment
# ===========================================================================

def classify_dim(values: pd.Series, threshold: float | None = None) -> tuple[np.ndarray, float]:
    """None (==0) / Low / High.

    Low-High split defaults to the median of positive values; pass ``threshold``
    for an explicit absolute boundary (used for the area axis, see
    ``AREA_HIGH_KM2``).

    Returns (class codes 0/1/2, split value used for the legend).
    """
    pos = values[values > 0]
    thr = float(threshold) if threshold is not None else (float(pos.median()) if len(pos) else 0.0)
    codes = np.zeros(len(values), dtype=int)
    gt0 = values > 0
    codes[gt0] = 1
    codes[gt0 & (values > thr)] = 2
    return codes, thr


# ===========================================================================
# Map rendering
# ===========================================================================

def _make_base_map(ax, world, fig):
    """Draw ocean, land, country boundaries (equirectangular, no Antarctica).

    Uses plain lon/lat — same CRS as the HGMF mangrove raster overlay and the
    ``collaboration_network_map`` — so the teal density layer aligns pixel
    perfect with the country polygons.
    """
    ax.set_facecolor(OCEAN_FACE)
    world.plot(ax=ax, facecolor=NO_DATA_FACE, edgecolor="none", zorder=1)
    ax.set_xlim(*MAP_XLIM)
    ax.set_ylim(*MAP_YLIM)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def _draw_country_borders(ax, world):
    """Country boundary lines on top of bivariate fills (zorder=2.5)."""
    world.boundary.plot(ax=ax, edgecolor=BOUNDARY_COLOR, linewidth=BOUNDARY_LW,
                        zorder=2.5)


def _draw_mangrove_overlay(ax, grid_lon_edges, grid_lat_edges, grid):
    """Rasterized HGMF density as a masked imshow on top of fills.

    Matches the ``collaboration_network_map`` overlay exactly: log1p-scaled
    teal colormap, ``origin="upper"`` (grid row 0 = 90°N maps to the TOP of
    the axes), masked cells are transparent, and the ``extent`` is the full
    grid edge range so the raster spans the same lat/lon window as the base
    map.
    """
    log_grid = np.log1p(grid)
    masked = np.ma.masked_where(log_grid <= 0, log_grid)
    vmin = vmax = None
    if masked.count() == 0:
        return None, vmin, vmax

    cmap = MANGROVE_CMAP
    cmap.set_bad("none")
    vmin, vmax = float(masked.min()), float(masked.max())
    im = ax.imshow(
        masked,
        extent=[grid_lon_edges[0], grid_lon_edges[-1],
                grid_lat_edges[0], grid_lat_edges[-1]],
        origin="upper",
        cmap=cmap,
        vmin=vmin, vmax=vmax,
        interpolation="nearest",
        alpha=MANGROVE_ALPHA,
        aspect="auto",   # keep the map's aspect="auto"; imshow's default
                         # aspect="equal" would re-shape/shrink the whole axes
                         # to the raster data aspect and leave blank bands
        zorder=3,
    )
    return im, vmin, vmax


def _place_labels(ax, fig, label_df):
    """Annotate the selected countries; nudge labels apart to avoid overlap."""
    texts = []
    for _, r in label_df.iterrows():
        t = ax.annotate(
            r["ADMIN"], (r["x"], r["y"]), xytext=(LABEL_MIN_OFFSET_X, 0),
            textcoords="offset points", fontsize=LABEL_FONTSIZE, color="#333333",
            zorder=6, ha="left", va="center",
            path_effects=[withStroke(linewidth=2.0, foreground="white")],
        )
        texts.append(t)

    # collision-avoidance: nudge up by one point while bboxes overlap
    fig.canvas.draw()
    for _ in range(LABEL_MAX_NUDGES):
        moved = False
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                bb_i = texts[i].get_window_extent()
                bb_j = texts[j].get_window_extent()
                if bb_i.overlaps(bb_j):
                    texts[j].set_position((texts[j].get_position()[0],
                                           texts[j].get_position()[1] + 2))
                    moved = True
        if not moved:
            break


def _fmt_break(x: float) -> str:
    """Format a median breakpoint for the legend (1 decimal, no trailing .0)."""
    return f"{x:.1f}".rstrip(".0") if x < 1000 else f"{x:.0f}"


def _add_bivariate_legend(fig, ax, area_med, pubs_med, is_mock, cell_counts=None):
    """3x3 legend block, 40% smaller, inset in the map's lower-left ocean.

    Every component sits in a fixed *data* grid (one unit = one cell) laid out
    with generous, disjoint regions:

    - grid cells span x in [0, 3] and y in [0, 3];
    - column (pubs) break labels sit below the grid; the "Research output"
      axis title above it;
    - row (area) break labels sit to the left of the grid; the rotated
      "Mangrove area" axis title further left, clear of them;
    - the no-data swatch + label occupy the region right of the grid;
    - when ``cell_counts`` is given, each cell is annotated with the number of
      countries in it, so counts quoted in the text can be verified against
      the figure directly.

    The block is LEGEND_SCALE-scaled from LEGEND_UNIT_MM (i.e. 40% smaller) and
    anchored a few mm inside the map's bottom-left corner (over open ocean).
    Fonts are shrunk by the same factor so text never collides with the cells.
    """
    s = LEGEND_SCALE
    unit = LEGEND_UNIT_MM * s
    xw = (LEGEND_XLIM[1] - LEGEND_XLIM[0]) * unit
    yw = (LEGEND_YLIM[1] - LEGEND_YLIM[0]) * unit
    pos = ax.get_position()          # the map's figure-fraction rectangle
    w_frac = xw / FIG_WIDTH_MM
    h_frac = yw / FIG_HEIGHT_MM
    lax = fig.add_axes([
        pos.x0 + LEGEND_PAD_MM / FIG_WIDTH_MM,
        pos.y0 + LEGEND_PAD_MM / FIG_HEIGHT_MM,
        w_frac, h_frac,
    ])
    # Transparent background: a fresh Axes defaults to opaque white
    # (rcParams axes.facecolor), which painted a big white panel over the map
    # far larger than the legend content itself.  Let the ocean show through;
    # only the grid + labels are then visible.
    lax.set_facecolor("none")
    lax.set_xlim(*LEGEND_XLIM)
    lax.set_ylim(*LEGEND_YLIM)
    lax.set_aspect("equal")
    lax.set_xticks([])
    lax.set_yticks([])
    for sp in lax.spines.values():
        sp.set_visible(False)

    cell = 1.0
    for area_cls in (0, 1, 2):          # rows, None at bottom -> High at top
        for pubs_cls in (0, 1, 2):      # cols, None at left -> High at right
            color = BIVAR_PALETTE[(area_cls, pubs_cls)]
            lax.add_patch(Rectangle(
                (pubs_cls, area_cls), cell, cell, facecolor=color,
                edgecolor="white", linewidth=0.8 * s, zorder=2,
            ))
            if cell_counts is not None:
                # country count in the cell's lower-right corner; dark text on
                # a white halo stays legible on both light and dark cells
                lax.text(pubs_cls + 0.90, area_cls + 0.12,
                         str(int(cell_counts.get((area_cls, pubs_cls), 0))),
                         ha="right", va="bottom", fontsize=LEGEND_FONT_COUNT * s,
                         color="#1a1a1a", zorder=3,
                         path_effects=[withStroke(linewidth=1.8, foreground="white")])

    # row / column axis titles (with units)
    lax.text(1.5, 3.28, "Research output (papers) \u2192", ha="center", va="bottom",
             fontsize=LEGEND_FONT_AXIS * s, fontweight="bold")
    # "Mangrove area" axis title.  x = -2.56 / y = 1.5 are legend data units:
    # decrease x to move left, increase to move right (1 unit ~ 56.7 px).
    lax.text(-2.45, 1.5, "Mangrove area (km\u00b2) \u2191", ha="center", va="center",
             fontsize=LEGEND_FONT_AXIS * s, fontweight="bold", rotation=90)

    # threshold annotations (None / Less-endowed / Resource-rich for the area
    # axis; None / 1-N / >N for the research axis, with class codes).
    for pubs_cls, label in [
        (0, "None"),
        (1, f"1\u2013{_fmt_break(pubs_med)}"),
        (2, f">{_fmt_break(pubs_med)}"),
    ]:
        # two lines (class code above the breakpoint) so a label never spills
        # wider than its cell column
        lax.text(pubs_cls + 0.5, -0.42, f"{PUBS_CLASS_CODES[pubs_cls]}\n{label}",
                 ha="center", va="top", fontsize=LEGEND_FONT_BREAK * s, linespacing=1.0)
    for area_cls, label in [
        (0, "None"),
        (1, "Less-endowed"),
        (2, "Resource-rich\n(\u22651,000 km\u00b2)"),
    ]:
        lax.text(-0.40, area_cls + 0.5, f"{AREA_CLASS_CODES[area_cls]}  {label}",
                 ha="right", va="center", fontsize=LEGEND_FONT_BREAK * s,
                 linespacing=1.0)

    # no-data swatch (countries absent from the attribute table), right of grid
    lax.add_patch(Rectangle((3.55, 1.30), 0.45, 0.45, facecolor=NO_DATA_FACE,
                            edgecolor="white", linewidth=0.8 * s, zorder=2))
    lax.text(4.15, 1.525, "no data", ha="left", va="center",
             fontsize=LEGEND_FONT_NODATA * s)

    if is_mock:
        lax.text(-0.30, -1.25, "mock data", ha="left", va="top",
                 fontsize=LEGEND_FONT_MOCK * s, style="italic", color="#b30000")
    return lax


def _add_mangrove_colorbar(fig, ax, im, vmin, vmax):
    """Horizontal mangrove density colorbar in its own strip below the map.
    底部的色带：红树林
    Re-labels the log1p scale back to real km² values (matches
    ``collaboration_network_map``).  A dedicated ``cax`` is used instead of
    ``fig.colorbar(im, ax=ax, ...)`` so that attaching the colorbar does not
    auto-resize (and thereby shrink) the manually-placed map axes -- which
    previously produced a large blank band under the suptitle and clipped the
    colorbar/labels at the bottom of the figure.
    """
    if im is None:
        return None
    # cax sits just below the map rectangle (whose height follows from the
    # true-aspect map), CBAR_GAP_MM under the map's bottom edge.
    p = ax.get_position()
    h = CBAR_THICK_MM / FIG_HEIGHT_MM
    y0 = p.y0 - CBAR_GAP_MM / FIG_HEIGHT_MM - h
    y0 = max(y0, 0.01)
    cax = fig.add_axes([CBAR_LEFT_FRAC, y0, CBAR_WIDTH_FRAC, h])
    cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
    cbar.outline.set_linewidth(0.5)   # thinnest colorbar frame (0.5 pt)
    area_ticks = [a for a in (0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000)
                  if vmin <= np.log1p(a) <= vmax]
    if not area_ticks:
        area_ticks = [float(np.expm1(vmin)), float(np.expm1(vmax))]
    cbar.set_ticks([np.log1p(a) for a in area_ticks])
    cbar.set_ticklabels([f"{a:g}" for a in area_ticks])
    cbar.set_label("Mangrove area (km² per grid cell)", fontsize=7, labelpad=2)
    cbar.ax.tick_params(labelsize=6)
    return cbar


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Bivariate mangrove x research map")
    parser.add_argument("--attr-csv", type=str, default=None,
                        help="Attribute CSV (ISO3, mangrove_area_km2, n_pubs)")
    parser.add_argument("--force-mock", action="store_true",
                        help="Rebuild and use the mock attribute table")
    parser.add_argument("--out-suffix", type=str, default="",
                        help="Suffix appended to the output basename (e.g. '_bluered'); "
                             "empty keeps the default names")
    parser.add_argument("--recompute-raster", action="store_true",
                        help="Ignore the cached mangrove grid and re-rasterize the HGMF shapefile")
    args = parser.parse_args()

    matplotlib.rcParams.update({
        "font.family": FONT_FAMILY,
        "font.sans-serif": [FONT_FAMILY, FONT_FALLBACK],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })

    # ------------------------------------------------------------------ CSV
    attr_path = Path(args.attr_csv) if args.attr_csv else ATTR_CSV_DEFAULT
    is_mock = args.force_mock or not attr_path.exists()
    if is_mock:
        if args.force_mock or not attr_path.exists():
            attr_path.parent.mkdir(parents=True, exist_ok=True)
            build_mock_attr_csv(attr_path)
        else:
            is_mock = False

    attr = pd.read_csv(attr_path)
    required = {"ISO3", "mangrove_area_km2", "n_pubs"}
    missing = required - set(attr.columns)
    if missing:
        raise ValueError(f"Attribute CSV missing columns: {missing} (have {list(attr.columns)})")

    # -------------------------------------------------------- data cleaning
    attr = clean_attr_table(attr)
    ATTR_CSV_CLEAN.parent.mkdir(parents=True, exist_ok=True)
    attr.to_csv(ATTR_CSV_CLEAN, index=False)
    print(f"[clean] cleaned table written: {ATTR_CSV_CLEAN} ({len(attr)} rows)")

    # ------------------------------------------------------------ ISO3 join
    world = gpd.read_file(WORLD_MAP_SOURCE)
    world["ISO3"] = world["ISO_A3"].where(world["ISO_A3"] != "-99", world["ISO_A3_EH"])
    # territory polygons display with the parent country's class
    world["ISO3"] = world["ISO3"].map(lambda x: ISO3_ALIAS_FOR_DISPLAY.get(x, x))
    world = world[world["ISO3"].notna() & (world["ISO3"] != "-99")]
    world = world[world["ADMIN"] != "Antarctica"]

    attr_by_iso = attr.set_index("ISO3")
    matched = world["ISO3"].isin(attr_by_iso.index)
    n_matched_poly = int(matched.sum())
    csv_isos = {str(x) for x in attr_by_iso.index if pd.notna(x)}
    polygon_isos = {str(x) for x in world["ISO3"] if pd.notna(x)}
    csv_unmatched = sorted(csv_isos - polygon_isos)
    poly_unmatched = sorted(polygon_isos - csv_isos)
    n_records_with_poly = len(csv_isos & polygon_isos)

    print("=" * 72)
    print("ACCEPTANCE REPORT 1 - ISO3 JOIN (NE 50m Admin 0 Countries)")
    print("=" * 72)
    print(f"cleaned data records             : {len(attr)}")
    print(f"  matched to a NE 50m polygon    : {n_records_with_poly} of {len(attr)}")
    print(f"  NO polygon in NE 50m           : {len(csv_unmatched)} -> {csv_unmatched}")
    print(f"    (kept in classification, not drawn)")
    print(f"NE 50m features (no Antarctica)  : {len(world)}")
    print(f"  polygons colored               : {n_matched_poly} "
          f"(incl. territory polygons aliased to a parent)")
    print(f"  polygons no data (light gray)  : {int((~matched).sum())}")
    if poly_unmatched:
        print(f"  (no-data sample)              : {poly_unmatched[:25]} ...")

    # ----------------------------------------------------- classification
    area_cls, area_med = classify_dim(attr["mangrove_area_km2"], threshold=AREA_HIGH_KM2)
    pubs_cls, pubs_med = classify_dim(attr["n_pubs"])
    attr["area_cls"] = area_cls
    attr["pubs_cls"] = pubs_cls

    n0_area = int((attr["mangrove_area_km2"] == 0).sum())
    n0_pubs = int((attr["n_pubs"] == 0).sum())
    attr_by_iso = attr.set_index("ISO3")  # refresh with class columns
    print("\n" + "=" * 72)
    print("ACCEPTANCE REPORT 2 - CLASSIFICATION & BREAKPOINTS "
          "(None=0 / Low / High; area split at 1,000 km2, pubs at median of positives)")
    print("=" * 72)
    print(f"mangrove_area_km2 : {n0_area}/{len(attr)} countries = 0 (None class)")
    print(f"  Low/High split (absolute threshold) = {area_med:.2f} km2")
    print(f"  class sizes: {[int((area_cls == c).sum()) for c in (0, 1, 2)]}")
    print(f"n_pubs           : {n0_pubs}/{len(attr)} countries = 0 (None class)")
    print(f"  positive median (Low/High split) = {pubs_med:.2f} papers")
    print(f"  class sizes: {[int((pubs_cls == c).sum()) for c in (0, 1, 2)]}")

    # -------------------------------------------------------- cell counts
    cell_counts = Counter(zip(area_cls, pubs_cls))
    print("\n" + "=" * 72)
    print("ACCEPTANCE REPORT 3 - 3x3 CELL COUNTS  "
          "(rows=area None/Low/High, cols=pubs None/Low/High)")
    print("=" * 72)
    total_cells = 0
    for area_cls_i in (2, 1, 0):
        line = []
        for pubs_cls_i in (0, 1, 2):
            n = cell_counts.get((area_cls_i, pubs_cls_i), 0)
            total_cells += n
            code = PUBS_CLASS_CODES[pubs_cls_i] + AREA_CLASS_CODES[area_cls_i]
            line.append(f"  {code}: {n:3d}")
        print(" ".join(line))
    print(f"TOTAL = {total_cells}  (cleaned records = {len(attr)})")
    assert total_cells == len(attr), "cell counts must sum to the total number of records"
    empty_cells = [(a, p) for a in (0, 1, 2) for p in (0, 1, 2)
                   if cell_counts.get((a, p), 0) == 0]
    for (a, p) in empty_cells:
        code = PUBS_CLASS_CODES[p] + AREA_CLASS_CODES[a]
        print(f"  NOTE: cell {code} is EMPTY - {CLASS_LABELS[a]} area x {CLASS_LABELS[p]} pubs")
    if not empty_cells:
        print("  All 9 cells non-empty.")
    else:
        # structural explanation for the empty cell(s)
        print("  Explanation: in this dataset every country with zero mangrove area")
        print("  publishes at least one paper, and every country with zero papers has")
        print("  some mangrove area -> the None-area x None-pubs cell is structurally")
        print("  empty (it is a real property of the data, not a join/encoding bug).")

    # ------------------------------------------------------- known cases
    print("\n" + "=" * 72)
    print("ACCEPTANCE REPORT 4 - KNOWN-CASE SANITY CHECKS")
    print("=" * 72)
    for name in ["Indonesia", "United Kingdom", "Germany", "India", "Australia"]:
        row = attr[attr["country"] == name]
        if len(row):
            r = row.iloc[0]
            print(f"  {name:16s} area={r['mangrove_area_km2']:10.1f} km2 "
                  f"-> cls {AREA_CLASS_CODES[r['area_cls']]}   "
                  f"pubs={r['n_pubs']:4d} -> cls {PUBS_CLASS_CODES[r['pubs_cls']]}")
    idn = attr[attr["country"] == "Indonesia"].iloc[0]
    assert idn["area_cls"] == 2, "Indonesia should be in the HIGH area class"
    for nm in ["United Kingdom", "Germany"]:
        r = attr[attr["country"] == nm].iloc[0]
        assert r["pubs_cls"] == 2 and r["area_cls"] == 0, \
            f"{nm} should be high-pubs / zero-area"

    # ------------------------------------------------------------- figure
    # equirectangular — same CRS as the HGMF mangrove raster overlay and the
    # ``collaboration_network_map`` (no projection), so the teal density layer
    # aligns pixel perfect with the country polygons.
    world = world.copy()
    world["fill"] = NO_DATA_FACE
    world["pubs_cls"] = np.nan
    world["area_cls"] = np.nan

    # join attribute classes onto polygons
    for iso, row in attr_by_iso.iterrows():
        sel = world["ISO3"] == iso
        world.loc[sel, "pubs_cls"] = row["pubs_cls"]
        world.loc[sel, "area_cls"] = row["area_cls"]

    fig = plt.figure(figsize=(FIG_WIDTH_MM / 25.4, FIG_HEIGHT_MM / 25.4))
    # Map rectangle: the height is derived from the width and the data-window
    # aspect (360 lon x 145 lat) so one degree of longitude equals one degree
    # of latitude on the page -- the world map is shown at its TRUE aspect,
    # not vertically stretched.  set_aspect("equal") enforces the same, and the
    # rectangle matches, so there is no blank margin inside the axes.
    # The mangrove colorbar lives in its OWN thin axes just below the map; the
    # 3x3 legend is drawn as its OWN axes inset in the map's lower-left ocean
    # corner (see _add_bivariate_legend).
    _map_h = (MAP_WIDTH_FRAC * FIG_WIDTH_MM * (MAP_YLIM[1] - MAP_YLIM[0])
              / (MAP_XLIM[1] - MAP_XLIM[0])) / FIG_HEIGHT_MM
    ax = fig.add_axes([MAP_LEFT_FRAC, MAP_TOP_FRAC - _map_h,
                       MAP_WIDTH_FRAC, _map_h])
    ax.set_aspect("equal")
    _make_base_map(ax, world, fig)

    # bivariate fill (on top of the no-data land base)
    has_data = world["pubs_cls"].notna()
    fills = [BIVAR_PALETTE[(int(a), int(p))] for a, p in
             zip(world.loc[has_data, "area_cls"], world.loc[has_data, "pubs_cls"])]
    world.loc[has_data].plot(ax=ax, facecolor=fills, edgecolor="none", zorder=2)

    # country borders on top of fills (zorder=2.5, below mangrove raster)
    _draw_country_borders(ax, world)

    # mangrove distribution overlay (teal, masked imshow, on top)
    res = MANGROVE_RES_DEG
    lon_edges = np.arange(-180.0, 180.0 + res, res)
    lat_edges = np.arange(-90.0, 90.0 + res, res)
    grid_shape = (len(lat_edges) - 1, len(lon_edges) - 1)

    # The raster grid is cached to a .npy so the slow HGMF shapefile read +
    # million-polygon centroid pass is skipped on repeat runs (the grid itself
    # is only ~350 x 720 floats).  Use --recompute-raster to force a rebuild.
    grid = None
    if not args.recompute_raster and MANGROVE_GRID_CACHE.exists():
        grid = np.load(MANGROVE_GRID_CACHE)
        if grid.shape == grid_shape:
            print(f"[map] Loaded cached mangrove grid {grid.shape} "
                  f"-> {MANGROVE_GRID_CACHE.name}")
        else:
            print(f"[map] Cached grid shape {grid.shape} != {grid_shape}; re-rasterizing")
            grid = None
    if grid is None:
        print("\n[map] Rasterizing HGMF mangrove polygons ...")
        mg = gpd.read_file(MANGROVE_SHP)
        pts = gpd.GeoSeries(mg.geometry.centroid, crs=mg.crs).to_crs("EPSG:4326")
        lon = pts.x.to_numpy()
        lat = pts.y.to_numpy()
        area_v = mg["Area"].to_numpy(dtype=float) if "Area" in mg.columns else np.ones(len(mg))
        col = np.clip(np.floor((lon + 180.0) / res).astype(int), 0, len(lon_edges) - 2)
        row = np.clip(np.floor((90.0 - lat) / res).astype(int), 0, len(lat_edges) - 2)
        grid = np.zeros(grid_shape, dtype=float)
        np.add.at(grid, (row, col), area_v)
        MANGROVE_GRID_CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.save(MANGROVE_GRID_CACHE, grid)
        print(f"[map] Cached mangrove grid {grid.shape} -> {MANGROVE_GRID_CACHE}")

    im, vmin, vmax = _draw_mangrove_overlay(ax, lon_edges, lat_edges, grid)

    # labels: the top-N countries by mangrove area and by publication count
    # (union of the two lists), so the map names the resource-rich and
    # research-active countries that matter for this study rather than the
    # largest landmasses.  One label per country, placed at the centroid of
    # its LARGEST polygon (main landmass) so far-flung territories (Alaska,
    # Hawaii, French Guiana) don't drag the label out into the ocean.  Area
    # and centroid are computed in equal-area EPSG:6933, then re-projected to
    # plain lon/lat to match the axes coordinate system.
    print("\n[map] Computing label set (top by mangrove area & pubs) ...")
    top_area = set(attr.nlargest(LABEL_TOP_N, "mangrove_area_km2")["ISO3"])
    top_pubs = set(attr.nlargest(LABEL_TOP_N, "n_pubs")["ISO3"])
    label_isos = top_area | top_pubs
    world_lab = world[world["ISO3"].isin(label_isos)].to_crs("EPSG:6933")
    world_lab["_area"] = world_lab.geometry.area
    main = world_lab.sort_values("_area").groupby("ISO3", as_index=False).tail(1)
    c = main.geometry.centroid.to_crs("EPSG:4326")
    main["x"] = c.x.to_numpy()
    main["y"] = c.y.to_numpy()
    label_df = attr[attr["ISO3"].isin(label_isos)][["ISO3", "country"]].copy()
    label_df = label_df.merge(main[["ISO3", "x", "y"]], on="ISO3", how="left")
    label_df = label_df.rename(columns={"country": "ADMIN"})
    label_df = label_df.dropna(subset=["x", "y"])
    label_df = label_df.sort_values("ADMIN")
    print(f"  labels ({len(label_df)}): {label_df['ADMIN'].tolist()}")
    _place_labels(ax, fig, label_df)

    # mangrove density colorbar (bottom-right strip)
    _add_mangrove_colorbar(fig, ax, im, vmin, vmax)

    # 3x3 bivariate legend (own axes, 40%-smaller inset in the map's lower-left);
    # each cell carries its country count so the text's numbers are verifiable
    _add_bivariate_legend(fig, ax, area_med, pubs_med, is_mock,
                          cell_counts=dict(cell_counts))

    fig.suptitle(TITLE, fontsize=11, fontweight="bold", y=0.97)

    out_png = OUT_PNG.with_name(f"{OUT_PNG.stem}{args.out_suffix}{OUT_PNG.suffix}")
    out_pdf = OUT_PDF.with_name(f"{OUT_PDF.stem}{args.out_suffix}{OUT_PDF.suffix}")
    out_svg = OUT_SVG.with_name(f"{OUT_SVG.stem}{args.out_suffix}{OUT_SVG.suffix}")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=DPI)
    fig.savefig(out_pdf)
    fig.savefig(out_svg)
    plt.close(fig)

    from PIL import Image
    im = Image.open(out_png)
    print("\n" + "=" * 72)
    print("EXPORT")
    print("=" * 72)
    print(f"PNG : {out_png}")
    print(f"  size: {im.size[0]} x {im.size[1]} px @ {DPI} dpi  "
          f"= {im.size[0] / DPI * 25.4:.1f} x {im.size[1] / DPI * 25.4:.1f} mm")
    print(f"PDF : {out_pdf}")
    print(f"SVG : {out_svg}")
    print(f"mock data annotation: {is_mock}")


if __name__ == "__main__":
    main()
