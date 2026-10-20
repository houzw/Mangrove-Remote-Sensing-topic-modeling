"""International collaboration network overlaid on the HGMF 2020 mangrove map.

Builds a country-level co-authorship network from the ``CU`` field of the
cleaned WoS data, then renders it on top of the global mangrove distribution
shapefile (HGMF_2020, Jia et al. 2023).

- Nodes = countries, anchored at their Natural Earth centroid, sized by
  publication count and coloured by continent.
- Edges = co-authored papers (Bézier arcs bowing toward the map centre,
  width/alpha scaled by weight).

Usage:
    python -m src.collaboration_network_map              # uses config.yaml
    python -m src.collaboration_network_map --config path/to/config.yaml
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

from src.config import Config, get_config
from src.utils import Timer, canonical_countries, ensure_dir, setup_logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Country name resolution
# ---------------------------------------------------------------------------

# Regions/territories folded into their administering country BEFORE the network
# is built (applied in ``build_collaboration_network``). Keeps Taiwan, Puerto
# Rico and French overseas territories from appearing as standalone nodes, and
# credits their papers/co-authorships to the parent country.
REGION_TO_COUNTRY: Dict[str, str] = {
    "Taiwan": "China",
    "Puerto Rico": "United States",
    # French overseas departments / collectivities → metropolitan France
    "French Guiana": "France",
    "Guadeloupe": "France",
    "Martinique": "France",
    "Mayotte": "France",
    "New Caledonia": "France",
}

# Residual false positives in the ``CU`` field.  Person-name leaks ("Jamali"→
# Mali, "Jordan M."→Jordan, "Montenegro J."→Montenegro, "Moldovanu"→Moldova,
# "Lomonaco"→Monaco, "Israel R."→Israel) were eliminated at the source:
# ``_extract_country`` now matches only affiliation text (see CLAUDE.md "Known
# Issues").  What remains are place/state-name collisions inside real
# affiliations (Niger Delta/State→Nigeria, Georgia (US state) institutions,
# Inner Mongolia→China).  Dropped so they do not create phantom nodes/edges
# (e.g. "Niger Delta" → a spurious "Niger" country that otherwise links to
# Nigeria).  Mirrors ``area_pub_deficit._CU_FALSE_POSITIVES``.
CU_FALSE_POSITIVES: set[str] = {"Niger", "Georgia", "Mongolia"}

# Cleaned ``CU`` name → Natural Earth ``ADMIN`` name.
CU_TO_NE: Dict[str, str] = {
    "United States": "United States of America",
    "Republic of Korea": "South Korea",
    "Tanzania": "United Republic of Tanzania",
    "Congo": "Republic of the Congo",
    "Cote d'Ivoire": "Ivory Coast",
    "Czech Republic": "Czechia",
    "Bahamas": "The Bahamas",
    "Trinidad & Tobago": "Trinidad and Tobago",
}

# Small island nations / territories absent from Natural Earth 110m.
# Value: (longitude, latitude) approximate centroid.
MANUAL_COORDS: Dict[str, Tuple[float, float]] = {
    "Singapore": (103.8198, 1.3521),
    "Bahrain": (50.5577, 26.0667),
    "Barbados": (-59.5432, 13.1939),
    "Maldives": (73.5093, 3.2028),
    "Mauritius": (57.5522, -20.3484),
    "Seychelles": (55.4920, -4.6796),
    "Tonga": (-175.1982, -21.1790),
    "Micronesia": (158.2150, 6.9167),
    "Palau": (134.5825, 7.5150),
    "Monaco": (7.4246, 43.7384),
    "Grenada": (-61.6750, 12.1165),
}

# Continent assignment for manually-placed islands (NE doesn't have them).
MANUAL_CONTINENT: Dict[str, str] = {
    "Singapore": "Asia",
    "Bahrain": "Asia",
    "Barbados": "North America",
    "Maldives": "Asia",
    "Mauritius": "Africa",
    "Seychelles": "Africa",
    "Tonga": "Oceania",
    "Micronesia": "Oceania",
    "Palau": "Oceania",
    "Monaco": "Europe",
    "Grenada": "North America",
}

CONTINENT_COLORS: Dict[str, str] = {
    "Asia": "#d62728",
    "Africa": "#ff7f0e",
    "Europe": "#1f77b4",
    "North America": "#2ca02c",
    "South America": "#9467bd",
    "Oceania": "#8c564b",
}

# ---------------------------------------------------------------------------
# Network construction
# ---------------------------------------------------------------------------


def build_collaboration_network(df: pd.DataFrame) -> Tuple[Counter, Dict[Tuple[str, str], int]]:
    """Build a country co-authorship network from the ``CU`` field.

    Args:
        df: Cleaned DataFrame with a ``CU`` column of semicolon-separated
            country names.

    Returns:
        ``(node_pubs, edge_weights)`` where *node_pubs* maps country → number
        of papers and *edge_weights* maps ``(a, b)`` (sorted) → number of
        co-authored papers.
    """
    node_pubs: Counter = Counter()
    edge_weights: Dict[Tuple[str, str], int] = defaultdict(int)

    for cu_val in df["CU"].dropna():
        # One canonical country space for the whole pipeline: fold the
        # administered territories (Taiwan / Puerto Rico / French overseas)
        # into their metropoles, drop the residual false positives and
        # de-duplicate (see src.utils.canonical_countries).
        countries = sorted(canonical_countries(
            cu_val, fold=REGION_TO_COUNTRY, drop=CU_FALSE_POSITIVES
        ))
        if not countries:
            continue
        for c in countries:
            node_pubs[c] += 1
        for a, b in combinations(countries, 2):
            edge_weights[(a, b)] += 1

    logger.info(
        "Collaboration network: %d countries, %d links, %d papers involved",
        len(node_pubs), len(edge_weights), len(df),
    )
    return node_pubs, edge_weights


# ---------------------------------------------------------------------------
# Geographic anchoring
# ---------------------------------------------------------------------------


def resolve_country_geo(
    countries: List[str],
    world: "gpd.GeoDataFrame",  # noqa: F821
) -> Tuple[Dict[str, Tuple[float, float]], Dict[str, str], List[str]]:
    """Map country names to ``(lon, lat)`` centroids and continents.

    Args:
        countries: Unique cleaned ``CU`` country names.
        world: Natural Earth GeoDataFrame with ``ADMIN`` and centroid columns.

    Returns:
        ``(coords, continents, missing)`` — dicts keyed by cleaned name, plus a
        list of country names that could not be located.
    """
    admin_to_xy: Dict[str, Tuple[float, float]] = dict(
        zip(world["ADMIN"], zip(world["lon"], world["lat"]))
    )
    admin_to_continent: Dict[str, str] = dict(zip(world["ADMIN"], world["CONTINENT"]))

    coords: Dict[str, Tuple[float, float]] = {}
    continents: Dict[str, str] = {}
    missing: List[str] = []

    for name in countries:
        ne_name = CU_TO_NE.get(name, name)
        if ne_name in admin_to_xy:
            coords[name] = admin_to_xy[ne_name]
            continents[name] = admin_to_continent[ne_name]
        elif name in MANUAL_COORDS:
            coords[name] = MANUAL_COORDS[name]
            continents[name] = MANUAL_CONTINENT.get(name, "Unknown")
        else:
            missing.append(name)

    if missing:
        logger.warning("Could not locate %d countries: %s", len(missing), missing)
    return coords, continents, missing


def load_country_centroids(world_map_source: str) -> "gpd.GeoDataFrame":  # noqa: F821
    """Load Natural Earth and attach equal-area centroid lon/lat columns."""
    world = gpd.read_file(world_map_source)
    # Compute centroids in an equal-area projection so lon/lat aren't treated
    # as planar (avoids wildly wrong centres for countries like Russia).
    world_ea = world.to_crs("EPSG:6933")
    centroids = world_ea.geometry.centroid
    centroids_geo = gpd.GeoSeries(centroids, crs="EPSG:6933").to_crs("EPSG:4326")
    world["lon"] = centroids_geo.x
    world["lat"] = centroids_geo.y
    return world


def load_mangrove_grid(
    shapefile_path: str, res_deg: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rasterize HGMF mangrove polygons to a coarse area-density grid.

    Each polygon contributes its ``Area`` (km²) to the grid cell containing its
    representative point.  Returns ``(lon_edges, lat_edges, grid)`` where
    ``grid[i, j]`` is the summed mangrove area in the cell bounded by
    ``lat_edges[i:i+2]`` and ``lon_edges[j:j+2]``.
    """
    gdf = gpd.read_file(shapefile_path)
    logger.info("Loaded %d mangrove polygons (crs=%s)", len(gdf), gdf.crs.name)

    pts_geo = gpd.GeoSeries(
        gdf.geometry.representative_point(), crs=gdf.crs
    ).to_crs("EPSG:4326")
    lon = pts_geo.x.to_numpy()
    lat = pts_geo.y.to_numpy()
    area = (
        gdf["Area"].to_numpy(dtype=float)
        if "Area" in gdf.columns
        else np.ones(len(gdf))
    )

    lon_edges = np.arange(-180.0, 180.0 + res_deg, res_deg)
    lat_edges = np.arange(-90.0, 90.0 + res_deg, res_deg)

    col = np.floor((lon + 180.0) / res_deg).astype(int)
    row = np.floor((90.0 - lat) / res_deg).astype(int)
    col = np.clip(col, 0, len(lon_edges) - 2)
    row = np.clip(row, 0, len(lat_edges) - 2)

    grid = np.zeros((len(lat_edges) - 1, len(lon_edges) - 1), dtype=float)
    np.add.at(grid, (row, col), area)
    logger.info(
        "Rasterized to %dx%d grid (%d cells with mangrove)",
        grid.shape[1], grid.shape[0], int((grid > 0).sum()),
    )
    return lon_edges, lat_edges, grid


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


# Reference point toward which edge arcs bow — roughly the centre of the
# visible map window (lon 0, mid-latitude). Bowing toward here routes
# trans-Pacific links (US↔China, US↔Australia) through Europe/Atlantic instead
# of the ±180° seam or the poles.
_CURVE_CENTER_LON = 0.0
_CURVE_CENTER_LAT = 15.0
_CURVE_BOW = 0.15  # fraction of chord length used as the perpendicular offset


def _bezier_arc(
    lon1: float, lat1: float, lon2: float, lat2: float, n: int = 48
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample a quadratic Bézier arc between two points, bowing toward the map centre.

    The control point sits on the perpendicular bisector of the chord, offset by
    ``_CURVE_BOW * chord_length`` on the side facing the map centre. Short links
    stay nearly straight; long trans-Pacific links bow through Europe/Atlantic.
    """
    mx, my = 0.5 * (lon1 + lon2), 0.5 * (lat1 + lat2)
    dx, dy = lon2 - lon1, lat2 - lat1
    length = np.hypot(dx, dy)
    if length < 1e-9:
        return np.array([lon1, lon2]), np.array([lat1, lat2])

    # Unit perpendicular to the chord (rotated +90°).
    px, py = -dy / length, dx / length
    # Flip so the perpendicular points toward the map centre.
    cx, cy = _CURVE_CENTER_LON - mx, _CURVE_CENTER_LAT - my
    if px * cx + py * cy < 0:
        px, py = -px, -py

    k = _CURVE_BOW * length
    cpx, cpy = mx + px * k, my + py * k

    t = np.linspace(0.0, 1.0, n)
    lon = (1.0 - t) ** 2 * lon1 + 2.0 * (1.0 - t) * t * cpx + t ** 2 * lon2
    lat = (1.0 - t) ** 2 * lat1 + 2.0 * (1.0 - t) * t * cpy + t ** 2 * lat2
    return lon, lat


def plot_collaboration_network(
    lon_edges: np.ndarray,
    lat_edges: np.ndarray,
    mangrove_grid: np.ndarray,
    world: "gpd.GeoDataFrame",  # noqa: F821
    node_pubs: Counter,
    edge_weights: Dict[Tuple[str, str], int],
    coords: Dict[str, Tuple[float, float]],
    continents: Dict[str, str],
    config: Config,
    output_path: Path,
) -> None:
    """Render the mangrove base map with the collaboration network overlay."""
    vis = config.visualization
    cn = config.collaboration_network
    plt.rcParams["font.family"] = vis.font_family

    fig, ax = plt.subplots(figsize=(16, 9))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#eaf2f8")  # ocean

    # --- base: land fill -------------------------------------------------
    world.plot(ax=ax, facecolor="#f4f1ec", edgecolor="none", zorder=1)

    # --- base: mangrove area-density raster ------------------------------
    # log1p transform gives a log-like scale without LogNorm's masked-array
    # rendering bug (LogNorm renders masked cells as opaque white).
    log_grid = np.log1p(mangrove_grid)
    masked = np.ma.masked_where(log_grid <= 0, log_grid)
    cmap = LinearSegmentedColormap.from_list(
        "mangrove", ["#a5d6b5", "#4caf6a", "#1b5e3a"]
    )
    cmap.set_bad("none")
    im = None
    vmin = vmax = None
    if masked.count() > 0:
        vmin, vmax = float(masked.min()), float(masked.max())
        im = ax.imshow(
            masked,
            extent=[lon_edges[0], lon_edges[-1], lat_edges[0], lat_edges[-1]],
            # origin="upper" (default): grid row 0 = 90°N must map to the TOP of
            # the axes. origin="lower" would mirror the raster across the equator.
            origin="upper",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
            zorder=2,
        )

    # --- country boundaries ----------------------------------------------
    world.boundary.plot(ax=ax, edgecolor="#b7b7b7", linewidth=0.3, zorder=3)

    # --- edges -----------------------------------------------------------
    edges = sorted(edge_weights.items(), key=lambda kv: kv[1], reverse=True)[: cn.top_n_edges]
    if edges:
        max_w = edges[0][1]
        min_w = edges[-1][1]
        span = max(max_w - min_w, 1)
        for (a, b), w in edges:
            if a not in coords or b not in coords:
                continue
            lon_a, lat_a = coords[a]
            lon_b, lat_b = coords[b]
            lw = 0.3 + 2.7 * (w - min_w) / span
            alpha = 0.15 + 0.55 * (w - min_w) / span
            arc_lon, arc_lat = _bezier_arc(lon_a, lat_a, lon_b, lat_b)
            ax.plot(arc_lon, arc_lat,
                    color="#5d5d5d", linewidth=lw, alpha=alpha,
                    zorder=4, solid_capstyle="round")

    # --- nodes -----------------------------------------------------------
    present = [c for c in node_pubs if c in coords]
    pubs = np.array([node_pubs[c] for c in present], dtype=float)
    color = [CONTINENT_COLORS.get(continents.get(c, "Unknown"), "#777777") for c in present]
    size = 10 + 320 * np.sqrt(pubs / pubs.max())

    ax.scatter(
        [coords[c][0] for c in present],
        [coords[c][1] for c in present],
        s=size, c=color, edgecolor="white", linewidth=0.5,
        alpha=0.9, zorder=5,
    )

    # --- labels ----------------------------------------------------------
    for c in present:
        if node_pubs[c] >= cn.min_pubs_for_label:
            ax.annotate(
                c, (coords[c][0], coords[c][1]),
                xytext=(4, 4), textcoords="offset points",
                fontsize=7, color="0.15", zorder=6,
            )

    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 85)
    ax.set_aspect("auto")
    # Hide ticks/spines manually — set_axis_off() would also drop the axes
    # facecolor patch (ocean background), rendering it white.
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # --- mangrove density colorbar --------------------------------------
    if im is not None:
        cbar = fig.colorbar(im, ax=ax, orientation="horizontal",
                            fraction=0.045, pad=0.02, aspect=40)
        # Re-label the log1p scale in real km² values.
        area_ticks = [a for a in (0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000)
                      if vmin <= np.log1p(a) <= vmax]
        if not area_ticks:
            area_ticks = [float(np.expm1(vmin)), float(np.expm1(vmax))]
        cbar.set_ticks([np.log1p(a) for a in area_ticks])
        cbar.set_ticklabels([f"{a:g}" for a in area_ticks])
        cbar.set_label("Mangrove area (km² per grid cell)", fontsize=8)
        cbar.ax.tick_params(labelsize=7)

    # --- legends ---------------------------------------------------------
    _add_legend(ax, pubs, present, continents, max_w)

    title = (
        "International Collaboration Network on Global Mangrove Distribution\n"
        "(HGMF_2020 base map; nodes = countries, links = co-authored papers)"
    )
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)

    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=vis.dpi, format=vis.figure_format, bbox_inches="tight")
    plt.close(fig)
    logger.info("Collaboration network map saved: %s", output_path)


def _add_legend(ax, pubs, present, continents, max_w) -> None:
    """Add continent-colour and node-size legends."""
    legend_artists: List[Line2D] = []
    legend_labels: List[str] = []

    # Continent colour legend
    for cont, col in CONTINENT_COLORS.items():
        if cont in {continents.get(c) for c in present}:
            legend_artists.append(Line2D([0], [0], marker="o", color="w",
                                         markerfacecolor=col, markersize=9, linestyle=""))
            legend_labels.append(cont)

    # Node-size legend (three reference sizes)
    ref_pubs = [25, 100, 500]
    for rp in ref_pubs:
        if rp > pubs.max():
            continue
        s = 10 + 320 * np.sqrt(rp / pubs.max())
        legend_artists.append(Line2D([0], [0], marker="o", color="w",
                                     markerfacecolor="none", markeredgecolor="grey",
                                     markersize=2 * np.sqrt(s / np.pi), linestyle=""))
        legend_labels.append(f"{rp} papers")

    if legend_artists:
        leg = ax.legend(legend_artists, legend_labels, loc="lower left",
                        fontsize=7, framealpha=0.9, title="Countries", title_fontsize=8)
        leg.get_frame().set_linewidth(0.5)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_collaboration_network(config: Config) -> None:
    paths = config.paths
    cn = config.collaboration_network

    cleaned_path = paths.processed_dir / config.preprocessing.output_file
    if not cleaned_path.exists():
        raise FileNotFoundError(f"Cleaned data not found: {cleaned_path}. Run preprocessing first.")

    shp_path = Path(cn.mangrove_shapefile)
    if not shp_path.is_absolute():
        shp_path = config.project_root / shp_path
    if not shp_path.exists():
        raise FileNotFoundError(
            f"Mangrove shapefile not found: {shp_path}. "
            f"Set collaboration_network.mangrove_shapefile in config.yaml."
        )

    df = pd.read_csv(cleaned_path, encoding="utf-8-sig")

    with Timer("Build collaboration network"):
        node_pubs, edge_weights = build_collaboration_network(df)

    with Timer("Load Natural Earth centroids"):
        world = load_country_centroids(cn.world_map_source)

    coords, continents, _missing = resolve_country_geo(list(node_pubs), world)

    with Timer("Load & rasterize mangrove shapefile"):
        lon_edges, lat_edges, grid = load_mangrove_grid(str(shp_path), cn.raster_res_deg)

    output_path = paths.figures_dir / cn.output_file
    plot_collaboration_network(
        lon_edges, lat_edges, grid, world, node_pubs, edge_weights,
        coords, continents, config, output_path,
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Collaboration network over mangrove map")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    config = get_config(args.config)
    setup_logging(config.system.log_level, config.system.log_file)

    try:
        run_collaboration_network(config)
        logger.info("Collaboration network map complete.")
    except Exception:
        logger.exception("Collaboration network map failed")
        raise


if __name__ == "__main__":
    main()
