"""Non-mangrove country analysis — "technology export" from countries without
native mangroves.

Mangrove remote-sensing research is produced not only by the ~100 countries that
host mangroves, but also by temperate / northern countries (Ireland, Russia,
Germany, France, the Netherlands, …) that have no native mangroves yet publish
mangrove studies through remote-sensing methodology or international
collaboration. This module quantifies that "north → south" technology export.

What it produces (into ``data/processed`` and ``outputs/figures``):

* ``non_mangrove_yearly.csv``            — publication counts by country × year.
* ``non_mangrove_collaboration_edges.csv`` — co-authorship links between a
  non-mangrove country and a mangrove-hosting country.
* ``non_mangrove_study_regions.csv``     — study-region matrix (non-mangrove
  country → the mangrove regions it studies, parsed from titles/abstracts).
* ``non_mangrove_collaboration_network.png`` — bipartite co-authorship network.
* ``non_mangrove_study_region_flow.png`` — Sankey flow of study regions.

Country classification
----------------------
The mangrove-hosting set is derived from ``HGMF_country_area.csv`` (the 104
territories with mapped mangrove area). Overseas territories are folded into
their metropole (e.g. French Guiana → France), so a country counts as
"mangrove-hosting" only if it has *native* mangroves. Taiwan, which hosts
mangroves but is absent from the HGMF table (folded into China), is added back
explicitly. Authors from overseas territories are likewise attributed to their
metropole so France / the Netherlands / the UK are correctly treated as
non-mangrove countries.
"""

from __future__ import annotations

import argparse
import logging
import re
from collections import Counter
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import plotly.graph_objects as go

from src.config import Config, get_config
from src.utils import setup_logging

logger = logging.getLogger(__name__)

# =============================================================================
# Country / territory name tables
# =============================================================================

# Residual false positives in the CU column produced by data_preprocessing.
# Person-name leaks ("Jamali"→Mali, "Jordan M."→Jordan, "Montenegro J."→
# Montenegro, "Moldovanu"→Moldova, "Lomonaco"→Monaco, "Israel R."→Israel) were
# fixed at the source: _extract_country now strips author-name brackets and
# matches affiliation text only, so those countries are no longer excluded and
# genuine affiliations (e.g. an "Amman, Jordan" address) count again.  What
# remains are place/state-name collisions inside real affiliations that a
# body-wide substring match cannot tell apart from the sovereign country:
#   * "Niger Delta/State"→Niger (Nigeria), "Georgia, USA"→Georgia,
#     "Inner Mongolia"→Mongolia (China)
# These are excluded here; the root cause (body-wide, position-unaware matching)
# lives in data_preprocessing.py.
_CU_FALSE_POSITIVES: set[str] = {"Niger", "Georgia", "Mongolia"}

# Countries that host native mangroves but are absent from the HGMF table:
# Taiwan (folded into China) and Maldives (too small for the HGMF 2020 layer).
_NATIVE_MANGROVE_EXTRAS: set[str] = {"Taiwan", "Maldives"}

# Overseas territories in the HGMF table → their metropole. Used both to build
# the native-mangrove country set and to attribute authors to their metropole.
_OVERSEAS_TERRITORY_PARENT: Dict[str, str] = {
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

# HGMF typos / abbreviations → canonical name (must match the preprocessing
# canonical country names so author-country classification lines up).
_HGMF_NAME_FIX: Dict[str, str] = {
    "Camerocn": "Cameroon",
    "Venzuela": "Venezuela",
    "Solomon Is.": "Solomon Islands",
    "Congo.DRC": "Congo",
    "Cote d'Ivory": "Cote d'Ivoire",
    "The Bahamas": "Bahamas",
    "The Gambia": "Gambia",
}

# Extra string variants used when scanning titles/abstracts for study regions.
# Keyed by the canonical display name (after _HGMF_NAME_FIX).
_REGION_SEARCH_ALIASES: Dict[str, List[str]] = {
    "Congo": ["Democratic Republic of the Congo", "DR Congo", "DRC"],
    "Cote d'Ivoire": ["Côte d'Ivoire", "Ivory Coast"],
    "Virgin Is.": ["Virgin Islands", "US Virgin Islands"],
    "British Virgin Is.": ["British Virgin Islands"],
    "Cayman Is.": ["Cayman Islands"],
    "Turks & Caicos Is.": ["Turks and Caicos"],
    "St. Vincent & the Grenadines": ["St. Vincent and the Grenadines"],
    "Trinidad & Tobago": ["Trinidad and Tobago"],
}


# =============================================================================
# Data loading & classification
# =============================================================================


def _load_cleaned(config: Config) -> pd.DataFrame:
    """Load cleaned_data (prefer fast parquet, fall back to CSV)."""
    parquet = config.paths.processed_dir / config.preprocessing.output_file.replace(".csv", ".parquet")
    if parquet.exists():
        return pd.read_parquet(parquet, columns=["CU", "PY", "TI", "AB", "SO", "TC"])
    csv = config.paths.processed_dir / config.preprocessing.output_file
    return pd.read_csv(csv, encoding="utf-8-sig", usecols=["CU", "PY", "TI", "AB", "SO", "TC"])


def _build_native_mangrove_hosts(config: Config) -> set[str]:
    """Return the set of countries with *native* mangroves (metropole level)."""
    area_path = config.paths.external_dir / config.country_analysis.mangrove_area_file
    hgmf = pd.read_csv(area_path)["Country"].astype(str)

    hosts: set[str] = set()
    for name in hgmf:
        if name in _OVERSEAS_TERRITORY_PARENT:
            continue  # overseas territory → folded into metropole
        hosts.add(_HGMF_NAME_FIX.get(name, name))
    hosts |= _NATIVE_MANGROVE_EXTRAS
    return hosts


def _normalise_author_country(name: str) -> str:
    """Attribute an author country to its metropole (overseas territories)."""
    return _OVERSEAS_TERRITORY_PARENT.get(name, name)


def _split_countries(cu: str) -> List[str]:
    """Split a ';'-joined CU field into canonical country names.

    Drops known false positives (person/place-name substrings) and ``Unknown``.
    """
    if not isinstance(cu, str) or not cu.strip():
        return []
    return [
        p.strip()
        for p in cu.split(";")
        if p.strip() and p.strip() != "Unknown" and p.strip() not in _CU_FALSE_POSITIVES
    ]


# =============================================================================
# Study-region extraction (rule-based country-name matching)
# =============================================================================


def _build_study_region_pattern(config: Config) -> Tuple[List[str], re.Pattern]:
    """Build a regex over mangrove-hosting territory names for study-region scan.

    Returns ``(display_names, pattern)``. Longer names are placed first in the
    alternation so that "Papua New Guinea" wins over "Guinea" (non-overlapping
    ``finditer`` consumes the longer match).
    """
    area_path = config.paths.external_dir / config.country_analysis.mangrove_area_file
    hgmf = pd.read_csv(area_path)["Country"].astype(str)

    display_names: List[str] = []
    search_terms: set[str] = set()
    for name in hgmf:
        display = _HGMF_NAME_FIX.get(name, name)
        if display not in display_names:
            display_names.append(display)
        for term in _REGION_SEARCH_ALIASES.get(display, [display]):
            search_terms.add(term)
    # Taiwan hosts mangroves but is absent from HGMF.
    display_names.append("Taiwan")
    search_terms.add("Taiwan")

    # Longest-first alternation to avoid substring overlap.
    ordered = sorted(search_terms, key=len, reverse=True)
    pattern = re.compile(
        r"(?<![A-Za-z])(" + "|".join(re.escape(t) for t in ordered) + r")(?![A-Za-z])",
        flags=re.IGNORECASE,
    )
    return display_names, pattern


def _canonical_region(matched_term: str, display_names: List[str]) -> str:
    """Map a matched search term back to a display name.

    Prefers the display name when the term equals it (ignoring case); otherwise
    falls back to a title-cased copy of the match.
    """
    for disp in display_names:
        if disp.lower() == matched_term.lower():
            return disp
    for disp, aliases in _REGION_SEARCH_ALIASES.items():
        if matched_term.lower() in {a.lower() for a in aliases}:
            return disp
    return matched_term


# =============================================================================
# Computations
# =============================================================================


def _classify(cleaned: pd.DataFrame, hosts: set[str]) -> pd.DataFrame:
    """Add per-document country lists, split into mangrove / non-mangrove."""
    rows = []
    for cu, py in zip(cleaned["CU"], cleaned["PY"]):
        non_mangrove: List[str] = []
        mangrove: List[str] = []
        for raw in _split_countries(cu):
            country = _normalise_author_country(raw)
            if country in hosts:
                mangrove.append(country)
            else:
                non_mangrove.append(country)
        rows.append((py, non_mangrove, mangrove))
    out = pd.DataFrame(rows, columns=["PY", "NonMangrove", "Mangrove"], index=cleaned.index)
    return out


def _compute_yearly(classified: pd.DataFrame) -> pd.DataFrame:
    """Country × year publication counts (full and fractional)."""
    full: Counter = Counter()
    frac: Counter = Counter()
    for py, non_mangrove in zip(classified["PY"], classified["NonMangrove"]):
        if not non_mangrove:
            continue
        w = 1.0 / len(non_mangrove)
        for country in non_mangrove:
            full[(country, py)] += 1
            frac[(country, py)] += w

    rows = [
        {"Country": c, "Year": y, "N_pubs": full[(c, y)], "N_pubs_frac": round(frac[(c, y)], 3)}
        for (c, y) in full
    ]
    return pd.DataFrame(rows).sort_values(["Country", "Year"]).reset_index(drop=True)


def _compute_collaboration_edges(classified: pd.DataFrame) -> pd.DataFrame:
    """Co-authorship edges between non-mangrove and mangrove countries."""
    edge: Counter = Counter()
    for non_mangrove, mangrove in zip(classified["NonMangrove"], classified["Mangrove"]):
        if not non_mangrove or not mangrove:
            continue
        for src in non_mangrove:
            for tgt in mangrove:
                edge[(src, tgt)] += 1
    rows = [{"Source": s, "Target": t, "Weight": w} for (s, t), w in edge.items()]
    return pd.DataFrame(rows).sort_values("Weight", ascending=False).reset_index(drop=True)


def _compute_study_regions(
    classified: pd.DataFrame,
    cleaned: pd.DataFrame,
    region_pattern: re.Pattern,
    display_names: List[str],
    max_regions_per_doc: int = 6,
) -> pd.DataFrame:
    """Non-mangrove country → study region matrix (from titles + abstracts)."""
    region_counter: Counter = Counter()

    for idx, row in classified.iterrows():
        non_mangrove = row["NonMangrove"]
        if not non_mangrove:
            continue
        text = " ".join(
            str(cleaned.at[idx, col]) for col in ("TI", "AB") if pd.notna(cleaned.at[idx, col])
        )
        matches = [m.group(1) for m in region_pattern.finditer(text)]
        regions = sorted({_canonical_region(m, display_names) for m in matches})

        if not regions:
            continue
        if len(regions) > max_regions_per_doc:
            regions = ["Global / multi-region"]
        for country in non_mangrove:
            for region in regions:
                region_counter[(country, region)] += 1

    rows = [
        {"Country": c, "Study_Region": r, "N_pubs": region_counter[(c, r)]}
        for (c, r) in region_counter
    ]
    return pd.DataFrame(rows).sort_values("N_pubs", ascending=False).reset_index(drop=True)


# =============================================================================
# Plots
# =============================================================================


def _plot_collaboration_network(edges: pd.DataFrame, out_path, min_pubs: int = 3):
    """Bipartite co-authorship network (networkx + matplotlib)."""
    if edges.empty:
        logger.warning("No collaboration edges to plot")
        return

    # Keep non-mangrove countries with enough output and their top partners.
    src_counts = edges.groupby("Source")["Weight"].sum()
    keep_src = src_counts[src_counts >= min_pubs].index.tolist()
    sub = edges[edges["Source"].isin(keep_src)]
    top_targets = sub.groupby("Target")["Weight"].sum().nlargest(20).index.tolist()
    sub = sub[sub["Target"].isin(top_targets)]

    G = nx.Graph()
    for _, r in sub.iterrows():
        G.add_edge(r["Source"], r["Target"], weight=r["Weight"])

    left = [n for n in G if n in set(keep_src)]
    right = [n for n in G if n not in set(keep_src)]
    pos = nx.bipartite_layout(G, left, aspect_ratio=1.6)

    weights = [G[u][v]["weight"] for u, v in G.edges()]
    max_w = max(weights) if weights else 1

    fig, ax = plt.subplots(figsize=(14, 9))
    nx.draw_networkx_edges(
        G, pos, ax=ax, edge_color=weights, edge_cmap=plt.cm.viridis,
        edge_vmin=1, edge_vmax=max_w, alpha=0.55, width=[0.5 + 3.5 * w / max_w for w in weights],
    )
    nx.draw_networkx_nodes(
        G, pos, ax=ax, nodelist=left, node_color="#d9534f", node_size=700, alpha=0.95
    )
    nx.draw_networkx_nodes(
        G, pos, ax=ax, nodelist=right, node_color="#0275d8", node_size=700, alpha=0.95
    )
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=8)

    from matplotlib.lines import Line2D
    legend = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#d9534f", markersize=10,
               label="Non-mangrove (technology-exporting)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#0275d8", markersize=10,
               label="Mangrove-hosting (study region / partner)"),
    ]
    ax.legend(handles=legend, loc="upper left", fontsize=9, frameon=False)
    ax.set_title("Mangrove RS co-authorship: non-mangrove ↔ mangrove countries", fontsize=13)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Collaboration network saved: %s", out_path)


def _plot_study_region_flow(region_df: pd.DataFrame, out_path, min_pubs: int = 3, top_regions: int = 20):
    """Sankey: non-mangrove country → study region (plotly, kaleido PNG)."""
    if region_df.empty:
        logger.warning("No study-region data to plot")
        return

    src_counts = region_df.groupby("Country")["N_pubs"].sum()
    keep_src = src_counts[src_counts >= min_pubs].index.tolist()
    sub = region_df[region_df["Country"].isin(keep_src)]
    top_regions_list = sub.groupby("Study_Region")["N_pubs"].sum().nlargest(top_regions).index.tolist()
    sub = sub[sub["Study_Region"].isin(top_regions_list)]

    sources = sorted(sub["Country"].unique())
    targets = sorted(sub["Study_Region"].unique())
    label = sources + targets
    node_index = {name: i for i, name in enumerate(label)}

    flows = sub[["Country", "Study_Region", "N_pubs"]].values.tolist()
    fig = go.Figure(
        go.Sankey(
            arrangement="snap",
            node=dict(
                pad=12, thickness=14,
                line=dict(color="black", width=0.4),
                label=label,
                color=["#d9534f"] * len(sources) + ["#0275d8"] * len(targets),
            ),
            link=dict(
                source=[node_index[s] for s, _, _ in flows],
                target=[node_index[t] for _, t, _ in flows],
                value=[int(w) for _, _, w in flows],
            ),
        )
    )
    fig.update_layout(title_text="Mangrove study-region flow from non-mangrove countries", font_size=11)
    fig.write_image(str(out_path), width=1400, height=900)
    logger.info("Study-region flow saved: %s", out_path)


# =============================================================================
# Orchestration
# =============================================================================


def run_non_mangrove_analysis(config: Config, min_pubs: int = 3) -> None:
    paths = config.paths
    processed = paths.processed_dir
    figures = paths.figures_dir

    logger.info("Loading cleaned data …")
    cleaned = _load_cleaned(config)

    hosts = _build_native_mangrove_hosts(config)
    logger.info("Native mangrove-hosting countries: %d", len(hosts))

    classified = _classify(cleaned, hosts)
    n_non = classified["NonMangrove"].apply(len).sum()
    n_docs_non = (classified["NonMangrove"].apply(len) > 0).sum()
    logger.info("Non-mangrove author appearances: %d across %d documents", n_non, n_docs_non)

    # --- CSV outputs ---
    yearly = _compute_yearly(classified)
    yearly.to_csv(processed / "non_mangrove_yearly.csv", index=False, encoding="utf-8-sig")
    logger.info("Saved non_mangrove_yearly.csv (%d rows)", len(yearly))

    edges = _compute_collaboration_edges(classified)
    edges.to_csv(processed / "non_mangrove_collaboration_edges.csv", index=False, encoding="utf-8-sig")
    logger.info("Saved non_mangrove_collaboration_edges.csv (%d rows)", len(edges))

    display_names, region_pattern = _build_study_region_pattern(config)
    regions = _compute_study_regions(classified, cleaned, region_pattern, display_names)
    regions.to_csv(processed / "non_mangrove_study_regions.csv", index=False, encoding="utf-8-sig")
    logger.info("Saved non_mangrove_study_regions.csv (%d rows)", len(regions))

    # --- Figure outputs ---
    _plot_collaboration_network(edges, figures / "non_mangrove_collaboration_network.png", min_pubs=min_pubs)
    _plot_study_region_flow(regions, figures / "non_mangrove_study_region_flow.png", min_pubs=min_pubs)

    logger.info("Non-mangrove analysis complete.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Non-mangrove country 'technology export' analysis"
    )
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--min-pubs", type=int, default=3, help="Min pubs to keep a country in figures")
    args = parser.parse_args()

    config = get_config(args.config)
    setup_logging(config.system.log_level, config.system.log_file)
    run_non_mangrove_analysis(config, min_pubs=args.min_pubs)


if __name__ == "__main__":
    main()
