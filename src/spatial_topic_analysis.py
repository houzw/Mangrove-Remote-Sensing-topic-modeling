"""Step 5: Spatial Topic Analysis — country-level topic specialization using
fractional counting and the Specialization Index (SI).

Perianes-Rodríguez, Waltman & Van Eck (2016): fractional counting w_{d,c} = 1/m_d.
NSF/NCSES; Rousseau & Yang (2012): SI_{c,k} = (N_{c,k}/N_c) / (N_k/N),
equivalent to Location Quotient / Activity Index / RCA.

Usage:
    python src/spatial_topic_analysis.py              # uses default config.yaml
    python src/spatial_topic_analysis.py --config path/to/config.yaml
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import Config, get_config
from src.utils import Timer, canonical_countries, ensure_dir

logger = logging.getLogger(__name__)

# =============================================================================
# Country name → ISO-3 mapping (for Plotly choropleth)
# =============================================================================

# Mapping from canonical names (as produced by data_preprocessing + bibliometric
# name mapping) to ISO 3166-1 alpha-3 codes recognised by Plotly.
_COUNTRY_ISO3: Dict[str, str] = {
    "Afghanistan": "AFG", "Albania": "ALB", "Algeria": "DZA", "Angola": "AGO",
    "Argentina": "ARG", "Armenia": "ARM", "Australia": "AUS", "Austria": "AUT",
    "Azerbaijan": "AZE", "Bahamas": "BHS", "Bahrain": "BHR",
    "Bangladesh": "BGD", "Barbados": "BRB", "Belarus": "BLR", "Belgium": "BEL",
    "Belize": "BLZ", "Benin": "BEN", "Bhutan": "BTN", "Bolivia": "BOL",
    "Botswana": "BWA", "Brazil": "BRA", "Brunei Darussalam": "BRN",
    "Brunei": "BRN", "Bulgaria": "BGR", "Burkina Faso": "BFA", "Burundi": "BDI",
    "Cambodia": "KHM", "Cameroon": "CMR", "Canada": "CAN", "Chile": "CHL",
    "China": "CHN", "Colombia": "COL", "Comoros": "COM", "Congo": "COG",
    "Costa Rica": "CRI", "Cote d'Ivoire": "CIV", "Côte d'Ivoire": "CIV",
    "Croatia": "HRV", "Cuba": "CUB", "Cyprus": "CYP",
    "Czech Republic": "CZE", "Dem. Rep. Korea": "PRK",
    "Democratic Republic of the Congo": "COD", "Denmark": "DNK",
    "Djibouti": "DJI", "Dominican Republic": "DOM", "Ecuador": "ECU",
    "Egypt": "EGY", "El Salvador": "SLV", "Equatorial Guinea": "GNQ",
    "Eritrea": "ERI", "Estonia": "EST", "Ethiopia": "ETH", "Fiji": "FJI",
    "Finland": "FIN", "France": "FRA", "French Guiana": "GUF", "Gabon": "GAB",
    "Gambia": "GMB", "Georgia": "GEO", "Germany": "DEU", "Ghana": "GHA",
    "Greece": "GRC", "Grenada": "GRD", "Guadeloupe": "GLP", "Guatemala": "GTM",
    "Guinea": "GIN", "Guinea-Bissau": "GNB", "Guyana": "GUY", "Haiti": "HTI",
    "Honduras": "HND", "Hungary": "HUN", "Iceland": "ISL", "India": "IND",
    "Indonesia": "IDN", "Iran": "IRN",
    "Iran (Islamic Republic of)": "IRN", "Iraq": "IRQ", "Ireland": "IRL",
    "Israel": "ISR", "Italy": "ITA", "Jamaica": "JAM", "Japan": "JPN",
    "Jordan": "JOR", "Kazakhstan": "KAZ", "Kenya": "KEN", "Kuwait": "KWT",
    "Kyrgyzstan": "KGZ", "Laos": "LAO",
    "Lao People's Democratic Republic": "LAO", "Latvia": "LVA",
    "Lebanon": "LBN", "Liberia": "LBR", "Libya": "LBY", "Lithuania": "LTU",
    "Luxembourg": "LUX", "Madagascar": "MDG", "Malawi": "MWI",
    "Malaysia": "MYS", "Maldives": "MDV", "Mali": "MLI", "Malta": "MLT",
    "Martinique": "MTQ", "Mauritania": "MRT", "Mauritius": "MUS",
    "Mayotte": "MYT", "Mexico": "MEX", "Micronesia": "FSM",
    "Moldova": "MDA", "Monaco": "MCO", "Mongolia": "MNG",
    "Montenegro": "MNE", "Morocco": "MAR", "Mozambique": "MOZ",
    "Myanmar": "MMR", "Namibia": "NAM", "Nepal": "NPL",
    "Netherlands": "NLD", "New Caledonia": "NCL", "New Zealand": "NZL",
    "Nicaragua": "NIC", "Niger": "NER", "Nigeria": "NGA",
    "North Korea": "PRK", "North Macedonia": "MKD", "Norway": "NOR",
    "Oman": "OMN", "Pakistan": "PAK", "Palau": "PLW", "Palestine": "PSE",
    "Panama": "PAN", "Papua New Guinea": "PNG", "Paraguay": "PRY",
    "Peru": "PER", "Philippines": "PHL", "Poland": "POL", "Portugal": "PRT",
    "Puerto Rico": "PRI", "Qatar": "QAT", "Republic of Korea": "KOR",
    "Republic of the Congo": "COG", "Romania": "ROU",
    "Russian Federation": "RUS", "Russia": "RUS", "Rwanda": "RWA",
    "Saudi Arabia": "SAU", "Senegal": "SEN", "Serbia": "SRB",
    "Seychelles": "SYC", "Sierra Leone": "SLE", "Singapore": "SGP",
    "Slovakia": "SVK", "Slovenia": "SVN", "Solomon Islands": "SLB",
    "Somalia": "SOM", "South Africa": "ZAF", "South Korea": "KOR",
    "Spain": "ESP", "Sri Lanka": "LKA", "St. Lucia": "LCA",
    "St. Vincent & the Grenadines": "VCT", "Sudan": "SDN",
    "Suriname": "SUR", "Sweden": "SWE", "Switzerland": "CHE",
    "Syria": "SYR", "Syrian Arab Republic": "SYR", "Taiwan": "TWN",
    "Tajikistan": "TJK", "Tanzania": "TZA",
    "United Republic of Tanzania": "TZA", "Thailand": "THA",
    "Timor Leste": "TLS", "Togo": "TGO", "Tonga": "TON",
    "Trinidad & Tobago": "TTO", "Trinidad and Tobago": "TTO",
    "Tunisia": "TUN", "Turkey": "TUR", "Turkmenistan": "TKM",
    "Uganda": "UGA", "Ukraine": "UKR", "United Arab Emirates": "ARE",
    "United Kingdom": "GBR", "United States": "USA",
    "United States of America": "USA", "Uruguay": "URY",
    "Uzbekistan": "UZB", "Vanuatu": "VUT", "Venezuela": "VEN",
    "Viet Nam": "VNM", "Vietnam": "VNM", "Yemen": "YEM",
    "Zambia": "ZMB", "Zimbabwe": "ZWE",
}

# =============================================================================
# Country name mapping (preprocessing canonical → Natural Earth / Plotly names)
# =============================================================================

_COUNTRY_NAME_MAPPING_DEFAULT: Dict[str, str] = {
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

# Residual false positives in the CU field.  Person-name leaks ("Jamali"→Mali,
# "Jordan M."→Jordan, "Montenegro J."→Montenegro, "Moldovanu"→Moldova,
# "Lomonaco"→Monaco, "Israel R."→Israel) were eliminated at the source:
# ``_extract_country`` now matches only affiliation text.  Kept in sync with
# ``area_pub_deficit._CU_FALSE_POSITIVES`` so LQ and deficit use identical
# country weighting.  What remains are place/state-name collisions inside real
# affiliations (Niger Delta/State→Nigeria, Georgia (US state) institutions,
# Inner Mongolia→China).  Dropped so they do not absorb fractional weight from
# the real countries on a paper.
_CU_FALSE_POSITIVES: set[str] = {"Niger", "Georgia", "Mongolia"}


# =============================================================================
# Data loading & merging
# =============================================================================


def load_topic_labels(config: Config) -> Dict[int, str]:
    """Build {topic_id: short_label} from topics_info.csv.

    Prefers ``Short_Label`` column if available (human-curated labels for
    publication figures). Falls back to top-3 words from ``Representation``.
    Final fallback: ``'Topic {id}'``.
    """
    import ast

    topics_csv = config.paths.processed_dir / config.topic_modeling.output.topics_info
    if not topics_csv.exists():
        logger.warning("topics_info.csv not found — topic labels will be numeric IDs")
        return {}

    df = pd.read_csv(topics_csv, encoding="utf-8-sig")
    labels: Dict[int, str] = {}

    for _, row in df.iterrows():
        tid = int(row.get("Topic", -1))
        if tid == -1:
            continue

        # 1. Prefer curated short label
        short = row.get("Short_Label", None)
        if pd.notna(short) and isinstance(short, str) and short.strip():
            labels[tid] = short.strip()
            continue

        # 2. Fall back to top-3 representation words
        try:
            rep = ast.literal_eval(row["Representation"])
            if isinstance(rep, list) and len(rep) > 0:
                top_words = [str(w) for w in rep[:3]]
                labels[tid] = "_".join(top_words)
            else:
                labels[tid] = f"Topic_{tid}"
        except (ValueError, SyntaxError, KeyError):
            labels[tid] = f"Topic_{tid}"

    return labels


def merge_country_topic(config: Config) -> pd.DataFrame:
    """Merge cleaned_data (CU) with doc_topic_dist (Topic) by doc_index.

    Returns:
        DataFrame with columns: ``doc_index, Topic, CU, PY``.
        Topic -1 (outliers) is dropped.
    """
    proc = config.paths.processed_dir

    # Load cleaned data (prefer parquet)
    cleaned_pq = proc / config.preprocessing.output_file.replace(".csv", ".parquet")
    if cleaned_pq.exists():
        df = pd.read_parquet(cleaned_pq)
    else:
        cleaned_csv = proc / config.preprocessing.output_file
        if not cleaned_csv.exists():
            raise FileNotFoundError(f"Cleaned data not found: {cleaned_csv}")
        df = pd.read_csv(cleaned_csv, encoding="utf-8-sig")

    # Load doc-topic distribution
    doc_topic_csv = proc / config.topic_modeling.output.doc_topic_dist
    if not doc_topic_csv.exists():
        raise FileNotFoundError(
            f"doc_topic_dist.csv not found: {doc_topic_csv}. Run topic modeling first."
        )
    doc_topic = pd.read_csv(doc_topic_csv, encoding="utf-8-sig")

    # Merge: positional alignment (same row order)
    merged = doc_topic[["doc_index", "Topic"]].copy()
    merged["CU"] = df["CU"].values[:len(merged)]
    if "PY" in df.columns:
        merged["PY"] = df["PY"].values[:len(merged)]

    # Drop outliers
    n_before = len(merged)
    merged = merged[merged["Topic"] != -1].copy()
    if len(merged) < n_before:
        logger.info(
            "Dropped %d outlier (-1) documents for spatial analysis",
            n_before - len(merged),
        )

    logger.info(
        "Merged data: %d documents, %d unique topics, %d countries (raw)",
        len(merged),
        merged["Topic"].nunique(),
        merged["CU"].dropna().apply(
            lambda x: len(str(x).split(";"))
        ).sum() if len(merged) > 0 else 0,
    )
    return merged


# =============================================================================
# Fractional counting
# =============================================================================


def fractional_country_topic(
    df: pd.DataFrame,
    mapping: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Apply fractional counting: each country on a paper gets weight 1/m_d.

    Perianes-Rodríguez, Waltman & Van Eck (2016).

    Args:
        df: Merged DataFrame with ``CU`` and ``Topic`` columns.
        mapping: Optional country name normalization dict.

    Returns:
        Long-form DataFrame: ``Country, Topic, N_ck`` (sum of fractional weights).
    """
    if mapping is None:
        mapping = {}

    logger.info("Applying fractional counting ...")
    records: List[Dict] = []

    total_docs = 0
    unknown_docs = 0
    single_country = 0
    multi_country = 0

    for _, row in df.iterrows():
        cu_val = str(row["CU"]) if pd.notna(row["CU"]) else "Unknown"
        # Fold (via mapping) and de-duplicate: "China;Taiwan" is ONE country for
        # the fractional weight, not two (see src.utils.canonical_countries).
        countries = canonical_countries(
            cu_val, fold=mapping, drop=_CU_FALSE_POSITIVES
        )

        if not countries:
            unknown_docs += 1
            continue

        m_d = len(countries)
        weight = 1.0 / m_d
        topic = int(row["Topic"])

        total_docs += 1
        if m_d == 1:
            single_country += 1
        else:
            multi_country += 1

        for canonical in countries:
            records.append({
                "Country": canonical,
                "Topic": topic,
                "Weight": weight,
            })

    logger.info(
        "  %d docs with country info (%d single, %d multi-country, %d unknown)",
        total_docs, single_country, multi_country, unknown_docs,
    )

    result = pd.DataFrame(records)
    # Aggregate: sum weights per (Country, Topic)
    result = result.groupby(["Country", "Topic"], as_index=False)["Weight"].sum()
    result = result.rename(columns={"Weight": "N_ck"})
    logger.info("  %d unique (country, topic) pairs", len(result))
    return result


# =============================================================================
# Specialization Index
# =============================================================================


def compute_specialization_index(
    fractional_df: pd.DataFrame,
    min_docs_per_country: int = 5,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute SI_{c,k} = (N_{c,k} / N_c) / (N_k / N).

    Args:
        fractional_df: Long-form DataFrame from ``fractional_country_topic``.
        min_docs_per_country: Filter countries with N_c < this threshold.

    Returns:
        (si_matrix, n_ck_matrix, n_c_series):
          - si_matrix: rows=Country, columns=Topic, values=SI.
          - n_ck_matrix: rows=Country, columns=Topic, values=N_{c,k}.
          - n_c_series: total fractional docs per country, sorted descending.
    """
    logger.info("Computing Specialization Index ...")

    # Pivot to country × topic matrix
    n_ck = fractional_df.pivot_table(
        index="Country",
        columns="Topic",
        values="N_ck",
        fill_value=0.0,
    )

    # Ensure topics are sorted
    n_ck = n_ck.reindex(sorted(n_ck.columns), axis=1)

    # N_c = total fractional docs per country (row sums)
    n_c = n_ck.sum(axis=1)
    # N_k = total fractional docs per topic (column sums)
    n_k = n_ck.sum(axis=0)
    # N = global total
    n_total = n_ck.values.sum()

    logger.info(
        "  %d countries, %d topics, N=%.1f fractional docs",
        len(n_ck), len(n_ck.columns), n_total,
    )

    # Filter by min docs
    valid_countries = n_c >= min_docs_per_country
    n_dropped = (~valid_countries).sum()
    if n_dropped > 0:
        dropped_names = n_c[~valid_countries].index.tolist()
        logger.info(
            "  Dropping %d countries with < %d fractional docs: %s",
            n_dropped, min_docs_per_country, dropped_names,
        )
        n_ck = n_ck.loc[valid_countries].copy()
        n_c = n_c.loc[valid_countries]

    # SI = (N_{c,k} / N_c) / (N_k / N)
    # Avoid division by zero: N_k can be zero for very sparse topics
    n_k_safe = n_k.replace(0, np.nan)
    si = n_ck.div(n_c, axis=0).div(n_k_safe / n_total, axis=1)
    si = si.fillna(0.0)

    # N_c sorted descending
    n_c_sorted = n_c.sort_values(ascending=False)

    logger.info(
        "  SI matrix: %d countries × %d topics (SI range: %.2f – %.2f)",
        len(si), len(si.columns), si.values.min(), si.values.max(),
    )
    return si, n_ck, n_c_sorted


# =============================================================================
# Reporting
# =============================================================================


def top_countries_per_topic(
    si_matrix: pd.DataFrame,
    n_ck_matrix: pd.DataFrame,
    topic_labels: Dict[int, str],
    n_c: pd.Series,
    top_n: int = 3,
    min_si: float = 1.0,
) -> pd.DataFrame:
    """For each topic, find the top-N countries by SI.

    Returns:
        DataFrame: columns = [Topic, Topic_Label, Rank, Country, SI, N_ck, N_c].
    """
    rows = []
    for topic_col in si_matrix.columns:
        tid = int(topic_col)
        si_col = si_matrix[topic_col].dropna()
        # Filter to SI >= min_si
        si_col = si_col[si_col >= min_si].sort_values(ascending=False)
        top = si_col.head(top_n)

        for rank, (country, si_val) in enumerate(top.items(), start=1):
            n_ck_val = n_ck_matrix.loc[country, topic_col] if country in n_ck_matrix.index else 0.0
            n_c_val = n_c.get(country, 0.0)
            rows.append({
                "Topic": tid,
                "Topic_Label": topic_labels.get(tid, f"Topic_{tid}"),
                "Rank": rank,
                "Country": country,
                "SI": round(si_val, 3),
                "N_ck": round(n_ck_val, 2),
                "N_c": round(n_c_val, 2),
            })

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(["Topic", "Rank"]).reset_index(drop=True)
    return result


def log_specialization_summary(summary_df: pd.DataFrame) -> None:
    """Log top-N specializing countries per topic to console."""
    logger.info("=" * 60)
    logger.info("Top specializing countries per topic (SI >= 1.0):")
    logger.info("=" * 60)

    current_topic = None
    for _, row in summary_df.iterrows():
        tid = row["Topic"]
        label = row["Topic_Label"]
        if tid != current_topic:
            current_topic = tid
            logger.info("  Topic %d (%s):", tid, label)
        logger.info(
            "    #%d  %-30s  SI=%.2f  N_ck=%.1f  N_c=%.1f",
            row["Rank"], row["Country"], row["SI"], row["N_ck"], row["N_c"],
        )


# =============================================================================
# Visualization: Heatmap [ARCHIVED — see src/archived_viz/spatial_topic_heatmap.py]
# =============================================================================


# =============================================================================
# Visualization: Choropleth
# =============================================================================


def plot_si_choropleth(
    si_matrix: pd.DataFrame,
    topic_labels: Dict[int, str],
    config: Config,
    output_path: Path,
) -> None:
    """Interactive Plotly choropleth with per-topic dropdown.

    Each topic is a separate choropleth trace; the dropdown menu switches
    between them.
    """
    import plotly.graph_objects as go

    sa_cfg = config.spatial_topic_analysis

    # Melt SI matrix to long form
    si_long = si_matrix.reset_index().melt(
        id_vars=["Country"],
        var_name="Topic",
        value_name="SI",
    )

    # Add ISO-3 codes
    si_long["iso_alpha"] = si_long["Country"].map(_COUNTRY_ISO3)
    n_missing = si_long["iso_alpha"].isna().sum()
    if n_missing > 0:
        missing_countries = si_long[si_long["iso_alpha"].isna()]["Country"].unique()
        logger.warning(
            "  %d countries without ISO-3 codes (will be omitted from choropleth): %s",
            n_missing, list(missing_countries),
        )
        si_long = si_long.dropna(subset=["iso_alpha"])

    # Get topics, sorted by global N_k (which correlates with column order in SI matrix)
    all_topics = sorted(si_long["Topic"].unique())
    top_n = min(len(all_topics), sa_cfg.top_n_topics)
    # Keep the top topics by global fractional count (first N columns of SI matrix)
    topics = list(si_matrix.columns[:top_n])

    # Build one trace per topic
    traces = []
    visible_flags = []
    for i, topic in enumerate(topics):
        tid = int(topic)
        topic_data = si_long[si_long["Topic"] == topic].copy()
        label = topic_labels.get(tid, f"Topic {tid}")
        visible = i == 0

        traces.append(go.Choropleth(
            locations=topic_data["iso_alpha"],
            z=topic_data["SI"],
            zmid=1.0,
            colorscale="RdBu_r",
            colorbar=dict(
                title="SI",
                tickformat=".1f",
                len=0.6,
            ),
            visible=visible,
            name=label,
            hovertemplate=(
                f"<b>{label}</b><br>"
                "%{location}<br>"
                "SI = %{z:.2f}<br>"
                "<extra></extra>"
            ),
        ))
        visible_flags.append(visible)

    if not traces:
        logger.warning("No choropleth traces to plot (all countries missing ISO-3?)")
        return

    fig = go.Figure(data=traces)

    # Build dropdown
    buttons = []
    for i, topic in enumerate(topics):
        tid = int(topic)
        label = topic_labels.get(tid, f"Topic {tid}")
        visibility = [False] * len(traces)
        visibility[i] = True
        buttons.append(dict(
            label=label,
            method="update",
            args=[{"visible": visibility}],
        ))

    fig.update_layout(
        title={
            "text": "Country-Level Topic Specialization Index (SI)",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 16},
        },
        geo=dict(
            showframe=False,
            showcoastlines=True,
            coastlinecolor="#444444",
            projection_type="equirectangular",
        ),
        updatemenus=[dict(
            active=0,
            buttons=buttons,
            x=1.05,
            y=1.0,
            xanchor="left",
            yanchor="top",
            direction="down",
            showactive=True,
            bgcolor="#ffffff",
            bordercolor="#cccccc",
            font={"size": 11},
        )],
        template="plotly_white",
        height=600,
        width=1000,
        margin={"r": 0, "t": 50, "l": 0, "b": 0},
    )

    ensure_dir(output_path.parent)
    fig.write_html(str(output_path))
    logger.info("SI choropleth saved: %s", output_path)


# =============================================================================
# Top topics per country (Hu et al. 2025 Fig. 9 style)
# =============================================================================


def plot_top_topics_per_country(
    n_ck_matrix: pd.DataFrame,
    topic_labels: Dict[int, str],
    n_c: pd.Series,
    config: Config,
    output_path: Path,
    top_n_countries: int = 8,
    top_n_topics: int = 5,
) -> None:
    """Grouped bar chart of top topics for selected countries.

    Inspired by Hu et al. (2025) Fig. 9. For each of the top-N countries
    (by total publications), the top-M topics are shown as grouped bars.

    Args:
        n_ck_matrix: Country × topic document count matrix.
        topic_labels: {topic_id: short_label} mapping.
        n_c: Total fractional docs per country (sorted descending).
        config: Validated configuration.
        output_path: Where to save the PNG.
        top_n_countries: Number of countries to include.
        top_n_topics: Number of top topics per country.
    """
    import matplotlib.pyplot as plt

    vis = config.visualization
    plt.rcParams["font.family"] = vis.font_family

    selected_countries = n_c.head(top_n_countries).index.tolist()
    selected_countries = [c for c in selected_countries if c in n_ck_matrix.index]

    if len(selected_countries) == 0:
        logger.warning("No countries selected — skipping top-topics plot")
        return

    # Gather data: for each country, find top topics
    records = []
    for country in selected_countries:
        row = n_ck_matrix.loc[country]
        top_topics = row.sort_values(ascending=False).head(top_n_topics)
        for topic_id, count in top_topics.items():
            tid = int(topic_id)
            label = topic_labels.get(tid, f"T{tid}")
            records.append({
                "Country": country,
                "Topic": label,
                "Topic_ID": tid,
                "Count": float(count),
            })

    plot_df = pd.DataFrame(records)

    # Create grouped bar chart using matplotlib
    n_countries = len(selected_countries)
    n_topics = top_n_topics
    fig, axes = plt.subplots(1, n_countries, figsize=(max(14, n_countries * 2.8), 6), sharey=True)
    if n_countries == 1:
        axes = [axes]

    cmap = plt.cm.viridis
    colors = cmap(np.linspace(0.2, 0.8, n_topics))

    for idx, country in enumerate(selected_countries):
        ax = axes[idx]
        sub = plot_df[plot_df["Country"] == country].sort_values("Count", ascending=False)
        y_pos = np.arange(len(sub))
        bars = ax.barh(y_pos, sub["Count"].values, color=colors[:len(sub)], edgecolor="white", height=0.6)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sub["Topic"].values, fontsize=8)
        ax.invert_yaxis()
        ax.set_title(country, fontsize=10, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", linestyle=":", alpha=0.4)

        # Add count labels
        for bar, val in zip(bars, sub["Count"].values):
            ax.text(bar.get_width() + max(sub["Count"].values) * 0.01,
                    bar.get_y() + bar.get_height() / 2.0,
                    f"{val:.1f}", va="center", fontsize=7, color="0.3")

    axes[0].set_ylabel("Topic", fontsize=10)
    fig.suptitle("Top Topics by Country (Hu et al. 2025 Fig. 9 Style)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=vis.dpi, format=vis.figure_format, bbox_inches="tight")
    plt.close(fig)
    logger.info("Top topics per country plot saved: %s", output_path)


# =============================================================================
# Topic popularity heatmap (Hu et al. 2025 Fig. 9 style)
# =============================================================================


def compute_topic_popularity(
    merged: pd.DataFrame,
    mapping: Optional[Dict[str, str]] = None,
    min_papers_per_year: int = 3,
) -> pd.DataFrame:
    """Compute Hu et al. (2025) style topic popularity per country.

    Topic popularity = sum across all years of [yearly fractional proportion].
    For each country c, year y, topic t:

        popularity(c, y, t) = N_ck(c, y, t) / N_c(c, y)

    Then sum over years: total_popularity(c, t) = Σ_y popularity(c, y, t).

    This gives each year equal weight in the cumulative sum, preventing
    high-publication years from dominating the popularity measure.

    Args:
        merged: DataFrame with ``CU``, ``Topic``, ``PY`` columns.
        mapping: Country name normalization dict.
        min_papers_per_year: Skip country-year bins with too few papers
            (proportions from small samples are unreliable).

    Returns:
        DataFrame with columns: Country, Topic, Popularity (sum of yearly proportions).
        Pivot-friendly for heatmap.
    """
    if mapping is None:
        mapping = {}

    logger.info("Computing Hu et al. (2025) style topic popularity ...")

    # --- Step 1: Fractional counting per (country, topic, year) ---
    records: List[Dict] = []
    for _, row in merged.iterrows():
        cu_val = str(row["CU"]) if pd.notna(row["CU"]) else "Unknown"
        # De-duplicate after folding (see src.utils.canonical_countries); this
        # path deliberately does not apply the *_CU_FALSE_POSITIVES filter.
        countries = canonical_countries(cu_val, fold=mapping)
        if not countries:
            continue

        m_d = len(countries)
        weight = 1.0 / m_d
        topic = int(row["Topic"])
        py = row.get("PY", None)
        if pd.isna(py):
            continue

        for canonical in countries:
            records.append({
                "Country": canonical,
                "Topic": topic,
                "Year": int(py),
                "Weight": weight,
            })

    df = pd.DataFrame(records)
    # Aggregate to (Country, Year, Topic) level
    yearly = df.groupby(["Country", "Year", "Topic"], as_index=False)["Weight"].sum()
    yearly = yearly.rename(columns={"Weight": "N_ck_y"})

    # --- Step 2: Compute yearly proportions ---
    # Total fractional papers per (Country, Year)
    n_c_yearly = yearly.groupby(["Country", "Year"], as_index=False)["N_ck_y"].sum()
    n_c_yearly = n_c_yearly.rename(columns={"N_ck_y": "N_c_y"})

    yearly = yearly.merge(n_c_yearly, on=["Country", "Year"])
    # Filter small bins
    yearly = yearly[yearly["N_c_y"] >= min_papers_per_year].copy()

    # Yearly proportion
    yearly["Proportion_y"] = yearly["N_ck_y"] / yearly["N_c_y"]

    # --- Step 3: Sum proportions across years ---
    popularity = yearly.groupby(["Country", "Topic"], as_index=False)["Proportion_y"].sum()
    popularity = popularity.rename(columns={"Proportion_y": "Popularity"})

    n_countries = popularity["Country"].nunique()
    n_topics = popularity["Topic"].nunique()
    n_pairs = len(popularity)
    logger.info(
        "  Computed topic popularity: %d countries × %d topics = %d pairs "
        "(range: %.3f – %.3f)",
        n_countries, n_topics, n_pairs,
        popularity["Popularity"].min(), popularity["Popularity"].max(),
    )

    return popularity


def plot_popular_topics_by_country(
    popularity_df: pd.DataFrame,
    topic_labels: Dict[int, str],
    config: Config,
    output_path: Path,
    top_n_topics: int = 10,
    top_n_countries: int = 12,
    n_cols: int = 4,
) -> None:
    """Small-multiples horizontal bar chart: top-N topics per country.

    Hu et al. (2025) Fig. 9 style — each country gets its own panel showing
    the top-N most popular topics as horizontal bars, ordered by popularity
    (sum of yearly fractional proportions).

    Countries are ranked by total cumulative topic popularity.

    Args:
        popularity_df: Long-form DataFrame from ``compute_topic_popularity``
            with columns ``Country, Topic, Popularity``.
        topic_labels: {topic_id: short_label} mapping.
        config: Validated configuration.
        output_path: Where to save the PNG.
        top_n_topics: Number of top topics to show per country (default 10).
        top_n_countries: Number of countries to include (default 12).
        n_cols: Grid columns (default 4, giving 4×3 for 12 countries).
    """
    vis = config.visualization
    plt.rcParams["font.family"] = vis.font_family

    # --- Select countries: use popularity data's own ranking ---
    country_total_pop = popularity_df.groupby("Country")["Popularity"].sum().sort_values(ascending=False)
    selected_countries = country_total_pop.head(top_n_countries).index.tolist()

    if len(selected_countries) == 0:
        logger.warning("No countries with popularity data — skipping Fig. 9 plot")
        return

    # --- Compute grid layout ---
    n_rows = int(np.ceil(len(selected_countries) / n_cols))

    # --- Plot ---
    fig_height = max(5.5, n_rows * 3.6)
    fig_width = max(20, n_cols * 5.5)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(fig_width, fig_height),
        sharex=False,
    )
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    for idx, country in enumerate(selected_countries):
        ax = axes[idx]
        sub = popularity_df[popularity_df["Country"] == country].copy()
        top = sub.nlargest(top_n_topics, "Popularity").sort_values("Popularity", ascending=True)

        if len(top) == 0:
            ax.set_visible(False)
            continue

        # Build labels (full text, no truncation) and single color
        bar_color = "#2177B2"
        bar_labels = []
        for _, row in top.iterrows():
            tid = int(row["Topic"])
            label = topic_labels.get(tid, f"T{tid}")
            bar_labels.append(label)

        values = top["Popularity"].values
        y_pos = np.arange(len(bar_labels))
        ax.barh(y_pos, values, color=bar_color,
                edgecolor="white", height=0.65, linewidth=0.5)

        # Topic labels at bar tips (Hu et al. 2025 Fig. 9 style)
        # No explicit numeric values — topic label replaces the number
        xmax = values.max() * 1.55  # headroom for full topic labels
        for j, (val, bar) in enumerate(zip(values, ax.patches)):
            ax.text(
                bar.get_width() + xmax * 0.012,
                bar.get_y() + bar.get_height() / 2.0,
                bar_labels[j],
                va="center", fontsize=6.5, color="0.25",
            )

        # Remove y-axis tick labels (we show them at bar tips instead)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([""] * len(bar_labels))
        ax.invert_yaxis()
        ax.set_xlim(0, xmax)
        ax.set_title(country, fontsize=9, fontweight="bold", pad=4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", labelsize=6)
        ax.grid(axis="x", linestyle=":", alpha=0.35)

        # Add total popularity as subtle text
        nc_val = country_total_pop.get(country, 0.0)
        ax.text(
            0.98, 0.02, f"Σ={nc_val:.1f}",
            transform=ax.transAxes, fontsize=5.5, color="0.4",
            ha="right", va="bottom",
        )

    # Hide unused axes
    for j in range(len(selected_countries), len(axes)):
        axes[j].set_visible(False)

    # Shared x-axis label
    fig.text(0.5, 0.015, "Cumulative Topic Popularity (Σ yearly proportion)",
             ha="center", fontsize=10, fontstyle="italic")

    fig.suptitle(
        "Top 10 Popular Topics by Country",
        fontsize=15,
        fontweight="bold",
        y=0.995,
    )

    fig.tight_layout(rect=[0.01, 0.03, 0.99, 0.97])
    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=vis.dpi, format=vis.figure_format, bbox_inches="tight")
    plt.close(fig)
    logger.info(
        "Popular topics small-multiples bar chart saved: %s (%d countries)",
        output_path, len(selected_countries),
    )


# =============================================================================
# Visualization: Small-multiples choropleth [ARCHIVED — see src/archived_viz/spatial_small_multiples.py]
# =============================================================================


# ARCHIVED — plot_dominant_topic_map moved to src/archived_viz/spatial_dominant_topic_map.py


# ARCHIVED — plot_country_topic_portfolio moved to src/archived_viz/spatial_country_portfolio.py


# =============================================================================
# Main entry point
# =============================================================================


def run_spatial_topic_analysis(config: Config) -> None:
    """Run the complete spatial topic analysis.

    Steps:
        1. Merge cleaned_data (CU) with doc_topic_dist (Topic).
        2. Apply fractional counting.
        3. Compute SI matrix.
        4. Report top-3 specializing countries per topic.
        5. Export SI matrix CSV.
        6. Plot SI heatmap.
        7. Plot interactive SI choropleth.
        8. Plot top topics per country bar chart.
        9. Plot static small-multiples choropleth grid.
       10. Plot dominant topic categorical map.
       11. Plot country research portfolio dot plot.
       12. Compute Hu et al. (2025) style topic popularity.
       13. Plot top-10 popular topics × countries heatmap.
    """
    sa_cfg = config.spatial_topic_analysis
    paths = config.paths

    # 1. Merge
    with Timer("Merging country-topic data"):
        merged = merge_country_topic(config)

    # 2. Build combined country name mapping
    mapping = {
        **_COUNTRY_NAME_MAPPING_DEFAULT,
        **config.country_analysis.country_name_mapping,
    }

    # 3. Fractional counting
    with Timer("Fractional counting"):
        fractional_df = fractional_country_topic(merged, mapping=mapping)

    # 4. Compute SI
    with Timer("Computing Specialization Index"):
        si_matrix, n_ck_matrix, n_c = compute_specialization_index(
            fractional_df,
            min_docs_per_country=sa_cfg.min_docs_per_country,
        )

    # 5. Load topic labels
    topic_labels = load_topic_labels(config)

    # 6. Top countries per topic
    with Timer("Reporting top countries per topic"):
        summary = top_countries_per_topic(
            si_matrix,
            n_ck_matrix,
            topic_labels,
            n_c,
            top_n=3,
            min_si=sa_cfg.min_si_threshold,
        )
        log_specialization_summary(summary)

    # 7. Export SI matrix
    with Timer("Saving SI matrix"):
        si_csv = paths.processed_dir / sa_cfg.output.si_matrix
        si_matrix.to_csv(si_csv, encoding="utf-8-sig", float_format="%.4f")
        logger.info("SI matrix saved: %s (%d countries × %d topics)", si_csv, len(si_matrix), len(si_matrix.columns))

    # 8. Visualizations
    # ARCHIVED: heatmap — see src/archived_viz/spatial_topic_heatmap.py
    # with Timer("Plotting SI heatmap"):
    #     heatmap_path = paths.figures_dir / sa_cfg.output.heatmap
    #     plot_si_heatmap(si_matrix, n_c, topic_labels, config, heatmap_path)

    with Timer("Plotting SI choropleth"):
        choropleth_path = paths.figures_dir / sa_cfg.output.choropleth_html
        plot_si_choropleth(si_matrix, topic_labels, config, choropleth_path)

    with Timer("Plotting top topics per country"):
        top_topics_path = paths.figures_dir / sa_cfg.output.top_topics_per_country
        plot_top_topics_per_country(
            n_ck_matrix, topic_labels, n_c, config, top_topics_path,
            top_n_countries=8, top_n_topics=5,
        )

    # ARCHIVED: small_multiples — see src/archived_viz/spatial_small_multiples.py
    # with Timer("Plotting SI small multiples"):
    #     small_multiples_path = paths.figures_dir / sa_cfg.output.small_multiples
    #     plot_si_small_multiples(si_matrix, topic_labels, config, small_multiples_path)

    # ARCHIVED: dominant_topic_map — see src/archived_viz/spatial_dominant_topic_map.py
    # with Timer("Plotting dominant topic map"):
    #     dominant_map_path = paths.figures_dir / sa_cfg.output.dominant_topic_map
    #     plot_dominant_topic_map(si_matrix, topic_labels, config, dominant_map_path)

    # ARCHIVED: country_portfolio — see src/archived_viz/spatial_country_portfolio.py
    # with Timer("Plotting country topic portfolio"):
    #     portfolio_path = paths.figures_dir / sa_cfg.output.country_portfolio
    #     plot_country_topic_portfolio(
    #         si_matrix, n_c, topic_labels, config, portfolio_path,
    #         top_n_countries=10,
    #     )

    # 9. Hu et al. (2025) Fig. 9 style: topic popularity heatmap
    with Timer("Computing Hu-style topic popularity"):
        popularity_df = compute_topic_popularity(merged, mapping=mapping)

    # SUPERSEDED 2026-09-26: this 21 x 11 in 12-panel popularity bar chart was the
    # file the manuscript's Fig. 7 caption pointed at, but it is a different metric
    # drawn at 3x the printed width, so its 6.5 pt bar labels printed at 2 pt. The
    # manuscript figure is now the country x topic heatmap below; the plotter above is
    # kept for reference but is no longer run.
    # with Timer("Plotting popular topics by country"):
    #     popularity_heatmap_path = paths.figures_dir / sa_cfg.output.popular_topics_heatmap
    #     plot_popular_topics_by_country(
    #         popularity_df, topic_labels, config, popularity_heatmap_path,
    #         top_n_topics=10, top_n_countries=12,
    #     )

    # Imported lazily: src.country_topic_heatmap imports this module for the matrix.
    from src.country_topic_heatmap import plot_country_topic_heatmap

    with Timer("Plotting country x topic heatmap (Fig. 7)"):
        heatmap_path = paths.figures_dir / sa_cfg.output.country_topic_heatmap
        plot_country_topic_heatmap(config, heatmap_path)

    logger.info("Spatial topic analysis complete.")


def main() -> None:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Step 5: Spatial Topic Analysis (country-level SI)"
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to config.yaml"
    )
    args = parser.parse_args()

    config = get_config(args.config)
    from src.utils import setup_logging

    setup_logging(config.system.log_level, config.system.log_file)

    try:
        run_spatial_topic_analysis(config)
    except Exception:
        logger.exception("Spatial topic analysis failed")
        raise


if __name__ == "__main__":
    main()