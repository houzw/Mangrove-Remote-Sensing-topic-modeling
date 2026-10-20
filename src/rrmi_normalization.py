"""Size-adjusted RRMI — separating capacity from field-specific deficit.

The headline index (``src.area_pub_deficit``) is

    RRMI_c = a_c - p_c,     a_c = area share, p_c = fractional publication share

and its normative benchmark is the assumption that a country's share of the
field's output should track its share of the resource.  A large positive RRMI
therefore mixes two different situations: a country that neglects mangrove
remote sensing *relative to its own scientific output*, and a country whose
whole research system is small.  This module separates them with an external
size denominator:

    s_c   all-field share of scientific and technical journal articles
    RRMI_adj_c = a_c - s_c          (capacity-normalized mismatch)
    RCA_c      = p_c / s_c          (Balassa revealed comparative advantage)

Because the two terms share ``s_c``, the headline index decomposes exactly:

    RRMI_c = (a_c - s_c) + (s_c - p_c)
           = capacity term + field-specific term

``a_c - s_c > 0`` means the country's research system is small relative to its
mangrove endowment; ``s_c - p_c > 0`` means it publishes less in this field
than its own overall output would predict.  ``RCA_c > 1`` marks a country that
is already *over*-specialized in mangrove remote sensing, so its positive RRMI
cannot be read as field-specific neglect.

Data
----
World Bank World Development Indicators (cached under ``data/external/worldbank``):

* ``IP.JRN.ARTC.SC``  scientific and technical journal articles, all fields
                      (the NSF Science & Engineering Indicators series) — ``s_c``
* ``NY.GDP.MKTP.CD``  GDP, current US$ — alternative denominator
* ``SP.POP.TOTL``     population, total — alternative denominator
* ``GB.XPD.RSDV.GD.ZS`` R&D expenditure (% of GDP) — coverage is reported but the
                      indicator is **not** used as a denominator, because it is
                      missing for far too many of the 76 countries to support a
                      ranking comparison (see the run report).

Primary window 2021--2023 (latest three available years), a long-window
variant averages every available year from 1996 to 2023.  Only countries whose
ISO3 code is a real country enter; World Bank aggregates are dropped.

Counting conventions: ``a_c``, ``p_c`` and ``s_c`` are all normalized within the
same 76-country table, so the three shares are commensurable and the
decomposition holds row by row.  ``p_c`` is fractional (author-country weight
``1/m_d``) while the all-field series is a whole-article count attributed to the
country of the author's address; the two are not the same estimator, which the
manuscript states as a caveat.

Outputs
-------
``data/processed/rrmi_normalization.csv``      per-country decomposition (76 rows)
``data/processed/rrmi_normalization_summary.json`` headline numbers + provenance

These are the *alternative* denominators (all-field articles, GDP, population).
The primary normalization is the parent field, which the same database and the
same query structure supply: ``src.rrmi_parent_field`` computes it, owns
Supplementary Tables S7--S8, and reads this module's CSV for the external rows.

Usage
-----
    pixi run python -m src.rrmi_normalization            # cache-first
    pixi run python -m src.rrmi_normalization --refresh  # re-download
"""

from __future__ import annotations

import argparse
import json
import logging
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.config import Config, get_config
from src.utils import Timer, ensure_dir, setup_logging

logger = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================

WB_API = (
    "https://api.worldbank.org/v2/country/all/indicator/{indicator}"
    "?format=json&per_page=20000&date={start}:{end}"
)

#: indicator key -> (World Bank code, human-readable label, denominator role)
INDICATORS: Dict[str, Tuple[str, str, str]] = {
    "articles": (
        "IP.JRN.ARTC.SC",
        "Scientific and technical journal articles (all fields)",
        "primary",
    ),
    "gdp": ("NY.GDP.MKTP.CD", "GDP (current US$)", "alternative"),
    "population": ("SP.POP.TOTL", "Population, total", "alternative"),
    "rd": (
        "GB.XPD.RSDV.GD.ZS",
        "Research and development expenditure (% of GDP)",
        "coverage-only",
    ),
}

#: Primary reference window: the latest three available years.
LATEST_WINDOW: Tuple[int, int] = (2021, 2023)
#: Long-window variant: every year the payload carries.
LONG_WINDOW: Tuple[int, int] = (1996, 2023)

#: Countries in ``area_pub_deficit.csv`` -> World Bank economy name.  Explicit,
#: because the World Bank spells several of them differently ("Viet Nam",
#: "Iran, Islamic Rep.", ...).  ``Congo`` joins HGMF's Congo DRC *area* row to
#: the publications recorded under ``Congo`` (Republic of the Congo); the
#: all-field denominator follows the publication side.
WB_COUNTRY_NAME: Dict[str, str] = {
    "Bahamas": "Bahamas, The",
    "Brunei": "Brunei Darussalam",
    "Congo": "Congo, Rep.",
    "Cote d'Ivoire": "Cote d'Ivoire",
    "Egypt": "Egypt, Arab Rep.",
    "Gambia": "Gambia, The",
    "Iran": "Iran, Islamic Rep.",
    "Micronesia": "Micronesia, Fed. Sts.",
    "Trinidad & Tobago": "Trinidad and Tobago",
    "Venezuela": "Venezuela, RB",
    "Vietnam": "Viet Nam",
}

#: Aggregates in the World Bank payload are identified by a non-ISO3 code.
_ISO3 = r"^[A-Z]{3}$"


# =============================================================================
# World Bank retrieval (cache-first, so the pipeline runs offline)
# =============================================================================


def _cache_dir(config: Config) -> Path:
    return config.paths.external_dir / "worldbank"


def _cache_path(config: Config, indicator: str, start: int, end: int) -> Path:
    return _cache_dir(config) / f"wb_{indicator}_{start}_{end}.json"


def _download(indicator: str, start: int, end: int) -> dict:
    url = WB_API.format(indicator=indicator, start=start, end=end)
    logger.info("Downloading World Bank indicator %s (%d-%d)", indicator, start, end)
    try:
        with urllib.request.urlopen(url, timeout=120) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError) as exc:  # pragma: no cover
        raise RuntimeError(
            f"World Bank download failed for {indicator} ({start}-{end}): {exc}. "
            "Re-run with network access, or restore the cached payload under "
            f"data/external/worldbank/wb_{indicator}_{start}_{end}.json"
        ) from exc
    if not isinstance(payload, list) or len(payload) < 2 or payload[1] is None:
        raise RuntimeError(f"Unexpected World Bank payload for {indicator}: {str(payload)[:200]}")
    return payload


COUNTRY_API = "https://api.worldbank.org/v2/country?format=json&per_page=400"


def load_country_metadata(config: Config, refresh: bool = False) -> dict:
    """Real economies only, from the World Bank country list.

    The indicator payload also carries aggregates (``WLD``, ``AFE``, ``EUU`` ...)
    whose codes are 3 uppercase letters, so a code-shape test cannot separate
    them: the country endpoint marks each aggregate with ``region.id == "NA"``.
    """
    path = _cache_dir(config) / "wb_countries.json"
    if refresh or not path.exists():
        with urllib.request.urlopen(COUNTRY_API, timeout=120) as response:
            payload = json.loads(response.read().decode("utf-8"))
        ensure_dir(path.parent)
        path.write_text(json.dumps(payload), encoding="utf-8")
        source = "downloaded"
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        source = "cache"
    entries = payload[1] if isinstance(payload, list) and len(payload) > 1 else payload
    real = {e["id"] for e in entries if (e.get("region") or {}).get("id") != "NA"}
    logger.info("World Bank country list: %d economies, %d non-aggregate (%s)",
                len(entries), len(real), source)
    return {"iso3": real, "source": source}


def load_indicator(
    config: Config,
    key: str,
    start: int,
    end: int,
    refresh: bool = False,
) -> Tuple[pd.DataFrame, dict]:
    """Load one indicator as a long frame ``country, iso3, year, value``.

    Cache-first: the raw JSON under ``data/external/worldbank`` is the artifact
    of record, so a run without network access reproduces the numbers exactly.
    """
    indicator = INDICATORS[key][0]
    real_iso3 = load_country_metadata(config, refresh=refresh)["iso3"]
    path = _cache_path(config, indicator, start, end)
    if refresh or not path.exists():
        payload = _download(indicator, start, end)
        ensure_dir(path.parent)
        path.write_text(json.dumps(payload), encoding="utf-8")
        source = "downloaded"
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        source = "cache"
    meta = {
        "key": key,
        "indicator": indicator,
        "label": INDICATORS[key][1],
        "role": INDICATORS[key][2],
        "window": [start, end],
        "source": source,
        "cache_file": path.name,
        "last_updated": payload[0].get("lastupdated") if isinstance(payload[0], dict) else None,
    }

    rows: List[dict] = []
    for rec in payload[1]:
        iso3 = (rec.get("countryiso3code") or "").strip()
        if not re_fullmatch_iso3(iso3) or iso3 not in real_iso3:
            continue  # aggregate (region / income group / "World")
        if rec.get("value") is None:
            continue
        rows.append(
            {
                "country": rec["country"]["value"],
                "iso3": iso3,
                "year": int(rec["date"]),
                "value": float(rec["value"]),
            }
        )
    frame = pd.DataFrame(rows)
    logger.info(
        "Indicator %s: %d country-years, %d economies (%s)",
        indicator, len(frame), frame["iso3"].nunique(), source,
    )
    return frame, meta


def re_fullmatch_iso3(value: str) -> bool:
    """True when ``value`` is a 3-letter ISO code (aggregates are not)."""
    return len(value) == 3 and value.isalpha() and value.isupper()


def _series(frame: pd.DataFrame, wb_name: str) -> Dict[int, float]:
    sub = frame[frame["country"] == wb_name]
    return {int(r.year): float(r.value) for r in sub.itertuples()}


def _window_mean(series: Dict[int, float], years: Sequence[int]) -> Optional[float]:
    vals = [series[y] for y in years if y in series]
    if not vals:
        return None
    return float(np.mean(vals))


def _latest_years(series: Dict[int, float], n: int = 3, ceiling: int = 2023) -> List[int]:
    usable = sorted(y for y in series if y <= ceiling)
    return usable[-n:]


def _all_years(series: Dict[int, float], window: Tuple[int, int]) -> List[int]:
    return sorted(y for y in series if window[0] <= y <= window[1])


# =============================================================================
# Statistics (Spearman and its permutation null without scipy)
# =============================================================================


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation via ranks + Pearson, so no scipy import.

    ``scipy.stats`` cannot be imported in this environment (see CLAUDE.md), and
    the rank transform makes the Pearson formula exact for ties.
    """
    a = pd.Series(list(x), dtype=float).rank()
    b = pd.Series(list(y), dtype=float).rank()
    return float(np.corrcoef(a.to_numpy(), b.to_numpy())[0, 1])


def permutation_p(
    x: Sequence[float],
    y: Sequence[float],
    n_perm: int = 10000,
    seed: int = 20261008,
) -> float:
    """Two-sided permutation p-value for :func:`spearman`."""
    observed = abs(spearman(x, y))
    rng = np.random.default_rng(seed)
    yv = np.asarray(list(y), dtype=float)
    xv = np.asarray(list(x), dtype=float)
    exceed = 0
    for _ in range(n_perm):
        if abs(spearman(xv, rng.permutation(yv))) >= observed:
            exceed += 1
    return float((exceed + 1) / (n_perm + 1))


def top_set(frame: pd.DataFrame, column: str, n: int = 14, ascending: bool = False) -> List[str]:
    return list(frame.sort_values(column, ascending=ascending)["country"].head(n))


def overlap(a: Iterable[str], b: Iterable[str]) -> Tuple[int, float]:
    sa, sb = set(a), set(b)
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter, (inter / union if union else 0.0)


# =============================================================================
# Computation
# =============================================================================


def compute(config: Config, refresh: bool = False) -> Tuple[pd.DataFrame, dict]:
    """Build the per-country decomposition and the summary report."""
    deficit_path = config.paths.processed_dir / "area_pub_deficit.csv"
    if not deficit_path.exists():
        raise FileNotFoundError(
            f"{deficit_path} missing — run `pixi run python -m src.area_pub_deficit` first"
        )
    base = pd.read_csv(deficit_path, encoding="utf-8-sig").rename(
        columns={"deficit": "rrmi"}
    )
    n_countries = len(base)

    frames: Dict[str, pd.DataFrame] = {}
    metas: Dict[str, dict] = {}
    for key in INDICATORS:
        frames[key], metas[key] = load_indicator(config, key, LONG_WINDOW[0], LONG_WINDOW[1], refresh)

    def indicator_values(key: str, window: Optional[Tuple[int, int]] = None):
        """Country-level value for one indicator, plus the years it used."""
        frame = frames[key]
        values: List[Optional[float]] = []
        years_used: List[str] = []
        for country in base["country"]:
            wb_name = WB_COUNTRY_NAME.get(country, country)
            series = _series(frame, wb_name)
            if not series:
                values.append(np.nan)
                years_used.append("")
                continue
            years = _latest_years(series, n=3) if window is None else _all_years(series, window)
            value = _window_mean(series, years)
            values.append(value if value is not None else np.nan)
            years_used.append(f"{min(years)}-{max(years)}" if years else "")
        return pd.Series(values, index=base.index, dtype=float), years_used

    arts, arts_years = indicator_values("articles")
    gdp, _ = indicator_values("gdp")
    pop, _ = indicator_values("population")
    rd, _ = indicator_values("rd")
    base["articles_years"] = arts_years

    # Shares are normalized within the same 76-country table (the denominator of
    # area_share / pub_share), so the decomposition is row-exact.
    base["articles"] = arts
    base["articles_share"] = arts / arts.sum(skipna=True)
    base["gdp"] = gdp
    base["gdp_share"] = gdp / gdp.sum(skipna=True)
    base["population"] = pop
    base["population_share"] = pop / pop.sum(skipna=True)
    base["rd"] = rd
    # R&D is sparse by construction: shares are not normalized, coverage only.

    base["capacity_term"] = base["area_share"] - base["articles_share"]
    base["field_term"] = base["articles_share"] - base["pub_share"]
    base["rrmi_adj"] = base["area_share"] - base["articles_share"]
    base["rca"] = base["pub_share"] / base["articles_share"]
    base["rrmi_gdp"] = base["area_share"] - base["gdp_share"]
    base["rrmi_pop"] = base["area_share"] - base["population_share"]

    # Long-window all-field variant (every retrieved year, 1996-2023).
    long_arts, _ = indicator_values("articles", window=LONG_WINDOW)
    base["articles_long_share"] = long_arts / long_arts.sum(skipna=True)
    base["rrmi_adj_long"] = base["area_share"] - base["articles_long_share"]

    summary: dict = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_countries": int(n_countries),
        "indicators": metas,
        "windows": {"latest": list(LATEST_WINDOW), "long": list(LONG_WINDOW)},
        "coverage": {
            "articles": int(base["articles"].notna().sum()),
            "gdp": int(base["gdp"].notna().sum()),
            "population": int(base["population"].notna().sum()),
            "rd": int(base["rd"].notna().sum()),
        },
        "articles_years_used": sorted(base["articles_years"].unique().tolist()),
    }

    # Fraction of the world's all-field output the 76-country table covers.
    world = frames["articles"]
    world_total = world[world["year"].between(*LATEST_WINDOW)].groupby("iso3")["value"].mean().sum()
    summary["articles_world_share_of_table"] = float(
        base["articles"].sum(skipna=True) / world_total
    )

    # --- headline correlations and top-14 comparisons -----------------------
    have_arts = base["articles"].notna()
    top14 = top_set(base, "rrmi", 14)
    summary["n_with_articles"] = int(have_arts.sum())

    comparisons: Dict[str, dict] = {}
    for label, column in (
        ("articles_latest", "rrmi_adj"),
        ("articles_long", "rrmi_adj_long"),
        ("gdp", "rrmi_gdp"),
        ("population", "rrmi_pop"),
    ):
        mask = base[column].notna()
        s = base[mask]
        rho = spearman(s["rrmi"], s[column])
        alt_top14 = top_set(s, column, 14)
        inter, jac = overlap(top14, alt_top14)
        comparisons[label] = {
            "n": int(mask.sum()),
            "spearman_vs_rrmi": round(rho, 4),
            "perm_p": permutation_p(s["rrmi"], s[column], n_perm=2000),
            "top14_overlap": inter,
            "top14_jaccard": round(jac, 4),
            "top14_area_share": round(float(s.set_index("country").loc[alt_top14, "area_share"].sum()), 4),
            "top14_pub_share": round(float(s.set_index("country").loc[alt_top14, "pub_share"].sum()), 4),
            "top14": alt_top14,
            "top14_only_in_variant": sorted(set(alt_top14) - set(top14)),
            "top14_only_in_headline": sorted(set(top14) - set(alt_top14)),
        }
    summary["comparisons"] = comparisons

    # --- the decomposition over the RRMI top-14 ------------------------------
    t14 = base[base["country"].isin(top14)]
    summary["top14_rrmi"] = {
        "countries": top14,
        "area_share": round(float(t14["area_share"].sum()), 4),
        "pub_share": round(float(t14["pub_share"].sum()), 4),
        "capacity_term_sum": round(float(t14["capacity_term"].sum()), 4),
        "field_term_sum": round(float(t14["field_term"].sum()), 4),
        "n_capacity_positive": int((t14["capacity_term"] > 0).sum()),
        "n_field_positive": int((t14["field_term"] > 0).sum()),
        "n_rca_above_one": int((t14["rca"] > 1).sum()),
        "rca_median": round(float(t14["rca"].median()), 3),
        "rca_min": round(float(t14["rca"].min()), 3),
    }
    by_country = base.set_index("country")
    summary["top14_rrmi"]["countries_rca_below_one"] = [
        c for c in top14 if by_country.loc[c, "rca"] < 1
    ]
    summary["rca_above_one_overall"] = int((base["rca"] > 1).sum())
    summary["rca_below_one"] = (
        base.loc[base["rca"] < 1,
                 ["country", "rca", "rrmi", "area_share", "pub_share", "articles_share"]]
        .sort_values("rca").round(4).to_dict(orient="records")
    )
    summary["articles_sum"] = float(base["articles"].sum(skipna=True))

    # The Reviewer's named cases: area-rich, low-output countries.
    watch = [
        "Papua New Guinea", "Guinea", "Guinea-Bissau", "Madagascar", "Mozambique",
        "Nigeria", "Indonesia", "Brazil", "Myanmar", "Venezuela", "Senegal",
        "Sierra Leone", "Gabon", "Cambodia", "Solomon Islands",
    ]
    cols = ["country", "area", "area_share", "pub_share", "articles_share",
            "rrmi", "capacity_term", "field_term", "rca", "rrmi_adj"]
    summary["watch_list"] = (
        base[base["country"].isin(watch)][cols]
        .sort_values("rrmi", ascending=False)
        .round(4)
        .to_dict(orient="records")
    )

    return base, summary


# =============================================================================
# Reporting
# =============================================================================


def log_report(base: pd.DataFrame, summary: dict) -> None:
    """Print every number the manuscript quotes, from the computed frame."""
    logger.info("=" * 78)
    logger.info("RRMI normalization — coverage")
    logger.info("=" * 78)
    logger.info("countries in the RRMI table          : %d", summary["n_countries"])
    for key, n in summary["coverage"].items():
        logger.info("  %-11s with data                  : %d of %d",
                    key, n, summary["n_countries"])
    logger.info("all-field article years used         : %s",
                ", ".join(summary["articles_years_used"]))
    logger.info("all-field articles in the table      : %.0f (world share %.1f%%)",
                summary["articles_sum"], 100 * summary["articles_world_share_of_table"])

    logger.info("")
    logger.info("=" * 78)
    logger.info("Normative-baseline variants vs the headline RRMI")
    logger.info("=" * 78)
    logger.info("%-18s %4s %8s %10s %10s %10s",
                "denominator", "n", "rho", "top14 ovl", "Jaccard", "area% of")
    for label, c in summary["comparisons"].items():
        logger.info("%-18s %4d %+8.3f %10d %10.3f %9.1f%%",
                    label, c["n"], c["spearman_vs_rrmi"], c["top14_overlap"],
                    c["top14_jaccard"], 100 * c["top14_area_share"])
        logger.info("      swapped in: %s | swapped out: %s",
                    ", ".join(c["top14_only_in_variant"]) or "(none)",
                    ", ".join(c["top14_only_in_headline"]) or "(none)")

    t = summary["top14_rrmi"]
    logger.info("")
    logger.info("=" * 78)
    logger.info("Decomposition of the RRMI top-14 (n=%d)", len(t["countries"]))
    logger.info("=" * 78)
    logger.info("area share %.1f%% ; publication share %.1f%%",
                100 * t["area_share"], 100 * t["pub_share"])
    logger.info("summed capacity term (a-s)  : %+.4f", t["capacity_term_sum"])
    logger.info("summed field term   (s-p)   : %+.4f", t["field_term_sum"])
    logger.info("capacity term positive      : %d of 14", t["n_capacity_positive"])
    logger.info("field term positive         : %d of 14", t["n_field_positive"])
    logger.info("RCA > 1 (over-specialized)  : %d of 14 (median RCA %.2f)",
                t["n_rca_above_one"], t["rca_median"])
    logger.info("RCA range within the top-14  : %.2f to the maximum",
                t["rca_min"])
    logger.info("top-14 members with RCA < 1 : %d -> %s",
                len(t["countries_rca_below_one"]),
                ", ".join(t["countries_rca_below_one"]) or "(none)")
    logger.info("countries with RCA > 1 overall: %d of %d",
                summary["rca_above_one_overall"], summary["n_countries"])
    logger.info("countries with RCA < 1 (field-specific under-publishers):")
    for r in summary["rca_below_one"]:
        logger.info("   %-20s RCA %5.2f  RRMI %+.4f  area share %.4f",
                    r["country"], r["rca"], r["rrmi"], r["area_share"])

    logger.info("")
    logger.info("Watch list (area-rich, low-output countries):")
    logger.info("%-18s %9s %8s %8s %8s %8s %6s",
                "country", "RRMI", "a-s", "s-p", "RCA", "adj", "area")
    for r in summary["watch_list"]:
        logger.info("%-18s %+8.4f %+8.4f %+8.4f %8.2f %+8.4f %6.0f",
                    r["country"], r["rrmi"], r["capacity_term"], r["field_term"],
                    r["rca"], r["rrmi_adj"], r["area"])


# =============================================================================
# Orchestration
# =============================================================================


CSV_COLUMNS = [
    "country", "area", "pubs", "area_share", "pub_share", "rrmi",
    "articles", "articles_share", "capacity_term", "field_term", "rrmi_adj", "rca",
    "articles_long_share", "rrmi_adj_long",
    "gdp", "gdp_share", "rrmi_gdp", "population", "population_share", "rrmi_pop",
    "rd", "rank",
]


def run_rrmi_normalization(config: Config, refresh: bool = False) -> Tuple[pd.DataFrame, dict]:
    with Timer("Size-adjusted RRMI"):
        base, summary = compute(config, refresh=refresh)

    log_report(base, summary)

    csv_path = config.paths.processed_dir / "rrmi_normalization.csv"
    base[CSV_COLUMNS].to_csv(csv_path, index=False, encoding="utf-8-sig", float_format="%.6f")
    logger.info("Saved %s (%d countries)", csv_path.name, len(base))

    json_path = config.paths.processed_dir / "rrmi_normalization_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Saved %s", json_path.name)

    return base, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Size-adjusted RRMI (capacity vs field-specific deficit)")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--refresh", action="store_true", help="Re-download the World Bank payloads")
    args = parser.parse_args()

    config = get_config(args.config)
    setup_logging(config.system.log_level, config.system.log_file)
    run_rrmi_normalization(config, refresh=args.refresh)


if __name__ == "__main__":
    main()
