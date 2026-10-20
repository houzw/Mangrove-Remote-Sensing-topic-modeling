"""Parent-field (remote-sensing) normalization of the research--resource mismatch.

The headline RRMI (``a_c - p_c``) compares a country's share of *mangrove*
remote-sensing publications with its share of the mangrove resource.  This
module adds the denominator the field itself supplies: the country's share of
the **parent field**, i.e. the remote-sensing literature retrieved with the same
search string and the same database, with only the mangrove block removed:

    TS=(("remote sensing" OR "earth observation" OR ... OR "RGB"))

Three properties make this the primary normalization:

* **Same database, same query structure.**  The Web of Science's English and
  indexing biases act on numerator and denominator alike, so they largely cancel
  in the ratio — which is the answer to the single-database objection.
* **Closer to the mechanism.**  Section 3.3 of the manuscript attributes the
  mismatch to analytical skill and compute infrastructure; remote-sensing
  capacity is that mediating variable, and normalizing by the parent field
  measures it directly rather than through all-science output.
* **One counting convention.**  The WoS Refine/Analyze panels return
  whole-record counts per country, so both sides of the ratio are full-count
  records rather than a fractional numerator over a full-count denominator.

Definitions, for each country c, with P_c the parent-field records and M_c the
corpus's full-count publications:

    s_c   = P_c / Sigma P        parent-field share
    m_c   = M_c / Sigma M        mangrove full-count share
    RRMI_norm_c = s_c - m_c      positive = within-discipline under-attention
    A_c   = m_c / s_c            RCA-style ratio, >1 = over-attention

``RRMI_norm`` is built like the headline ``a_c - p_c`` (benchmark minus the
country's own output share), with the parent field in place of the resource:
positive means the country's remote-sensing weight exceeds its weight in this
field, so the field is under-attended given the capacity it already has.

Both shares are reported on two bases. The **world** basis divides by the parent
field's total over every country in the export and by the corpus's full count
over every country, which is how a share of global output is normally quoted and
what ``A``(China) means in the review. The **76-country** basis re-normalizes
both shares to the mangrove-holding countries of the RRMI table, the denominator
convention of the published index. The world basis is primary; the other is the
check, and the two can put a country on opposite sides of A = 1.

Input
-----
``data/external/wos_parent_field/`` holds the country counts exported from the
parent-field search (any of ``.csv``/``.tsv``/``.txt``/``.xlsx``; see
``docs/review_round_*_rrmi_parent_field.md`` for the export recipe).  The reader
is tolerant: it sniffs the delimiter, finds the country and count columns by
name or position, understands the ``COUNTRY (1234)`` paste format, and sums
across year rows when a year column is present.  A missing input raises, because
the parent field is the primary denominator of the revised analysis.

Outputs
-------
``data/processed/rrmi_parent_field.csv``            per-country parent-field table
``data/processed/rrmi_parent_field_summary.json``   headline numbers + provenance
``manuscript/els-cas-templates/rrmi_parent_field_table.tex``  Tables S7--S8

Usage
-----
    pixi run python -m src.rrmi_parent_field
    pixi run python -m src.rrmi_parent_field --input path/to/counts.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.area_pub_deficit import _fractional_country_pubs, _load_cleaned
from src.config import Config, get_config
from src.generate_supplementary_tables import comma, signed, tex
from src.rrmi_normalization import permutation_p, spearman, top_set
from src.utils import Timer, ensure_dir, setup_logging

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT_DIR = PROJECT_ROOT / "manuscript" / "els-cas-templates"

#: Where the WoS parent-field export lives, and the file stems accepted.
PARENT_FIELD_DIRNAME = "wos_parent_field"
PARENT_FIELD_STEMS: Tuple[str, ...] = (
    "parent_field_countries", "rs_countries", "remote_sensing_countries", "parent_field",
)
PARENT_FIELD_SUFFIXES: Tuple[str, ...] = (".csv", ".tsv", ".txt", ".xlsx", ".xls")

#: WoS country spellings -> the spellings used by ``area_pub_deficit.csv``.
#: WoS separates the United Kingdom's constituent countries and Taiwan, which the
#: corpus folds (Taiwan into China, as in ``area_pub_deficit._PUB_ONLY_FOLD``).
WOS_NAME_ALIAS: Dict[str, str] = {
    "PEOPLES R CHINA": "China",
    "P R CHINA": "China",
    "CHINA": "China",
    "TAIWAN": "China",
    "USA": "United States",
    "UNITED STATES OF AMERICA": "United States",
    "ENGLAND": "United Kingdom",
    "SCOTLAND": "United Kingdom",
    "WALES": "United Kingdom",
    "NORTH IRELAND": "United Kingdom",
    "U K": "United Kingdom",
    "IRAN": "Iran",
    "EGYPT": "Egypt",
    "VIETNAM": "Vietnam",
    "VENEZUELA": "Venezuela",
    "BAHAMAS": "Bahamas",
    "GAMBIA": "Gambia",
    "BRUNEI": "Brunei",
    "MICRONESIA": "Micronesia",
    "TRINIDAD TOBAGO": "Trinidad & Tobago",
    "TRINIDAD AND TOBAGO": "Trinidad & Tobago",
    "COTE D IVOIRE": "Cote d'Ivoire",
    "IVORY COAST": "Cote d'Ivoire",
    "CONGO": "Congo",
    "REP CONGO": "Congo",
    "CONGO REP": "Congo",
    # Territory folds, identical to area_pub_deficit._TERRITORY_TO_METROPOLE and
    # _PUB_ONLY_FOLD, so the parent side aggregates exactly like the corpus side.
    "HONG KONG SAR": "China",
    "HONG KONG": "China",
    "MACAO": "China",
    "FRENCH GUIANA": "France",
    "GUADELOUPE": "France",
    "MARTINIQUE": "France",
    "MAYOTTE": "France",
    "NEW CALEDONIA": "France",
    "WALLIS FUTUNA": "France",
    "REUNION": "France",
    "NETH ANTILLES": "Netherlands",
    "NETHERLANDS ANTILLES": "Netherlands",
    "ARUBA": "Netherlands",
    "PUERTO RICO": "United States",
    "AMERICAN SAMOA": "United States",
    "VIRGIN IS": "United States",
    "BRITISH VIRGIN IS": "United Kingdom",
    "CAYMAN IS": "United Kingdom",
    "TURKS CAICOS IS": "United Kingdom",
}

_COUNTRY_COLUMNS = (
    "countries/regions", "country/region", "countries", "country", "region", "name", "field",
)
_COUNT_COLUMNS = (
    "record count", "records", "rec count", "count", "n", "articles", "publications",
    "documents", "records count", "no. of records",
)
_YEAR_COLUMNS = ("year", "publication year", "py", "years")


# =============================================================================
# Input discovery and tolerant parsing
# =============================================================================


def find_input(config: Config, override: Optional[Path] = None) -> Path:
    """Locate the parent-field country-counts export, or fail loudly."""
    if override is not None:
        if not Path(override).exists():
            raise FileNotFoundError(f"--input file not found: {override}")
        return Path(override)

    directory = config.paths.external_dir / PARENT_FIELD_DIRNAME
    candidates: List[Path] = []
    if directory.is_dir():
        for path in sorted(directory.iterdir()):
            if path.suffix.lower() in PARENT_FIELD_SUFFIXES and not path.name.startswith("."):
                candidates.append(path)
    # Also accept the export as supplied by the author (``data/refine.xlsx``).
    data_dir = config.paths.external_dir.parent
    for name in ("refine.xlsx", "refine.csv", "refine.txt", "refine.tsv"):
        candidate = data_dir / name
        if candidate.exists():
            candidates.append(candidate)

    if not candidates:
        raise FileNotFoundError(
            f"No parent-field country counts found in {directory}. "
            "Export the countries/regions counts of the parent-field search "
            "(TS with the remote-sensing terms only) as CSV/TSV/XLSX and save it as "
            f"{directory / 'parent_field_countries.csv'}. "
            "Expected columns: a country column and a record-count column "
            "(an optional year column is summed over)."
        )
    preferred = [
        p for p in candidates
        if p.stem.lower() in PARENT_FIELD_STEMS or any(s in p.stem.lower() for s in PARENT_FIELD_STEMS)
    ]
    chosen = (preferred or candidates)[0]
    if len(preferred) > 1:
        logger.warning("Several parent-field files found; using %s", chosen.name)
    return chosen


def _read_table(path: Path) -> pd.DataFrame:
    """Read CSV/TSV/TXT/XLSX without assuming a delimiter or a header spelling."""
    if path.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(path)
    last_error: Optional[Exception] = None
    for sep in (None, "\t", ",", ";"):
        try:
            frame = pd.read_csv(
                path, sep=sep, engine="python", encoding="utf-8-sig",
                skip_blank_lines=True, comment=None,
            )
        except Exception as exc:  # noqa: BLE001 - try the next delimiter
            last_error = exc
            continue
        if frame.shape[1] >= 2:
            return frame
    if last_error is not None:
        raise RuntimeError(f"Could not parse {path}: {last_error}")
    raise RuntimeError(f"Could not parse {path}: fewer than two columns")


def _pick_column(frame: pd.DataFrame, names: Sequence[str], numeric: bool) -> Optional[str]:
    lower = {str(c).strip().lower(): c for c in frame.columns}
    for name in names:
        if name in lower:
            return lower[name]
    for column in frame.columns:
        if pd.api.types.is_numeric_dtype(frame[column]) == numeric:
            return column
    return None


def parse_counts(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize a raw export into ``country_raw, count, year`` (long form).

    Handles three shapes: a proper two-column export, a year-by-country matrix,
    and the ``COUNTRY (1234)`` paste from the Refine panel (single text column).
    """
    frame = frame.dropna(how="all")
    country_col = _pick_column(frame, _COUNTRY_COLUMNS, numeric=False)
    count_col = _pick_column(frame, _COUNT_COLUMNS, numeric=True)
    year_col = None
    lower = {str(c).strip().lower(): c for c in frame.columns}
    for name in _YEAR_COLUMNS:
        if name in lower:
            year_col = lower[name]
            break

    # Refine-panel paste: one text column of "COUNTRY (1234)" (rank bars dropped).
    if country_col is None:
        non_numeric = [c for c in frame.columns if not pd.api.types.is_numeric_dtype(frame[c])]
        country_col = non_numeric[0] if non_numeric else frame.columns[0]

    # Refine-panel paste: one text column of "COUNTRY (1234)", rank bars dropped.
    if count_col is None or frame.shape[1] == 1:
        text = frame[frame.columns[0]].astype(str)
        parsed = text.str.extract(r"^(?P<country>.+?)\s*\(?\s*(?P<count>[\d,]+)\s*\)?\s*$")
        parsed = parsed.dropna()
        parsed["count"] = parsed["count"].str.replace(",", "", regex=False).astype(float)
        return pd.DataFrame({
            "country_raw": parsed["country"].str.strip(),
            "count": parsed["count"],
            "year": np.nan,
        })

    out = pd.DataFrame({
        "country_raw": frame[country_col].astype(str).str.strip(),
        "count": pd.to_numeric(
            frame[count_col].astype(str).str.replace(",", "", regex=False).str.replace("%", "", regex=False),
            errors="coerce",
        ),
        "year": pd.to_numeric(frame[year_col], errors="coerce") if year_col is not None else np.nan,
    })
    out = out.dropna(subset=["count"])
    out = out[out["country_raw"].str.len() > 0]
    # Drop the header/legend rows a WoS export repeats ("Field: ...", "Record Count").
    out = out[~out["country_raw"].str.lower().str.startswith(("field", "record", "total", "%"))]
    return out.reset_index(drop=True)


def _canonical(raw: str, known: Dict[str, str]) -> Optional[str]:
    """Map a WoS country spelling onto a name in the 76-country table."""
    key = re.sub(r"[^A-Z0-9 ]+", " ", raw.upper()).strip()
    key = re.sub(r"\s+", " ", key)
    if key in WOS_NAME_ALIAS:
        return WOS_NAME_ALIAS[key]
    if key in known:
        return known[key]
    # No approximate matching. A unique-containment fallback once mapped ROMANIA
    # onto OMAN and absorbed Romania's records into Oman's parent-field share, so
    # an unrecognised spelling stays unmapped and is logged with its count.
    return None


# =============================================================================
# Computation
# =============================================================================


def compute_parent_field(
    config: Config,
    path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, dict]:
    """Build the per-country parent-field table and the summary report."""
    deficit_path = config.paths.processed_dir / "area_pub_deficit.csv"
    if not deficit_path.exists():
        raise FileNotFoundError(
            f"{deficit_path} missing - run `pixi run python -m src.area_pub_deficit` first"
        )
    base = pd.read_csv(deficit_path, encoding="utf-8-sig").rename(columns={"deficit": "rrmi"})

    pubs = _fractional_country_pubs(_load_cleaned(config))
    base = base.merge(
        pubs[["Country", "Pubs_int"]].rename(columns={"Country": "country"}),
        on="country", how="left",
    )
    if base["Pubs_int"].isna().any():
        missing = base.loc[base["Pubs_int"].isna(), "country"].tolist()
        raise RuntimeError(f"full-count publications missing for {missing}")

    input_path = find_input(config, path)
    raw = _read_table(input_path)
    counts = parse_counts(raw)
    # The parent field's own total, over every country in the export: this is the
    # denominator of the world-basis share, so it is taken before any row is
    # dropped for falling outside the 76-country table.
    p_world = float(counts["count"].sum())
    known = {re.sub(r"[^A-Z0-9 ]+", " ", c.upper()).strip(): c for c in base["country"]}
    counts["country"] = counts["country_raw"].map(lambda v: _canonical(v, known))

    unmapped = counts[counts["country"].isna()]
    if len(unmapped):
        # Expected: these are countries with no row in the HGMF mangrove-area
        # table (Germany, Italy, Canada ...), which the RRMI table does not hold.
        # Printed in full for the largest ones because a *wrong* alias would land
        # here too: the containment fallback that mapped ROMANIA onto OMAN showed
        # up in exactly this list.
        logger.info(
            "Parent-field countries outside the 76-country table (no mangrove-area row): "
            "%d of %d rows, %.1f%% of the parent field. Largest: %s",
            len(unmapped), len(counts), 100 * unmapped["count"].sum() / counts["count"].sum(),
            ", ".join(f"{r.country_raw}={r.count:,.0f}" for r in unmapped.head(12).itertuples()),
        )
    counts = counts.dropna(subset=["country"])
    if counts.empty:
        raise RuntimeError(
            f"No parent-field row in {input_path.name} matched a country in the RRMI table; "
            "check the country spellings and the column order."
        )

    frame = base.merge(
        counts.groupby("country", as_index=False)["count"].sum().rename(columns={"count": "P_c"}),
        on="country", how="left",
    ).rename(columns={"Pubs_int": "M_c"})
    years = sorted(int(y) for y in counts["year"].dropna().unique())

    # --- the two bases ------------------------------------------------------
    m_world = float(pubs["Pubs_int"].sum())      # corpus full count, every country
    p_table = float(frame["P_c"].sum(skipna=True))
    m_table = float(frame["M_c"].sum())

    frame["s_c"] = frame["P_c"] / p_world
    frame["m_c"] = frame["M_c"] / m_world
    frame["s_c_76"] = frame["P_c"] / p_table
    frame["m_c_76"] = frame["M_c"] / m_table
    frame["rrmi_norm"] = frame["s_c"] - frame["m_c"]
    frame["A"] = frame["m_c"] / frame["s_c"]
    frame["rrmi_norm_76"] = frame["s_c_76"] - frame["m_c_76"]
    frame["A_76"] = frame["m_c_76"] / frame["s_c_76"]
    frame["rrmi_full"] = frame["area_share"] - frame["m_c_76"]

    # --- comparisons against the published index ----------------------------
    have = frame["P_c"].notna() & (frame["P_c"] > 0)
    sub = frame[have]
    top14_pub = top_set(base, "rrmi", 14)
    top14_norm = top_set(sub, "rrmi_norm", 14)
    top14_norm76 = top_set(sub, "rrmi_norm_76", 14)
    inter, jac = _overlap(top14_pub, top14_norm)
    inter76, jac76 = _overlap(top14_pub, top14_norm76)
    same_sign = np.sign(sub["rrmi_norm"]) == np.sign(sub["rrmi"])
    t14 = frame[frame["country"].isin(top14_pub)]
    bottom10 = list(base.sort_values("rrmi")["country"].head(10))
    bottom = frame[frame["country"].isin(bottom10)].sort_values("rrmi")

    def _row(country: str) -> dict:
        r = frame.set_index("country").loc[country]
        return {
            "country": country,
            "P_c": float(r["P_c"]),
            "M_c": float(r["M_c"]),
            "s_c": round(float(r["s_c"]), 5),
            "m_c": round(float(r["m_c"]), 5),
            "rrmi_norm": round(float(r["rrmi_norm"]), 5),
            "A": round(float(r["A"]), 3),
            "s_c_76": round(float(r["s_c_76"]), 5),
            "m_c_76": round(float(r["m_c_76"]), 5),
            "rrmi_norm_76": round(float(r["rrmi_norm_76"]), 5),
            "A_76": round(float(r["A_76"]), 3),
            "rrmi_published": round(float(r["rrmi"]), 5),
            "area_share": round(float(r["area_share"]), 5),
        }

    summary: dict = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "input_file": input_path.name,
        "input_rows_mapped": int(len(counts)),
        "n_countries": int(len(frame)),
        "n_with_parent_records": int(have.sum()),
        "parent_records_all_countries": p_world,
        "parent_records_in_table": p_table,
        "parent_records_table_share": round(p_table / p_world, 4),
        "corpus_full_count_all_countries": m_world,
        "corpus_full_count_in_table": m_table,
        "years_in_input": years,
        "convention": "full-count records on both sides (WoS Refine/Analyze counts)",
        "bases": {
            "primary": "world: shares of the parent field's and the corpus's totals over all countries",
            "check": "76-country table: both shares re-normalized to the RRMI table",
        },
        "comparison": {
            "n": int(have.sum()),
            "top14_overlap": inter,
            "top14_jaccard": round(jac, 4),
            "top14_overlap_76": inter76,
            "top14_jaccard_76": round(jac76, 4),
            "top14_by_norm": top14_norm,
            "top14_only_in_variant": sorted(set(top14_norm) - set(top14_pub)),
            "top14_only_in_headline": sorted(set(top14_pub) - set(top14_norm)),
            "spearman_rrmi_vs_rrmi_norm": round(spearman(sub["rrmi"], sub["rrmi_norm"]), 4),
            "spearman_rrmi_vs_A": round(spearman(sub["rrmi"], sub["A"]), 4),
            "perm_p": permutation_p(sub["rrmi"], sub["rrmi_norm"], n_perm=2000),
            "sign_agreement_n": int(same_sign.sum()),
            "sign_agreement_frac": round(float(same_sign.mean()), 4),
            "sign_discordant": sorted(sub.loc[~same_sign, "country"]),
            "spearman_A_world_vs_76": round(spearman(sub["A"], sub["A_76"]), 4),
            "n_A_above_one": int((sub["A"] > 1).sum()),
            "A_median": round(float(sub["A"].median()), 3),
        },
        "top14_published": {
            "countries": top14_pub,
            "area_share": round(float(t14["area_share"].sum()), 4),
            "m_c_mean_basis_sum": round(float(t14["m_c"].sum()), 4),
            "rrmi_norm_positive": int((t14["rrmi_norm"] > 0).sum()),
            "A_above_one": int((t14["A"] > 1).sum()),
            "A_median": round(float(t14["A"].median()), 3),
            "A_min": round(float(t14["A"].min()), 3),
            "A_max": round(float(t14["A"].max()), 3),
        },
        "china": _row("China"),
        "over_publishers": [_row(c) for c in bottom["country"]],
        "watch_list": [_row(c) for c in (
            "Papua New Guinea", "Guinea", "Guinea-Bissau", "Madagascar", "Mozambique",
            "Nigeria", "Indonesia", "Brazil", "Myanmar", "Venezuela", "Mexico", "Cuba",
            "Bangladesh", "Colombia", "India", "United States", "United Kingdom", "Japan",
            "France", "Netherlands", "Singapore", "Saudi Arabia", "Vietnam",
        )],
    }
    return frame, summary


def _overlap(a: Sequence[str], b: Sequence[str]) -> Tuple[int, float]:
    sa, sb = set(a), set(b)
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter, (inter / union if union else 0.0)


# =============================================================================
# Reporting
# =============================================================================


def log_report(frame: pd.DataFrame, summary: dict) -> None:
    c = summary["comparison"]
    t = summary["top14_published"]
    logger.info("=" * 78)
    logger.info("Parent-field normalization - %s", summary["input_file"])
    logger.info("=" * 78)
    logger.info("countries mapped to the RRMI table : %d (%d with >=1 parent record)",
                summary["n_countries"], summary["n_with_parent_records"])
    logger.info("parent field, all countries        : %.0f records",
                summary["parent_records_all_countries"])
    logger.info("   held by the 76-country table    : %.0f (%.1f%%)",
                summary["parent_records_in_table"], 100 * summary["parent_records_table_share"])
    logger.info("corpus full count, all countries   : %.0f ; in the table %.0f",
                summary["corpus_full_count_all_countries"], summary["corpus_full_count_in_table"])
    logger.info("years in the input                 : %s",
                ", ".join(str(y) for y in summary["years_in_input"]) or "(none)")
    logger.info("convention                         : %s", summary["convention"])

    logger.info("")
    logger.info("1) TOP-14 OVERLAP (world basis)")
    logger.info("   shared with the published RRMI top-14 : %d of 14 (Jaccard %.3f)",
                c["top14_overlap"], c["top14_jaccard"])
    logger.info("   76-country basis                      : %d of 14 (Jaccard %.3f)",
                c["top14_overlap_76"], c["top14_jaccard_76"])
    logger.info("   swapped in  : %s", ", ".join(c["top14_only_in_variant"]) or "(none)")
    logger.info("   swapped out : %s", ", ".join(c["top14_only_in_headline"]) or "(none)")
    logger.info("   rho(published RRMI, RRMI_norm) %.3f (perm p %.4f); rho vs A %.3f",
                c["spearman_rrmi_vs_rrmi_norm"], c["perm_p"], c["spearman_rrmi_vs_A"])

    logger.info("")
    logger.info("2) SIGN CONSISTENCY")
    logger.info("   same sign as the published RRMI : %d of %d (%.0f%%)",
                c["sign_agreement_n"], c["n"], 100 * c["sign_agreement_frac"])
    logger.info("   discordant                      : %s",
                ", ".join(c["sign_discordant"]) or "(none)")
    logger.info("   published top-14 with RRMI_norm>0: %d of 14 ; with A>1: %d of 14",
                t["rrmi_norm_positive"], t["A_above_one"])

    logger.info("")
    logger.info("3) A (m_c / s_c), the RCA-style ratio")
    logger.info("   A(China) = %.2f world basis, %.2f on the 76-country basis "
                "(M=%.0f, P=%.0f)", summary["china"]["A"], summary["china"]["A_76"],
                summary["china"]["M_c"], summary["china"]["P_c"])
    logger.info("   A > 1 in %d of %d countries ; median A %.2f ; rho(A_world, A_76) %.3f",
                c["n_A_above_one"], c["n"], c["A_median"], c["spearman_A_world_vs_76"])
    logger.info("   published top-14: median A %.2f (%.2f-%.2f), %d of 14 above 1",
                t["A_median"], t["A_min"], t["A_max"], t["A_above_one"])
    logger.info("")
    logger.info("   the ten most negative published RRMI values:")
    logger.info("      %-18s %8s %8s %8s %8s %8s %9s", "country", "RRMI", "s_c", "m_c",
                "rrmi_norm", "A", "A(76)")
    for r in summary["over_publishers"]:
        logger.info("      %-18s %+8.4f %8.5f %8.5f %+8.5f %8.2f %9.2f",
                    r["country"], r["rrmi_published"], r["s_c"], r["m_c"],
                    r["rrmi_norm"], r["A"], r["A_76"])
    logger.info("")
    logger.info("   the published top-14:")
    logger.info("      %-18s %8s %8s %8s %8s %8s %9s", "country", "RRMI", "s_c", "m_c",
                "rrmi_norm", "A", "A(76)")
    by = frame.set_index("country")
    for name in summary["top14_published"]["countries"]:
        r = by.loc[name]
        logger.info("      %-18s %+8.4f %8.5f %8.5f %+8.5f %8.2f %9.2f",
                    name, r["rrmi"], r["s_c"], r["m_c"], r["rrmi_norm"], r["A"], r["A_76"])


# =============================================================================
# LaTeX tables
# =============================================================================


def _fmt_ratio(v: float) -> str:
    if pd.isna(v):
        return "n/a"
    if v >= 100:
        return f"{v:,.0f}".replace(",", "{,}")
    return f"{v:.2f}"


def build_table_s7(frame: pd.DataFrame, summary: dict) -> List[str]:
    rows = frame.sort_values("rrmi_norm", ascending=False)
    cap = (
        r"\caption{Per-country mismatch against the parent field (the remote-sensing literature "
        r"retrieved with the same search string and database, mangrove block removed) for the 76 "
        r"countries with both publication and mangrove-area data. $P_c$ is the country's parent-field "
        r"record count and $M_c$ its full-count publications in this corpus; $s_c=P_c/\sum P$ and "
        r"$m_c=M_c/\sum M$ are the two shares, both on the world basis (every country in the export "
        r"for $s_c$, every country in the corpus for $m_c$: the parent field holds "
        + f"{summary['parent_records_all_countries']:,.0f}".replace(",", "{,}")
        + r" records in total, of which the 76-country table supplies "
        + f"{100 * summary['parent_records_table_share']:.1f}"
        + r"\%), and $a_c$ the mangrove-area share of Table~\ref{tab:S3}. "
        r"$\mathrm{RRMI}_{norm}=s_c-m_c$ is positive when the country's remote-sensing weight "
        r"exceeds its weight in this field, that is when the field is under-attended given the "
        r"capacity the country already has; $A=m_c/s_c$ is the same statement as a ratio, and is "
        r"above one when the field is over-attended. Both are record counts, so the ratio carries "
        r"one counting convention. Caption continues in Table~\ref{tab:S8} for the 76-country basis "
        r"and the external denominators.}"
    )
    L = [
        "% Table S7: parent-field (remote-sensing) normalization "
        "(auto-generated by src/rrmi_parent_field.py)",
        r"\begingroup\footnotesize",
        r"\setlength{\tabcolsep}{4pt}%",
        r"\setlength{\LTleft}{0pt}%",
        r"\setlength{\LTright}{0pt}%",
        r"\setlength{\LTcapwidth}{\textwidth}%",
        r"\begin{longtable}{@{\extracolsep{\fill}}l r r r r r r r@{}}",
        cap,
        r"\label{tab:S7}\\",
        r"\toprule",
        r"Country & $a_c$ & $M_c$ & $P_c$ & $m_c$ & $s_c$ & $\mathrm{RRMI}_{norm}$ & $A$ \\",
        r"\midrule",
        r"\endfirsthead",
        r"\multicolumn{8}{l}{\emph{Table~\ref{tab:S7} (continued)}}\\",
        r"\toprule",
        r"Country & $a_c$ & $M_c$ & $P_c$ & $m_c$ & $s_c$ & $\mathrm{RRMI}_{norm}$ & $A$ \\",
        r"\midrule",
        r"\endhead",
    ]
    for r in rows.itertuples():
        p = "n/a" if pd.isna(r.P_c) else comma(float(r.P_c), 0)
        s = "n/a" if pd.isna(r.s_c) else f"{r.s_c:.5f}"
        norm = "n/a" if pd.isna(r.rrmi_norm) else signed(float(r.rrmi_norm), 5)
        L.append(
            f"{tex(r.country)} & {r.area_share:.4f} & {comma(float(r.M_c), 0)} & {p} & "
            f"{r.m_c:.5f} & {s} & {norm} & {_fmt_ratio(float(r.A))} \\\\"
        )
    L += [r"\bottomrule", r"\end{longtable}", r"\endgroup"]
    return L


def build_table_s8(config: Config, summary: dict) -> List[str]:
    c = summary["comparison"]
    pf = pd.read_csv(config.paths.processed_dir / "rrmi_parent_field.csv",
                     encoding="utf-8-sig")
    # The 76-country row re-normalizes both shares to the mangrove-holding
    # countries of the RRMI table.  Seven of those countries have no
    # parent-field record, so their rrmi_norm_76 is NaN.  Correlate on the
    # countries that do have one -- the same subset the world-basis row uses --
    # rather than handing NaN to np.corrcoef, which returns nan for the whole
    # column.  The sign agreement is computed on that subset for the same
    # reason (it was previously left blank).
    sub76 = pf[pf["P_c"].notna() & (pf["P_c"] > 0)]
    rho76 = round(spearman(sub76["rrmi"].values, sub76["rrmi_norm_76"].values), 4)
    sign76 = round(float((np.sign(sub76["rrmi_norm_76"]) == np.sign(sub76["rrmi"])).mean()), 4)
    rows: List[Tuple[str, int, float, int, float, float]] = [
        ("Parent field, world basis (primary)", c["n"], c["spearman_rrmi_vs_rrmi_norm"],
         c["top14_overlap"], c["top14_jaccard"], c["sign_agreement_frac"]),
        ("Parent field, 76-country basis", int(len(sub76)), rho76,
         c["top14_overlap_76"], c["top14_jaccard_76"], sign76),
    ]
    normal_path = config.paths.processed_dir / "rrmi_normalization.csv"
    rd_note = ""
    if normal_path.exists():
        norm = pd.read_csv(normal_path, encoding="utf-8-sig")
        meta_path = config.paths.processed_dir / "rrmi_normalization_summary.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        pub_top14 = top_set(norm, "rrmi", 14)
        for column, label in (
            ("rrmi_adj", "All-field articles, 2021--2023 (area-relative)"),
            ("rrmi_gdp", "GDP (area-relative)"),
            ("rrmi_pop", "Population (area-relative)"),
        ):
            s = norm[norm[column].notna()]
            top = top_set(s, column, 14)
            inter, jac = _overlap(pub_top14, top)
            sign = float((np.sign(s[column]) == np.sign(s["rrmi"])).mean())
            rows.append((label, int(len(s)), round(spearman(s["rrmi"], s[column]), 4),
                         inter, round(jac, 4), round(sign, 4)))
        if meta:
            rd_note = (
                " R\\&D expenditure is not used as a denominator: the World Bank series is "
                f"populated for only {meta['coverage']['rd']} of {meta['n_countries']} countries "
                "over the reference window, so it cannot support a ranking comparison."
            )
    L = [
        "% Table S8: normalization agreement with the headline RRMI",
        r"\begin{table}[H]",
        (
            r"\caption{Agreement between the published (fractional, area-relative) RRMI and the "
            r"normalizations of Table~\ref{tab:S7}. $\rho$ is the Spearman rank correlation over "
            r"the countries with data; ``top-14 overlap'' is the number of members shared with the "
            r"14 largest positive RRMI values, ``Jaccard'' the set overlap, and ``sign'' the share "
            r"of countries placed on the same side of zero by both indicators. The parent-field "
            r"rows use $\mathrm{RRMI}_{norm}=s_c-m_c$ and the 76-country rows re-normalize both "
            r"shares to the mangrove-holding countries of the RRMI table; the remaining rows "
            r"replace the mangrove-area benchmark instead, so their variant index is $a_c-s_c$ for "
            r"that denominator." + rd_note + "}"
        ),
        r"\label{tab:S8}",
        r"\begin{tabular*}{\tblwidth}{@{\extracolsep{\fill}}l r r r r r@{}}",
        r"\toprule",
        r"Normalization & $n$ & $\rho$ vs RRMI & Top-14 overlap & Jaccard & Sign \\",
        r"\midrule",
    ]
    for i, (label, n, rho, inter, jac, sign) in enumerate(rows):
        if i == 2:
            L.append(r"\midrule")
        sign_txt = "n/a" if pd.isna(sign) else f"{100 * sign:.0f}\\%"
        L.append(f"{label} & {n} & {signed(rho, 3)} & {inter} & {jac:.2f} & {sign_txt} \\\\")
    L += [r"\bottomrule", r"\end{tabular*}", r"\end{table}"]
    return L


# =============================================================================
# Orchestration
# =============================================================================


CSV_COLUMNS = [
    "country", "area", "area_share", "pubs", "pub_share", "rrmi", "rank",
    "M_c", "m_c", "m_c_76", "P_c", "s_c", "s_c_76",
    "rrmi_norm", "A", "rrmi_norm_76", "A_76", "rrmi_full",
]


def run_parent_field(config: Config, path: Optional[Path] = None) -> Tuple[pd.DataFrame, dict]:
    with Timer("Parent-field RRMI normalization"):
        frame, summary = compute_parent_field(config, path=path)

    log_report(frame, summary)

    csv_path = config.paths.processed_dir / "rrmi_parent_field.csv"
    frame[CSV_COLUMNS].to_csv(csv_path, index=False, encoding="utf-8-sig", float_format="%.6f")
    logger.info("Saved %s (%d countries)", csv_path.name, len(frame))

    json_path = config.paths.processed_dir / "rrmi_parent_field_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Saved %s", json_path.name)

    tex_path = MANUSCRIPT_DIR / "rrmi_parent_field_table.tex"
    parts = [
        "% Auto-generated by src/rrmi_parent_field.py -- do not edit by hand.",
        "% Tables S7--S8. \\input at the END of manuscript-v3-appendix.tex, after the",
        "% study-region table (S6), so the S1--S6 numbering is not disturbed.",
        "",
    ]
    parts += build_table_s7(frame, summary)
    parts.append("")
    parts += build_table_s8(config, summary)
    parts.append("")
    ensure_dir(tex_path.parent)
    tex_path.write_text("\n".join(parts), encoding="utf-8")
    logger.info("Saved %s", tex_path.name)
    return frame, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parent-field (remote-sensing) normalization of the RRMI"
    )
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--input", type=str, default=None,
                        help="Country-counts file (default: data/external/wos_parent_field/)")
    args = parser.parse_args()

    config = get_config(args.config)
    setup_logging(config.system.log_level, config.system.log_file)
    run_parent_field(config, path=Path(args.input) if args.input else None)


if __name__ == "__main__":
    main()
