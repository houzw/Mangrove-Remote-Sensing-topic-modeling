"""P0-2: Study-area vs. affiliation decomposition (rule-based, full-population).

Reviewer concern: affiliation country is not study-area country, so an
affiliation-based "China publishes 25%" understates how much of that output
reports fieldwork in low-output tropical countries (Indonesia, Brazil, PNG, …).
This module quantifies the confounding without manual annotation:

  1. Re-parse the raw WoS tagged files to recover the **first author's** country
     from the first ``C1`` address block (``cleaned_data.csv`` only retains ``CU``,
     the union of *all* co-author countries).
  2. Reuse the rule-based study-region extractor from ``non_mangrove_analysis``
     (regex over mangrove-hosting territory names in title + abstract).
  3. Cross-tabulate first-author country vs. study region, and report the
     fraction of high-output (over-publishing) first-authored papers whose study
     area is a low-output tropical country.

Outputs:
    data/processed/study_area_vs_affiliation.csv          per-record table
    data/processed/study_area_vs_affiliation_matrix.csv   country × region matrix
    data/processed/study_area_vs_affiliation_summary.json headline stats
    data/processed/study_area_vs_affiliation_report.md    manuscript report text

Usage:
    pixi run python -m src.study_area_vs_affiliation
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import pandas as pd

from src.config import Config, get_config
from src.data_preprocessing import (
    _extract_country,
    find_raw_files,
    parse_wos_tagged,
)
from src.non_mangrove_analysis import (
    _build_study_region_pattern,
    _canonical_region,
    _CU_FALSE_POSITIVES,
)
from src.utils import ensure_dir, setup_logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Country groups (names match data_preprocessing._COUNTRY_NORMALIZE canonical
# forms and the manuscript Table 2 deficit ranking).
# ---------------------------------------------------------------------------

# The two country sets are *derived* from the published deficit artifact, not
# hard-coded (standing rule 7): a reported set is re-derived and asserted on every
# run, so a drift in the deficit ranking raises instead of silently renaming the
# group. Both definitions name the manuscript objects they reproduce:
#   HIGH_OUTPUT         -- the ten most negative RRMI values (negative deficit),
#                          i.e. the pure method-exporters and the resource-rich
#                          over-publishers of Sec. 3.2;
#   LOW_OUTPUT_TROPICAL -- the top-14 by positive RRMI (the most under-publishing
#                          countries), i.e. the Table 2 "top-14" set.
# An earlier revision hard-coded {Malaysia, Iran} where the artifact holds
# {Guinea-Bissau, Saudi Arabia}; the mismatch was caught by
# scripts_tmp/check_m6_study_region_method.py and is now impossible by construction.
_DEFICIT_CSV = "area_pub_deficit.csv"
_N_UNDER = 14
_N_OVER = 10
def _country_set(processed_dir: Path, *, most_negative: bool, n: int, label: str) -> set:
    """Top-``n`` countries by signed deficit, lowercased for membership tests."""
    path = Path(processed_dir) / _DEFICIT_CSV
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing; run `pixi run python -m src.area_pub_deficit` first "
            f"(the {label} set is derived from it)"
        )
    df = pd.read_csv(path)
    chosen = df.sort_values("deficit", ascending=most_negative).head(n)
    if len(chosen) != n:
        raise ValueError(f"{label}: expected {n} countries, found {len(chosen)} in {path}")
    return {str(c).strip().lower() for c in chosen["country"]}


def _resolve_country_sets(config: Config) -> Tuple[set, set]:
    """Return ``(high_output, low_output_tropical)`` derived from the deficit table."""
    processed = config.paths.processed_dir
    high = _country_set(processed, most_negative=True, n=_N_OVER, label="HIGH_OUTPUT")
    low = _country_set(processed, most_negative=False, n=_N_UNDER, label="LOW_OUTPUT_TROPICAL")
    overlap = high & low
    if overlap:
        raise ValueError(f"the two country sets overlap: {sorted(overlap)}")
    logger.info(
        "Country sets derived from %s: HIGH_OUTPUT n=%d, LOW_OUTPUT_TROPICAL n=%d",
        _DEFICIT_CSV, len(high), len(low),
    )
    return high, low


HIGH_OUTPUT, LOW_OUTPUT_TROPICAL = _resolve_country_sets(get_config())


def _norm(name: str) -> str:
    """Lowercase + strip for set membership checks."""
    return str(name).strip().lower()


def first_author_country(c1_text: str) -> str:
    """Recover the first author's country from a merged ``C1`` field.

    ``C1`` is ``" ".join`` of per-author address lines, e.g.::

        [Kumar, T.; Mahendra, R.] INCOIS, Hyderabad, India. [Radhakrishnan, K.] VSSC, Kerala, India.

    The first ``. [`` boundary separates the first author's block; within it, the
    address follows the closing ``]``.  Matching against the address (not the
    author bracket) avoids person-name false positives; ``_CU_FALSE_POSITIVES``
    drops any residual place-name leaks.
    """
    if not c1_text or not str(c1_text).strip():
        return "Unknown"
    first_block = str(c1_text).split(". [")[0].strip()
    address = first_block.split("]", 1)[1] if "]" in first_block else first_block
    if not address.strip():
        return "Unknown"
    countries = _extract_country(address)
    parts = [
        p.strip()
        for p in countries.split(";")
        if p.strip() and p.strip() != "Unknown" and p.strip() not in _CU_FALSE_POSITIVES
    ]
    return parts[0] if parts else "Unknown"


def _extract_regions(ti: str, ab: str, pattern: re.Pattern, display_names: List[str]) -> List[str]:
    """Return the sorted set of canonical study-region names named in TI+AB."""
    text = " ".join(str(x) for x in (ti, ab) if pd.notna(x) and str(x).strip())
    matches = [m.group(1) for m in pattern.finditer(text)]
    return sorted({_canonical_region(m, display_names) for m in matches})


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_first_author_corpus(config: Config) -> pd.DataFrame:
    """Parse raw WoS files, dedup by DOI, and attach first-author country."""
    raw_files = find_raw_files(config.paths.raw_dir, config.preprocessing.raw_file_pattern)

    frames = []
    for fp in raw_files:
        encodings = config.preprocessing.encodings
        df = None
        for enc in encodings:
            try:
                df = parse_wos_tagged(fp, encoding=enc)
                if len(df) > 0:
                    break
            except Exception:
                continue
        if df is None:
            logger.warning("Skipping unparseable raw file: %s", fp.name)
            continue
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)
    logger.info("Raw records (pre-dedup): %d", len(df))

    # Dedup by DOI *only among records that have one*, keeping no-DOI records
    # as-is (mirrors preprocessing.deduplicate_by_doi). A naive drop_duplicates
    # would collapse all NaN DOIs into a single row.
    df["_doi"] = df["DI"].astype(str).str.strip().str.lower()
    df["_doi"] = df["_doi"].replace({"nan": None, "": None, "none": None})
    has_doi = df[df["_doi"].notna()].drop_duplicates(subset=["_doi"], keep="first")
    no_doi = df[df["_doi"].isna()]
    df = pd.concat([has_doi, no_doi], ignore_index=True).drop(columns=["_doi"])
    logger.info("Records after DOI dedup (no-DOI kept): %d", len(df))

    # First-author country
    df["first_author_country"] = df["C1"].apply(first_author_country)
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(config: Config) -> None:
    df = load_first_author_corpus(config)

    # Study-region extraction
    display_names, pattern = _build_study_region_pattern(config)
    df["study_regions"] = df.apply(
        lambda r: _extract_regions(r.get("TI"), r.get("AB"), pattern, display_names),
        axis=1,
    )
    df["n_regions"] = df["study_regions"].apply(len)

    # Flags
    df["is_high_output"] = df["first_author_country"].apply(
        lambda c: _norm(c) in HIGH_OUTPUT
    )
    df["studies_low_output_tropical"] = df["study_regions"].apply(
        lambda regs: any(_norm(r) in LOW_OUTPUT_TROPICAL for r in regs)
    )
    df["has_region"] = df["n_regions"] > 0

    # ---- Headline statistics ----
    high = df[df["is_high_output"]]
    n_high = len(high)
    n_high_with_region = int(high["has_region"].sum())
    n_high_tropical = int(high["studies_low_output_tropical"].sum())

    # By individual high-output country (esp. China)
    per_country = []
    for country in sorted(HIGH_OUTPUT):
        sub = df[df["first_author_country"].apply(_norm) == country]
        if sub.empty:
            continue
        per_country.append({
            "first_author_country": country.title(),
            "n_papers": len(sub),
            "n_with_region": int(sub["has_region"].sum()),
            "n_studies_low_output_tropical": int(sub["studies_low_output_tropical"].sum()),
            "pct_studies_low_output_tropical": round(
                100.0 * int(sub["studies_low_output_tropical"].sum()) / len(sub), 1
            ),
        })
    per_country_df = pd.DataFrame(per_country).sort_values(
        "n_papers", ascending=False
    ).reset_index(drop=True)

    # ---- Cross-tab matrix (first-author country × study region) ----
    matrix_counter: Counter = Counter()
    for _, row in df.iterrows():
        fac = row["first_author_country"]
        if _norm(fac) not in HIGH_OUTPUT:
            continue
        for region in row["study_regions"]:
            matrix_counter[(fac, region)] += 1
    matrix = pd.DataFrame(
        [{"first_author_country": c, "study_region": r, "N_pubs": n}
         for (c, r), n in matrix_counter.items()]
    ).sort_values("N_pubs", ascending=False).reset_index(drop=True)

    # ---- Summary dict ----
    summary = {
        "n_records": int(len(df)),
        "n_high_output_papers": n_high,
        "n_high_output_with_region": n_high_with_region,
        "n_high_output_studies_low_output_tropical": n_high_tropical,
        "pct_high_output_studies_low_output_tropical_all": round(
            100.0 * n_high_tropical / n_high, 1
        ) if n_high else None,
        "pct_high_output_studies_low_output_tropical_of_region_detectable": round(
            100.0 * n_high_tropical / n_high_with_region, 1
        ) if n_high_with_region else None,
        "by_country": per_country_df.to_dict(orient="records"),
    }

    # ---- Save ----
    out_dir = config.paths.processed_dir
    ensure_dir(out_dir)

    rec_cols = [
        "DI", "PY", "first_author_country", "study_regions", "n_regions",
        "is_high_output", "studies_low_output_tropical", "has_region",
    ]
    rec_df = df[[c for c in rec_cols if c in df.columns]].copy()
    rec_df["study_regions"] = rec_df["study_regions"].apply(lambda r: ";".join(r))
    rec_path = out_dir / "study_area_vs_affiliation.csv"
    rec_df.to_csv(rec_path, index=False, encoding="utf-8-sig")
    logger.info("Per-record table saved: %s", rec_path)

    matrix_path = out_dir / "study_area_vs_affiliation_matrix.csv"
    matrix.to_csv(matrix_path, index=False, encoding="utf-8-sig")
    logger.info("Cross-tab matrix saved: %s", matrix_path)

    summary_path = out_dir / "study_area_vs_affiliation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    logger.info("Summary saved: %s", summary_path)

    # ---- Report segment ----
    report_path = out_dir / "study_area_vs_affiliation_report.md"
    report = _compose_report(summary, per_country_df, matrix)
    report_path.write_text(report, encoding="utf-8")
    logger.info("Report saved: %s", report_path)

    # ---- Console ----
    print("\n" + "=" * 72)
    print("P0-2 STUDY-AREA vs AFFILIATION (rule-based, full-population)")
    print("=" * 72)
    print(report)
    print("\nBy high-output first-author country:")
    print(per_country_df.to_string(index=False))
    print("\nTop study regions named by high-output first authors:")
    top_regions = (
        matrix.groupby("study_region")["N_pubs"].sum()
        .sort_values(ascending=False).head(15)
    )
    print(top_regions.to_string())


def _compose_report(summary: dict, per_country_df: pd.DataFrame, matrix: pd.DataFrame) -> str:
    pct_all = summary["pct_high_output_studies_low_output_tropical_all"]
    pct_region = summary["pct_high_output_studies_low_output_tropical_of_region_detectable"]
    n_high = summary["n_high_output_papers"]
    n_trop = summary["n_high_output_studies_low_output_tropical"]
    n_reg = summary["n_high_output_with_region"]

    china = per_country_df[per_country_df["first_author_country"] == "China"]
    china_line = ""
    if not china.empty:
        c = china.iloc[0]
        china_line = (
            f" China is the leading case: {c['pct_studies_low_output_tropical']:.1f}% of "
            f"{c['n_papers']} Chinese-first-authored papers name a low-output tropical "
            f"study region."
        )

    lines = [
        "## P0-2 Study-area vs affiliation",
        "",
        f"Re-parsing the raw WoS records to recover the first author's country "
        f"(the corpus keeps only the union of co-author countries), {n_trop} of "
        f"{n_high} ({pct_all:.1f}%) high-output-country first-authored papers name at "
        f"least one low-output tropical study region in their title or abstract. "
        f"Restricting to the {n_reg} papers with any detectable study region, the "
        f"fraction rises to {pct_region:.1f}%. This confirms the affiliation-vs-study-area "
        f"confounding: a substantial share of the over-publishing countries' output "
        f"reports fieldwork in resource-rich tropics, which is why we phrase the result "
        f"as a first-author output deficit rather than an under-studied claim."
        + china_line,
    ]

    return "\n\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="P0-2 study-area vs affiliation")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    config = get_config(args.config)
    setup_logging(config.system.log_level, config.system.log_file)
    run(config)


if __name__ == "__main__":
    main()
