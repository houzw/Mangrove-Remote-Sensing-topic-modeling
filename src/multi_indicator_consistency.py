"""#2-C: Multi-indicator gap consistency (deficit + pub density + citation gap).

Cross-checks the research--resource deficit against two alternative gap
metrics to show the finding is not an artefact of any single denominator:

1. ``deficit``        = area_share - pub_share        (already in area_pub_deficit.csv)
2. ``pub_density``    = pubs / area                   (area-weighted output)
3. ``citation_gap``   = area_share - citation_share   (citation-side gap)

``citation_share_c`` uses fractional counting of WoS times-cited (TC):
C_c = sum_d w_{d,c} * TC_d. Pairwise Spearman rank correlations quantify
whether the three indicators agree on under/over-research ordering.

Outputs:
    data/processed/multi_indicator_consistency.csv   (Supplementary Table S2)
    data/processed/multi_indicator_consistency_summary.json

Usage:
    python src/multi_indicator_consistency.py
"""

from __future__ import annotations

import json
import logging
from collections import Counter

import pandas as pd

from src.area_pub_deficit import (
    _CU_FALSE_POSITIVES,
    _PUB_ONLY_FOLD,
    _TERRITORY_TO_METROPOLE,
    _load_area,
)
from src.config import Config, get_config
from src.utils import Timer, canonical_countries

logger = logging.getLogger(__name__)


def _fractional_country_citations(cleaned: pd.DataFrame) -> pd.DataFrame:
    """Fractional-count citations per country (C_c = sum_d w_{d,c} * TC_d)."""
    frac: Counter = Counter()
    for _, row in cleaned[["CU", "TC"]].iterrows():
        cu = str(row["CU"]) if pd.notna(row["CU"]) else "Unknown"
        tc = row["TC"] if pd.notna(row["TC"]) else 0
        # Fold and de-duplicate (see src.utils.canonical_countries) so the
        # citation-side share uses the same country space as the deficit.
        countries = canonical_countries(
            cu,
            fold=(_TERRITORY_TO_METROPOLE, _PUB_ONLY_FOLD),
            drop=_CU_FALSE_POSITIVES,
        )
        if not countries:
            continue
        w = 1.0 / len(countries)
        for c in countries:
            frac[c] += w * float(tc)
    return pd.DataFrame({"Country": list(frac.keys()), "Citations": list(frac.values())})


def run_multi_indicator_consistency(config: Config) -> pd.DataFrame:
    paths = config.paths

    # Load cleaned data with citations.
    csv = paths.processed_dir / config.preprocessing.output_file
    cleaned = pd.read_csv(csv, encoding="utf-8-sig", usecols=["CU", "TC"])
    cleaned["TC"] = pd.to_numeric(cleaned["TC"], errors="coerce").fillna(0)

    # Load deficit table (already has area, pubs, shares, deficit).
    deficit = pd.read_csv(paths.processed_dir / "area_pub_deficit.csv", encoding="utf-8-sig")
    deficit = deficit.rename(columns={"country": "Country"})

    with Timer("Fractional citation counting"):
        citations = _fractional_country_citations(cleaned)

    merged = deficit.merge(citations, on="Country", how="inner")
    merged["pub_density"] = merged["pubs"] / merged["area"]
    total_citations = merged["Citations"].sum()
    total_area = merged["area"].sum()
    merged["citation_share"] = merged["Citations"] / total_citations
    merged["area_share_self"] = merged["area"] / total_area
    merged["citation_gap"] = merged["area_share_self"] - merged["citation_share"]

    # scipy.stats is imported here for the same reason as in combined_viz: the
    # extension DLLs of scipy.stats do not load in this environment.
    from scipy.stats import spearmanr

    # Pairwise Spearman rank correlations among the three indicators.
    rho_dd, _ = spearmanr(merged["deficit"], merged["pub_density"])
    rho_dc, _ = spearmanr(merged["deficit"], merged["citation_gap"])
    rho_cd, _ = spearmanr(merged["pub_density"], merged["citation_gap"])

    summary = {
        "n_countries": int(len(merged)),
        "spearman_deficit_pub_density": round(float(rho_dd), 4),
        "spearman_deficit_citation_gap": round(float(rho_dc), 4),
        "spearman_pub_density_citation_gap": round(float(rho_cd), 4),
    }

    out = merged[["Country", "area", "pubs", "Citations", "deficit", "pub_density", "citation_gap"]].copy()
    out = out.sort_values("deficit", ascending=False).reset_index(drop=True)

    out.to_csv(paths.processed_dir / "multi_indicator_consistency.csv", index=False, encoding="utf-8-sig")
    (paths.processed_dir / "multi_indicator_consistency_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    logger.info("Multi-indicator consistency (%d countries):", summary["n_countries"])
    logger.info("  Spearman(deficit, pub_density)      = %.3f", rho_dd)
    logger.info("  Spearman(deficit, citation_gap)     = %.3f", rho_dc)
    logger.info("  Spearman(pub_density, citation_gap) = %.3f", rho_cd)
    return out


def main() -> None:
    import argparse

    from src.utils import setup_logging

    parser = argparse.ArgumentParser(description="Multi-indicator gap consistency (#2-C)")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    config = get_config(args.config)
    setup_logging(config.system.log_level, config.system.log_file)
    try:
        run_multi_indicator_consistency(config)
    except Exception:
        logger.exception("Multi-indicator consistency failed")
        raise


if __name__ == "__main__":
    main()
