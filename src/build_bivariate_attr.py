"""Rebuild ``data/processed/bivariate_attr.csv`` — HGMF mangrove area x WoS publications.

Why this module exists
----------------------
The attribute table consumed by :mod:`src.bivariate_choropleth` was a hand-built
*orphan input*: no module in ``src/`` produced it, so when ``_extract_country``
was fixed (2026-09-04, completed 2026-09-21) the table kept the old publication
counts and the choropleth — and the 3x3 cell counts quoted in the manuscript —
silently reported stale numbers.  This module makes the derivation reproducible;
it is the standing remedy for the "orphan input" lesson recorded in CLAUDE.md
(any CSV consumed by a module must be produced by a module).

Inputs / conventions
--------------------
* **Area side** — preserved verbatim from the existing table.  It is the HGMF
  2020 per-territory mangrove area, keyed by HGMF's own country names (typos
  such as ``Camerocn`` / ``Venzuela`` / ``Cote d'Ivory`` included).  The country
  extractor fix does not touch geography, so this column is not recomputed.
* **Publications side** — recomputed as *full-count* papers per country from
  ``data/processed/cleaned_data.csv`` via
  :func:`src.collaboration_network_map.build_collaboration_network`, i.e. the
  exact country space of the collaboration network (``REGION_TO_COUNTRY``
  folding of Taiwan / Puerto Rico / French overseas territories into their
  metropoles, and the same ``CU_FALSE_POSITIVES`` exclusions).  Table, network
  and deficit index therefore agree on what a country is.

Row semantics
-------------
A row whose ``country`` is not a current WoS country name (HGMF spelling
variants) keeps ``n_pubs = 0``; the consumer merges duplicate-ISO3 rows, so the
area row and the publication row of e.g. Cameroon still sum correctly.  WoS
countries missing from the table are appended with ``mangrove_area_km2 = 0``.

Usage:
    pixi run python -m src.build_bivariate_attr
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.collaboration_network_map import build_collaboration_network
from src.config import Config, get_config
from src.utils import Timer, setup_logging

logger = logging.getLogger(__name__)

ATTR_CSV = Path("data/processed/bivariate_attr.csv")


def rebuild_bivariate_attr(config: Config, attr_csv: Path = ATTR_CSV) -> pd.DataFrame:
    """Recompute the ``n_pubs`` column of the bivariate attribute table."""
    paths = config.paths
    attr_path = attr_csv if attr_csv.is_absolute() else config.project_root / attr_csv
    if not attr_path.exists():
        raise FileNotFoundError(f"Attribute table not found: {attr_path}")

    cleaned_path = paths.processed_dir / config.preprocessing.output_file
    if not cleaned_path.exists():
        raise FileNotFoundError(
            f"Cleaned data not found: {cleaned_path}. Run preprocessing first."
        )

    attr = pd.read_csv(attr_path, encoding="utf-8-sig")
    required = {"country", "mangrove_area_km2", "n_pubs"}
    if not required.issubset(attr.columns):
        raise ValueError(f"{attr_path.name} is missing columns: {required - set(attr.columns)}")

    df = pd.read_csv(cleaned_path, encoding="utf-8-sig")
    with Timer("Build collaboration country space"):
        node_pubs, _edge_weights = build_collaboration_network(df)

    # --- recompute the publication column -------------------------------------
    old_pubs = attr["n_pubs"].astype(int).copy()
    attr["n_pubs"] = attr["country"].map(lambda c: int(node_pubs.get(c, 0)))
    diff = pd.DataFrame({
        "country": attr["country"],
        "old": old_pubs.values,
        "new": attr["n_pubs"].values,
    })
    diff = diff.loc[diff["old"] != diff["new"]]

    # --- append WoS countries absent from the HGMF-keyed row space ------------
    missing = sorted(set(node_pubs) - set(attr["country"]))
    if missing:
        extra = pd.DataFrame({
            "ISO3": [""] * len(missing),
            "country": missing,
            "mangrove_area_km2": [0.0] * len(missing),
            "n_pubs": [int(node_pubs[c]) for c in missing],
        })
        attr = pd.concat([attr, extra], ignore_index=True)
        logger.info("Appended %d WoS-only countries: %s", len(missing), missing)

    # --- report ---------------------------------------------------------------
    print("=" * 72)
    print("BIVARIATE ATTRIBUTE TABLE - REBUILD REPORT")
    print("=" * 72)
    print(f"source cleaned data       : {cleaned_path.name} "
          f"({len(df)} documents)")
    print(f"rows                      : {len(attr)} (was {len(old_pubs)})")
    print(f"countries with n_pubs > 0 : {(attr['n_pubs'] > 0).sum()}")
    print(f"total mangrove area (km2) : {attr['mangrove_area_km2'].sum():,.1f}")
    print(f"rows with changed n_pubs  : {len(diff)}")
    pretty = diff.rename(columns={"old": "old_n_pubs", "new": "new_n_pubs"})
    for _, row in pretty.sort_values("country").iterrows():
        print(f"    {row['country']:28s} {int(row['old_n_pubs']):4d} -> "
              f"{int(row['new_n_pubs']):4d}")
    if missing:
        print(f"appended rows             : {', '.join(missing)}")
    print()
    print("top-12 countries by publications (new):")
    top = attr.sort_values("n_pubs", ascending=False).head(12)
    for _, row in top.iterrows():
        print(f"    {row['country']:28s} {int(row['n_pubs']):4d}  "
              f"area={row['mangrove_area_km2']:.1f} km2")
    print()

    attr.to_csv(attr_path, index=False, encoding="utf-8-sig")
    logger.info("Bivariate attribute table written: %s", attr_path)
    return attr


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild data/processed/bivariate_attr.csv (HGMF area x WoS pubs)"
    )
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--attr-csv", type=str, default=str(ATTR_CSV))
    args = parser.parse_args()

    config = get_config(args.config)
    setup_logging(config.system.log_level, config.system.log_file)
    try:
        rebuild_bivariate_attr(config, Path(args.attr_csv))
    except Exception:
        logger.exception("Bivariate attribute rebuild failed")
        raise


if __name__ == "__main__":
    main()
