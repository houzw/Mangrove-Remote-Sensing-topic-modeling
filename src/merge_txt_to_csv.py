"""Merge WoS tagged-format .txt files from data/raw/ into a single CSV.

Usage:
    python -m src.merge_txt_to_csv
    python -m src.merge_txt_to_csv --output data/raw/merged_wos.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.data_preprocessing import parse_wos_tagged, find_raw_files

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge WoS tagged .txt files into a single CSV"
    )
    parser.add_argument(
        "--input-dir", type=str, default="data/raw",
        help="Directory containing WoS .txt files (default: data/raw)",
    )
    parser.add_argument(
        "--output", type=str, default="data/raw/merged_wos.csv",
        help="Output CSV path (default: data/raw/merged_wos.csv)",
    )
    parser.add_argument(
        "--pattern", type=str, default="*.txt",
        help="Glob pattern for input files (default: *.txt)",
    )
    parser.add_argument(
        "--keep-all", action="store_true",
        help="Keep all record types; default filters to journal articles (PT=J) only",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-7s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    raw_dir = Path(args.input_dir)
    files = find_raw_files(raw_dir, args.pattern)
    logger.info("Processing %d file(s): %s", len(files), [f.name for f in files])

    dfs = []
    for fp in files:
        df = parse_wos_tagged(fp)
        dfs.append(df)

    merged = pd.concat(dfs, ignore_index=True)
    logger.info("Merged: %d records total", len(merged))

    if not args.keep_all and "PT" in merged.columns:
        n_before = len(merged)
        merged = merged[merged["PT"] == "J"].copy()
        logger.info("Filtered to journal articles (PT=J): %d → %d", n_before, len(merged))

    output_path = Path(args.output)
    merged.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info("Saved: %s (%d rows × %d columns)", output_path, len(merged), len(merged.columns))

    # Quick stats
    if "PY" in merged.columns:
        logger.info("Year range: %s – %s", merged["PY"].min(), merged["PY"].max())
    logger.info("Columns: %s", list(merged.columns))


if __name__ == "__main__":
    main()
