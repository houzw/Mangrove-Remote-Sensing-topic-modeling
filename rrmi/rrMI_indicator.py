#!/usr/bin/env python3
"""Reference implementation of the research--resource mismatch index (RRMI).

Companion code for
    Hou et al., "A research--resource mismatch index for mangrove remote
    sensing: topic evolution, emerging fronts, and who leads the science",
    submitted.

The index is deliberately one line of arithmetic over two shares:

    RRMI_c = a_c - p_c

where ``a_c`` is country ``c``'s share of the ecological resource (here,
global mangrove area) and ``p_c`` is its share of fractional scientific
publications.  Positive = the country holds more of the resource than it
produces of the literature (under-publishing); negative = the reverse.

Everything else in this file is bookkeeping: how the two shares are counted,
and the diagnostics that keep the index honest.

Inputs
------
publications CSV
    One row per publication.  Required column ``countries`` holding the
    affiliation countries, separated by ``;``.  Optional column ``year``
    (used by ``--min-year`` / ``--max-year``).
areas CSV
    Columns ``country`` and ``area_km2``.

Counting conventions (Section 2.9 and Table 1 of the paper)
-----------------------------------------------------------
* Fractional counting: a publication with ``m`` distinct affiliation
  countries gives 1/m to each, so the weights of every publication sum to 1.
* ``p_c`` is a fractional share.  Full-count (integer) publication volumes are
  reported alongside it in the output but never enter the index.
* Both shares are normalised **inside the analysis table**: the denominators
  are the countries that hold both an area row and at least one publication
  (76 countries in the paper).  ``--global-basis`` switches to the world
  denominators, i.e. the whole resource map and the whole corpus, including
  countries that never enter the table.  The two bases are close here (the
  76 countries hold 143{,}970 of the map's 145{,}068 km$^2$ and 2{,}087.17
  of the corpus's 2{,}332 fractional publications) but they are not
  interchangeable, and the published values are on the table basis.
* No publication-volume floor: every country with both an area row and at
  least one publication enters the index.
* Administered territories are folded into their metropole on **both** the
  area and the publication side, so a country's resource and its research are
  on equal footing (French Guiana's mangroves and its papers both count
  towards France).  Taiwan is folded on the publication side only, because the
  resource map has no Taiwan row.  ``--no-fold`` switches all folding off.

Usage
-----
    python rrMI_indicator.py --publications pubs.csv --areas areas.csv \
        --out rrmi.csv

    # the world-basis variant (not the published one)
    python rrMI_indicator.py --publications pubs.csv --areas areas.csv \
        --global-basis --out rrmi_global.csv

    # restrict to a publication window (used for the 2016--2025 robustness check)
    python rrMI_indicator.py --publications pubs.csv --areas areas.csv \
        --min-year 2016 --max-year 2025 --out rrmi_recent.csv

Licence: MIT.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy.stats import spearmanr
except ImportError:  # scipy is optional; the index itself needs only pandas
    spearmanr = None


# ---------------------------------------------------------------------------
# Country-name tables
#
# These mirror the tables of ``src/area_pub_deficit.py`` exactly, so the
# reference implementation and the published pipeline agree country by country.
# ---------------------------------------------------------------------------

# Spellings in the resource map (HGMF 2020) that differ from the canonical
# names used by the publication side, so the two join on the same key.
HGMF_NAME_FIX = {
    "camerocn": "Cameroon",
    "venzuela": "Venezuela",
    "solomon is.": "Solomon Islands",
    "congo.drc": "Congo",
    "cote d'ivory": "Cote d'Ivoire",
    "the bahamas": "Bahamas",
    "the gambia": "Gambia",
}

# Administered territories -> metropole.  Applied to the area *and* the
# publication side.  Keys are lower-cased, folded spellings.
TERRITORY_TO_METROPOLE = {
    "french guiana": "France",
    "guadeloupe": "France",
    "martinique": "France",
    "mayotte": "France",
    "new caledonia": "France",
    "wallis & futuna": "France",
    "wallis and futuna": "France",
    "reunion": "France",
    "puerto rico": "United States",
    "american samoa": "United States",
    "virgin is.": "United States",
    "virgin islands": "United States",
    "united states virgin islands": "United States",
    "netherlands antilles": "Netherlands",
    "aruba": "Netherlands",
    "british virgin is.": "United Kingdom",
    "british virgin islands": "United Kingdom",
    "cayman is.": "United Kingdom",
    "cayman islands": "United Kingdom",
    "turks & caicos is.": "United Kingdom",
    "turks and caicos islands": "United Kingdom",
}

# Taiwan hosts native mangroves but has no row of its own in the resource map,
# so it can only be folded into China on the publication side.
PUB_ONLY_FOLD = {"taiwan": "China"}

# Publication side: everything above, keyed on the lower-cased country name.
DEFAULT_FOLD = {
    **{k: v for k, v in TERRITORY_TO_METROPOLE.items()},
    **PUB_ONLY_FOLD,
}

# Place/state names inside real affiliations that a substring match cannot tell
# apart from the sovereign country: Niger Delta/State (Nigeria), Georgia (US
# state) institutions, Inner Mongolia (China).  Dropped so they do not absorb
# fractional weight from the real countries.
FALSE_POSITIVES = {"niger", "georgia", "mongolia"}


def _norm(name: str) -> str:
    return " ".join(str(name).strip().lower().split())


def _area_key(name: str, fold: dict[str, str] | None) -> str:
    """Canonical join key for a country name coming from the area file."""
    n = _norm(name)
    n = _norm(HGMF_NAME_FIX.get(n, n))
    if fold:
        n = _norm(fold.get(n, n))
    return n


def _pub_key(name: str, fold: dict[str, str] | None) -> str:
    """Canonical join key for a country name coming from the publication file."""
    n = _norm(name)
    if fold:
        n = _norm(fold.get(n, n))
    return n


def load_publications(
    path: Path, sep: str = ";", fold: dict[str, str] | None = None
) -> pd.DataFrame:
    """Return one row per (publication, country) with a fractional weight."""
    if fold is None:
        fold = DEFAULT_FOLD
    df = pd.read_csv(path)
    if "countries" not in df.columns:
        raise SystemExit("publications CSV must have a 'countries' column")
    df = df.copy()
    df["_id"] = np.arange(len(df))
    pairs = []
    for _, row in df.iterrows():
        raw = [_pub_key(c, fold) for c in str(row["countries"]).split(sep) if str(c).strip()]
        countries = sorted({c for c in raw if c not in FALSE_POSITIVES})
        if not countries:
            continue
        w = 1.0 / len(countries)
        for c in countries:
            pairs.append({"_id": row["_id"], "country": c, "frac": w, "full": 1})
    long = pd.DataFrame(pairs)
    return long.merge(df.drop(columns=["countries"]), on="_id", how="left")


def load_areas(path: Path, fold: dict[str, str] | None = None) -> pd.DataFrame:
    """Return one row per country with its folded, aggregated resource area."""
    if fold is None:
        fold = TERRITORY_TO_METROPOLE
    area = pd.read_csv(path)
    area.columns = [c.strip().lower() for c in area.columns]
    if not {"country", "area_km2"} <= set(area.columns):
        raise SystemExit("areas CSV must have 'country' and 'area_km2' columns")
    area = area.copy()

    def _display(name: str) -> str:
        """Readable output name: corrected spelling, or the metropole if folded."""
        n = _norm(name)
        if n in HGMF_NAME_FIX:
            return HGMF_NAME_FIX[n]
        if fold and n in fold:
            return fold[n]
        return str(name).strip()

    area["_display"] = area["country"].map(_display)
    area["country"] = area["country"].map(lambda s: _area_key(s, fold))
    area = (
        area.groupby("country", as_index=False)
        .agg(area_km2=("area_km2", "sum"), _display=("_display", "first"))
    )
    return area


def compute_rrmi(
    publications: Path,
    areas: Path,
    fold: dict[str, str] | None = None,
    min_year: int | None = None,
    max_year: int | None = None,
    basis: str = "table",
) -> pd.DataFrame:
    """Compute the RRMI table for every country holding both data types.

    ``basis="table"`` (the published convention) normalises both shares inside
    the analysis table; ``basis="world"`` divides by the whole resource map and
    the whole corpus, including countries that never enter the table.
    """
    if basis not in {"table", "world"}:
        raise SystemExit("--basis must be 'table' (default) or 'world'")

    long = load_publications(publications, fold=fold)
    if "year" in long.columns:
        if min_year is not None:
            long = long[long["year"] >= min_year]
        if max_year is not None:
            long = long[long["year"] <= max_year]

    # Fractional and full-count output per country.
    g = long.groupby("country", as_index=False).agg(
        frac_pubs=("frac", "sum"), full_pubs=("full", "sum")
    )

    area = load_areas(areas, fold=fold)

    out = area.merge(g, on="country", how="inner")
    if out.empty:
        raise SystemExit("no country has both an area row and at least one publication")

    # Denominators.  The table basis is the published convention: both shares
    # are taken over the countries that actually enter the index, so a_c and
    # p_c sum to 1 over the table.  The world basis keeps the map's and the
    # corpus's own totals, which are larger because countries without a
    # counterpart row still contribute.
    if basis == "table":
        total_area = out["area_km2"].sum()
        total_frac = out["frac_pubs"].sum()
    else:
        total_area = area["area_km2"].sum()
        total_frac = g["frac_pubs"].sum()

    out["area_share"] = out["area_km2"] / total_area
    out["pub_share"] = out["frac_pubs"] / total_frac
    out["RRMI"] = out["area_share"] - out["pub_share"]

    # Ratio-form counterpart: fractional publications per 10^3 km^2.
    # Reported for the rank-correlation diagnostic, never as the index.
    out["pubs_per_1000km2"] = np.where(
        out["area_km2"] > 0, out["frac_pubs"] / out["area_km2"] * 1e3, np.nan
    )

    out = out.sort_values("RRMI", ascending=False).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    out["country"] = out["_display"]

    print(
        f"basis = {basis} | {len(out)} countries | "
        f"area denominator {total_area:,.1f} km2 | "
        f"fractional-publication denominator {total_frac:,.2f}",
        file=sys.stderr,
    )
    if spearmanr is not None:
        rho, pval = spearmanr(out["RRMI"], out["pubs_per_1000km2"], nan_policy="omit")
        print(
            f"n = {len(out)} countries | Spearman(RRMI, pubs per 10^3 km^2) "
            f"= {rho:.3f} (p = {pval:.2e})",
            file=sys.stderr,
        )
    return out[
        [
            "rank",
            "country",
            "area_km2",
            "area_share",
            "frac_pubs",
            "pub_share",
            "RRMI",
            "full_pubs",
            "pubs_per_1000km2",
        ]
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--publications", required=True, type=Path)
    ap.add_argument("--areas", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--min-year", type=int, default=None)
    ap.add_argument("--max-year", type=int, default=None)
    ap.add_argument(
        "--no-fold",
        action="store_true",
        help="do not fold administered territories into their metropole",
    )
    ap.add_argument(
        "--global-basis",
        action="store_true",
        help="normalise the shares over the whole resource map and the whole "
             "corpus instead of over the analysis table (not the published "
             "convention)",
    )
    args = ap.parse_args()

    table = compute_rrmi(
        args.publications,
        args.areas,
        fold={} if args.no_fold else None,
        min_year=args.min_year,
        max_year=args.max_year,
        basis="world" if args.global_basis else "table",
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False, float_format="%.6g")
    print(f"wrote {args.out} ({len(table)} countries)")


if __name__ == "__main__":
    main()
