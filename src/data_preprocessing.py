"""Step 2: Data Preprocessing — read, clean, deduplicate, and fuse WoS export data.

Supports both:
- WoS classic tagged plaintext format (``.txt``, fields like ``TI``, ``AB``, ``ER``)
- WoS CSV/Excel export format (``.csv``, ``.xlsx``)

Usage:
    python -m src.data_preprocessing              # uses default config.yaml
    python -m src.data_preprocessing --config path/to/config.yaml
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.config import Config, get_config
from src.utils import Timer, ensure_dir, log_df_stats

logger = logging.getLogger(__name__)


# =============================================================================
# WoS tagged-format parser
# =============================================================================

# Tags we skip (recognize as new tag sections but don't store)
_SKIP_TAGS = {
    "AU", "AF", "BA", "BF", "CA", "CT", "CY", "C3", "CL",
    "DA", "DT", "EM", "EP", "FN", "FU", "FX", "LA",
    "NR", "OI", "PD", "PG", "PI", "PM", "PN", "PU", "PA",
    "RI", "RP", "SC", "SN", "EI", "BN", "J9", "JI",
    "VL", "IS", "BP", "EP", "AR", "GA", "UT", "WC", "WE",
    "Z9", "ZB", "ZS",
}

def _is_tag_line(line: str) -> bool:
    """Check if line starts with a known WoS 2-character tag."""
    return len(line) >= 3 and line[2] == " " and line[:2].isalpha() and line[:2].isupper()


_WANTED_TAGS = {
    "PT",   # Publication type (J=article, B=book review, etc.)
    "TI",   # Title
    "SO",   # Source (journal name)
    "DE",   # Author keywords
    "ID",   # Keywords Plus (WoS-indexed keywords)
    "AB",   # Abstract
    "C1",   # Author addresses (→ extract country)
    "PY",   # Publication year
    "TC",   # Times cited
    "DI",   # DOI
    "DT",   # Document type
    "LA",   # Language
}

# Known country names for reliable extraction from C1 addresses.
# We match substrings of the address text against this set, then
# normalize via COUNTRY_ALIASES to a canonical form.
_COUNTRY_NAMES: set[str] = {
    "Afghanistan", "Albania", "Algeria", "Angola", "Argentina", "Armenia",
    "Australia", "Austria", "Azerbaijan",
    "Bahamas", "Bahrain", "Bangladesh", "Barbados", "Belarus", "Belgium",
    "Belize", "Benin", "Bhutan", "Bolivia", "Botswana", "Brazil",
    "Brunei Darussalam", "Brunei", "Bulgaria", "Burkina Faso", "Burundi",
    "Cambodia", "Cameroon", "Canada", "Chile", "China", "Colombia",
    "Comoros", "Congo", "Costa Rica", "Cote d'Ivoire", "Croatia", "Cuba",
    "Cyprus", "Czech Republic",
    "Dem. Rep. Korea", "Dem. Rep. Congo", "Denmark", "Djibouti",
    "Dominican Republic",
    "Ecuador", "Egypt", "El Salvador", "Equatorial Guinea", "Eritrea",
    "Estonia", "Ethiopia",
    "Fiji", "Finland", "France", "French Guiana",
    "Gabon", "Gambia", "Georgia", "Germany", "Ghana", "Greece", "Grenada",
    "Guadeloupe", "Guatemala", "Guinea", "Guinea-Bissau", "Guyana",
    "Haiti", "Honduras", "Hungary",
    "Iceland", "India", "Indonesia", "Iran", "Iraq", "Ireland", "Israel",
    "Italy", "Ivory Coast",
    "Jamaica", "Japan", "Jordan",
    "Kazakhstan", "Kenya", "Kuwait", "Kyrgyzstan",
    "Laos", "Latvia", "Lebanon", "Liberia", "Libya", "Lithuania",
    "Luxembourg",
    "Madagascar", "Malawi", "Malaysia", "Maldives", "Mali", "Malta",
    "Martinique", "Mauritania", "Mauritius", "Mayotte", "Mexico",
    "Micronesia", "Moldova", "Monaco", "Mongolia", "Montenegro",
    "Morocco", "Mozambique", "Myanmar",
    "Namibia", "Nepal", "Netherlands", "New Caledonia", "New Zealand",
    "Nicaragua", "Niger", "Nigeria", "North Macedonia", "Norway",
    "Oman",
    "Pakistan", "Palau", "Palestine", "Panama", "Papua New Guinea",
    "Paraguay", "Peru", "Philippines", "Poland", "Portugal",
    "Puerto Rico",
    "Qatar",
    "Republic of Korea", "Romania", "Russia", "Rwanda",
    "Saudi Arabia", "Senegal", "Serbia", "Seychelles", "Sierra Leone",
    "Singapore", "Slovakia", "Slovenia", "Solomon Islands", "Somalia",
    "South Africa", "South Korea", "Spain", "Sri Lanka",
    "St. Lucia", "St. Vincent & the Grenadines", "Sudan", "Suriname",
    "Sweden", "Switzerland", "Syria",
    "Taiwan", "Tajikistan", "Tanzania", "Thailand",
    "Timor Leste", "Togo", "Tonga", "Trinidad & Tobago",
    "Trinidad and Tobago", "Tunisia", "Turkey", "Turkmenistan",
    "Uganda", "Ukraine", "United Arab Emirates", "United Kingdom",
    "United States", "Uruguay", "Uzbekistan",
    "Vanuatu", "Venezuela", "Vietnam",
    "Yemen",
    "Zambia", "Zimbabwe",
    # WoS common variants
    "USA", "U Arab Emirates", "U K", "England", "Scotland", "Wales",
    "North Ireland", "Peoples R China", "P.R. China", "P R China",
    "Rep of Korea", "Russian Federation", "Viet Nam",
    "Iran (Islamic Republic of)", "Islamic Republic of Iran",
    "Netherlands Antilles", "Papua N Guinea", "Trinidad & Tobago",
    "Cote Ivoire", "Côte d'Ivoire", "French Guiana", "New Caledonia",
    "Solomon Is.", "Dem Rep Congo", "Congo.DRC", "Venzuela",
    "Camerocn", "The Gambia", "The Bahamas",
}

# Normalize matched names to a canonical short form.
_COUNTRY_NORMALIZE: dict[str, str] = {
    # UK variants
    "England": "United Kingdom", "Scotland": "United Kingdom",
    "Wales": "United Kingdom", "North Ireland": "United Kingdom",
    "U K": "United Kingdom",
    # US variants
    "USA": "United States",
    # China variants
    "Peoples R China": "China", "P.R. China": "China", "P R China": "China",
    # Korea
    "South Korea": "Republic of Korea", "Rep of Korea": "Republic of Korea",
    "Dem. Rep. Korea": "North Korea",
    # Russia
    "Russian Federation": "Russia",
    # Iran
    "Iran (Islamic Republic of)": "Iran", "Islamic Republic of Iran": "Iran",
    # Congo
    "Dem Rep Congo": "Congo", "Congo.DRC": "Congo",
    # UAE
    "U Arab Emirates": "United Arab Emirates",
    # Vietnam
    "Viet Nam": "Vietnam",
    # Papua New Guinea
    "Papua N Guinea": "Papua New Guinea",
    # Côte d'Ivoire
    "Cote Ivoire": "Cote d'Ivoire", "Côte d'Ivoire": "Cote d'Ivoire",
    "Ivory Coast": "Cote d'Ivoire",
    # Trinidad
    "Trinidad and Tobago": "Trinidad & Tobago",
    # Gambia / Bahamas
    "The Gambia": "Gambia", "The Bahamas": "Bahamas",
    # Typo corrections in source data
    "Venzuela": "Venezuela", "Camerocn": "Cameroon",
    "Solomon Is.": "Solomon Islands",
    "Netherlands Antilles": "Netherlands",
}

# Multi-word phrases whose inner word is a shorter country name.  Key: the
# phrase; value: canonical country for *real* compound countries (the glued
# underscore form is registered as a matchable name in ``_GLUED_NAMES``), or
# ``None`` for sub-national places that must never match anything (their
# glued form is deliberately left unregistered, so no country can fire
# inside them).  Applied to the C1 text before word-boundary matching.
_PHRASE_GUARDS: dict[str, Optional[str]] = {
    # Real compound countries — protect the phrase, suppress the inner word.
    "Papua New Guinea": "Papua New Guinea",
    "Papua N Guinea": "Papua New Guinea",
    "Guinea-Bissau": "Guinea-Bissau",
    "Guinea Bissau": "Guinea-Bissau",
    "Equatorial Guinea": "Equatorial Guinea",
    # Sub-national places — suppress outright.
    "New Mexico": None,       # US state (→ phantom Mexico)
    "New England": None,      # US region (→ phantom United Kingdom)
    "New South Wales": None,  # Australian state (→ phantom United Kingdom)
    "New Ireland": None,      # Papua New Guinea province (→ phantom Ireland)
}

# Canonical country for each protected phrase, keyed by its underscore-glued
# form (the form the C1 text is rewritten to before matching).
_GLUED_NAMES: dict[str, str] = {
    phrase.replace(" ", "_"): canonical
    for phrase, canonical in _PHRASE_GUARDS.items()
    if canonical is not None
}

# Word-boundary matchers for every known country name.  Plain substring
# matching fabricated phantom countries out of longer words — "india" inside
# "Indian River Lagoon"/"Indian Institute" (inflated India and its edges),
# "oman" inside "Romania", "sudan" inside "Sudanese", "mali" inside "Komali",
# "u k" inside "qinzhou key lab" (→ phantom United Kingdom) etc. — while
# word-boundary matching keeps genuine name occurrences only.  The lookarounds
# exclude every word character *including underscore*, so the underscore-glued
# phrase forms of ``_PHRASE_GUARDS`` really cannot fire their inner country
# name.  Phrases are rewritten beforehand (see ``_extract_country``).
# Residual *word-level* collisions (Niger Delta→Niger, Georgia the US state,
# Inner Mongolia→Mongolia) cannot be resolved here and stay with the
# ``*_CU_FALSE_POSITIVES`` sets in the consumer modules.
_COUNTRY_PATTERNS: list[tuple[str, "re.Pattern[str]"]] = [
    (name, re.compile(r"(?<!\w)" + re.escape(name) + r"(?!\w)",
                      re.IGNORECASE))
    for name in list(_COUNTRY_NAMES) + list(_GLUED_NAMES)
]


def _canonical_country(name: str) -> str:
    """Canonical form of a matched country-name token."""
    return _COUNTRY_NORMALIZE.get(name) or _GLUED_NAMES.get(name) or name


def _affiliation_only(text: str) -> str:
    """Drop every ``[Author names]`` bracket segment from a C1 string.

    WoS ``C1`` entries have the shape ``[Surname, Initials] Institution, ...,
    Country`` and are ``" ".join``-ed across authors.  Author names routinely
    contain country-like substrings (``Jamali``→Mali, ``Jordan M.``→Jordan,
    ``Montenegro J.``→Montenegro, ``Moldovanu``→Moldova, ``Lomonaco``→Monaco,
    ``Israel R.``→Israel) which leaked into CU when the bracket segment was
    matched as if it were affiliation text.  Country extraction must only ever
    see the *affiliation* text, so strip all ``[...]`` segments up front.

    Two bracket shapes occur and both must go:

    * closed segments ``[Long, Jordan]`` — removed by the first regex;
    * *dangling* unclosed fragments ``"... USA. [Long, Jordan"`` — the
      ``"] "-anchored`` block splitter of ``_extract_country`` cuts right
      after a closing bracket, so the next author's bracket reaches the end
      of the block still open.  A closed-bracket-only regex cannot remove
      it, and the person name inside (``Long, Jordan``, ``Orimoloye, Israel
      Ropo``, ``Malik, Abdul``) leaked the same false countries a second
      time even after the first fix; the second regex strips it.
    """
    text = re.sub(r"\[[^\]]*\]", " ", text)  # closed [Author names] segments
    text = re.sub(r"\[[^\]]*$", " ", text)   # dangling unclosed trailing "[Names"
    return text.strip()


def _extract_country(c1_text: str) -> str:
    """Extract country names from a C1 address field via known-name matching.

    Strategy:
        1. Split C1 into individual address entries on ``"] "`` / ``". ["``
           boundaries (the ``]``-anchored splitter leaves dangling author-only
           fragments such as ``"[Jamali, Sadegh"`` and glues the next author's
           bracket onto the previous affiliation's tail).
        2. Match only against affiliation text: every ``[Author names]`` bracket
           segment — *closed or dangling-unclosed* — is stripped from each block
           before matching, so person-name false positives never reach the
           country matcher.
        3. For each remaining affiliation block, check every known country name
           on *word boundaries* (substring matching fabricated phantoms:
           "india" in "Indian River Lagoon", "oman" in "Romania"); normalise
           matches to canonical form.  Multi-word phrases carrying a shorter
           country name as an inner word are rewritten first ("New Mexico",
           "New England", "New South Wales" → suppressed; "Papua New Guinea",
           "Guinea-Bissau" → protected as glued names).
        4. Fall back: if nothing matched, try the last comma-segment of the
           bracket-stripped text against the list (same word-boundary rule).

    Person-name false positives (author names in brackets) and substring
    false positives (country names inside longer words) are both eliminated
    at the source.  Residual *word-level* place/state-name collisions inside
    real affiliations (Niger Delta/State→Niger, Georgia (US state)
    institutions, Inner Mongolia→Mongolia) are handled by the
    ``*_CU_FALSE_POSITIVES`` sets in the consumer modules.
    """
    if pd.isna(c1_text) or not str(c1_text).strip():
        return "Unknown"

    text = str(c1_text).strip()
    countries: set[str] = set()

    # Rewrite multi-word phrases that contain a shorter country name as an
    # inner word BEFORE matching ("New Mexico"→"New_Mexico" so Mexico cannot
    # fire; "Papua New Guinea"→"Papua_New_Guinea", a registered name, so the
    # phrase still counts as Papua New Guinea but bare "Guinea" cannot fire).
    # Protected phrases resolve to their canonical country via _GLUED_NAMES
    # at match time; suppressed phrases match nothing at all.
    for phrase in _PHRASE_GUARDS:
        text = re.sub(
            r"(?<!\w)" + re.escape(phrase) + r"(?!\w)",
            phrase.replace(" ", "_"),
            text,
            flags=re.IGNORECASE,
        )

    # Split into individual address blocks on "] " (the bracket close) or, when
    # only one block results, on ". [" (period before a new author bracket).
    blocks = re.split(r"\]\.?\s+", text)
    if len(blocks) <= 1:
        blocks = re.split(r"\.\s+\[", text)

    for block in blocks:
        # Drop every author-name bracket segment, wherever it sits in the block.
        # A "] "-anchored split glues the next author's "[Names" onto the tail of
        # the current affiliation ("...Sweden. [Jamali, Sadegh"), and a lone
        # author bracket can be its own fragment; author names must never reach
        # the country matcher ("...Jamali..." would otherwise leak "Mali").
        block = _affiliation_only(block).strip().rstrip(".")
        if not block or len(block) < 3:
            continue

        # Case-insensitive word-boundary match of every known country name.
        # Substring matching produced phantom countries from longer words
        # ("india" inside "Indian River Lagoon"/"Indian Institute", "oman"
        # inside "Romania", "sudan" inside "Sudanese"); word boundaries keep
        # real name occurrences only.
        for name, pat in _COUNTRY_PATTERNS:
            if pat.search(block):
                countries.add(_canonical_country(name))

    # Fallback: if nothing found, try the "last comma segment" approach on the
    # bracket-stripped text, but only accept it if it matches a known country.
    if not countries:
        last_seg = _affiliation_only(text).split(",")[-1].strip().rstrip(".")
        last_seg = last_seg.split(";")[-1].strip()  # handle trailing C3 fragments
        for name, pat in _COUNTRY_PATTERNS:
            if pat.search(last_seg):
                countries.add(_canonical_country(name))
                break

    return ";".join(sorted(countries)) if countries else "Unknown"


def _merge_tagged_lines(tag: str, lines: List[str]) -> str:
    """Merge multi-line tagged field into a single string.

    Tagged fields may span multiple lines; continuation lines start with 3 spaces.
    """
    result_parts: List[str] = []
    for line in lines:
        if line.startswith(tag + " "):
            # New occurrence of this tag — take the content after the tag
            content = line[3:].strip()
        elif line.startswith("   ") or line.startswith("\t"):
            # Continuation line
            content = line.strip()
        else:
            continue
        if content:
            result_parts.append(content)
    return " ".join(result_parts)


def parse_wos_tagged(filepath: Path, encoding: str = "utf-8") -> pd.DataFrame:
    """Parse a WoS classic tagged-format export file into a DataFrame.

    The tagged format uses 2-character field codes (``TI``, ``AB``, ``PY``, etc.)
    at the start of each line.  Records begin with ``PT J`` (or similar) and end
    with ``ER``.  Continuation lines are indented with 3 spaces.

    Args:
        filepath: Path to the ``.txt`` WoS export file.
        encoding: File encoding.

    Returns:
        DataFrame with columns: ``PT, TI, SO, DE, ID, AB, PY, TC, DI, DT, LA, CU``.
    """
    with open(filepath, "r", encoding=encoding, errors="replace") as fh:
        raw_lines = fh.readlines()

    records: List[Dict[str, str]] = []
    current_record: Dict[str, List[str]] = {}
    in_record = False
    _last_tag: Optional[str] = None

    for line in raw_lines:
        # Detect record start: "PT J" (or "PT B", "PT S", etc.)
        if line.startswith("PT ") and not in_record:
            in_record = True
            current_record = {tag: [] for tag in _WANTED_TAGS}
            current_record["PT"].append(line)   # keep prefix for _merge_tagged_lines
            _last_tag = "PT"
            continue

        if in_record:
            # Record end
            if line.startswith("ER"):
                in_record = False
                # Merge multi-line fields and build flat dict
                flat: Dict[str, str] = {}
                for tag in _WANTED_TAGS:
                    flat[tag] = _merge_tagged_lines(tag, current_record[tag])
                records.append(flat)
                continue

            # Check if this line starts a known tag
            tag_matched = False
            for tag in _WANTED_TAGS:
                if line.startswith(tag + " "):
                    current_record[tag].append(line)
                    _last_tag = tag
                    tag_matched = True
                    break

            if not tag_matched:
                # Check if this is ANY tag line (2 uppercase letters + space).
                # If so, treat it as a section boundary to avoid content bleed.
                if _is_tag_line(line):
                    _last_tag = f"_skip_{line[:2]}"
                    tag_matched = True

            if not tag_matched:
                # Continuation line — append to the last matched tag
                stripped = line.strip()
                if stripped and not stripped.startswith("FN ") and not stripped.startswith("VR "):
                    if _last_tag is not None and _last_tag in current_record:
                        current_record[_last_tag].append(line)

    df = pd.DataFrame(records)

    # Extract country from C1
    if "C1" in df.columns:
        df["CU"] = df["C1"].apply(
            lambda x: _extract_country(str(x)) if pd.notna(x) and str(x).strip() else "Unknown"
        )
        logger.info("Extracted country (CU) from C1 addresses")
    else:
        df["CU"] = "Unknown"

    logger.info("Parsed %s: %d records [encoding=%s]", filepath.name, len(df), encoding)
    return df


# =============================================================================
# CSV / Excel reader (fallback)
# =============================================================================


def read_wos_csv(filepath: Path, encoding: str) -> Optional[pd.DataFrame]:
    """Read a WoS CSV/Excel export, normalising column names to short tags."""
    df = pd.read_csv(filepath, encoding=encoding, low_memory=False)
    if df is not None and len(df) > 0:
        logger.info("Read %s [CSV, encoding=%s]: %d records", filepath.name, encoding, len(df))
        return df
    return None


# =============================================================================
# File discovery and dispatch
# =============================================================================


def find_raw_files(raw_dir: Path, pattern: str = "*.txt") -> List[Path]:
    """Find all WoS export files matching *pattern* in *raw_dir*.

    Skips non-data files (``.jpg``, ``.xlsx`` citation reports, etc.).
    """
    all_files = sorted(raw_dir.glob(pattern))
    # Filter out non-data files
    skip_patterns = [".jpg", ".png", ".pdf", "citation report"]
    files = [
        f for f in all_files
        if not any(s in f.name.lower() for s in skip_patterns)
    ]
    if not files:
        raise FileNotFoundError(
            f"No data files matching '{pattern}' found in {raw_dir}. "
            f"Please place your WoS export file(s) in {raw_dir}."
        )
    logger.info("Found %d raw file(s): %s", len(files), [f.name for f in files])
    return files


def read_wos_file(filepath: Path, encodings: List[str]) -> pd.DataFrame:
    """Read a WoS export file, auto-detecting format (tagged .txt or CSV).

    Args:
        filepath: Path to the WoS export file.
        encodings: Encodings to try in order.

    Returns:
        DataFrame with standardised WoS columns.
    """
    suffix = filepath.suffix.lower()

    # --- Tagged plaintext format ---
    if suffix == ".txt":
        for enc in encodings:
            try:
                df = parse_wos_tagged(filepath, encoding=enc)
                if len(df) > 0:
                    return df
            except Exception:
                logger.debug("Failed to parse %s as tagged with encoding %s", filepath.name, enc)
                continue
        raise ValueError(f"Could not parse {filepath} as WoS tagged format.")

    # --- CSV format ---
    if suffix == ".csv":
        for enc in encodings:
            try:
                df = read_wos_csv(filepath, encoding=enc)
                if df is not None and len(df) > 0:
                    return df
            except Exception:
                logger.debug("Failed to read %s as CSV with encoding %s", filepath.name, enc)
                continue
        raise ValueError(f"Could not read {filepath} as CSV.")

    raise ValueError(f"Unsupported file format: {suffix} for {filepath}")


# =============================================================================
# Deduplication and cleaning
# =============================================================================


def deduplicate_by_doi(df: pd.DataFrame, doi_col: str) -> pd.DataFrame:
    """Deduplicate by DOI, keeping the record with the highest citation count.

    Records with missing DOI are kept (cannot determine uniqueness).
    """
    n_before = len(df)
    logger.info("Deduplicating by '%s' — %d records before", doi_col, n_before)

    df = df.copy()
    df["_doi_norm"] = df[doi_col].astype(str).str.strip().str.lower()
    df["_doi_norm"] = df["_doi_norm"].replace({"nan": None, "": None, "none": None})

    has_doi = df[df["_doi_norm"].notna()].copy()
    no_doi = df[df["_doi_norm"].isna()].copy()

    logger.info("  Records with DOI: %d", len(has_doi))
    logger.info("  Records without DOI: %d (kept as-is)", len(no_doi))

    if len(has_doi) > 0:
        if "TC" in has_doi.columns:
            has_doi["TC"] = pd.to_numeric(has_doi["TC"], errors="coerce").fillna(0).astype(int)
        else:
            has_doi["TC"] = 0

        has_doi = has_doi.sort_values("TC", ascending=False)
        n_before_doi = len(has_doi)
        has_doi = has_doi.drop_duplicates(subset=["_doi_norm"], keep="first")
        logger.info("  DOI duplicates removed: %d -> %d", n_before_doi, len(has_doi))

    df_clean = pd.concat([has_doi, no_doi], ignore_index=True)
    df_clean = df_clean.drop(columns=["_doi_norm"])
    logger.info("Deduplication: %d -> %d records (removed %d)", n_before, len(df_clean), n_before - len(df_clean))
    return df_clean


def extract_fields(df: pd.DataFrame, core_fields: List[str], fillna_values: dict) -> pd.DataFrame:
    """Select required columns, fill missing, enforce dtypes."""
    logger.info("Extracting core fields: %s", core_fields)

    # Keep only columns that exist
    missing = [c for c in core_fields if c not in df.columns]
    if missing:
        logger.warning("Columns not found in WoS export: %s — will be empty", missing)
        for c in missing:
            df[c] = ""

    df = df[core_fields].copy()

    for col, fill_val in fillna_values.items():
        if col in df.columns:
            df[col] = df[col].fillna(fill_val).replace("", fill_val)

    if "PY" in df.columns:
        df["PY"] = pd.to_numeric(df["PY"], errors="coerce")
        current_year = pd.Timestamp.now().year
        df = df.dropna(subset=["PY"])
        df = df[(df["PY"] >= 1900) & (df["PY"] <= current_year + 1)]
        df["PY"] = df["PY"].astype(int)
        if len(df) > 0:
            logger.info("  PY range: %.0f – %.0f", float(df["PY"].min()), float(df["PY"].max()))

    if "TC" in df.columns:
        df["TC"] = pd.to_numeric(df["TC"], errors="coerce").fillna(0).astype(int)

    logger.info("Fields extracted: %d records", len(df))
    return df


def merge_text_columns(df: pd.DataFrame, text_fields: List[str]) -> pd.DataFrame:
    """Merge title, abstract, and keywords into a single ``text`` column.

    Also converts ``ID`` (Keywords Plus) to ``DE_ID`` for broader keyword coverage.
    """
    logger.info("Merging text fields: %s", text_fields)

    # If both DE and ID exist, merge them for richer keywords
    if "ID" in df.columns and "DE" in df.columns:
        df["DE_ID"] = df.fillna("").apply(
            lambda r: "; ".join([x for x in [str(r.get("DE", "")).strip(), str(r.get("ID", "")).strip()] if x]),
            axis=1,
        )
        # Replace DE in text_fields with the merged version
        text_fields = ["DE_ID" if f == "DE" else f for f in text_fields]
        # Also ensure the merged field is in text_fields
        if "DE_ID" not in text_fields:
            text_fields = [f for f in text_fields if f != "DE"] + ["DE_ID"]

    available = [c for c in text_fields if c in df.columns]
    if not available:
        logger.warning("No text fields available — text column will be empty")
        df["text"] = ""
        return df

    # Combine text columns: use apply for compatibility with newer pandas
    df["text"] = df[available].fillna("").apply(lambda row: " ".join(row), axis=1)
    df["text"] = df["text"].str.replace(r"\s+", " ", regex=True).str.strip()

    n_before = len(df)
    df = df[df["text"].str.len() > 0].copy()
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        logger.warning("Dropped %d records with empty text after merge", n_dropped)

    logger.info("Text merge: %d documents (fields: %s)", len(df), available)
    return df


def save_cleaned(df: pd.DataFrame, output_path: Path) -> Path:
    """Save cleaned DataFrame to CSV and Parquet."""
    ensure_dir(output_path.parent)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info("Cleaned data saved: %s (%d records)", output_path, len(df))
    pq_path = output_path.with_suffix(".parquet")
    df.to_parquet(pq_path, index=False)
    logger.info("Parquet copy saved: %s", pq_path)
    return output_path


# =============================================================================
# Main entry point
# =============================================================================


def preprocess(config: Config) -> Path:
    """Run the full preprocessing pipeline."""
    cfg = config.preprocessing
    paths = config.paths

    with Timer("Step 2 — Data Preprocessing"):
        # 1. Find raw files
        files = find_raw_files(paths.raw_dir, cfg.raw_file_pattern)

        # 2. Read and concatenate all files
        dfs = []
        for fp in files:
            try:
                dfs.append(read_wos_file(fp, cfg.encodings))
            except Exception as exc:
                logger.warning("Skipping %s: %s", fp.name, exc)
        if not dfs:
            raise RuntimeError("No valid WoS data files could be read.")
        df = pd.concat(dfs, ignore_index=True)
        log_df_stats(df, "Raw data")

        # 3. Keep all document types (journal articles, proceedings, reviews, etc.)
        logger.info("Keeping all %d records (all document types)", len(df))

        # 4. Deduplicate by DOI
        if cfg.dedup_column in df.columns:
            df = deduplicate_by_doi(df, cfg.dedup_column)
        else:
            logger.warning("Dedup column '%s' not found — skipping. Available: %s", cfg.dedup_column, list(df.columns))

        # 5. Extract core fields
        df = extract_fields(df, cfg.core_fields, cfg.fillna_values)

        # 6. Merge text columns (TI + AB + DE + ID)
        df = merge_text_columns(df, cfg.text_fields)

        # 7. Save
        output_path = paths.processed_dir / cfg.output_file
        save_cleaned(df, output_path)
        log_df_stats(df, "Cleaned data")

    return output_path


def main() -> None:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Step 2: Data Preprocessing")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    config = get_config(args.config)
    from src.utils import setup_logging
    setup_logging(config.system.log_level, config.system.log_file)

    try:
        output = preprocess(config)
        logger.info("Preprocessing complete. Output: %s", output)
    except Exception:
        logger.exception("Preprocessing failed")
        raise


if __name__ == "__main__":
    main()
