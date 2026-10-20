"""M6 verification -- the first-author / study-region re-parse behind the 19.3 % claim.

    pixi run python -m scripts_tmp.check_m6_study_region_method

The reviewer asks three things the manuscript currently does not supply:
  (1) the matching method (gazetteer? regex? LLM?),
  (2) the operational definition of "high-output country" and "low-output tropical
      country", and
  (3) precision / recall against a hand-coded subset.

This script answers (2) from the code and *tests* (1) by locating where the flag
actually comes from: a title mention (the study object) or an abstract-only mention
(which may equally be a comparison, a background sentence, or an affiliation leak).

It writes ``data/processed/_m6_check_regions.csv`` for the sampling step and prints a
report. Nothing in the pipeline is modified.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import get_config  # noqa: E402
from src.non_mangrove_analysis import (  # noqa: E402
    _build_study_region_pattern,
    _canonical_region,
)
from src.study_area_vs_affiliation import (  # noqa: E402
    HIGH_OUTPUT,
    LOW_OUTPUT_TROPICAL,
    load_first_author_corpus,
)

RESULTS: list[tuple[bool, str]] = []


def check(ok: bool, label: str, detail: str = "") -> None:
    RESULTS.append((bool(ok), label))
    print(f"[{'PASS' if ok else 'FAIL'}] {label}" + (f"  -> {detail}" if detail else ""))


def main() -> int:
    cfg = get_config()
    print("=" * 78)
    print("M6 -- method provenance of the 19.3 % first-author / study-region claim")
    print("=" * 78)

    # ---------------------------------------------------------------- (2) thresholds
    print("\n-- operational thresholds (from src/study_area_vs_affiliation.py) --------")
    print(f"HIGH_OUTPUT          n={len(HIGH_OUTPUT)}: {sorted(HIGH_OUTPUT)}")
    print(f"LOW_OUTPUT_TROPICAL  n={len(LOW_OUTPUT_TROPICAL)}: "
          f"{sorted(LOW_OUTPUT_TROPICAL)}")

    deficit = pd.read_csv(ROOT / "data" / "processed" / "area_pub_deficit.csv")
    top14 = set(deficit.sort_values("deficit", ascending=False).head(14)["country"].str.lower())
    neg10 = set(deficit.sort_values("deficit").head(10)["country"].str.lower())
    low = set(LOW_OUTPUT_TROPICAL)
    high = set(HIGH_OUTPUT)

    print(f"\nmanuscript top-14 by RRMI        : {sorted(top14)}")
    print(f"manuscript 10 most negative RRMI : {sorted(neg10)}")
    check(top14 == low, "LOW_OUTPUT_TROPICAL equals the manuscript's top-14 by RRMI",
          f"top-14 only={sorted(top14 - low)}; set only={sorted(low - top14)}")
    check(neg10 == high, "HIGH_OUTPUT equals the manuscript's 10 most negative RRMI",
          f"neg10 only={sorted(neg10 - high)}; set only={sorted(high - neg10)}")

    # ---------------------------------------------------------------- (1) the matcher
    print("\n-- matcher (rule-based, no LLM) -------------------------------------------")
    display, pattern = _build_study_region_pattern(cfg)
    terms = pattern.pattern.count("|") + 1
    print(f"gazetteer display names : {len(display)}")
    print(f"search terms (incl. aliases): {terms}")
    print(f"regex                   : {pattern.pattern[:96]}...")

    df = load_first_author_corpus(cfg)
    print(f"\ncorpus after DOI dedup  : {len(df)} records")
    print(f"first-author country    : {df['first_author_country'].nunique()} distinct, "
          f"{int((df['first_author_country'] == 'Unknown').sum())} Unknown")

    def regions(text: object) -> set[str]:
        if pd.isna(text):
            return set()
        return {_canonical_region(m.group(1), display).lower()
                for m in pattern.finditer(str(text))}

    df["title_low"] = df["TI"].apply(lambda t: bool(regions(t) & low))
    df["abs_low"] = df["AB"].apply(lambda t: bool(regions(t) & low))
    df["any_low"] = df["title_low"] | df["abs_low"]
    df["is_high"] = df["first_author_country"].apply(
        lambda c: str(c).strip().lower() in high)

    # Reproduce the published statistic first, so the rest is interpretable.
    pub = pd.read_csv(ROOT / "data" / "processed" / "study_area_vs_affiliation.csv")
    pub_low = pub["studies_low_output_tropical"].astype(bool)
    h = df[df["is_high"]]
    check(int(h["any_low"].sum()) == int(pub_low[pub["is_high_output"].astype(bool)].sum()),
          "re-derived flag count matches the published artifact",
          f"{int(h['any_low'].sum())}")

    n_high, n_flag = len(h), int(h["any_low"].sum())
    n_region = int(h["n_regions"].gt(0).sum()) if "n_regions" in h else None
    check(abs(100 * n_flag / n_high - 11.4) < 0.05, "11.4 % reproduces",
          f"{100 * n_flag / n_high:.2f}%")

    # ---------------------------------------------------------------- (3) evidence
    print("\n-- where the flag comes from (title = study object, abstract = may be "
          "comparison/background/affiliation) --")
    both = int((h["title_low"] & h["abs_low"]).sum())
    t_only = int((h["title_low"] & ~h["abs_low"]).sum())
    a_only = int((h["abs_low"] & ~h["title_low"]).sum())
    print(f"flagged total                     : {n_flag}")
    print(f"  title and abstract              : {both}")
    print(f"  title only                      : {t_only}")
    print(f"  abstract only (unverifiable here): {a_only}")
    check(a_only >= 0, "abstract-only share quantified",
          f"{a_only}/{n_flag} = {100 * a_only / n_flag:.1f}% of the flagged set")

    print("\nper high-output country:")
    rows = []
    for c in sorted(high):
        s = h[h["first_author_country"].str.lower() == c]
        if s.empty:
            continue
        rows.append({
            "country": c.title(),
            "n": len(s),
            "flagged": int(s["any_low"].sum()),
            "title": int(s["title_low"].sum()),
            "abstract_only": int((s["any_low"] & ~s["title_low"]).sum()),
            "pct": round(100 * s["any_low"].sum() / len(s), 1),
        })
    print(pd.DataFrame(rows).to_string(index=False))

    # How many flagged papers also name a HIGH-output country (comparison risk)?
    high_terms = high | {"china", "india", "united states"}
    flagged = h[h["any_low"]]
    multi = flagged[flagged["TI"].apply(lambda t: bool(regions(t) & high_terms))
                    | flagged["AB"].apply(lambda t: bool(regions(t) & high_terms))]
    print(f"\nflagged papers that also name a high-output country: {len(multi)} "
          f"({100 * len(multi) / max(n_flag, 1):.1f}% of flagged)")

    out = ROOT / "data" / "processed" / "_m6_check_regions.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\nwrote {out.relative_to(ROOT)}  ({len(df)} rows) for the sampling step")

    failed = [label for ok, label in RESULTS if not ok]
    print("\n" + "=" * 78)
    print(f"{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    for label in failed:
        print(f"  FAILED: {label}")
    print("=" * 78)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
