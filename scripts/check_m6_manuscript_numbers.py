"""Verify every number the revised Sec. 4.3 sentence now asserts (standing rule 1).

    pixi run python -m scripts_tmp.check_m6_manuscript_numbers

Checks the claim sentences against the *regenerated* artifacts and prints the
title/abstract split and the corpus-scope screen that Supplementary Table S6 needs.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.non_mangrove_analysis import (  # noqa: E402
    _build_study_region_pattern,
    _REGION_SEARCH_ALIASES,
)
from src.study_area_vs_affiliation import (  # noqa: E402
    HIGH_OUTPUT,
    LOW_OUTPUT_TROPICAL,
    _resolve_country_sets,
)
from src.config import get_config  # noqa: E402

RESULTS: list[tuple[bool, str]] = []


def check(ok: bool, label: str, detail: str = "") -> None:
    RESULTS.append((bool(ok), label))
    print(f"[{'PASS' if ok else 'FAIL'}] {label}" + (f"  -> {detail}" if detail else ""))


def main() -> int:
    cfg = get_config()
    summary = json.loads(
        (ROOT / "data" / "processed" / "study_area_vs_affiliation_summary.json")
        .read_text(encoding="utf-8")
    )
    tex = (ROOT / "manuscript" / "els-cas-templates" / "manuscript-v4.tex").read_text(
        encoding="utf-8")

    print("=" * 78)
    print("Sec. 4.3 numbers -- manuscript text vs regenerated artifacts")
    print("=" * 78)

    # (a) headline numbers under the adopted fractional accounting
    summary_llm = json.loads(
        (ROOT / "data" / "processed" / "study_area_llm_summary.json")
        .read_text(encoding="utf-8"))
    frac = summary_llm["n_study_low_output_fractional"]
    pct = summary_llm["pct_of_all_high_output_fractional"]
    binary = summary_llm["n_study_low_output_binary"]
    check(f"{pct:.1f}\\%" in tex, f"fractional share {pct:.1f} % is in the text")
    check(f"equivalent of {frac:.0f} study-area papers" in tex,
          "the fractional count is stated as an equivalent, not a paper count")
    check(f"against {binary} (" in tex, "the binary sensitivity number is in the text")
    check("contributes $1/n$ to each" in tex, "the 1/n rule is stated in the methods")
    h = summary_llm["human_spotcheck"]
    check(f"{h['human_yes']} of them against the model's {h['model_yes']}" in tex,
          "the human spot-check counts are in the text")
    check(f"{h['kappa']:.3f}" in tex, "the human-model kappa is in the text")
    check(f"{h['fp']} disagreements" in tex, "the overcount number is in the text")
    for stale in ("75.7\\%", "89.4\\%", "66.9--82.7", "65.5\\%", "97/148",
                  "116 of the 1{,}353", "8.6\\%) place their"):
        check(stale not in tex, f"stale value {stale} is gone")

    # (c) the country sets are the artifact-derived ones (not the old hard-coded pair)
    high, low = _resolve_country_sets(cfg)
    check(high == set(HIGH_OUTPUT) and low == set(LOW_OUTPUT_TROPICAL),
          "module sets equal the derived sets")
    check("guinea-bissau" in low and "malaysia" not in low,
          "LOW_OUTPUT_TROPICAL = Guinea-Bissau, not Malaysia", sorted(low))
    check("saudi arabia" in high and "iran" not in high,
          "HIGH_OUTPUT = Saudi Arabia, not Iran", sorted(high))
    check(len(high) == 10 and len(low) == 14, "set sizes 10 / 14",
          f"{len(high)} / {len(low)}")
    check(not (high & low), "the two sets are disjoint")

    # (d) gazetteer size claims: 105 territories + Taiwan, 109 search terms
    display, pattern = _build_study_region_pattern(cfg)
    n_terms = pattern.pattern.count("|") + 1
    check(len(display) == 105, "gazetteer = 104 HGMF territories + Taiwan, 105 after alias merge",
          f"{len(display)} display names")
    check(n_terms == 109, "109 search terms", f"{n_terms}")
    check(sorted(_REGION_SEARCH_ALIASES) and n_terms > len(display),
          "aliases are folded into the term list",
          f"{len(display)} names -> {n_terms} terms")

    # (e) title vs abstract split -- the S6 input
    reg = pd.read_csv(ROOT / "data" / "processed" / "_m6_check_regions.csv")
    h = reg  # the audit is already high-output scoped
    fl = h[h["flag_any"] == True]  # noqa: E712
    both = int((fl["flag_title"] & fl["flag_abstract"]).sum())
    t_only = int((fl["flag_title"] & ~fl["flag_abstract"]).sum())
    a_only = int((fl["flag_abstract"] & ~fl["flag_title"]).sum())
    print("\n-- flagged-set split (Supplementary Table S6 input) --")
    print(f"flagged total        : {len(fl)}")
    print(f"  title (and abstract): {both}")
    print(f"  title only          : {t_only}")
    print(f"  abstract only       : {a_only}")
    print(f"  named in title      : {both + t_only} "
          f"({100 * (both + t_only) / len(fl):.1f}% of flagged)")
    check(both + t_only + a_only == len(fl), "the three strata account for every flagged paper")

    failed = [label for ok, label in RESULTS if not ok]
    print("\n" + "=" * 78)
    print(f"{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    for label in failed:
        print(f"  FAILED: {label}")
    print("=" * 78)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
