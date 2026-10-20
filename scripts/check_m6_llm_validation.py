"""Validate the LLM adjudication against the hand-coded gold set.

    pixi run python -m scripts_tmp.check_m6_llm_validation

Gold: ``m6_validation_coded.csv`` (148 flagged papers, hand-coded provisional labels,
32 of them read in full). Method: ``study_region_llm.csv`` (LLM adjudication of every
high-output paper). Reports agreement, the LLM's own precision and recall against the
gold labels, and the resulting revision of the 18.5 % / 10.9 % pair.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.study_area_vs_affiliation import LOW_OUTPUT_TROPICAL  # noqa: E402

LLM = ROOT / "data" / "processed" / "study_region_llm.csv"
GOLD = ROOT / "data" / "processed" / "m6_validation_coded.csv"
AUDIT = ROOT / "data" / "processed" / "_m6_check_regions.csv"
SUMMARY = ROOT / "data" / "processed" / "study_area_vs_affiliation_summary.json"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 100.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (100 * (c - h), 100 * (c + h))


def main() -> int:
    if not LLM.exists():
        print("run scripts_tmp.llm_adjudicate_regions first")
        return 2

    llm = pd.read_csv(LLM)
    llm["study_low"] = llm["study_countries"].apply(
        lambda v: bool({str(x).strip().lower() for x in json.loads(v.replace("'", '"'))}
                       & LOW_OUTPUT_TROPICAL) if isinstance(v, str) and v.startswith("[")
        else bool({str(x).strip().lower() for x in eval(v)} & LOW_OUTPUT_TROPICAL))

    gold = pd.read_csv(GOLD)
    audit = pd.read_csv(AUDIT).drop_duplicates(subset=["DI"], keep="first")

    m = gold.merge(llm[["doi", "study_low", "llm_ran"]], left_on="DI", right_on="doi",
                   how="left")
    m["gold_yes"] = m["provisional_label"] == "yes"
    m = m[m["llm_ran"] == True]  # noqa: E712
    n = len(m)
    tp = int((m["study_low"] & m["gold_yes"]).sum())
    fp = int((m["study_low"] & ~m["gold_yes"]).sum())
    fn = int((~m["study_low"] & m["gold_yes"]).sum())
    tn = int((~m["study_low"] & ~m["gold_yes"]).sum())
    print("=" * 74)
    print("LLM adjudication vs hand-coded gold (flagged papers with an LLM verdict)")
    print("=" * 74)
    print(f"papers compared : {n}")
    print(f"confusion       : TP={tp} FP={fp} FN={fn} TN={tn}")
    agree = 100 * (tp + tn) / max(n, 1)
    print(f"agreement       : {agree:.1f}%")
    if tp + fp:
        lo, hi = wilson(tp, tp + fp)
        print(f"LLM precision   : {100 * tp / (tp + fp):.1f}%  (95% CI {lo:.1f}-{hi:.1f})")
    if tp + fn:
        lo, hi = wilson(tp, tp + fn)
        print(f"LLM recall      : {100 * tp / (tp + fn):.1f}%  (95% CI {lo:.1f}-{hi:.1f})")
    print(f"\nbaseline (rule-based): precision 65.5%, recall ~100%")

    # Revised headline numbers, if the LLM is adopted as the method.
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    high = llm[llm["llm_ran"] == True]  # noqa: E712
    n_llm_low = int(high["study_low"].sum())
    n_high = summary["n_high_output_papers"]
    n_region = summary["n_high_output_with_region"]
    print(f"\nLLM-based counts over {len(llm)} high-output records:")
    print(f"  study area in a low-output tropical country : {n_llm_low}")
    print(f"  of all high-output publications             : "
          f"{100 * n_llm_low / n_high:.1f}%   (rule-based 10.9%)")
    print(f"  of region-detectable publications           : "
          f"{100 * n_llm_low / n_region:.1f}%   (rule-based 18.5%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
