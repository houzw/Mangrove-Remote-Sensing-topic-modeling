"""Recall probe for M6: which control papers actually study a low-output country?

    pixi run python -m scripts_tmp.check_m6_recall

Control population: high-output papers WITH a detectable study region but WITHOUT the
low-output-tropical flag. A paper whose TITLE names a low-output tropical country is a
certain false negative (the study area is right there in the title and the gazetteer
did not flag it). That gives a defensible lower bound on the false-negative rate
without hand-coding 1,205 abstracts.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.study_area_vs_affiliation import LOW_OUTPUT_TROPICAL  # noqa: E402


def main() -> None:
    df = pd.read_csv(ROOT / "data" / "processed" / "_m6_check_regions.csv")
    high = df[df["is_high"] == True].copy()  # noqa: E712
    flagged = high[high["flag_any"] == True]  # noqa: E712
    ctrl = high[~high["flag_any"].astype(bool)].copy()

    print(f"flagged (positives)              : {len(flagged)}")
    print(f"control (unflagged high-output)  : {len(ctrl)}")

    def title_low(title: object) -> bool:
        t = str(title)
        return any(re.search(rf"(?<![A-Za-z]){re.escape(n)}(?![A-Za-z])", t, re.I)
                   for n in LOW_OUTPUT_TROPICAL)

    ctrl["title_names_low"] = ctrl["TI"].apply(title_low)
    fn = ctrl[ctrl["title_names_low"]]
    print(f"\ncertain false negatives (low-output country in the TITLE): {len(fn)}")
    for _, r in fn.iterrows():
        print(f"  [{r['first_author_country']} {r['PY']}] {str(r['TI'])[:92]}")

    # Recall is estimated over the flagged-plus-certain-FN frame, which is the
    # closest this design gets to a gold standard without full hand coding.
    tp = len(flagged)
    recall = 100 * tp / (tp + len(fn))
    print(f"\nrecall (lower bound, title rule) = {tp}/({tp}+{len(fn)}) = {recall:.1f}%")
    print("  note: abstract-only false negatives are not counted here, so the true "
          "recall is at or below this value.")

    # Precision impact: of the flagged, how many are confirmed by a title mention?
    conf = flagged[flagged["flag_title"].astype(bool)]
    print(f"\nflagged with the country in the title (near-certain) : {len(conf)} "
          f"({100 * len(conf) / len(flagged):.1f}%)")
    print(f"flagged abstract-only (precision 65.5% applies)      : "
          f"{len(flagged) - len(conf)}")

    coded = pd.read_csv(ROOT / "data" / "processed" / "m6_validation_coded.csv")
    yes = int((coded["provisional_label"] == "yes").sum())
    n = len(coded)
    print(f"\nprecision on the flag = {yes}/{n} = {100 * yes / n:.1f}%")
    print(f"implied confirmed share of region-detectable high-output papers:")
    base = 18.5
    print(f"  18.5% x {100 * yes / n:.1f}% = {base * yes / n:.1f}%")
    print(f"  10.9% x {100 * yes / n:.1f}% = {10.9 * yes / n:.1f}%")


if __name__ == "__main__":
    main()
