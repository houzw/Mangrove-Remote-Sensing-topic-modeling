"""Inter-coder reliability: deepseek-flash vs deepseek-v4-pro on the flagged set.

    pixi run python -m scripts_tmp.check_m6_intercoder

Both coders saw the same prompt and the same 143 flagged papers (the 5 flagged papers
without a DOI are missing from both, so the pairwise comparison is identity-aligned and
the gap is reported rather than hidden). Reports the 2x2 table, raw agreement and
Cohen's kappa, plus every disagreement with each coder's accepted country list.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.study_area_vs_affiliation import LOW_OUTPUT_TROPICAL  # noqa: E402

C1 = ROOT / "data" / "processed" / "study_region_llm.csv"
C2 = ROOT / "data" / "processed" / "study_region_llm_coder2.csv"
AUDIT = ROOT / "data" / "processed" / "_m6_check_regions.csv"
OUT = ROOT / "data" / "processed" / "m6_intercoder_disagreements.csv"


def parse(v):
    if isinstance(v, list):
        return v
    try:
        return ast.literal_eval(v) if isinstance(v, str) and v.strip().startswith("[") else []
    except Exception:  # noqa: BLE001
        return []


def low_yes(series) -> pd.Series:
    return series.apply(lambda v: bool({str(x).strip().lower() for x in parse(v)}
                                       & LOW_OUTPUT_TROPICAL))


def kappa(tp: int, fp: int, fn: int, tn: int) -> float:
    n = tp + fp + fn + tn
    if n == 0:
        return float("nan")
    po = (tp + tn) / n
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (n * n)
    return (po - pe) / (1 - pe) if pe != 1 else float("nan")


def main() -> int:
    a = pd.read_csv(AUDIT)
    flagged = a[a["flag_any"] == True].copy()  # noqa: E712  (audit is high-output only)
    c1 = pd.read_csv(C1)
    c2 = pd.read_csv(C2)
    c1["yes1"] = low_yes(c1["study_countries"])
    c2["yes2"] = low_yes(c2["study_countries"])

    m = (flagged[["DI", "PY", "first_author_country", "TI", "flag_title", "regions_all"]]
         .merge(c1[["doi", "yes1", "study_countries"]], left_on="DI", right_on="doi")
         .merge(c2[["doi", "yes2", "study_countries"]], on="doi", suffixes=("_c1", "_c2")))
    n = len(m)
    tp = int((m["yes1"] & m["yes2"]).sum())
    fp = int((m["yes1"] & ~m["yes2"]).sum())
    fn = int((~m["yes1"] & m["yes2"]).sum())
    tn = int((~m["yes1"] & ~m["yes2"]).sum())

    po = (tp + tn) / n
    k = kappa(tp, fp, fn, tn)
    print("=" * 76)
    print("Inter-coder reliability -- deepseek-flash (coder 1) vs deepseek-v4-pro (coder 2)")
    print("=" * 76)
    print(f"papers both coders scored : {n} of {len(flagged)} flagged "
          f"({len(flagged) - n} have no DOI and are absent from both)")
    print(f"confusion (coder1 x coder2): TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"raw agreement             : {100 * po:.1f}%")
    print(f"Cohen's kappa             : {k:.3f}")
    print(f"coder 1 positives {int(m['yes1'].sum())} | coder 2 positives {int(m['yes2'].sum())}")

    dis = m[m["yes1"] != m["yes2"]].copy()
    dis["only_coder1"] = dis.apply(
        lambda r: "yes" if r["yes1"] else "no", axis=1)
    print(f"\ndisagreements: {len(dis)} "
          f"({100 * len(dis) / n:.1f}%; coder 1 accepts, coder 2 rejects "
          f"{fp}; the reverse {fn})")
    for _, r in dis.head(12).iterrows():
        print(f"  [{r['first_author_country']} {r['PY']}] {str(r['TI'])[:66]}")
        print(f"      c1={r['study_countries_c1']}  c2={r['study_countries_c2']}")

    print("\nstratum breakdown (does agreement hold where it matters?):")
    for label, sub in (("low country in TITLE", m[m["flag_title"] == True]),   # noqa: E712
                       ("abstract only", m[m["flag_title"] == False])):       # noqa: E712
        if sub.empty:
            continue
        t = int((sub["yes1"] & sub["yes2"]).sum())
        f_ = int((sub["yes1"] & ~sub["yes2"]).sum())
        n_ = int((~sub["yes1"] & sub["yes2"]).sum())
        tn_ = int((~sub["yes1"] & ~sub["yes2"]).sum())
        tot = len(sub)
        print(f"  {label:22s} n={tot:3d}  agreement={100 * (t + tn_) / tot:5.1f}%  "
              f"kappa={kappa(t, f_, n_, tn_):+.3f}  (c1 pos {int(sub['yes1'].sum())}, "
              f"c2 pos {int(sub['yes2'].sum())})")

    keep = ["DI", "PY", "first_author_country", "TI", "flag_title", "regions_all",
            "yes1", "yes2", "study_countries_c1", "study_countries_c2"]
    dis[keep].to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
