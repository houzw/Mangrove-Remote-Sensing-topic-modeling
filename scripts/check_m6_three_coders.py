"""Cross-vendor agreement on the flagged set: DeepSeek flash, DeepSeek v4-pro, GLM, authors.

    pixi run python -m scripts_tmp.check_m6_three_coders

All three models ran the same prompt over the same 148 flagged publications at temperature 0
with JSON output, and the authors hand-coded the same set. Every coder is mapped onto the
flagged keys in one fixed order before any pairwise statistic is computed, so no comparison can
silently use a different frame.
"""

from __future__ import annotations

import ast
import itertools
from pathlib import Path

import pandas as pd

ROOT = Path(".")
AUDIT = ROOT / "data/processed/_m6_check_regions.csv"
C1 = ROOT / "data/processed/study_region_llm.csv"
C2 = ROOT / "data/processed/study_region_llm_coder2.csv"
GLM = ROOT / "data/processed/study_region_llm_glm.csv"
SHEET = ROOT / "data/processed/m6_spotcheck_148_with_evidence.csv"
OUT = ROOT / "data/processed/m6_three_coder_agreement.csv"


def parse(v):
    if isinstance(v, list):
        return v
    if isinstance(v, str) and v.strip()[:1] in "[{":
        try:
            return ast.literal_eval(v)
        except Exception:  # noqa: BLE001
            return []
    return []


def read_any(p: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(p, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(p, encoding="gbk")


def kappa(a: pd.Series, b: pd.Series) -> tuple:
    m = pd.DataFrame({"a": a, "b": b}).dropna()
    n = len(m)
    if n == 0:
        return (float("nan"), float("nan"), 0)
    m = m.astype(bool)
    tp = int((m["a"] & m["b"]).sum())
    fp = int((~m["a"] & m["b"]).sum())
    fn = int((m["a"] & ~m["b"]).sum())
    tn = int((~m["a"] & ~m["b"]).sum())
    po = (tp + tn) / n
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (n * n)
    k = (po - pe) / (1 - pe) if pe != 1 else float("nan")
    return (round(100 * po, 1), round(k, 3), n)


def main() -> int:
    low = set(pd.read_csv(ROOT / "data/processed/area_pub_deficit.csv")
              .sort_values("deficit", ascending=False).head(14)["country"]
              .str.strip().str.lower())

    audit = pd.read_csv(AUDIT).reset_index(drop=True)
    flagged = audit[audit["flag_any"]].reset_index(drop=True)
    keys = flagged["m6_key"].astype(str).tolist()
    print(f"flagged publications in the comparison frame: {len(keys)}")

    def flag_map(df: pd.DataFrame, key_col: str) -> dict:
        return {str(k): bool({str(x).strip().lower() for x in parse(v)} & low)
                for k, v in zip(df[key_col], df["study_countries"])}

    flash = flag_map(pd.read_csv(C1), "doi")
    glm = flag_map(pd.read_csv(GLM), "m6_key")
    v4 = flag_map(pd.read_csv(C2), "doi")

    sheet = read_any(SHEET)
    for col in sheet.columns:
        sheet[col] = sheet[col].fillna("").astype(str).replace("nan", "")
    human = {str(k): (c.strip().lower() in ("yes", "y"))
             for k, c in zip(sheet["m6_key"], sheet["your_call"])}

    frame = pd.DataFrame({
        "DeepSeek-flash": [flash.get(k) for k in keys],
        "DeepSeek-v4-pro": [v4.get(k) for k in keys],
        "GLM-5.3-Flash": [glm.get(k) for k in keys],
        "Authors": [human.get(k) for k in keys],
    })

    print("=" * 78)
    print("Cross-vendor agreement on the 148 flagged publications")
    print("=" * 78)
    rows = []
    for a, b in itertools.combinations(frame.columns, 2):
        po, k, n = kappa(frame[a], frame[b])
        rows.append({"coder_a": a, "coder_b": b, "n": n,
                     "agreement_pct": po, "kappa": k})
        print(f"{a:16s} vs {b:16s} n={n:3d}  agreement={po:5.1f}%  kappa={k:+.3f}")
    pd.DataFrame(rows).to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"\nwrote {OUT.relative_to(ROOT)}")

    print("\npositives and coverage per coder:")
    for name in frame.columns:
        s = frame[name]
        print(f"  {name:16s} positives {int(s.fillna(False).sum()):3d} / "
              f"scored {int(s.notna().sum()):3d}")

    dis = frame[(frame["GLM-5.3-Flash"].notna()) & (frame["DeepSeek-flash"].notna())
                & (frame["GLM-5.3-Flash"] != frame["DeepSeek-flash"])]
    print(f"\nGLM vs DeepSeek-flash disagreements: {len(dis)} of {len(keys)}")
    for i in dis.index[:10]:
        print(f"    flash={bool(frame.at[i, 'DeepSeek-flash'])} "
              f"glm={bool(frame.at[i, 'GLM-5.3-Flash'])} | {str(flagged.at[i, 'TI'])[:64]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
