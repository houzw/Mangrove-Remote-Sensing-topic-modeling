"""User (human) calls vs the model, on the 48 rows filled so far."""

from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd

ROOT = Path(".")
SHEET = ROOT / "data/processed/m6_spotcheck_148_with_evidence.csv"


def read_any(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="gbk")


d = read_any(SHEET)
low = set(pd.read_csv(ROOT / "data/processed/area_pub_deficit.csv")
          .sort_values("deficit", ascending=False).head(14)["country"]
          .str.strip().str.lower())


def norm(v):
    s = str(v).strip().lower()
    if s in ("yes", "y", "true", "1"):
        return True
    if s in ("no", "n", "false", "0"):
        return False
    return None


d["human"] = d["your_call"].apply(norm)
d["model"] = d["model_says_low_is_study_area"].astype(str).str.lower().map(
    {"true": True, "false": False})
m = d[d["human"].notna() & d["model"].notna()]
print(f"rows with both a human call and a model call: {len(m)}")

yes_h = int(m["human"].sum())
yes_m = int(m["model"].sum())
both = int((m["human"] & m["model"]).sum())
agree = (m["human"] == m["model"]).mean()
print(f"human YES {yes_h} | model YES {yes_m} | both YES {both} | "
      f"raw agreement {100 * agree:.1f}%")

tp = int((m["human"] & m["model"]).sum())
fp = int((~m["human"] & m["model"]).sum())   # model says study area, human says no
fn = int((m["human"] & ~m["model"]).sum())   # human says study area, model says no
tn = int((~m["human"] & ~m["model"]).sum())
n = len(m)
po = (tp + tn) / n
pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (n * n)
print(f"confusion vs the model: TP={tp} FP={fp} FN={fn} TN={tn}  kappa={(po-pe)/(1-pe):.3f}")
print(f"  model precision against your calls: {100 * tp / max(tp + fp, 1):.1f}%")

print("\nthe rows where you and the model differ:")
dis = m[m["human"] != m["model"]]
for _, r in dis.iterrows():
    print(f"  {r['review_id']} | you={'yes' if r['human'] else 'no':3s} "
          f"model={'yes' if r['model'] else 'no':3s} | {r['coding_author_country']} {r['year']}")
    print(f"      gz={r['gazetteer_low_countries']} | notes: {str(r['your_notes'])[:90]}")
    print(f"      {str(r['title'])[:88]}")
