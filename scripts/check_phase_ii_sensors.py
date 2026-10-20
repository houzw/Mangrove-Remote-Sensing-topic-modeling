"""Phase (ii) 1985-1998: was the era "optical", or genuinely multi-source?

Table 6 calls it "Multi-source optical" while its own cell lists ERS-1/2, JERS-1 and
RADARSAT-1 — all SAR. This counts, among the corpus records of that window, how many name an
optical platform and how many name a SAR one, so the name can be set to what the data shows.
"""

from __future__ import annotations

import pandas as pd

OPTICAL = ("landsat", "thematic mapper", " tm ", "spot", "irs-", "optical", "multispectral",
           "panchromatic", "photograph", "aerial")
RADAR = ("sar", "radar", "ers-1", "ers-2", "jers", "radarsat", "synthetic aperture", "backscatter",
         "polarimetric", "insar")

d = pd.read_csv("data/processed/cleaned_data.csv", encoding="utf-8-sig")
w = d[(d["PY"] >= 1985) & (d["PY"] <= 1998)]
text = w["text"].fillna("").str.lower()

n_opt = text.apply(lambda t: any(k in t for k in OPTICAL))
n_rad = text.apply(lambda t: any(k in t for k in RADAR))
print(f"corpus records 1985-1998: {len(w)}")
print(f"  name an optical platform : {int(n_opt.sum())}  ({n_opt.mean():.0%})")
print(f"  name a SAR/radar platform: {int(n_rad.sum())}  ({n_rad.mean():.0%})")
print(f"  both                     : {int((n_opt & n_rad).sum())}")
print(f"  neither                  : {int((~n_opt & ~n_rad).sum())}")

print("\nplatform terms in that window:")
for lab, keys in (("optical", OPTICAL), ("SAR", RADAR)):
    hits = {k: int(text.str.contains(k, regex=False).sum()) for k in keys}
    hits = {k: v for k, v in sorted(hits.items(), key=lambda kv: -kv[1]) if v}
    print(f"  {lab}: {hits}")

print("\nfirst records of the window:")
for _, r in w.sort_values("PY").head(8).iterrows():
    print(f"  PY{int(r['PY'])}  {str(r['TI'])[:88]}")
