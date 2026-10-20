"""What did the corpus actually use in phase (i), 1972-1984?

Table 6 names 1972-1984 "SAR-monolithic" but lists Landsat-1-5 MSS alongside Seasat SAR and
SIR-A/B. This prints the early records and the sensor vocabulary they use, so the phase name
can be checked against the evidence.
"""

from __future__ import annotations

import pandas as pd

SENSORS = ("landsat", "mss", "thematic mapper", "spot", "seasat", "sir-a", "sir-b", "shuttle imaging",
           "sar", "synthetic aperture", "radar", "aerial", "photograph", "airborne")
SENSOR_LABEL = ("Landsat", "MSS", "TM", "SPOT", "Seasat", "SIR-A", "SIR-B", "SAR", "radar", "aerial photo")

d = pd.read_csv("data/processed/cleaned_data.csv", encoding="utf-8-sig")
early = d[d["PY"] <= 1984].sort_values("PY")
print(f"corpus records 1972-1984: {len(early)} of {len(d)}")

for _, r in early.iterrows():
    text = str(r["text"]).lower()
    hits = [lab for lab in SENSOR_LABEL if lab.lower() in text]
    print(f"  PY{int(r['PY'])}  {str(r['TI'])[:86]}")
    print(f"        sensors: {hits or 'none named'}")
    print(f"        journal: {str(r['SO'])[:70]}")

print("\nSensor vocabulary for 1972-1984 records (whole corpus, for contrast):")
for lab in SENSOR_LABEL:
    n_early = early["text"].fillna("").str.lower().str.contains(lab.lower()).sum()
    n_all = d["text"].fillna("").str.lower().str.contains(lab.lower()).sum()
    print(f"  {lab:<12} {n_early:>3} early / {n_all:>4} corpus")
