"""Verify every claim in manuscript-v3.tex S\\ref{sec:emerging-technologies}.

Round of 2026-09-27. Addresses four review points:

  P2-1  the "2026 H1" cutoff is a real, code-backed constant (CONFIRM_YEAR),
        not a typo -- and the corpus retrieval date bounds it.
  P2-2  the two-group reading (3 + 5 = 8) and the "rather than horizontal
        extent mapping" clause: which topics carry an extent-mapping label,
        and what did the automaton do with them?
  P3-3  T17 vs T5 both carry "blue-carbon" in the canonical Short_Label.
  P3-4  "terminal burst" -- is it defined in the methods, and does it mean
        "the last burst interval" or "a burst still running"?
  +     label agreement between Table 4, the heatmap key and topics_info.csv.

Run:  pixi run python scripts_tmp/check_emerging_subsection.py
"""
from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.emerging_tech_detection import (  # noqa: E402
    CONFIRM_YEAR,
    AUTOMATON_END,
    compute_yearly_topic_counts,
    detect_emerging_tech_prototypes,
)
from src.country_topic_heatmap import TOPIC_ABBREV  # noqa: E402

OUT = Path(__file__).resolve().parent / "check_emerging_subsection_out.txt"
BUF = io.StringIO()


def P(*args, **kw) -> None:
    print(*args, **kw)
    print(*args, **kw, file=BUF)


def main() -> None:
    P("=" * 78)
    P("P2-1  CORPUS CUTOFF  ('2026 H1')")
    P("=" * 78)
    cleaned = pd.read_csv(ROOT / "data/processed/cleaned_data.csv")
    P(f"cleaned_data.csv rows                 : {len(cleaned)}")
    P(f"PY value counts (last 6 years)        :")
    for y, n in cleaned["PY"].value_counts().sort_index().tail(6).items():
        P(f"    {y}: {n}")
    P(f"AUTOMATON_END (last decoded year)     : {AUTOMATON_END}")
    P(f"CONFIRM_YEAR  (liveness year)         : {CONFIRM_YEAR}")
    P("retrieval date, per manuscript l.225/231 : 15 July 2026")
    P("=> 2026 slice = January..mid-July 2026. 'H1' is the code's own name for it")
    P("   (CONFIRM_YEAR docstrings, emerging_window_sensitivity.csv Config_Label")
    P("   'Main: hist 1970-2022, recent 2023-2026 (2026 H1 raw)'); it is NOT a typo.")
    P("   But the paper's prose elsewhere says 'the 2026 first half' (l.283, l.287)")
    P("   and 'the partially indexed 2026' (l.330, l.449, l.544). '2026 H1' occurs")
    P("   exactly twice (l.487, l.510) -> terminology drift, not a date error.")
    P(f"   Also: 2026 carries {int((cleaned['PY'] == 2026).sum())} publications, matching")
    P("   the 160 quoted in the Results (l.330); the methods say 'seven months'.")
    P("")

    # ---- recompute the automaton over all 72 topics ------------------------
    dt = pd.read_csv(ROOT / "data/processed/doc_topic_dist.csv", usecols=["Topic", "PY"])
    yearly_counts, total_per_year = compute_yearly_topic_counts(dt)
    with redirect_stdout(io.StringIO()):
        allt = detect_emerging_tech_prototypes(
            yearly_counts, total_per_year, return_all=True
        )
    labels = pd.read_csv(ROOT / "data/processed/topics_info.csv").set_index("Topic")[
        "Short_Label"
    ].to_dict()

    prod = pd.read_csv(ROOT / "data/processed/emerging_tech_prototypes.csv")
    P("=" * 78)
    P("PRODUCTION CANDIDATE LIST (emerging_tech_prototypes.csv) vs recomputation")
    P("=" * 78)
    P(f"{'T':>4} {'W_prod':>7} {'W_recomp':>9} {'burst':>12} {'active':>7} "
      f"{'2026H1':>7} {'live':>5}  label")
    rec = {int(r.Topic): r for _, r in allt.iterrows()}
    for _, r in prod.iterrows():
        t = int(r.Topic)
        rr = rec[t]
        P(f"{t:>4} {r.Burst_Strength:>7.2f} {rr.Burst_Strength:>9.2f} "
          f"{str(r.Burst_Interval):>12} {str(r.Active_Terminal):>7} "
          f"{int(r.Count_2026_H1):>7} {str(r.Liveness_2026_H1):>5}  {labels[t]}")
    P("")

    P("=" * 78)
    P("P3-4  'TERMINAL BURST' -- what the column means")
    P("=" * 78)
    P(f"topics whose LAST decoded state is burst (Active_Terminal=True) : "
      f"{int(allt.Active_Terminal.sum())} of {len(allt)}")
    def _end_year(v):
        s = str(v).strip()
        m = pd.Series([s]).str.extract(r"(\d{4})\s*-\s*(\d{4})")[0]
        if not m.isna().iloc[0]:
            return int(m.iloc[0])
        return int(s) if s.isdigit() else None

    def _span(v):
        s = str(v).strip()
        m = pd.Series([s]).str.extract(r"(\d{4})\s*-\s*(\d{4})")
        if not m.isna().iloc[0, 0]:
            return int(m.iloc[0, 0]), int(m.iloc[0, 1])
        return (int(s), int(s)) if s.isdigit() else (None, None)

    allt["_start"], allt["_end"] = zip(*allt.Burst_Interval.map(_span))
    allt["_end"] = allt.Burst_Interval.map(lambda v: _end_year(v))
    P(f"topics with a decoded burst interval                            : "
      f"{int(allt['_end'].notna().sum())} of {len(allt)}")
    P(f"topics with NO decoded burst                                    : "
      f"{int(allt['_end'].isna().sum())} of {len(allt)}")
    P(f"start year of the terminal burst, distribution                  : "
      f"{dict(sorted(allt['_start'].dropna().astype(int).value_counts().items()))}")
    P(f"END year of the terminal burst, distribution                    : "
      f"{dict(sorted(allt['_end'].dropna().astype(int).value_counts().items()))}")
    P("=> 'terminal' = the MOST RECENT burst interval, not 'still running'.")
    P("   Methods l.283 defines it: 'its terminal burst, its most recent burst")
    P("   interval'.  For all eight candidates that interval ends in 2025, the")
    P("   last year the automaton decodes -- so it is also ongoing, which is why")
    P("   the caption's 'terminal burst interval' and the text's 'remained active'")
    P("   read as the same fact.  They are not: 'remained active' is criterion")
    P("   (iii) liveness, evaluated on 2026, a separate test.")
    P("")

    P("=" * 78)
    P("P2-2  THE TWO GROUPS AND 'rather than horizontal extent mapping'")
    P("=" * 78)
    P(f"Group 1 blue-carbon-and-conservation : T17 T5 T57  -> "
      f"{allt.set_index('Topic').loc[[17, 5, 57], 'Passed_Filter'].sum()} passed")
    P(f"Group 2 fine-scale-mapping-and-DL    : T16 T8 T4 T35 T68 -> "
      f"{allt.set_index('Topic').loc[[16, 8, 4, 35, 68], 'Passed_Filter'].sum()} passed")
    P(f"total passing                        : {int(allt.Passed_Filter.sum())} "
      f"of 72  (3 + 5 = 8)")
    P("")
    KEY = ("extent", "cover", "change detection", "deforestation", "area",
           "mapping", "classification", "characterization", "monitoring")
    ext = [
        t for t, lab in labels.items()
        if any(k in str(lab).lower() for k in KEY)
    ]
    P("Topics whose canonical Short_Label names an extent/cover/mapping operation")
    P("(candidate reading of 'horizontal extent mapping'), with their automaton fate:")
    P(f"{'T':>4} {'burst':>12} {'active':>7} {'W':>7} {'Pass':>5}  fail_reason / label")
    for t in sorted(ext):
        r = rec[t]
        bi = r.Burst_Interval if pd.notna(r.Burst_Interval) else "-- none --"
        fr = r.Fail_Reason if isinstance(r.Fail_Reason, str) else ""
        P(f"{t:>4} {str(bi):>12} {str(r.Active_Terminal):>7} "
          f"{(r.Burst_Strength if pd.notna(r.Burst_Strength) else float('nan')):>7.2f} "
          f"{str(r.Passed_Filter):>5}  {fr:<28} {labels[t]}")
    P("")
    n_ext_burst = sum(1 for t in ext if rec[t]["_end"] is not None)
    P(f"=> {n_ext_burst} of these {len(ext)} mapping-labelled topics carry a decoded")
    P(f"   burst at all; {sum(1 for t in ext if rec[t].Passed_Filter)} pass all three criteria.")
    # The tight class the manuscript's clause is about
    EXTENT = {0: "Mangrove Change Detection",
              3: "Mangrove Deforestation in Southeast Asia",
              14: "Deforestation and forest cover change",
              29: "Coastal land cover change detection",
              53: "Mangrove characterization and mapping",
              58: "Mangrove Cover Trends and Change Detection",
              61: "Mangrove Vegetation Index Mapping",
              62: "Mangrove Remote Sensing Change Detection",
              63: "mangrove extent changes mapping",
              40: "Mangrove Remote Sensing Monitoring",
              41: "Mangrove remote sensing classification"}
    P("")
    P("TIGHT CLASS -- labels naming EXTENT / COVER / CHANGE mapping (the clause's target):")
    for t, lab in EXTENT.items():
        r = rec[t]
        P(f"   T{t:<3} burst={str(r['_end'] is not None):<5} pass={r.Passed_Filter!s:<5} "
          f"{r.Fail_Reason if isinstance(r.Fail_Reason, str) else '':<24} {lab}")
    nb = sum(1 for t in EXTENT if rec[t]["_end"] is not None)
    np_ = sum(1 for t in EXTENT if rec[t].Passed_Filter)
    P(f"   -> {nb} of {len(EXTENT)} carry a decoded burst; {np_} passes all three criteria.")
    P("   The manuscript's 'rather than horizontal extent mapping' asserts a")
    P("   NEGATIVE about a category; what the data support is the narrower claim")
    P("   that no extent-mapping topic meets the three criteria in this window.")
    P("")

    P("=" * 78)
    P("P3-3  T17 vs T5 : the shared 'blue carbon' token")
    P("=" * 78)
    for t in (5, 17):
        r = rec[t]
        P(f"T{t}: {labels[t]!r}")
        P(f"     heatmap key label : {TOPIC_ABBREV.get(t, '(not in top-15 key)')!r}")
        P(f"     burst {r.Burst_Interval}, W={r.Burst_Strength:.2f}, "
          f"p0={r.Baseline_Share:.4f} p1={r.Burst_Share:.4f}, "
          f"docs={int((dt.Topic == t).sum())}")
        rep = pd.read_csv(ROOT / "data/processed/topics_info.csv").set_index("Topic")
        P(f"     c-TF-IDF terms    : {rep.loc[t, 'Representation']}")
        P(f"     LLM raw label     : {rep.loc[t, 'LLM']}")
    P("")
    P("=> The shared token is 'blue carbon'; T5 is a resilience/sequestration")
    P("   vocabulary and T17 a conservation/policy one.  Both are in the top-15")
    P(f"   heatmap columns? T5={5 in TOPIC_ABBREV}, T17={17 in TOPIC_ABBREV}")
    P("")

    P("=" * 78)
    P("  TABLE 4 / HEATMAP / topics_info.csv LABEL AGREEMENT")
    P("=" * 78)
    tbl = {
        17: "Mangrove conservation and blue-carbon",
        16: "Shoreline change analysis",
        8: "Mangrove semantic segmentation",
        4: "ML mangrove mapping",
        5: "Mangrove resilience and blue-carbon",
        57: "Mangrove restoration and health assessment",
        35: "UAV-based mangrove species classification",
        68: "Deep learning for mangrove remote sensing",
    }
    P(f"{'T':>4} {'in heatmap 15':>13}  canonical / table / abbrev")
    for t in sorted(tbl, key=lambda x: list(tbl).index(x)):
        can = labels[t]
        P(f"{t:>4} {str(t in TOPIC_ABBREV):>13}  canonical={can!r}")
        P(f"{'':>4} {'':>13}  table4   ={tbl[t]!r}")
        P(f"{'':>4} {'':>13}  key      ={TOPIC_ABBREV.get(t, '(absent)')!r}")
    P("")
    P("Column check against the 15 heatmap columns actually drawn:")
    cols = pd.read_csv(ROOT / "data/processed/spatial_topic_si.csv")
    P(f"spatial_topic_si.csv columns/rows head:")
    P(cols.head(3).to_string()[:400])

    OUT.write_text(BUF.getvalue(), encoding="utf-8")
    print(f"\n[written] {OUT}")


if __name__ == "__main__":
    main()
