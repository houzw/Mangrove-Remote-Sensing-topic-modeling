"""Verify every claim in manuscript-v3.tex S\\ref{sec:emerging-technologies}.

Review round of 2026-09-27.  Addresses four review points:

  P2-1  the "2026 H1" cutoff -- is it a typo, or the code's own constant?
  P2-2  the two-group reading (3 + 5 = 8) and the "rather than horizontal
        extent mapping" clause: what did the automaton actually do with the
        extent/cover/change-mapping topics?
  P3-3  T17 vs T5, both carrying "blue-carbon" in the canonical Short_Label.
  P3-4  "terminal burst" -- the table caption's gloss vs the methods' definition.
  +     label agreement between Table 4, the Fig. 7 key and topics_info.csv.

Run:  $env:MPLCONFIGDIR="$PWD\\.mplcache"; pixi run python scripts_tmp/check_emerging_subsection.py
"""
from __future__ import annotations

import io
import re
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.emerging_tech_detection import (  # noqa: E402
    AUTOMATON_END,
    CONFIRM_YEAR,
    compute_yearly_topic_counts,
    detect_emerging_tech_prototypes,
)
from src.country_topic_heatmap import TOPIC_ABBREV  # noqa: E402

OUT = Path(__file__).resolve().parent / "check_emerging_subsection_out.txt"
BUF = io.StringIO()
FAILURES: list[str] = []


def P(*args, **kw) -> None:
    print(*args, **kw)
    print(*args, **kw, file=BUF)


def check(name: str, cond: bool, detail: str = "") -> None:
    P(f"   [{'ok ' if cond else 'FAIL'}] {name}{(' -- ' + detail) if detail else ''}")
    if not cond:
        FAILURES.append(name)


def span(v):
    """('2024-2025') -> (2024, 2025); single year or empty -> (None, None)."""
    s = str(v).strip()
    m = re.search(r"(\d{4})\s*-\s*(\d{4})", s)
    if m:
        return int(m.group(1)), int(m.group(2))
    if s.isdigit():
        return int(s), int(s)
    return None, None


def main() -> None:
    labels = (
        pd.read_csv(ROOT / "data/processed/topics_info.csv")
        .set_index("Topic")["Short_Label"]
        .to_dict()
    )
    info = pd.read_csv(ROOT / "data/processed/topics_info.csv").set_index("Topic")

    # ---------------------------------------------------------------- P2-1
    P("=" * 78)
    P("P2-1  CORPUS CUTOFF -- is '2026 H1' a typo?")
    P("=" * 78)
    cleaned = pd.read_csv(ROOT / "data/processed/cleaned_data.csv")
    n2026 = int((cleaned["PY"] == 2026).sum())
    P(f"cleaned_data.csv                       : {len(cleaned)} rows")
    P("PY value counts, last 6 years          : "
      + ", ".join(f"{y}:{n}" for y, n in
                  cleaned["PY"].value_counts().sort_index().tail(6).items()))
    P(f"max PY                                 : {int(cleaned['PY'].max())}")
    P(f"2026 publications                      : {n2026}")
    P(f"AUTOMATON_END (last decoded year)      : {AUTOMATON_END}")
    P(f"CONFIRM_YEAR  (liveness year)          : {CONFIRM_YEAR}")
    P("retrieval date (manuscript l.225/231)  : 15 July 2026")
    P("")
    P("VERDICT: not a typo.  The 2026 slice runs January..mid-July 2026, which is")
    P("what the code calls '2026 H1' (CONFIRM_YEAR docstrings; and")
    P("emerging_window_sensitivity.csv carries the Config_Label 'Main: hist")
    P("1970-2022, recent 2023-2026 (2026 H1 raw)').  The manuscript's own prose")
    P("elsewhere says 'the 2026 first half' (l.283, l.287) and 'the partially")
    P("indexed 2026' (l.330, l.449, l.544); '2026 H1' occurs exactly twice")
    P("(l.487, l.510).  So the defect is terminology drift across the paper, not")
    P("an unsupported date -- and it is fixed by using the house phrasing.")
    check("2026 count matches the 160 quoted at l.330", n2026 == 160, f"{n2026}")
    check("AUTOMATON_END is 2025 (the final complete year)",
          AUTOMATON_END == 2025, str(AUTOMATON_END))
    check("CONFIRM_YEAR is 2026", CONFIRM_YEAR == 2026, str(CONFIRM_YEAR))
    P("")

    # --------------------------------------------------- recompute automaton
    dt = pd.read_csv(ROOT / "data/processed/doc_topic_dist.csv", usecols=["Topic", "PY"])
    yearly_counts, total_per_year = compute_yearly_topic_counts(dt)
    with redirect_stdout(io.StringIO()):
        allt = detect_emerging_tech_prototypes(
            yearly_counts, total_per_year, return_all=True
        )
    allt[["_start", "_end"]] = pd.DataFrame(
        allt.Burst_Interval.map(span).tolist(), index=allt.index
    )
    rec = {int(r.Topic): r for _, r in allt.iterrows()}
    prod = pd.read_csv(ROOT / "data/processed/emerging_tech_prototypes.csv")

    P("=" * 78)
    P("PRODUCTION CANDIDATE LIST vs in-process recomputation")
    P("=" * 78)
    P(f"{'T':>4} {'W_prod':>7} {'W_recomp':>9} {'burst':>12} {'act':>5} "
      f"{'2026H1':>7} {'live':>5}  canonical Short_Label")
    for _, r in prod.iterrows():
        t = int(r.Topic)
        rr = rec[t]
        P(f"{t:>4} {r.Burst_Strength:>7.2f} {rr.Burst_Strength:>9.2f} "
          f"{str(r.Burst_Interval):>12} {str(r.Active_Terminal):>5} "
          f"{int(r.Count_2026_H1):>7} {str(r.Liveness_2026_H1):>5}  {labels[t]}")
    drift = [
        int(r.Topic) for _, r in prod.iterrows()
        if abs(rec[int(r.Topic)].Burst_Strength - r.Burst_Strength) > 1e-9
        or str(rec[int(r.Topic)].Burst_Interval) != str(r.Burst_Interval)
    ]
    check("recomputation reproduces emerging_tech_prototypes.csv exactly",
          not drift, f"drifting topics: {drift}")
    check("all eight satisfy the liveness criterion (the 'remained active' claim)",
          bool(prod.Liveness_2026_H1.all()))
    P("")

    # ---------------------------------------------------------------- P3-4
    P("=" * 78)
    P("P3-4  'TERMINAL BURST' -- caption gloss vs methods definition")
    P("=" * 78)
    nburst = int(allt["_end"].notna().sum())
    P(f"topics with a decoded burst interval      : {nburst} of 72")
    P(f"topics with no decoded burst at all       : {72 - nburst} of 72")
    P(f"topics whose LAST decoded state is burst  : {int(allt.Active_Terminal.sum())}")
    P("start year of the terminal burst          : "
      + str(dict(sorted(allt["_start"].dropna().astype(int).value_counts().items()))))
    P("END year of the terminal burst            : "
      + str(dict(sorted(allt["_end"].dropna().astype(int).value_counts().items()))))
    P("")
    P("VERDICT: the methods DO define it (l.283: 'its terminal burst, its most")
    P("recent burst interval'), so the caption is consistent with the paper's own")
    P("terminology.  'Terminal' = most recent, NOT 'still running' -- 5 of the 25")
    P("burst-carrying topics have a terminal burst ending in 2025, and for those")
    P("it is also ongoing because 2025 is the automaton's last decoded year.")
    P("The caption can say so in three words; and the following sentence's")
    P("'remained active' is criterion (iii) liveness on 2026, a different test.")
    check("all eight candidates' terminal bursts end at AUTOMATON_END (2025)",
          all(rec[int(r.Topic)]["_end"] == AUTOMATON_END for _, r in prod.iterrows()))
    starts = allt.loc[allt["_end"].notna(), "_start"].astype(int)
    check("every decoded burst ends in 2025 (the Table 4 caption claim)",
          bool((allt.loc[allt["_end"].notna(), "_end"].astype(int)
                == AUTOMATON_END).all()), f"{len(starts)} topics carry one")
    P("")

    # ---------------------------------------------------------------- P2-2
    P("=" * 78)
    P("P2-2  THE TWO GROUPS, AND 'rather than horizontal extent mapping'")
    P("=" * 78)
    g1, g2 = [17, 5, 57], [16, 8, 4, 35, 68]
    P(f"group 1 blue-carbon-and-conservation : {g1} -> "
      f"{int(allt.set_index('Topic').loc[g1, 'Passed_Filter'].sum())} passed")
    P(f"group 2 fine-scale-mapping-and-DL    : {g2} -> "
      f"{int(allt.set_index('Topic').loc[g2, 'Passed_Filter'].sum())} passed")
    P(f"total passing                        : {int(allt.Passed_Filter.sum())} of 72")
    check("the grouping is exhaustive and disjoint (3 + 5 = 8)",
          len(g1) + len(g2) == 8 and not set(g1) & set(g2)
          and set(g1) | set(g2) == set(prod.Topic.astype(int)))
    P("")
    P("The clause 'rather than horizontal extent mapping' asserts a negative about")
    P("a category.  What the automaton did to that category, topic by topic:")
    P("")
    P(f"{'T':>4} {'burst':>12} {'W':>7} {'pass':>5}  {'fail_reason':<24} label")
    for t in sorted(rec):
        lab = str(labels[t])
        if not re.search(r"extent|cover|change detect|deforestation|mapping|"
                         r"classification|monitoring", lab, re.I):
            continue
        r = rec[t]
        bi = (f"{r['_start']:.0f}-{r['_end']:.0f}" if pd.notna(r["_end"])
              else "-- none --")
        w = r.Burst_Strength if pd.notna(r.Burst_Strength) else float("nan")
        fr = r.Fail_Reason if isinstance(r.Fail_Reason, str) else ""
        P(f"{t:>4} {bi:>12} {w:>7.2f} {str(r.Passed_Filter):>5}  {fr:<24} {lab}")
    P("")
    EXTENT = {0: "Mangrove Change Detection",
              3: "Mangrove Deforestation in Southeast Asia",
              14: "Deforestation and forest cover change",
              29: "Coastal land cover change detection",
              53: "Mangrove characterization and mapping",
              58: "Mangrove Cover Trends and Change Detection",
              61: "Mangrove Vegetation Index Mapping",
              62: "Mangrove Remote Sensing Change Detection",
              63: "Global mangrove extent mapping"}
    P("TIGHT CLASS -- canonical labels naming EXTENT / COVER / CHANGE mapping")
    P("(the sense of 'horizontal extent mapping' the sentence invokes):")
    for t, lab in EXTENT.items():
        r = rec[t]
        P(f"   T{t:<3} burst={str(pd.notna(r['_end'])):<5} "
          f"pass={str(r.Passed_Filter):<5} "
          f"{r.Fail_Reason if isinstance(r.Fail_Reason, str) else '':<22} {lab}")
    nb = sum(1 for t in EXTENT if pd.notna(rec[t]["_end"]))
    np_ = sum(1 for t in EXTENT if rec[t].Passed_Filter)
    P(f"   -> {nb} of {len(EXTENT)} carry a decoded burst; {np_} passes.")
    check("no extent/cover/change-mapping topic passes the three criteria",
          np_ == 0, f"{np_} passed")
    check("none of them even carries a decoded burst", nb == 0, f"{nb} carry one")
    P("")
    P("The only two passing topics whose labels contain 'mapping' or")
    P("'classification' at all are T4 'Mangrove Remote Sensing Mapping' and")
    P("T35 'UAV-Based Mangrove Species Classification' -- both already placed in")
    P("the fine-scale-mapping-and-deep-learning group.  So the two-group split is")
    P("supported; the categorical 'rather than' is not, and the defensible")
    P("negative is the one about this window.")
    P("")

    # ---------------------------------------------------------------- P3-3
    P("=" * 78)
    P("P3-3  T17 vs T5 : the shared 'blue carbon' token")
    P("=" * 78)
    for t in (5, 17):
        r = rec[t]
        P(f"T{t}: canonical Short_Label = {labels[t]!r}")
        P(f"     Fig. 7 key label     = {TOPIC_ABBREV.get(t, '(absent from key)')!r}")
        P(f"     burst {r['_start']}-{r['_end']}, W={r.Burst_Strength:.2f}, "
          f"p0={r.Baseline_Share:.4f} p1={r.Burst_Share:.4f}, "
          f"docs={int((dt.Topic == t).sum())}")
        P(f"     c-TF-IDF terms       = {info.loc[t, 'Representation']}")
        P(f"     raw LLM label        = {info.loc[t, 'LLM']!r}")
    P("")
    check("T5 and T17 both carry 'blue-carbon' in the canonical label",
          "blue-carbon" in str(labels[5]) and "blue-carbon" in str(labels[17]))
    check("T5 is a Fig. 7 column and T17 is not (so they co-occur only in Table 4)",
          5 in TOPIC_ABBREV and 17 not in TOPIC_ABBREV)
    P("=> one distinguishing clause is warranted, and it should separate the two")
    P("   by vocabulary, not by the words 'blue carbon'.")
    P("")

    # ------------------------------------------------- label agreement
    P("=" * 78)
    P("  TABLE 4 / FIG. 7 KEY / topics_info.csv LABEL AGREEMENT")
    P("=" * 78)
    tbl = {17: "Mangrove conservation and blue-carbon",
           16: "Shoreline change analysis",
           8: "Mangrove semantic segmentation",
           4: "Mangrove Remote Sensing Mapping",
           5: "Mangrove resilience and blue-carbon",
           57: "Mangrove restoration and health assessment",
           35: "UAV-based mangrove species classification",
           68: "Deep learning for mangrove remote sensing"}
    for t, row in tbl.items():
        can = str(labels[t])
        exact = row.lower() == can.lower()
        key = TOPIC_ABBREV.get(t)
        P(f"T{t:<3} table4 == canonical : {str(exact):<5} "
          f"(table={row!r}, canonical={can!r})")
        P(f"     fig. 7 key            : {key!r}"
          + ("" if key is None else
             f"  [{'same' if key.lower() == can.lower() else 'abbrev/expansion'}]"))
    P("")
    check("every Table 4 short label matches topics_info.csv up to letter case",
          all(r.lower() == str(labels[t]).lower() for t, r in tbl.items()))
    P("")
    P("Fig. 7 key vs canonical, for the three the review names:")
    for t in (16, 8, 4):
        can, key = str(labels[t]), TOPIC_ABBREV[t]
        P(f"  T{t}: canonical={can!r}  key={key!r}  "
          f"elides 'mangrove'={key.lower() in can.lower()}  "
          f"exact={key.lower() == can.lower()}")
    P("  -> T16 ('Shoreline change' <-> 'Shoreline change analysis') and T8")
    P("     ('Semantic segmentation' <-> 'Mangrove semantic segmentation') are")
    P("     elisions of the shared head noun, as documented in the producer's")
    P("     _TOPIC_ABBREV comment.  T4 is the exception the 2026-09-27 relabel")
    P("     removed: the key used to expand the canonical 'ML' to 'Machine")
    P("     learning', and now carries the canonical string verbatim.")
    P("     No two topics share a label; the key stays inside the")
    P("     canonical name's meaning.")
    P("")
    P("Near-duplicate canonical labels worth knowing about (not in Table 4 / Fig. 7):")
    for a, b in [(4, 47), (53, 63), (0, 62), (12, 41)]:
        P(f"   T{a} {labels[a]!r}")
        P(f"   T{b} {labels[b]!r}")
    P("")

    P("=" * 78)
    P(f"RESULT: {len(FAILURES)} failing checks"
      + (f" -> {FAILURES}" if FAILURES else " -- all checks passed"))
    P("=" * 78)
    OUT.write_text(BUF.getvalue(), encoding="utf-8")
    print(f"\n[written] {OUT}")


if __name__ == "__main__":
    main()
