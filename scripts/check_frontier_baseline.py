"""Round check: outlier denominator (771 / 33.1%) and the newly-born-frontier
pseudocount question behind the "strongest signals" sentence.

Part 1 -- P2-2: rebuild the document-level outlier ledger from
``data/processed/doc_topic_dist.csv`` and check whether 771 and 33.1% share the
2,332-document corpus denominator (the reviewer's reverse-derivation implied an
unstated intermediate of ~2,330 entering HDBSCAN).

Part 2 -- P2-1: for each of the eight candidate fronts, split the automaton's
year series into the baseline state and the burst state, and report how many
documents actually sit in the baseline state.  For a newly born front with zero
baseline documents, p0 = (0 + 0.5) / (T_base + 1) is fixed by the Jeffreys
pseudocount alone; we then re-run the whole estimator at pseudocounts
0.05 / 0.5 / 5.0 to see how much of W is pseudocount-driven, and whether the
within-class ranking of the three newly born fronts survives.

Run: pixi run python -m scripts_tmp.check_frontier_baseline
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import burst_detection_kleinberg as bk  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DOC_TOPIC = ROOT / "data" / "processed" / "doc_topic_dist.csv"
PROTOTYPES = ROOT / "data" / "processed" / "emerging_tech_prototypes.csv"

AUTOMATON_END = 2025
GAMMA = 1.0


def load_yearly():
    df = pd.read_csv(DOC_TOPIC)
    df["PY"] = pd.to_numeric(df["PY"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["PY", "Topic"])
    df["Topic"] = df["Topic"].astype(int)

    print("=" * 78)
    print("PART 1 -- outlier / denominator ledger (doc_topic_dist.csv)")
    print("=" * 78)
    total = len(df)
    outlier = int((df["Topic"] == -1).sum())
    assigned = int((df["Topic"] != -1).sum())
    print(f"documents in file            : {total}")
    print(f"  Topic == -1 (unassigned)   : {outlier}   ({outlier / total:.4%})")
    print(f"  topic-assigned (0..71)     : {assigned}")
    print(f"  outlier + assigned         : {outlier + assigned}")
    print(f"distinct topic ids           : {df.loc[df['Topic'] != -1, 'Topic'].nunique()}")
    print(f"reviewer's implied denominator: 771 / 0.331 = {771 / 0.331:.1f}")
    print(f"771 / 2332                   : {771 / total:.4%}  -> rounds to 33.1%")
    print("no document is dropped between the corpus and HDBSCAN: the model was fit")
    print("on all 2,332 documents and -1 is one of its labels, so 771 and 33.1% share")
    print("the same denominator (no unstated ~2,330 intermediate exists).")

    counts = df.groupby(["PY", "Topic"]).size().unstack(fill_value=0)
    total_per_year = df.groupby("PY").size().to_dict()
    total_per_year = {int(k): int(v) for k, v in total_per_year.items()}

    print("\nannual totals are topic-inclusive (the -1 column stays in the")
    print("denominator, matching 'the annual corpus total T_y' in Sec 2.6):")
    for y in (2015, 2020, 2025, 2026):
        if y in total_per_year:
            n_out = int(counts.loc[y, -1]) if -1 in counts.columns else 0
            print(f"  {y}: T_y = {total_per_year[y]:4d}   of which -1 = {n_out:3d}")
    return counts, total_per_year


def year_series(counts: pd.DataFrame, total_per_year: dict):
    all_years = sorted(int(y) for y in counts.index)
    complete = [y for y in all_years if y <= AUTOMATON_END]
    T = np.array([total_per_year[y] for y in complete], dtype=float)
    return complete, T


def state_split(N: np.ndarray, T: np.ndarray, p0: float, p1: float):
    """Pointwise E-step split, exactly as _hard_em_once does it."""
    c0 = bk._neg_binom_ll(N, T, p0)
    c1 = bk._neg_binom_ll(N, T, p1)
    mask1 = c1 < c0
    return mask1


def main() -> None:
    counts, total_per_year = load_yearly()
    complete, T = year_series(counts, total_per_year)
    proto = pd.read_csv(PROTOTYPES)
    topics = [int(t) for t in proto["Topic"]]
    newly_born = {8, 16, 68}

    print("\n" + "=" * 78)
    print("PART 2 -- what actually drives p0 (and W) for each candidate front")
    print("=" * 78)
    hdr = (f"{'T':>4} {'class':>10} {'p0':>9} {'p1':>9} {'W':>7} "
           f"{'N_base':>7} {'T_base':>7} {'N_burst':>8} {'onset':>6}")
    print(hdr)
    print("-" * len(hdr))
    rows = {}
    viterbi_differs = []
    for tid in topics:
        N = np.array(
            [int(counts.loc[y, tid]) if tid in counts.columns else 0 for y in complete],
            dtype=float,
        )
        det = bk.detect_recent_burst(np.array(complete), N, T, gamma=GAMMA)
        p0, p1, W = det["p0"], det["p1"], det["burst_strength"]
        mask1 = state_split(N, T, p0, p1)
        n_base, t_base = float(N[~mask1].sum()), float(T[~mask1].sum())
        n_burst = float(N[mask1].sum())
        cls = "newly born" if tid in newly_born else "resurgent"
        tb = det["terminal_burst"]
        print(f"{tid:>4} {cls:>10} {p0:9.6f} {p1:9.6f} {W:7.2f} "
              f"{n_base:7.0f} {t_base:7.0f} {n_burst:8.0f} {tb[0]:>6}")
        # Does the Viterbi path used for W agree with the EM assignment that
        # sets p0?  If not, "baseline state" must be described as the EM
        # assignment, not as the decoded path.
        vit = np.asarray(det["states"]).astype(bool)
        n_base_vit = float(N[~vit].sum())
        if not np.array_equal(vit, mask1):
            viterbi_differs.append((tid, n_base, n_base_vit))
        rows[tid] = dict(cls=cls, p0=p0, p1=p1, W=W, n_base=n_base,
                         t_base=t_base, n_base_vit=n_base_vit)
        expected_p0 = (n_base + bk.JEFFREYS_PSEUDOCOUNT) / (t_base + 2 * bk.JEFFREYS_PSEUDOCOUNT)
        assert abs(expected_p0 - p0) < 1e-12, (tid, expected_p0, p0)

    print(f"\nEM baseline assignment vs Viterbi path: "
          f"{len(viterbi_differs)} of {len(topics)} candidates differ")
    for tid, n_em, n_vit in viterbi_differs:
        print(f"  T{tid}: baseline documents EM {n_em:.0f} vs Viterbi {n_vit:.0f}")

    print("\nidentity p0 = (N_base + 0.5) / (T_base + 1) holds for all eight.")
    for tid in sorted(newly_born):
        r = rows[tid]
        print(f"  T{tid}: N_base = {r['n_base']:.0f} documents over {r['t_base']:.0f} "
              f"baseline-year publications -> p0 = 0.5 / {r['t_base'] + 1:.0f} "
              f"= {r['p0']:.6f}, i.e. 100% pseudocount.")

    print("\n" + "=" * 78)
    print("PART 2b -- pseudocount sweep (the paper's value is 0.5)")
    print("=" * 78)
    sweep = {}
    original = bk.JEFFREYS_PSEUDOCOUNT
    try:
        for pc in (0.05, 0.5, 5.0):
            bk.JEFFREYS_PSEUDOCOUNT = pc
            out = {}
            for tid in topics:
                N = np.array(
                    [int(counts.loc[y, tid]) if tid in counts.columns else 0
                     for y in complete], dtype=float)
                det = bk.detect_recent_burst(np.array(complete), N, T, gamma=GAMMA)
                out[tid] = (det["p0"], det["p1"], det["burst_strength"], det["terminal_burst"])
            sweep[pc] = out
    finally:
        bk.JEFFREYS_PSEUDOCOUNT = original

    def fmt(v: float | None) -> str:
        return "  -  " if v is None else f"{v:5.1f}"

    print("burst strength W (a dash = no terminal burst at that setting)")
    print(f"{'pc':>6} | " + " | ".join(f"T{t:<4}" for t in topics))
    for pc, out in sweep.items():
        print(f"{pc:>6} | " + " | ".join(fmt(out[t][2]) for t in topics))

    print("\nterminal burst interval (and W change against pc=0.5)")
    print(f"{'T':>4} | " + " | ".join(f"{pc:>18}" for pc in sweep))
    for tid in topics:
        cells = []
        for pc, out in sweep.items():
            tb = out[tid][3]
            w = out[tid][2]
            ref = sweep[0.5][tid][2]
            if tb is None or w is None:
                cells.append(f"{'none':>18}")
            else:
                delta = "" if pc == 0.5 else f" {(w / ref - 1) * 100:+5.1f}%"
                cells.append(f"{str(tb[0]) + '-' + str(tb[1]):>12}{delta:>6}")
        print(f"{tid:>4} | " + " | ".join(cells))

    for label, members in (("resurgent", [t for t in topics if t not in newly_born]),
                           ("newly born", sorted(newly_born))):
        print(f"\nrank order within the {label} class, by pseudocount:")
        for pc, out in sweep.items():
            def key(t, out=out):
                w = out[t][2]
                return -1e9 if w is None else -w
            order = sorted(members, key=key)
            print(f"  pc = {pc:<5}: " + " > ".join(f"T{t}" for t in order))
        print("  pseudocount share of p0's numerator, 0.5 / (N_base + 0.5):")
        print("    " + ", ".join(
            f"T{t} {0.5 / (rows[t]['n_base'] + 0.5):.0%}" for t in members))

    print("\n" + "=" * 78)
    print("PART 3 -- how a topic with pre-2021 documents can hold none in the")
    print("          baseline state (T5, T57): the pointwise E-step sends a")
    print("          sparse early year with an above-burst share to state 1")
    print("=" * 78)
    for tid in (5, 57):
        N = np.array(
            [int(counts.loc[y, tid]) if tid in counts.columns else 0 for y in complete],
            dtype=float)
        det = bk.detect_recent_burst(np.array(complete), N, T, gamma=GAMMA)
        mask1 = state_split(N, T, det["p0"], det["p1"])
        docs = [(y, int(n), int(t), n / t if t else 0.0)
                for y, n, t in zip(complete, N, T) if n > 0]
        print(f"\nT{tid}: p0 = {det['p0']:.6f}, p1 = {det['p1']:.6f}, "
              f"terminal burst {det['terminal_burst']}")
        print("  years with >=1 document (year, docs, annual total, share, state):")
        for y, n, t, s in docs:
            st = "burst" if mask1[complete.index(y)] else "baseline"
            print(f"    {y}  n={n:3d}  T_y={t:4d}  share={s:.4f}  -> {st}")


if __name__ == "__main__":
    main()
