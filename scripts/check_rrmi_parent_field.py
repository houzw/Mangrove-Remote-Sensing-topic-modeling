"""Verify the parent-field (remote-sensing) normalization artifact and its claims.

Answers, from the artifacts rather than from the prose:

  P1  definitions: RRMI_norm = s_c - m_c, A = m_c / s_c, both bases, and the
      shares rebuilt from the recorded totals
  P2  the three numbers the revision reports: top-14 overlap, sign agreement,
      A(China) on both bases
  P3  provenance: the parent field's own total, the corpus full count (country
      counts exceed publications), and which countries have no parent record
  P4  why the two benchmarks are near-orthogonal: the exact identity
      RRMI_norm = (a_c - m_c) + (s_c - a_c)
  P5  the LaTeX fragment: both labels, one data row per country, no NUL bytes

Run: pixi run python scripts_tmp/check_rrmi_parent_field.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.generate_supplementary_tables import tex  # noqa: E402

PROC = Path("data/processed")
TEX = Path("manuscript/els-cas-templates/rrmi_parent_field_table.tex")
TOL = 2e-6

art = PROC / "rrmi_parent_field.csv"
if not art.exists():
    print("SKIP  data/processed/rrmi_parent_field.csv is absent - run")
    print("      `pixi run python -m src.rrmi_parent_field` first")
    sys.exit(0)

failures: list[str] = []


def check(condition: bool, label: str, detail: str = "") -> None:
    print(f"  [{'OK  ' if condition else 'FAIL'}] {label}{(' -- ' + detail) if detail else ''}")
    if not condition:
        failures.append(label)


base = pd.read_csv(art, encoding="utf-8-sig")
raw = pd.read_csv(PROC / "area_pub_deficit.csv", encoding="utf-8-sig")
summary = json.loads((PROC / "rrmi_parent_field_summary.json").read_text(encoding="utf-8"))
c = summary["comparison"]
with_p = base[base["P_c"].notna() & (base["P_c"] > 0)]
p_world = float(summary["parent_records_all_countries"])
m_world = float(summary["corpus_full_count_all_countries"])

print("=" * 78)
print("P1  definitions")
print("=" * 78)
check((base["rrmi_norm"] - (base["s_c"] - base["m_c"])).abs().max() < TOL,
      "RRMI_norm = s_c - m_c", f"max residual {(base['rrmi_norm'] - (base['s_c'] - base['m_c'])).abs().max():.2e}")
# Recompute A from the integer counts and the recorded totals: m_c and s_c are
# stored at 6 dp, and for the smallest shares that storage carries ~0.2% error,
# so comparing A against the stored shares would test the rounding, not the code.
_a_recomp = (with_p["M_c"] / m_world) / (with_p["P_c"] / p_world)
_a_resid = (with_p["A"] - _a_recomp).abs().max()
check(_a_resid < 1e-5, "A = (M_c / Sigma M) / (P_c / Sigma P)", f"max residual {_a_resid:.2e}")
check((base["rrmi_norm_76"] - (base["s_c_76"] - base["m_c_76"])).abs().max() < TOL,
      "76-country basis: RRMI_norm_76 = s_c_76 - m_c_76")
check((base["s_c"] - base["P_c"] / p_world).abs().max() < TOL, "s_c uses the parent field's world total",
      f"Sigma P = {p_world:,.0f}")
check((base["m_c"] - base["M_c"] / m_world).abs().max() < TOL, "m_c uses the corpus full count",
      f"Sigma M = {m_world:,.0f} country counts")
check(abs(with_p["s_c_76"].sum() - 1) < 1e-4 and abs(base["m_c_76"].sum() - 1) < 1e-4,
      "both 76-country shares sum to 1",
      f"s {with_p['s_c_76'].sum():.6f}, m {base['m_c_76'].sum():.6f}")

print()
print("=" * 78)
print("P2  the three reported numbers")
print("=" * 78)
top14_pub = list(raw.sort_values("deficit", ascending=False)["country"].head(14))
top14_norm = list(base.sort_values("rrmi_norm", ascending=False)["country"].head(14))
check(top14_norm == c["top14_by_norm"], "top-14 by RRMI_norm matches the summary")
check(len(set(top14_pub) & set(top14_norm)) == 0,
      "top-14 overlap with the published RRMI top-14",
      f"{len(set(top14_pub) & set(top14_norm))} of 14 shared (the two sets are disjoint)")
sign = (base.loc[with_p.index, "rrmi_norm"] > 0) == (base.loc[with_p.index, "rrmi"] > 0)
check(int(sign.sum()) == c["sign_agreement_n"] == 21,
      "sign agreement", f"{int(sign.sum())} of {len(with_p)} = {100 * sign.mean():.0f}%")
china = base.set_index("country").loc["China"]
check(abs(china["A"] - 0.649) < 0.002 and abs(china["A_76"] - 0.517) < 0.002,
      "A(China) on both bases", f"{china['A']:.3f} world, {china['A_76']:.3f} table")
t14 = base[base["country"].isin(top14_pub)]
check(int((t14["A"] > 1).sum()) == 12 and int(t14["A"].notna().sum()) == 12,
      "published top-14: A > 1 in all 12 with a parent record (2 undefined)",
      f"{int((t14['A'] > 1).sum())} of {int(t14['A'].notna().sum())} defined; median {t14['A'].median():.2f}")
check(abs(c["spearman_rrmi_vs_rrmi_norm"] - (-0.338)) < 0.002,
      "rho(published RRMI, RRMI_norm) is negative", f"{c['spearman_rrmi_vs_rrmi_norm']:+.3f}")

print()
print("=" * 78)
print("P3  provenance")
print("=" * 78)
check(summary["parent_records_in_table"] <= p_world,
      "table's parent records do not exceed the world total",
      f"{summary['parent_records_in_table']:,.0f} of {p_world:,.0f} "
      f"({100 * summary['parent_records_table_share']:.1f}%)")
n_docs = 2332
check(m_world > n_docs,
      "full-count total exceeds the publication count (multi-country papers)",
      f"Sigma M = {m_world:,.0f} country counts for {n_docs:,} publications")
no_parent = sorted(base.loc[base["P_c"].isna(), "country"])
check(len(no_parent) == 7, "countries with no parent record", ", ".join(no_parent))
check(float(with_p["P_c"].min()) >= 5,
      "smallest mapped parent count respects the export's display floor",
      f"min = {with_p['P_c'].min():.0f} (the export lists no country below 5)")

print()
print("=" * 78)
print("P4  why the two benchmarks are near-orthogonal")
print("=" * 78)
resid = (base["rrmi_norm"] - (base["area_share"] - base["m_c"]) - (base["s_c"] - base["area_share"])).abs().max()
check(resid < TOL, "RRMI_norm = (a_c - m_c) + (s_c - a_c) exactly",
      f"max residual {resid:.2e}; the second term is the capacity-versus-resource gap")

print()
print("=" * 78)
print("P5  the LaTeX fragment")
print("=" * 78)
check(TEX.exists(), f"{TEX.name} exists")
if TEX.exists():
    blob = TEX.read_bytes()
    check(b"\x00" not in blob, "no NUL bytes in the fragment")
    text = blob.decode("utf-8")
    check(r"\label{tab:S7}" in text and r"\label{tab:S8}" in text, "both labels present")
    # Neither ampersand counting nor line terminators identify a data row
    # reliably (a country name may contain "&"; caption and label lines also end
    # in "\\"), so check for the row each country must contribute.
    missing = [c for c in base["country"] if f"{tex(c)} & " not in text]
    check(not missing, "one data row per country",
          f"{len(base) - len(missing)} of {len(base)} present; missing: {missing or 'none'}")

print()
print("=" * 78)
if failures:
    print(f"FAILED {len(failures)} check(s): " + "; ".join(failures))
    sys.exit(1)
print("All checks passed.")
