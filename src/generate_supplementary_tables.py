"""Generate supplementary LaTeX tables (S1--S3) from data/processed CSVs.

Writes ``manuscript/els-cas-templates/supplementary_tables.tex``, which is
``\\input`` by ``manuscript-v3-appendix.tex``. Stdlib only.

Run: ``pixi run python -m src.generate_supplementary_tables``
"""

from __future__ import annotations

import csv
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "processed")
OUT = os.path.join(ROOT, "manuscript", "els-cas-templates", "supplementary_tables.tex")


def read_rows(name: str) -> list[dict]:
    with open(os.path.join(DATA, name), newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def load_short_labels() -> dict[int, str]:
    """Map topic id -> final Short_Label from topics_info.csv (author-curated).

    Keeps S1/S3 label columns aligned with the topic-label verification table
    and all label-bearing figures, which read the same column.
    """
    with open(os.path.join(DATA, "topics_info.csv"), newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        labels = {}
        for r in reader:
            tid = int(r["Topic"])
            if tid == -1:
                continue
            labels[tid] = r["Short_Label"].strip()
        return labels


def tex(s: str) -> str:
    return s.replace("&", r"\&").replace("%", r"\%").replace("#", r"\#").replace("_", r"\_")


def area(v: float) -> str:
    if v >= 100:
        return f"{v:,.0f}".replace(",", "{,}")
    if v >= 10:
        return f"{v:,.1f}".replace(",", "{,}")
    if v >= 1:
        return f"{v:.1f}"
    return f"{v:.2f}"


def signed(v: float, nd: int = 3) -> str:
    if abs(v) < 0.5 * 10 ** (-nd):
        return f"${0.0:.{nd}f}$"
    return f"${v:+.{nd}f}$"


def comma(v: float, nd: int) -> str:
    s = f"{v:,.{nd}f}"
    return s.replace(",", "{,}")


def dens1000(v: float) -> str:
    """pubs/km^2 -> pubs per 10^3 km^2, 3 significant figures with commas."""
    x = v * 1000.0
    if x >= 100:
        return f"{x:,.0f}".replace(",", "{,}")
    if x >= 10:
        return f"{x:,.1f}".replace(",", "{,}")
    if x >= 1:
        return f"{x:.2f}"
    return f"{x:.3f}"


def ci(v: float) -> str:
    if v >= 1000:
        return f"{v:,.0f}".replace(",", "{,}")
    return f"{v:.1f}"


def build_s1() -> list[str]:
    rows = read_rows("lq_robustness.csv")
    rows.sort(key=lambda r: -float(r["observed_LQ"]))
    # Theme column carries the topic's final Short_Label (author-curated),
    # keeping S1 aligned with the label-verification table and figures.
    labels = load_short_labels()
    L = [
        "% Table S1: full detail behind the main text's LQ niche table",
        r"\begin{table}[H]",
        r"\caption{Full detail behind the main text's LQ niche table: the seven measurable resource-rich countries with the largest positive RRMI and, for each, its signature niche ($\mathrm{LQ}>1$; the country's highest-LQ topic, except for Colombia---see the main text). $N_c$ is the country's topic-assigned fractional document count---unassigned documents are excluded, so it is smaller than the full-corpus count in Table~\ref{tab:S2}---and $N_{c,t}$ its count in the reported topic; perm.\ $p$ is the one-sided permutation-test $p$-value (1{,}000 replicates); shrunk LQ is the empirical-Bayes (beta-binomial) posterior. Nigeria (T50) rests on a single document in a sparse tsunami topic (see text).}\label{tab:S1}",
        r"\setlength{\tabcolsep}{4pt}%",
        r"\begin{tabular*}{\tblwidth}{@{\extracolsep{\fill}}l r l r r r r r@{}}",
        r"\toprule",
        r"Country & Topic & Theme & $N_c$ & $N_{c,t}$ & Observed LQ & Perm.\ $p$ & Shrunk LQ \\",
        r"\midrule",
    ]
    for r in rows:
        c = tex(r["Country"])
        theme = tex(labels[int(r["Topic"])])
        nc = float(r["N_c"])
        nck = float(r["N_ck"])
        lq = float(r["observed_LQ"])
        p = float(r["perm_p_value"])
        slq = float(r["shrunk_LQ"])
        L.append(f"{c} & T{r['Topic']} & {theme} & {nc:.2f} & {nck:.2f} & {lq:.2f} & {p:.3f} & {slq:.2f} \\\\")
    L += [r"\bottomrule", r"\end{tabular*}", r"\setlength{\tabcolsep}{6pt}", r"\end{table}"]
    return L


def build_s2() -> list[str]:
    rows = read_rows("multi_indicator_consistency.csv")  # already sorted by deficit desc
    header = r"Country & Area (km$^2$) & Pubs (frac.) & Citations (frac.) & Mismatch & Pubs per $10^3$~km$^2$ & Citation gap \\"
    L = [
        "% Longtable: multi-indicator consistency (76 countries). Numbering note: the \\label keys tab:S1--S3 follow the STANDALONE supplementary numbering (S1=LQ robustness, S2=this table, S3=bootstrap); in the main-manuscript appendix they follow the label-verification table and therefore DISPLAY as S2--S4. Every \\ref resolves to the correct table in each document.",
        r"\setlength{\tabcolsep}{4pt}%",
        # longtable has no tabular*-style width argument; stretch to full text
        # width with LTleft/LTright=0pt + \extracolsep{\fill} (longtable doc
        # recipe). Caption: use \textwidth, NOT \tblwidth -- the CAS class only
        # defines \tblwidth correctly *inside* a table environment (it resolves
        # to a stale \linewidth value at top level), which is why the S2 caption
        # stayed narrow while the table body stretched.
        r"\setlength{\LTleft}{0pt}%",
        r"\setlength{\LTright}{0pt}%",
        r"\setlength{\LTcapwidth}{\textwidth}%",
        r"\begin{longtable}{@{\extracolsep{\fill}}l r r r r r r@{}}",
        r"\caption{Multi-indicator consistency across the 76 countries with both publication and mangrove-area data, ranked by descending research--resource mismatch. Mismatch $=\mathrm{area\_share}-\mathrm{pub\_share}$; pubs per $10^3$~km$^2$ is the relative output indicator; citation gap $=\mathrm{area\_share}-\mathrm{citation\_share}$ (fractional citation counting). A positive value marks an under-publishing country relative to its own area.}\label{tab:S2}\\",
        r"\toprule",
        header,
        r"\midrule",
        r"\endfirsthead",
        # 76-country table spans pages; label continuation pages like the
        # verification table (renders "Table S3 (continued)" in the main
        # appendix, "Table S2 (continued)" in the standalone supplementary).
        r"\multicolumn{7}{l}{\emph{Table~\ref{tab:S2} (continued)}}\\",
        r"\toprule",
        header,
        r"\midrule",
        r"\endhead",
    ]
    for r in rows:
        c = tex(r["Country"])
        a = area(float(r["area"]))
        pubs = float(r["pubs"])
        cit = float(r["Citations"])
        mismatch = float(r["deficit"])
        d = dens1000(float(r["pub_density"]))
        gap = float(r["citation_gap"])
        L.append(f"{c} & {a} & {comma(pubs, 2)} & {comma(cit, 1)} & {signed(mismatch)} & {d} & {signed(gap)} \\\\")
    L += [r"\bottomrule", r"\end{longtable}", r"\setlength{\tabcolsep}{6pt}"]
    return L


def build_s3() -> list[str]:
    rows = read_rows("prototype_bootstrap_ci.csv")
    rows.sort(key=lambda r: -float(r["Burst_Strength"]))
    # Short label = final topic Short_Label (author-curated); p{} column wraps
    # the longer labels instead of overflowing the table width.
    labels = load_short_labels()
    L = [
        "% Table S3: year-stratified bootstrap 95% CI for candidate burst strengths",
        r"\begin{table}[H]",
        r"\caption{Bootstrap 95\% confidence intervals for the candidate emerging research fronts (1{,}000 year-stratified document resamples with yearly totals fixed; on every replicate the two rates are re-estimated by hard-EM and the state path is re-decoded). ``Re-detected'' is the percentage of replicates in which the terminal burst again satisfies the recency criterion in full; the interval is computed over those replicates only. ``Excludes 0'' indicates whether the interval excludes the null value of no burst ($W=0$).}\label{tab:S3}",
        # p{2.9cm} (was p{4.2cm}): at 4.2cm the 9-column table overran the
        # standalone-article text block (464.9pt) by 33.6pt even at \scriptsize.
        r"\begin{tabular*}{\tblwidth}{@{\extracolsep{\fill}}r r >{\raggedright\arraybackslash}p{2.9cm} r r r c r@{}}",
        r"\toprule",
        r"Rank & Topic & Short label & Burst strength $W$ & 95\% CI low & 95\% CI high & Excludes 0 & Re-detected (\%) \\",
        r"\midrule",
    ]
    for i, r in enumerate(rows, 1):
        label = tex(labels[int(r["Topic"])])
        bi = float(r["Burst_Strength"])
        lo = float(r["CI_low"])
        hi = float(r["CI_high"])
        ex = "Yes" if r["excludes_0"] == "True" else "No"
        det = float(r["pct_detected"])
        L.append(f"{i} & T{r['Topic']} & {label} & {bi:.0f} & {ci(lo)} & {ci(hi)} & {ex} & {det:.1f} \\\\")
    L += [r"\bottomrule", r"\end{tabular*}", r"\end{table}"]
    return L


def main() -> None:
    parts = [
        "% Auto-generated by src/generate_supplementary_tables.py -- do not edit by hand.",
        "% Tables S1--S3 for the mangrove remote-sensing manuscript supplementary material.",
        "",
    ]
    parts += build_s1()
    parts.append("")
    parts += build_s2()
    parts.append("")
    parts += build_s3()
    parts.append("")
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
