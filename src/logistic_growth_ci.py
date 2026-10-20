"""P1-2: Logistic growth K/t_mid 95% CI and annual-increments fit.

The cumulative 3-parameter logistic ``f(t)=L/(1+exp(-k(t-t0)))`` is fitted on
the complete years (the retrieval-truncated final corpus year is excluded from
the production fit and retained as a sensitivity entry), and the NLS covariance
matrix is used to derive 95% confidence intervals for the carrying capacity
``K``, the intrinsic growth rate ``r=k``, and the inflection year ``t_mid``.
A second fit of the *annual increments* (the derivative of the logistic) is
reported for both window choices; on the sparse pre-inflection data it does not
identify the asymptote, which is itself evidence of weak ``L``/``k``
identifiability.

Outputs:
    data/processed/logistic_growth_ci.json   fit + CI summary
    data/processed/logistic_growth_ci_report.md

Usage:
    pixi run python -m src.logistic_growth_ci
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit

from src.bibliometric_analysis import compute_cumulative, logistic_function
from src.config import get_config
from src.utils import ensure_dir, setup_logging

logger = logging.getLogger(__name__)


def _annual_increment(t: np.ndarray, L: float, k: float, t0: float) -> np.ndarray:
    """Derivative of the 3-param logistic: ``L*k*exp(-k(t-t0)) / (1+exp(-k(t-t0)))^2``."""
    z = np.exp(-k * (t - t0))
    return L * k * z / (1.0 + z) ** 2


def _fit(model, t: np.ndarray, y: np.ndarray, p0, bounds, maxfev: int):
    popt, pcov = curve_fit(model, t, y.astype(float), p0=p0, bounds=bounds, maxfev=maxfev)
    perr = np.sqrt(np.diag(pcov))
    return popt, pcov, perr


def run(config) -> dict:
    paths = config.paths
    cleaned_path = paths.processed_dir / config.preprocessing.output_file
    if not cleaned_path.exists():
        raise FileNotFoundError(f"Cleaned data not found: {cleaned_path}")

    df = pd.read_csv(cleaned_path, encoding="utf-8-sig")
    yearly = compute_cumulative(df)

    # The final corpus year (2026, retrieved 2026-07-15) is only partially
    # indexed: complete years enter the production fits, and fits that retain
    # the truncated year are kept as an explicit sensitivity entry.
    fit_end_year = 2025
    fit_mask = yearly["PY"] <= fit_end_year

    x = yearly.loc[fit_mask, "PY"].values.astype(float)
    cumulative = yearly.loc[fit_mask, "cumulative"].values.astype(float)
    annual = yearly.loc[fit_mask, "count"].values.astype(float)
    t_offset = float(x.min())
    t_norm = x - t_offset

    b = config.logistic_fit.bounds
    p0 = config.logistic_fit.p0
    bounds = ([b.L[0], b.k[0], b.t0[0]], [b.L[1], b.k[1], b.t0[1]])
    maxfev = config.logistic_fit.maxfev

    n = len(x)  # complete years used in the fits
    dfree = n - 3
    tcrit = stats.t.ppf(0.975, dfree)

    # ---- Cumulative fit ----
    popt, pcov, perr = _fit(logistic_function, t_norm, cumulative, p0, bounds, maxfev)
    L, k, t0 = popt
    t_mid = t_offset + t0
    y_pred = logistic_function(t_norm, *popt)
    ss_res = float(np.sum((cumulative - y_pred) ** 2))
    ss_tot = float(np.sum((cumulative - np.mean(cumulative)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    cum = {
        "K": float(L),
        "r": float(k),
        "t_mid": float(t_mid),
        "R2": r_squared,
        "K_ci95": [float(L - tcrit * perr[0]), float(L + tcrit * perr[0])],
        "r_ci95": [float(k - tcrit * perr[1]), float(k + tcrit * perr[1])],
        "t_mid_ci95": [float(t_mid - tcrit * perr[2]), float(t_mid + tcrit * perr[2])],
        "K_se": float(perr[0]),
        "r_se": float(perr[1]),
        "t_mid_se": float(perr[2]),
        "peak_annual": float(k * L / 4.0),
        "df": int(dfree),
    }

    # ---- Annual-increments fit (complete years) ----
    popt_inc, _pcov_inc, perr_inc = _fit(_annual_increment, t_norm, annual, p0, bounds, maxfev)
    L_i, k_i, t0_i = popt_inc
    inc_complete = {
        "K": float(L_i),
        "r": float(k_i),
        "t_mid": float(t_offset + t0_i),
        "peak_annual": float(k_i * L_i / 4.0),
        "K_se": float(perr_inc[0]),
        "r_se": float(perr_inc[1]),
        "t_mid_se": float(perr_inc[2]),
    }

    # ---- Sensitivity: re-fit including the truncated final year ----
    tn_all = yearly["PY"].values.astype(float) - t_offset
    cum_all = yearly["cumulative"].values.astype(float)
    ann_all = yearly["count"].values.astype(float)
    tcrit_all = stats.t.ppf(0.975, len(tn_all) - 3)
    popt_ct, _pcov_ct, perr_ct = _fit(logistic_function, tn_all, cum_all, p0, bounds, maxfev)
    L_ct, k_ct, t0_ct = popt_ct
    ss_res_ct = float(np.sum((cum_all - logistic_function(tn_all, *popt_ct)) ** 2))
    ss_tot_ct = float(np.sum((cum_all - np.mean(cum_all)) ** 2))
    cum_truncated = {
        "K": float(L_ct),
        "r": float(k_ct),
        "t_mid": float(t_offset + t0_ct),
        "R2": 1.0 - ss_res_ct / ss_tot_ct if ss_tot_ct > 0 else 0.0,
        "K_ci95": [float(L_ct - tcrit_all * perr_ct[0]), float(L_ct + tcrit_all * perr_ct[0])],
        "t_mid_ci95": [
            float(t_offset + t0_ct - tcrit_all * perr_ct[2]),
            float(t_offset + t0_ct + tcrit_all * perr_ct[2]),
        ],
        "note": "includes the retrieval-truncated final year",
    }
    popt_at, _pcov_at, perr_at = _fit(_annual_increment, tn_all, ann_all, p0, bounds, maxfev)
    L_at, k_at, t0_at = popt_at
    inc_truncated = {
        "K": float(L_at),
        "r": float(k_at),
        "t_mid": float(t_offset + t0_at),
        "peak_annual": float(k_at * L_at / 4.0),
        "K_se": float(perr_at[0]),
        "r_se": float(perr_at[1]),
        "t_mid_se": float(perr_at[2]),
        "note": "includes the retrieval-truncated final year",
    }

    # ---- Phase boundaries (10%/90%) from cumulative fit ----
    ln9 = np.log(9.0)
    t10 = t_offset + t0 - ln9 / k
    t90 = t_offset + t0 + ln9 / k

    # ---- Empirical peak year (complete years only) ----
    yearly_complete = yearly.loc[fit_mask]
    peak_year = int(yearly_complete.loc[yearly_complete["count"].idxmax(), "PY"])
    peak_count = int(yearly_complete["count"].max())

    summary = {
        "n_records": int(len(df)),
        "n_years": int(n),
        "n_years_all": int(len(yearly)),
        "t_offset": t_offset,
        "fit_end_year": fit_end_year,
        "cumulative_fit": cum,
        "cumulative_fit_with_truncated_year": cum_truncated,
        "annual_increments": inc_complete,
        "annual_increments_with_truncated_year": inc_truncated,
        "phase_boundaries": {"10%": t10, "90%": t90},
        "empirical_peak": {"year": peak_year, "count": peak_count},
    }

    out_dir = paths.processed_dir
    ensure_dir(out_dir)
    json_path = out_dir / "logistic_growth_ci.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    logger.info("CI summary saved: %s", json_path)

    report = _compose_report(summary)
    report_path = out_dir / "logistic_growth_ci_report.md"
    report_path.write_text(report, encoding="utf-8")
    logger.info("Report saved: %s", report_path)

    print("\n" + "=" * 72)
    print("P1-2 LOGISTIC GROWTH CI + ANNUAL-INCREMENTS FIT")
    print("=" * 72)
    print(report)
    return summary


def _fmt_ci(lo: float, hi: float, nd: int = 0) -> str:
    return f"[{lo:,.{nd}f}, {hi:,.{nd}f}]"


def _compose_report(s: dict) -> str:
    c = s["cumulative_fit"]
    ct = s["cumulative_fit_with_truncated_year"]
    inc = s["annual_increments"]
    inc_t = s["annual_increments_with_truncated_year"]
    lines = [
        "## P1-2 Logistic growth CI + annual-increments fit",
        "",
        f"Corpus: {s['n_records']} records; fits use the {s['n_years']} complete "
        f"years {s['t_offset']:.0f}--{s['fit_end_year']} "
        f"({s['n_years_all']} corpus-covered years including the truncated final year).",
        "",
        "### Cumulative 3-param logistic fit (complete years; production)",
        f"- K (carrying capacity) = {c['K']:,.0f}  "
        f"95% CI {_fmt_ci(*c['K_ci95'])}  (SE {c['K_se']:,.0f})",
        f"- r (growth rate) = {c['r']:.4f} yr^-1  95% CI "
        f"[{c['r_ci95'][0]:.4f}, {c['r_ci95'][1]:.4f}]",
        f"- t_mid (inflection) = {c['t_mid']:.1f}  95% CI "
        f"[{c['t_mid_ci95'][0]:.1f}, {c['t_mid_ci95'][1]:.1f}]  (SE {c['t_mid_se']:.2f})",
        f"- R^2 = {c['R2']:.4f}; peak annual = {c['peak_annual']:.0f}",
        "",
        "### Sensitivity: cumulative fit including the truncated final year",
        f"- K = {ct['K']:,.0f}  95% CI {_fmt_ci(*ct['K_ci95'])}; "
        f"t_mid = {ct['t_mid']:.1f}  95% CI [{ct['t_mid_ci95'][0]:.1f}, "
        f"{ct['t_mid_ci95'][1]:.1f}]; R^2 = {ct['R2']:.4f}",
        "",
        "### Annual-increments fit (derivative of the logistic)",
        f"- Complete years: K={inc['K']:,.0f}, r={inc['r']:.4f}, "
        f"t_mid={inc['t_mid']:.1f}, peak_annual={inc['peak_annual']:.0f}",
        f"- With truncated year: K={inc_t['K']:,.0f}, r={inc_t['r']:.4f}, "
        f"t_mid={inc_t['t_mid']:.1f}, peak_annual={inc_t['peak_annual']:.0f}",
        "",
        "### Phase boundaries (10% / 90% of K)",
        f"- 10% year = {s['phase_boundaries']['10%']:.1f}; 90% year = "
        f"{s['phase_boundaries']['90%']:.1f}",
        "",
        f"### Empirical annual peak (complete years): {s['empirical_peak']['year']} "
        f"({s['empirical_peak']['count']} publications)",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="P1-2 logistic CI")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    config = get_config(args.config)
    setup_logging(config.system.log_level, config.system.log_file)
    run(config)


if __name__ == "__main__":
    main()
