"""Six-cluster stacked area chart of mangrove remote-sensing topics over time.

Companion to ``src.topic_clusters``: the cluster mapping is the verified Ward
k=6 partition of the 72 topics, and the series is the share of *assigned*
publications per year, aggregated by cluster.

Only complete years are plotted (1972--2025); the partially indexed 2026
(January--July) is excluded from the primary series, consistent with the
manuscript's stated exclusion of 2026 from primary analyses.  A centred 5-year
moving average is applied to suppress the 0/1 proportions of the sparse early
years; because the yearly shares sum to one, the smoothed rows also sum to one.

Supersedes ``tech_evolution_stacked.py`` for the manuscript: that script draws a
three-tier (Traditional / Advanced / Emerging) chart built on the superseded
nine-prototype list and is retained for provenance only.

Usage:
    pixi run python -m src.cluster_evolution_stacked
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.config import get_config
from src.topic_clusters import CLUSTERS, BASE_YEAR, yearly_topic_shares

logger = logging.getLogger(__name__)

# Okabe-Ito colourblind-safe qualitative palette
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00"]

# The five manuscript phases (Table 6).  Phases (i)--(iv) are mutually exclusive, so their
# boundaries are drawn dotted; (v) emerges *inside* (iv), so its boundary is dash-dot and
# marks an overlap rather than a hand-over.
PHASES = [(1972, 1984), (1985, 1998), (1999, 2012), (2013, 2026), (2023, 2026)]
EXCLUSIVE_BOUNDARIES = (1984.5, 1998.5, 2012.5)
OVERLAP_BOUNDARY = 2022.5  # 2023 onward belongs to both (iv) and (v)


def cluster_yearly_shares(config, window: int = 5):
    """Return (years, cluster x year share matrix, cluster labels)."""
    shares = yearly_topic_shares(config)
    years = [y for y in sorted(shares.columns) if 1972 <= y <= BASE_YEAR]
    letters = [c["letter"] for c in CLUSTERS]
    names = [c["name"] for c in CLUSTERS]

    mat = np.zeros((len(letters), len(years)), dtype=float)
    for i, c in enumerate(CLUSTERS):
        members = [t for t in c["members"] if t in shares.index]
        for j, y in enumerate(years):
            mat[i, j] = float(shares.loc[members, y].sum()) if y in shares.columns else 0.0

    if window and window > 1:
        half = window // 2
        smoothed = np.zeros_like(mat)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                lo, hi = max(0, j - half), min(mat.shape[1], j + half + 1)
                smoothed[i, j] = mat[i, lo:hi].mean()
        col = smoothed.sum(axis=0)
        col[col == 0] = 1.0
        mat = smoothed / col  # preserve the composition (columns sum to 1)
    return np.asarray(years), mat, letters, names


def plot_cluster_evolution(config, output_path: Path, window: int = 5) -> Path:
    years, mat, letters, names = cluster_yearly_shares(config, window=window)

    plt.rcParams["font.family"] = config.visualization.font_family
    fig, ax = plt.subplots(figsize=(9.0, 4.8), dpi=300)

    ax.stackplot(
        years,
        mat,
        colors=PALETTE[: len(letters)],
        alpha=0.92,
        edgecolor="white",
        linewidth=0.35,
        labels=[f"{l}: {n} ({len(c['members'])})" for l, n, c in zip(letters, names, CLUSTERS)],
    )

    for x in EXCLUSIVE_BOUNDARIES:
        ax.axvline(x, color="#666666", linestyle=":", linewidth=0.7, alpha=0.55, zorder=5)
    ax.axvline(OVERLAP_BOUNDARY, color="#333333", linestyle="-.", linewidth=0.9,
               alpha=0.8, zorder=5)
    ax.text(OVERLAP_BOUNDARY - 0.5, 0.97, "(iv)/(v) overlap boundary",
            rotation=90, ha="center", va="top", fontsize=6.0, color="#333333",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=0.6))

    ax.set_xlim(1972, BASE_YEAR)
    ax.set_ylim(0, 1.0)
    ax.set_xticks(range(1975, BASE_YEAR + 1, 5))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlabel("Publication year", fontsize=10)
    ax.set_ylabel("Share of assigned publications", fontsize=10)
    ax.grid(axis="y", linestyle=":", alpha=0.4, color="#D5D5D5")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        fontsize=8,
        framealpha=0.9,
        edgecolor="#CCCCCC",
        ncol=3,
        title="Cluster (number of topics)",
        title_fontsize=8.5,
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    logger.info("Six-cluster stacked chart saved: %s", output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Six-cluster stacked evolution chart")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", default=None)
    parser.add_argument("--window", type=int, default=5)
    args = parser.parse_args()

    from src.utils import setup_logging

    config = get_config(args.config)
    setup_logging(config.system.log_level, config.system.log_file)

    out = Path(args.output) if args.output else (
        config.paths.figures_dir / "cluster_evolution_stacked.png"
    )
    years, mat, letters, names = cluster_yearly_shares(config, window=0)
    print("REPORT -- cluster shares of assigned publications (raw, selected years)")
    print(f"  {'year':>5s} " + " ".join(f"{l:>7s}" for l in letters) + "   sum")
    for y in (1980, 1990, 2000, 2010, 2015, 2020, 2024, 2025):
        if y not in years:
            continue
        j = list(years).index(y)
        print(f"  {y:5d} " + " ".join(f"{mat[i, j]:7.4f}" for i in range(len(letters)))
              + f" {mat[:, j].sum():6.4f}")
    plot_cluster_evolution(config, out, window=args.window)


if __name__ == "__main__":
    main()
