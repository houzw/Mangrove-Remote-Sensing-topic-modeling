"""Manuscript Fig. 7 (candidate B): top topics per country, redesigned for print.

This is a print-sized rebuild of ``plot_popular_topics_by_country`` in
``src/spatial_topic_analysis.py`` -- the artifact the Fig. 7 caption currently
points at. The metric is unchanged (topic *popularity*: the sum of a country's
yearly topic proportions, Hu et al. 2025 style), but the geometry is not:

* the old chart was laid out at 20 x 11 in for 12 countries and then placed at
  ``\\columnwidth``, a 0.33x downscale that turned every 6.5 pt bar label into
  ~2 pt of ink;
* the old chart wrote the topic label at the *bar tip*, so label length and bar
  length competed for the same space;
* this rebuild is drawn at the final 6.50-in printed width, moves the topic
  labels to the y-axis where their length is not coupled to the bar length, and
  prints 8 countries instead of 12 so each panel has room for a 5.5 pt label.

Run: ``pixi run python -m src.country_topic_bars``
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

from src.config import Config, get_config
from src.country_topic_heatmap import TOPIC_ABBREV
from src.spatial_topic_analysis import (
    _COUNTRY_NAME_MAPPING_DEFAULT,
    compute_topic_popularity,
    load_topic_labels,
    merge_country_topic,
)
from src.utils import ensure_dir

logger = logging.getLogger(__name__)

_PRINT_WIDTH_IN = 6.50


def build_country_topic_bars(
    config: Config,
    top_n_countries: int = 8,
    top_n_topics: int = 5,
) -> dict:
    """Rank countries by summed topic popularity and take each one's top topics."""
    merged = merge_country_topic(config)
    mapping = {
        **_COUNTRY_NAME_MAPPING_DEFAULT,
        **config.country_analysis.country_name_mapping,
    }
    popularity = compute_topic_popularity(merged, mapping=mapping)
    labels = load_topic_labels(config)

    totals = popularity.groupby("Country")["Popularity"].sum().sort_values(ascending=False)
    countries = totals.head(top_n_countries).index.tolist()

    panels = []
    for country in countries:
        sub = popularity[popularity["Country"] == country].nlargest(top_n_topics, "Popularity")
        rows = []
        for r in sub.itertuples():
            tid = int(r.Topic)
            short = TOPIC_ABBREV.get(tid, labels.get(tid, f"T{tid}"))
            rows.append((tid, f"T{tid} {short}", float(r.Popularity)))
        panels.append({
            "country": country,
            "total": float(totals[country]),
            "rows": rows,
        })

    return {
        "panels": panels,
        "countries": countries,
        "n_countries_available": int(len(totals)),
        "max_label_chars": max(
            (len(label) for p in panels for _, label, _ in p["rows"]), default=0
        ),
        "max_value": max((v for p in panels for _, _, v in p["rows"]), default=1.0),
    }


def plot_country_topic_bars(
    config: Config,
    output_path: Path,
    top_n_countries: int = 8,
    top_n_topics: int = 5,
    n_cols: int = 2,
) -> Optional[dict]:
    """Draw the small-multiples bar chart at final printed size."""
    prep = build_country_topic_bars(
        config, top_n_countries=top_n_countries, top_n_topics=top_n_topics
    )
    panels = prep["panels"]
    if not panels:
        logger.warning("No countries with popularity data - skipping")
        return None

    n = len(panels)
    n_rows = int(np.ceil(n / n_cols))
    fig_width = _PRINT_WIDTH_IN

    label_fs = 5.5
    tick_fs = 5.0
    value_fs = 5.0
    title_fs = 6.5

    panel_h_in = 1.12          # 5 bars at a 0.20-in pitch plus the x tick labels
    row_gap_in = 0.28          # panel title above + x tick labels below
    top_pad_in = 0.17          # clears the row-0 panel title from the canvas top
    outer_left_in = 0.05
    outer_right_in = 0.16       # holds the overhang of the rightmost x tick label
    col_gap_in = 0.34

    xlabel_pad_in = 0.24       # figure-level x label below the last row
    provisional_h = n_rows * (panel_h_in + row_gap_in) + top_pad_in + xlabel_pad_in
    fig = plt.figure(figsize=(fig_width, provisional_h))

    # Provisional axes, one per panel; the label strip is measured below and the
    # whole grid is repositioned so every panel gets the same geometry.
    axes = []
    for i in range(n):
        c, r = i % n_cols, i // n_cols
        rect = [0.02 + c * 0.5, 1.0 - (r + 1) * 0.2, 0.44, 0.16]
        axes.append(fig.add_axes(rect))

    for ax, panel in zip(axes, panels):
        rows = panel["rows"]
        label_texts = [label for _, label, _ in rows]
        values = np.array([value for _, _, value in rows])
        y = np.arange(len(rows))
        ax.barh(y, values, color="#2177B2", edgecolor="white", height=0.66, linewidth=0.5)
        ax.set_yticks(y)
        ax.set_yticklabels(label_texts, fontsize=label_fs)
        ax.set_ylim(len(rows) - 0.5, -0.5)
        # Per-panel x scale: popularity runs from 3.9 (China) to 0.3 (Japan), so a
        # shared axis would erase the smaller panels. Values are printed at the bar
        # tips, which is what makes the panels comparable despite the local scales.
        xmax = float(values.max()) * 1.34
        ax.set_xlim(0, xmax)
        for value, patch in zip(values, ax.patches):
            ax.text(patch.get_width() + xmax * 0.02,
                    patch.get_y() + patch.get_height() / 2.0,
                    f"{value:.2f}", va="center", ha="left",
                    fontsize=value_fs, color="0.25")
        ax.set_title(f"{panel['country']}  ($\\Sigma$ = {panel['total']:.0f})",
                     fontsize=title_fs, fontweight="bold", pad=2.5, loc="left")
        ax.tick_params(axis="x", labelsize=tick_fs, length=2, pad=1.5)
        ax.tick_params(axis="y", length=0, pad=1.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", linestyle=":", alpha=0.35, linewidth=0.5)
        ax.set_axisbelow(True)
        # Explicit ticks. MaxNLocator can return a tick just past the view limit, and
        # a tick *label* is not clipped by the axes patch, so that label printed outside
        # the canvas. Nice numbers are kept, but only those inside [0, xmax].
        ax.set_xticks(
            [t for t in MaxNLocator(3).tick_values(0.0, xmax) if 0.0 <= t <= xmax]
        )
        # One format for every panel: the locator otherwise renders China's axis as
        # "0 2 4" and the rest as "0.0 0.5 1.0", which reads as a scale difference.
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    # Measure the label strips from the rendered glyphs and lay the grid out for
    # real, so a 5.5 pt label group cannot push the bars off the canvas.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    px_to_in = 1.0 / fig.dpi
    label_w_in = max(
        (t.get_window_extent(renderer=renderer).width
         for ax in axes for t in ax.get_yticklabels()),
        default=0.0,
    ) * px_to_in
    label_strip_in = label_w_in + 0.12           # + tick pad

    panel_w_in = (
        fig_width - outer_left_in - outer_right_in
        - n_cols * label_strip_in - (n_cols - 1) * col_gap_in
    ) / n_cols
    if panel_w_in <= 0.8:
        raise ValueError(
            f"Measured label strip {label_strip_in:.2f} in leaves only "
            f"{panel_w_in:.2f} in per panel; shorten TOPIC_ABBREV."
        )
    fig_height = n_rows * (panel_h_in + row_gap_in) + top_pad_in + xlabel_pad_in
    fig.set_size_inches(fig_width, fig_height)

    for i, ax in enumerate(axes):
        c, r = i % n_cols, i // n_cols
        x_in = outer_left_in + c * (label_strip_in + panel_w_in + col_gap_in) + label_strip_in
        y_in = (fig_height - top_pad_in
                - (r + 1) * panel_h_in - r * row_gap_in)
        ax.set_position([
            x_in / fig_width, y_in / fig_height,
            panel_w_in / fig_width, panel_h_in / fig_height,
        ])

    fig.text(
        0.5, 0.010,
        "Topic popularity (\u03a3 of yearly proportions); the x scale is set per panel",
        ha="center", va="bottom", fontsize=6.0,
    )

    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=config.visualization.dpi, facecolor="white")
    plt.close(fig)

    logger.info(
        "Print-sized popularity bars saved: %s (%d countries x %d topics; "
        "canvas %.2f x %.2f in)",
        output_path, n, top_n_topics, fig_width, fig_height,
    )
    prep.update({
        "output_path": str(output_path),
        "fig_width": fig_width,
        "fig_height": fig_height,
        "panel_w_in": panel_w_in,
        "label_strip_in": label_strip_in,
    })
    return prep


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s | %(message)s")
    config = get_config()
    out = config.paths.figures_dir / "popular_topics_heatmap_print.png"
    prep = plot_country_topic_bars(config, out)
    print(f"written: {out}")
    print(f"  countries shown {len(prep['countries'])} of {prep['n_countries_available']} "
          f"with popularity data")
    print(f"  canvas {prep['fig_width']:.2f} x {prep['fig_height']:.2f} in; "
          f"panel {prep['panel_w_in']:.2f} in wide, label strip {prep['label_strip_in']:.2f} in")
    print(f"  longest y label: {prep['max_label_chars']} characters")


if __name__ == "__main__":
    main()
