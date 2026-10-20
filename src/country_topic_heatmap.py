"""Manuscript Fig. 7: country x topic heatmap of fractional output.

Replaces the artifact that used to sit under the Fig. 7 caption. The caption
promised a ``25 x 15`` heatmap of *fractional publication count*; the file it
pointed at (``popular_topics_heatmap.png``) was a 20-in-wide 12-panel bar chart of
a different metric (summed yearly popularity), which at ``\\columnwidth`` was
shrunk to ~0.33x and printed every bar label at ~2 pt.

Design constraints, mirroring the Fig. 6 heatmap in ``src/combined_viz.py``:

* **Designed 1:1 at the printed width.** CAS single-column sets
  ``\\columnwidth = 468.33 pt = 6.50 in``, so the canvas is built at exactly
  6.50 in and every font size prints at its nominal value.
* **Rows = the 25 most productive countries** by topic-assigned fractional
  documents (``N_c``), descending; each row label carries its ``N_c`` so the
  selection criterion is visible in the figure, not only in the caption.
* **Columns = the 15 topics with the largest global fractional frequency**,
  descending; the top marginal repeats that frequency for the same reason.
* **The axis carries only ``T<id>``; a key under the figure names each topic.**
  Fifteen descriptive labels cannot fit under a ~22 pt column pitch: at 45
  degrees they would have to be ~7 characters long. Moving the names into a key
  below the matrix lets the axis stay horizontal and lets the key use the full
  6.5-in width instead of a 0.30-in column.
* **Color ceiling at the 95th percentile of the non-zero cells.** The 25 x 15
  submatrix is severely right-skewed (median non-zero cell 1.0, p95 8.4, max
  51.3), so an unclipped linear scale would flatten ~95 % of the map. The top
  colourbar tick is therefore labelled ``>= ceiling``.
* **Every non-zero cell is annotated** with its value, because 40 % of the cells
  are exactly zero and a color scale alone cannot separate "0" from "1.0".

Run: ``pixi run python -m src.country_topic_heatmap``
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import Config, get_config
from src.spatial_topic_analysis import (
    _COUNTRY_NAME_MAPPING_DEFAULT,
    compute_specialization_index,
    fractional_country_topic,
    load_topic_labels,
    merge_country_topic,
)
from src.utils import ensure_dir

logger = logging.getLogger(__name__)

# Final printed width of the figure: CAS single-column \columnwidth = 468.33 pt.
# See src/combined_viz.py::_PRINT_WIDTH_IN for the same constant.
_PRINT_WIDTH_IN = 6.50

# How much of the column pitch a rotated tick label may occupy horizontally, and the
# shallowest rotation the auto rule will consider. With the axis reduced to "T<id>"
# the auto rule settles at 0 degrees (upright), which is why _MIN_TILT_DEG is 0.
_LABEL_CLEARANCE = 0.92
_MIN_TILT_DEG = 0.0

# Topic names for the key under the figure. The canonical ``Short_Label`` values run
# to 54 characters ("Mangrove Water Quality and Bio-Optical Remote Sensing"); these
# elide the word "mangrove" that every label shares and trim the rest. Any topic
# missing here falls back to its canonical label, so the figure still draws if the
# topic model changes -- the key just gets wider.
_TOPIC_ABBREV: Dict[int, str] = {
    0: "Change detection",
    1: "Canopy height / biomass",
    2: "Water quality / bio-optics",
    3: "Deforestation, SE Asia",
    4: "Mangrove Remote Sensing Mapping",
    5: "Resilience / blue carbon",
    6: "Hurricane damage recovery",
    7: "Land use / eco-services",
    8: "Semantic segmentation",
    9: "Drivers of loss",
    10: "Ecosystem remote sensing",
    11: "Remote sensing in India",
    12: "Image classification",
    13: "Coastal geomorphologic change",
    14: "Deforestation / cover",
    15: "Amazon mud-bank coasts",
    16: "Shoreline change",
    18: "Remote sensing applications",
    19: "Leaf area index estimation",
    20: "Tidal wetland elevation",
    21: "Biomass and carbon estimation",
    22: "Polarimetric SAR analysis",
    23: "Ecosystem biogeochemistry",
    24: "AGB estimation",
    25: "Degradation mapping",
    26: "Coastal response to SLR",
    27: "Coastal ecosystem RS",
    28: "Range limit dynamics",
    30: "Forest spatio-temporal",
    31: "Carbon stock assessment",
    32: "Spartina invasion monitoring",
    39: "RS of coastal ecosystems",
    40: "Remote sensing monitoring",
    41: "Remote sensing classification",
    43: "Sediment accumulation",
    44: "Coastal wetland RS",
    48: "Hyperspectral classification",
    52: "Black mangrove RS",
    53: "Characterization & mapping",
    58: "Cover trends / change",
    64: "Biomass carbon estimation",
}

# Public alias: both Fig. 7 candidates share one abbreviation table, so their topic
# labels cannot disagree with each other.
TOPIC_ABBREV = _TOPIC_ABBREV


def build_country_topic_matrix(
    config: Config,
    top_n_countries: int = 25,
    top_n_topics: int = 15,
) -> dict:
    """Rebuild the fractional country x topic matrix and select the Fig. 7 block.

    Goes through the same production path as
    :func:`src.spatial_topic_analysis.run_spatial_topic_analysis` -- merge,
    fractional counting, then the ``N_c >= min_docs_per_country`` filter -- so the
    submatrix cannot drift from ``spatial_topic_si.csv``.

    Returns:
        Dict with ``frame`` (countries x topics, fractional counts), the row and
        column labels, the marginals, the colour ceiling and provenance
        (``n_countries_total``, ``threshold``, the first excluded country, ...).
    """
    sa_cfg = config.spatial_topic_analysis

    merged = merge_country_topic(config)
    mapping = {
        **_COUNTRY_NAME_MAPPING_DEFAULT,
        **config.country_analysis.country_name_mapping,
    }
    fractional = fractional_country_topic(merged, mapping=mapping)
    _si, n_ck, n_c = compute_specialization_index(
        fractional, min_docs_per_country=sa_cfg.min_docs_per_country
    )
    labels = load_topic_labels(config)

    n_c_sorted = n_c.sort_values(ascending=False)
    countries = n_c_sorted.head(top_n_countries).index.tolist()
    global_freq = n_ck.sum(axis=0).sort_values(ascending=False)
    topics = list(global_freq.head(top_n_topics).index)

    frame = n_ck.loc[countries, topics]

    positive = frame.values[frame.values > 0]
    data_max = float(frame.values.max())
    vmax = float(np.percentile(positive, 95.0)) if positive.size else 1.0

    # The row label carries the country's exact N_c (topic-assigned fractional
    # documents), which is the criterion that put it in the figure at all.
    row_labels = [
        f"{c} ({n_c_sorted[c]:.1f})" if n_c_sorted[c] < 100 else f"{c} ({n_c_sorted[c]:.0f})"
        for c in countries
    ]

    return {
        "frame": frame,
        "countries": countries,
        "topics": topics,
        "row_labels": row_labels,
        "tick_labels": [f"T{int(t)}" for t in topics],
        "topic_labels": {
            int(t): _TOPIC_ABBREV.get(int(t), labels.get(int(t), f"T{int(t)}"))
            for t in topics
        },
        # The key is a lookup table, so it is ordered by topic id rather than by the
        # frequency order of the columns; the axis and the top marginal already show
        # the latter, and "what is T13?" is the question the key exists to answer.
        "key_labels": [
            f"T{t} {_TOPIC_ABBREV.get(t, labels.get(t, 'T' + str(t)))}"
            for t in sorted(int(x) for x in topics)
        ],
        "canonical_labels": {int(t): labels.get(int(t), f"T{int(t)}") for t in topics},
        "column_totals": global_freq.loc[topics],
        "row_totals": n_c_sorted.loc[countries],
        "vmax": vmax,
        "data_max": data_max,
        "n_countries_total": int(len(n_c)),
        "threshold": float(sa_cfg.min_docs_per_country),
        "first_excluded": (
            (n_c_sorted.index[top_n_countries], float(n_c_sorted.iloc[top_n_countries]))
            if len(n_c_sorted) > top_n_countries
            else None
        ),
        "topic_freq_of_first_excluded": (
            float(global_freq.iloc[top_n_topics]) if len(global_freq) > top_n_topics else None
        ),
        "share_of_matrix": float(frame.values.sum() / n_ck.values.sum()),
        "zero_fraction": float((frame.values == 0).mean()),
    }


def _annot_text(value: float) -> str:
    """Cell annotation: integers without a decimal, everything else to 1 d.p."""
    if value <= 0:
        return ""
    if abs(value - round(value)) < 0.05:
        return f"{int(round(value))}"
    return f"{value:.1f}"


def plot_country_topic_heatmap(
    config: Config,
    output_path: Path,
    top_n_countries: int = 25,
    top_n_topics: int = 15,
    annotate: bool = True,
    clip_percentile: float = 95.0,
    xlabel_rotation: Optional[float] = None,
    topic_key_cols: Optional[int] = 4,
) -> Optional[dict]:
    """Draw the country x topic heatmap at final printed size.

    Args:
        xlabel_rotation: Tick-label rotation in degrees. ``None`` (default) picks the
            shallowest rotation that still clears the neighbouring column, which for
            bare ``T<id>`` labels is 0 (upright); pass a number to force one.
        topic_key_cols: Number of columns in the topic key under the figure, or
            ``None`` to omit the key (the tick labels then have to carry the names).
    """
    prep = build_country_topic_matrix(
        config, top_n_countries=top_n_countries, top_n_topics=top_n_topics
    )
    frame: pd.DataFrame = prep["frame"]
    topics = prep["topics"]
    tick_labels = prep["tick_labels"]
    key_labels = prep["key_labels"]
    n_rows = len(prep["countries"])
    n_cols = len(topics)

    if clip_percentile != 95.0:
        positive = frame.values[frame.values > 0]
        prep["vmax"] = float(np.percentile(positive, clip_percentile)) if positive.size else 1.0
    vmax = prep["vmax"]

    fig_width = _PRINT_WIDTH_IN

    # Geometry, in inches. The row pitch is the one designed number (0.150 in =
    # 10.8 pt leaves a 5.5 pt annotation room to breathe); every other margin is
    # measured from the rendered text further down, so the canvas ends up exactly
    # as tall as its contents and no label is clipped at an edge. The axes are
    # positioned explicitly rather than through GridSpec because the measured
    # margins are known only after a draw.
    pitch_in = 0.150
    matrix_h_in = pitch_in * n_rows
    top_slab_in = 0.40          # top marginal (global frequency per topic)
    gap_in = 0.05               # matrix -> top marginal
    top_pad_in = 0.05
    cbar_w_in = 0.17
    cbar_furniture_in = 0.62    # colorbar tick labels + rotated axis label
    right_pad_in = 0.02
    key_line_in = 0.098         # key line pitch (~7.1 pt for 5.0 pt text)
    key_fontsize = 5.0          # the key is a lookup table, not prose
    key_pad_bottom_in = 0.05
    key_pad_side_in = 0.04       # key block -> canvas edge
    key_gap_in = 0.05           # key block -> tick labels

    key_rows = int(math.ceil(n_cols / topic_key_cols)) if topic_key_cols else 0
    key_h_in = key_rows * key_line_in if topic_key_cols else 0.0

    provisional_h = matrix_h_in + top_slab_in + gap_in + top_pad_in + key_h_in + 1.20
    fig = plt.figure(figsize=(fig_width, provisional_h))
    ax_main = fig.add_axes([0.22, 0.24, 0.52, 0.62])
    ax_cbar = fig.add_axes([0.76, 0.24, 0.03, 0.62])
    ax_top = fig.add_axes([0.22, 0.90, 0.52, 0.07], sharex=ax_main)

    display = frame.values.astype(float)
    annot = (
        pd.DataFrame(
            np.vectorize(_annot_text)(display),
            index=frame.index,
            columns=frame.columns,
        )
        if annotate
        else None
    )

    sns.heatmap(
        display,
        ax=ax_main,
        cmap="YlOrRd",
        vmin=0,
        vmax=vmax,
        xticklabels=tick_labels,
        yticklabels=prep["row_labels"],
        annot=annot,
        fmt="",
        annot_kws={"size": 5.5},
        cbar_ax=ax_cbar,
        linewidths=0.2,
        linecolor="#f0f0f0",
    )
    # Flip the annotation colours on the dark (clipped) cells so the numbers stay
    # legible instead of black-on-dark-red.
    if annotate:
        for text, value in zip(ax_main.texts, display.ravel(order="C")):
            if value > 0.55 * vmax:
                text.set_color("white")

    cbar = ax_main.collections[0].colorbar
    cbar_ticks = list(np.linspace(0.0, vmax, 5))
    cbar.set_ticks(cbar_ticks)
    cbar.set_ticklabels([f"{t:.1f}" for t in cbar_ticks[:-1]] + [f"\u2265{vmax:.1f}"])
    cbar.set_label("Fractional publications", fontsize=6.5)
    cbar.ax.tick_params(labelsize=5.5)
    cbar.outline.set_linewidth(0.4)
    # The colorbar's default aspect=20 resizes the bar inside whatever axes rect
    # it is given, which would undo the exact geometry set below.
    cbar.ax.set_aspect("auto")

    ax_main.set_xlabel("")
    ax_main.set_ylabel("")
    # Provisional 0 degrees: at zero rotation a label's bbox gives both the string
    # length (its width) and the glyph height (its height), which is what the rotation
    # rule needs. The final angle is chosen below, once the column pitch is known.
    ax_main.tick_params(axis="x", labelsize=5.5, rotation=0)
    ax_main.tick_params(axis="y", labelsize=5.5)

    # Top marginal: the selection criterion for the columns.
    col_totals = prep["column_totals"].values.astype(float)
    ax_top.bar(range(n_cols), col_totals, color="#4472C4", edgecolor="white",
               width=0.75, alpha=0.85)
    ax_top.set_ylabel("\u03a3 frac.", fontsize=5.5, labelpad=2)
    ax_top.set_ylim(0, float(col_totals.max()) * 1.25)
    ax_top.tick_params(axis="x", labelbottom=False)
    ax_top.tick_params(axis="y", labelsize=5.0)
    ax_top.set_yticks([0, float(col_totals.max())])
    ax_top.set_yticklabels(["0", f"{col_totals.max():.0f}"])
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)

    # No right marginal for the row criterion. The rows are already ordered by N_c
    # and each row label carries its exact N_c, whereas a bar encoding on a linear
    # axis is unreadable here: China holds 324.9 of the 1,435 fractional documents in
    # the full matrix, so on a 0-325 axis every other country is a sliver. The width
    # it would cost is better spent on the matrix, which is the informative part.

    # Measure the label strips from the rendered glyphs. The tick mark and its pad
    # (~7 pt) are not part of a Text's window extent, so each margin adds them back.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    px_to_in = 1.0 / fig.dpi
    px_to_pt = 72.0 / fig.dpi
    row_label_w_in = max(
        (t.get_window_extent(renderer=renderer).width for t in ax_main.get_yticklabels()),
        default=0.0,
    ) * px_to_in
    col_bboxes = [t.get_window_extent(renderer=renderer) for t in ax_main.get_xticklabels()]
    col_label_len_pt = max((b.width for b in col_bboxes), default=0.0) * px_to_pt
    col_label_h_pt = max((b.height for b in col_bboxes), default=0.0) * px_to_pt

    left_in = row_label_w_in + 0.16
    matrix_w_in = (
        fig_width - left_in - cbar_w_in - cbar_furniture_in - gap_in - right_pad_in
    )
    if matrix_w_in <= 1.0:
        raise ValueError(
            f"Measured label margins leave only {matrix_w_in:.2f} in for the matrix "
            f"(row labels {row_label_w_in:.2f} in); shorten the row labels."
        )
    pitch_pt = matrix_w_in / n_cols * 72.0

    # Rotate the tick labels only as far as the column pitch allows. Two neighbouring
    # labels sit one pitch apart and each occupies an axis-aligned box of width
    # L*cos(theta) + H*sin(theta) (L = string length, H = glyph height); they stay clear
    # while that box fits in _LABEL_CLEARANCE * pitch. With the axis reduced to "T<id>"
    # this settles at 0 degrees, which is the readable answer for 2-4 character labels.
    if xlabel_rotation is None:
        limit = _LABEL_CLEARANCE * pitch_pt
        angle = 90.0
        for step in range(0, 181):                      # 0 -> 90 in half degrees
            candidate = _MIN_TILT_DEG + 0.5 * step
            rad = math.radians(candidate)
            box = col_label_len_pt * math.cos(rad) + col_label_h_pt * math.sin(rad)
            if box <= limit:
                angle = candidate
                break
    else:
        angle = float(xlabel_rotation)

    def layout(theta: float) -> tuple:
        """Apply a rotation, size the canvas to hold its contents, and place the axes.

        Every measurement here has to be taken with the axes in their FINAL position:
        the provisional rect is narrower than the final one, so a clearance test run
        before this point measures the wrong pitch. (An earlier version did exactly
        that and over-steepened the labels all the way back to vertical.)

        The bottom margin holds, from the canvas edge upward: the topic key block, a
        gap, the tick labels, and a gap to the matrix.
        """
        ax_main.tick_params(axis="x", rotation=theta)
        fig.canvas.draw()
        rend = fig.canvas.get_renderer()
        label_h = max(
            (t.get_window_extent(renderer=rend).height
             for t in ax_main.get_xticklabels()),
            default=0.0,
        ) * px_to_in
        bottom = key_pad_bottom_in + key_h_in + key_gap_in + label_h + 0.10
        height = bottom + matrix_h_in + gap_in + top_slab_in + top_pad_in
        fig.set_size_inches(fig_width, height)
        ax_main.set_position([
            left_in / fig_width, bottom / height,
            matrix_w_in / fig_width, matrix_h_in / height,
        ])
        ax_top.set_position([
            left_in / fig_width,
            (bottom + matrix_h_in + gap_in) / height,
            matrix_w_in / fig_width, top_slab_in / height,
        ])
        ax_cbar.set_position([
            (left_in + matrix_w_in + gap_in) / fig_width,
            bottom / height,
            cbar_w_in / fig_width, matrix_h_in / height,
        ])
        fig.canvas.draw()
        rend = fig.canvas.get_renderer()
        boxes = [t.get_window_extent(renderer=rend)
                 for t in ax_main.get_xticklabels() if t.get_text()]
        worst = max(
            (min(a.x1, b.x1) - max(a.x0, b.x0) for a, b in zip(boxes, boxes[1:])),
            default=0.0,
        ) * px_to_pt
        return bottom, height, label_h, worst

    bottom_in, fig_height, col_label_h_in, worst_overlap = layout(angle)
    # The analytic AABB is only a starting guess, so steepen until the rendered boxes
    # genuinely clear each other.
    for _ in range(30):
        if worst_overlap <= 0.0 or angle >= 90.0:
            break
        angle = min(90.0, angle + 1.0)
        bottom_in, fig_height, col_label_h_in, worst_overlap = layout(angle)

    # Topic key. The axis carries only "T<id>", so the names live here, laid out across
    # the full canvas (not squeezed under a single 0.30-in column) on a pitch of
    # key_line_in. Using the whole 6.5 in rather than just the matrix width is what lets
    # four columns hold names as long as "Coastal geomorphologic change". Each entry is
    # centred in its own column slot, so the four columns read as a balanced grid rather
    # than four ragged-left lists. Drawn last because the y positions depend on the final
    # height; the smaller key font buys the clearance that centring spends, since two
    # centred neighbours collide once their half-widths together exceed the column pitch.
    key_texts = []
    if topic_key_cols:
        for i, label in enumerate(key_labels):
            row, col = divmod(i, topic_key_cols)
            key_texts.append(fig.text(
                (key_pad_side_in
                 + (col + 0.5) * (fig_width - 2 * key_pad_side_in) / topic_key_cols)
                / fig_width,
                (key_pad_bottom_in + (key_rows - 1 - row) * key_line_in + 0.010) / fig_height,
                label,
                fontsize=key_fontsize, ha="center", va="bottom", color="0.15",
            ))
        fig.canvas.draw()

    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=config.visualization.dpi, facecolor="white")
    plt.close(fig)

    logger.info(
        "Country-topic heatmap saved: %s (%d countries x %d topics; vmax=%.2f; "
        "canvas %.2f x %.2f in)",
        output_path, n_rows, n_cols, vmax, fig_width, fig_height,
    )
    prep.update({
        "output_path": str(output_path),
        "fig_height": fig_height,
        "fig_width": fig_width,
        "left_in": left_in,
        "bottom_in": bottom_in,
        "matrix_w_in": matrix_w_in,
        "cell_w_in": matrix_w_in / n_cols,
        "cell_h_in": matrix_h_in / n_rows,
        "longest_tick_label": max(tick_labels, key=len),
        "xlabel_rotation": angle,
        "xlabel_tilt_from_vertical_deg": 90.0 - angle,
        "column_pitch_pt": pitch_pt,
        "max_col_label_pt": col_label_len_pt,
        "col_label_glyph_h_pt": col_label_h_pt,
        "col_label_h_in": col_label_h_in,
        "worst_label_overlap_pt": worst_overlap,
        "topic_key_cols": topic_key_cols,
        "key_rows": key_rows,
        "key_h_in": key_h_in,
        "key_labels": key_labels,
        "n_key_texts": len(key_texts),
        "key_fontsize": key_fontsize,
    })
    return prep


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s | %(message)s")
    config = get_config()
    out = config.paths.figures_dir / "country_topic_heatmap.png"
    prep = plot_country_topic_heatmap(config, out)
    print(f"written: {out}")
    print(f"  countries={len(prep['countries'])} topics={len(prep['topics'])} "
          f"vmax={prep['vmax']:.2f} data_max={prep['data_max']:.2f}")
    print(f"  canvas {prep['fig_width']:.2f} x {prep['fig_height']:.2f} in; "
          f"axis tick labels {prep['tick_labels']}")
    print(f"  tick rotation {prep['xlabel_rotation']:.1f} deg; "
          f"key {prep['key_rows']} rows x {prep['topic_key_cols']} cols "
          f"({prep['n_key_texts']} entries)")
    for label in prep["key_labels"]:
        print(f"    {label}")


if __name__ == "__main__":
    main()
