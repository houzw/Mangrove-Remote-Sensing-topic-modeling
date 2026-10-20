"""Verify the candidate Fig. 7 artifacts the way the printed page will use them.

Checks, for both candidates, on the *rendered* canvas rather than on the axes
geometry (a design margin must be measured in the same frame as the thing it
protects -- see CLAUDE.md):

1. canvas size vs the printed size. CAS single-column ``\\columnwidth`` is
   468.33 pt = 6.5046 in; a figure designed 1:1 must be within a hair of that.
2. every Text artist lies inside the canvas, so nothing is clipped at an edge.
3. the smallest / largest font actually used, and the text-to-cell fit for the
   heatmap's 5.5 pt annotations.
4. the effective font size after the LaTeX downscale, which is the number that
   explains why the old figure was unreadable.

Run: pixi run python -m scripts_tmp.check_fig7_legibility
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str((Path("outputs") / ".mplcache").resolve()))

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.text import Text

from src.config import get_config

logging.basicConfig(level=logging.WARNING)

COLUMNWIDTH_PT = 468.33
COLUMNWIDTH_IN = COLUMNWIDTH_PT / 72.0

config = get_config()
figures = config.paths.figures_dir
px_to_pt = 72.0 / config.visualization.dpi


def capture(build):
    """Run a figure builder and return the live Figure instead of closing it."""
    captured = {}
    original_close = plt.close

    def fake_close(fig=None):
        if fig is not None:
            captured.setdefault("fig", fig)

    plt.close = fake_close
    try:
        meta = build()
    finally:
        plt.close = original_close
    return captured.get("fig"), meta


def audit(fig, title: str) -> dict:
    print("=" * 78)
    print(title)
    print("=" * 78)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    w_in, h_in = fig.get_size_inches()

    print(f"canvas            : {w_in:.3f} x {h_in:.3f} in "
          f"({w_in * 72:.1f} x {h_in * 72:.1f} pt)")
    scale = COLUMNWIDTH_IN / w_in
    print(f"placed at \\columnwidth -> downscale factor {scale:.4f}")
    print(f"effective font x  : {scale:.4f}  (1.0 = designed 1:1, no shrink)")

    canvas_bb = fig.bbox
    texts = [t for t in fig.findobj(Text) if t.get_text().strip() and t.get_visible()]
    sizes = sorted({round(t.get_fontsize(), 2) for t in texts})
    print(f"text artists      : {len(texts)}   font sizes used: {sizes}")
    print(f"smallest font     : {min(sizes)} pt  -> prints at {min(sizes) * scale:.2f} pt")

    outside = []
    for t in texts:
        bb = t.get_window_extent(renderer=renderer)
        if (bb.x0 < -0.5 or bb.y0 < -0.5
                or bb.x1 > canvas_bb.x1 + 0.5 or bb.y1 > canvas_bb.y1 + 0.5):
            outside.append((t.get_text()[:44], round(bb.x0, 1), round(bb.y0, 1),
                            round(bb.x1, 1), round(bb.y1, 1)))
    print(f"clipped text      : {len(outside)}"
          f"  (canvas is {canvas_bb.x1:.0f} x {canvas_bb.y1:.0f} px)")
    for label, x0, y0, x1, y1 in outside[:12]:
        print(f"    OUT {label!r} bbox=({x0},{y0})-({x1},{y1})")

    # Headroom actually left at each edge, in points, so a margin claim can be
    # checked instead of assumed.
    min_x = min(t.get_window_extent(renderer=renderer).x0 for t in texts)
    max_x = max(t.get_window_extent(renderer=renderer).x1 for t in texts)
    min_y = min(t.get_window_extent(renderer=renderer).y0 for t in texts)
    max_y = max(t.get_window_extent(renderer=renderer).y1 for t in texts)
    # window extents are in points at fig.dpi; convert px -> pt
    px_to_pt = 72.0 / fig.dpi
    print(f"text headroom     : left {min_x * px_to_pt:.1f} pt | "
          f"right {(canvas_bb.x1 - max_x) * px_to_pt:.1f} pt | "
          f"bottom {min_y * px_to_pt:.1f} pt | top {(canvas_bb.y1 - max_y) * px_to_pt:.1f} pt")

    return {
        "width_in": w_in, "height_in": h_in, "scale": scale,
        "font_sizes": sizes, "n_clipped": len(outside),
    }


report = {}

# ---------------------------------------------------------------- candidate A
from src.country_topic_heatmap import plot_country_topic_heatmap

fig_a, prep_a = capture(lambda: plot_country_topic_heatmap(
    config, figures / "country_topic_heatmap.png"))
if fig_a is not None:
    report["A_heatmap"] = audit(fig_a, "CANDIDATE A: country x topic heatmap (25 x 15)")
    frame = prep_a["frame"]
    print(f"matrix            : {frame.shape[0]} countries x {frame.shape[1]} topics")
    print(f"colour ceiling vmax = {prep_a['vmax']:.2f}  (data max {prep_a['data_max']:.2f}); "
          f"top tick labelled '>= {prep_a['vmax']:.1f}'")
    print(f"clipped cells     : {int((frame.values > prep_a['vmax']).sum())} of {frame.size} "
          f"({100 * (frame.values > prep_a['vmax']).mean():.1f} %); "
          f"zeros {100 * prep_a['zero_fraction']:.1f} %")
    print(f"submatrix holds   : {100 * prep_a['share_of_matrix']:.1f} % of all fractional "
          f"country-topic mass")
    measurable = int((prep_a["row_totals"] >= 10).sum())
    print(f"rows clearing the N_c >= 10 measurability threshold: "
          f"{measurable} of {len(prep_a['row_totals'])}")
    print(f"smallest included N_c: {prep_a['row_totals'].min():.2f}; "
          f"first excluded row: {prep_a['first_excluded']}")
    # Annotation fit: cell pitch vs the widest annotation string.
    n_rows, n_cols = frame.shape
    ax = [a for a in fig_a.axes if a.get_ylabel() == "" and len(a.texts) > 0][0]
    cell_w_pt = ax.get_window_extent().width / n_cols * px_to_pt
    cell_h_pt = ax.get_window_extent().height / n_rows * px_to_pt
    widest = max((t.get_window_extent(renderer=fig_a.canvas.get_renderer()).width
                  for t in ax.texts), default=0.0) * px_to_pt
    print(f"cell pitch        : {cell_w_pt:.1f} x {cell_h_pt:.1f} pt; "
          f"widest annotation {widest:.1f} pt -> fits: {widest < cell_w_pt}")
    # The tilt, and the label clearance it buys, re-measured on the rendered canvas.
    ax_heat = [a for a in fig_a.axes if len(a.texts) > 0][0]
    heat_pt = 72.0 / fig_a.dpi
    heat_boxes = [t.get_window_extent(renderer=fig_a.canvas.get_renderer())
                  for t in ax_heat.get_xticklabels() if t.get_text()]
    heat_overlap = max(
        (min(a.x1, b.x1) - max(a.x0, b.x0) for a, b in zip(heat_boxes, heat_boxes[1:])),
        default=0.0,
    ) * heat_pt
    _rot = prep_a["xlabel_rotation"]
    _rot_txt = "upright" if _rot == 0 else f"{_rot:.1f} deg"
    print(f"column-label tilt : {_rot_txt}; "
          f"pitch {prep_a['column_pitch_pt']:.1f} pt; longest tick label "
          f"{prep_a['max_col_label_pt']:.1f} pt (glyph height "
          f"{prep_a['col_label_glyph_h_pt']:.1f} pt)")
    print(f"label clearance   : worst adjacent overlap {heat_overlap:+.2f} pt "
          f"({'clear' if heat_overlap <= 0 else 'OVERLAP'})")
    # The topic key is figure-level text, so it is checked separately: any two entries
    # whose boxes intersect in BOTH axes are colliding.
    key_texts = [t for t in fig_a.texts if t.get_text().strip().startswith("T")]
    key_boxes = [t.get_window_extent(renderer=fig_a.canvas.get_renderer()) for t in key_texts]
    key_pairs = [(a, b) for i, a in enumerate(key_boxes) for b in key_boxes[i + 1:]]
    key_worst = max(
        (min(min(a.x1, b.x1) - max(a.x0, b.x0), min(a.y1, b.y1) - max(a.y0, b.y0))
         for a, b in key_pairs),
        default=0.0,
    )
    # Same-row neighbours are the ones that can actually collide, so report the
    # tightest horizontal gap between them -- a bare "0.00" cannot distinguish
    # "clear by nothing" from "no overlap at all".
    row_gaps, col_gaps = [], []
    for i, a in enumerate(key_boxes):
        for b in key_boxes[i + 1:]:
            if min(a.y1, b.y1) - max(a.y0, b.y0) > 0:      # same row, neighbours sideways
                left, right = (a, b) if a.x0 <= b.x0 else (b, a)
                row_gaps.append((right.x0 - left.x1) * heat_pt)
            if min(a.x1, b.x1) - max(a.x0, b.x0) > 0:      # same column, neighbours down
                lower, upper = (a, b) if a.y0 <= b.y0 else (b, a)
                col_gaps.append((upper.y0 - lower.y1) * heat_pt)
    widest_key = max((t.get_text() for t in key_texts), key=len)
    # Both axes are reported: `key_worst` is the max over pairs of min(overlap_x,
    # overlap_y), which for a clear grid is the tightest of the two gaps, and on its own
    # it cannot say which axis is tight.
    print(f"topic key         : {len(key_texts)} entries in "
          f"{prep_a['key_rows']} rows x {prep_a['topic_key_cols']} cols "
          f"@ {prep_a['key_fontsize']} pt; worst collision {key_worst * heat_pt:.2f} pt "
          f"({'clear' if key_worst <= 0 else 'OVERLAP'}); "
          f"tightest gaps: sideways {min(row_gaps):.2f} pt, down {min(col_gaps):.2f} pt")
    print(f"                  : longest entry {widest_key!r} "
          f"({max(b.width for b in key_boxes) * heat_pt:.1f} pt wide)")
    plt.close(fig_a)

# ---------------------------------------------------------------- candidate B
from src.country_topic_bars import plot_country_topic_bars

fig_b, prep_b = capture(lambda: plot_country_topic_bars(
    config, figures / "popular_topics_heatmap_print.png"))
if fig_b is not None:
    report["B_bars"] = audit(fig_b, "CANDIDATE B: top topics per country (8 panels)")
    print(f"panels            : {len(prep_b['panels'])} of "
          f"{prep_b['n_countries_available']} countries with popularity data")
    print(f"panel width       : {prep_b['panel_w_in']:.2f} in; "
          f"label strip {prep_b['label_strip_in']:.2f} in")
    print(f"longest y label   : {prep_b['max_label_chars']} characters")
    plt.close(fig_b)

# ------------------------------------------- reference: the shipped artifact
old = figures / "popular_topics_heatmap.png"
if old.exists():
    from PIL import Image

    with Image.open(old) as im:
        old_w_in = im.size[0] / config.visualization.dpi
    print()
    print("=" * 78)
    print("REFERENCE: the artifact the Fig. 7 caption currently points at")
    print("=" * 78)
    print(f"{old.name}: {old_w_in:.2f} in wide -> placed at \\columnwidth "
          f"(downscale {COLUMNWIDTH_IN / old_w_in:.3f})")
    print(f"  a nominal 6.5 pt bar label in that file prints at "
          f"{6.5 * COLUMNWIDTH_IN / old_w_in:.2f} pt")

print()
print("=" * 78)
print("SUMMARY")
print("=" * 78)
for name, info in report.items():
    print(f"{name:<14} {info['width_in']:.2f} x {info['height_in']:.2f} in | "
          f"downscale {info['scale']:.3f} | fonts {info['font_sizes']} | "
          f"smallest prints at {min(info['font_sizes']) * info['scale']:.2f} pt | "
          f"clipped {info['n_clipped']}")
