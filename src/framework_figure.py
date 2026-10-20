"""Figure 1 (``fig1-framework.png``) — render the draw.io source to PNG.

The manuscript's Figure 1 is authored in draw.io
(``manuscript/els-cas-templates/fig1-framework.drawio``). That file is the
single source of truth for the figure's text and layout; this script parses it
and rasterises it to the PNG that LaTeX includes, so a text correction in the
``.drawio`` reaches the PDF without a manual export step (draw.io itself is not
installed on the analysis machine).

Rendering is done with Pillow rather than matplotlib: ``matplotlib.ft2font``
fails to load its freetype DLL in this pixi environment ("DLL load failed while
importing ft2font"), while Pillow's freetype works. Arial is used because the
draw.io document specifies Helvetica; the DejaVu faces bundled with matplotlib
are the fallback.

Canvas check against the draw.io export of 2026-09-27 (2583 x 2895 px): the
1000 x 1010-unit page is drawn at 3 px per unit and cropped to the diagram
bounds, which reproduces both the export's canvas and its ~5.4 pt print font at
``width=0.8\\textwidth``.

Usage:
    pixi run python -m src.framework_figure
"""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
DRAWIO = ROOT / "manuscript" / "els-cas-templates" / "fig1-framework.drawio"
OUTS = (
    ROOT / "manuscript" / "els-cas-templates" / "figs" / "fig1-framework.png",
    ROOT / "manuscript" / "els-cas-templates" / "figs" / "framework.png",
)

PAGE_W, PAGE_H = 1000.0, 1010.0  # mxGraphModel pageWidth / pageHeight
S = 3.0                          # pixels per draw.io unit (300 dpi at 10 in)
PAD = 2.0                        # units of white margin kept around the diagram
SS = 3                           # supersampling factor (anti-aliasing)

FONT_FILES = {
    (False, False): (r"C:\Windows\Fonts\arial.ttf",
                     "matplotlib/mpl-data/fonts/ttf/DejaVuSans.ttf"),
    (True, False): (r"C:\Windows\Fonts\arialbd.ttf",
                    "matplotlib/mpl-data/fonts/ttf/DejaVuSans-Bold.ttf"),
    (False, True): (r"C:\Windows\Fonts\ariali.ttf",
                    "matplotlib/mpl-data/fonts/ttf/DejaVuSans-Oblique.ttf"),
    (True, True): (r"C:\Windows\Fonts\arialbi.ttf",
                   "matplotlib/mpl-data/fonts/ttf/DejaVuSans-BoldOblique.ttf"),
}


@lru_cache(maxsize=None)
def _font(size_px: int, bold: bool, italic: bool) -> ImageFont.FreeTypeFont:
    for candidate in FONT_FILES[(bold, italic)]:
        path = Path(candidate)
        if not path.is_absolute():
            path = ROOT / ".pixi/envs/default/Lib/site-packages" / candidate
        if path.exists():
            return ImageFont.truetype(str(path), size_px)
    return ImageFont.load_default()


def _parse_style(style: str | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for part in (style or "").split(";"):
        if "=" in part:
            key, val = part.split("=", 1)
            out[key.strip()] = val.strip()
    return out


def _plain_text(value: str | None) -> str:
    """draw.io markup -> plain text with explicit line breaks."""
    if not value:
        return ""
    text = re.sub(r"<br\s*/?>", "\n", value)
    text = re.sub(r"<div[^>]*>", "\n", text)
    text = re.sub(r"</div>", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("&nbsp;", " ").replace("\xa0", " ")
    lines = [ln.strip() for ln in text.split("\n")]
    return "\n".join(ln for ln in lines if ln)


def _load(path: Path) -> tuple[dict, list[dict]]:
    model = ET.parse(path).getroot().find(".//mxGraphModel")
    vertices: dict[str, dict] = {}
    edges: list[dict] = []
    for cell in model.iter("mxCell"):
        geom = cell.find("mxGeometry")
        if geom is None:
            continue
        if cell.get("vertex") == "1":
            vertices[cell.get("id")] = {
                "x": float(geom.get("x", 0)), "y": float(geom.get("y", 0)),
                "w": float(geom.get("width", 0)), "h": float(geom.get("height", 0)),
                "text": _plain_text(cell.get("value")),
                "style": _parse_style(cell.get("style")),
            }
        elif cell.get("edge") == "1":
            edges.append({
                "source": cell.get("source"), "target": cell.get("target"),
                "label": _plain_text(cell.get("value")),
                "style": _parse_style(cell.get("style")),
            })
    return vertices, edges


def _route(src: dict, dst: dict, style: dict) -> list[tuple[float, float]]:
    """Orthogonal polyline between the declared exit/entry anchors."""
    sx = src["x"] + float(style.get("exitX", 0.5)) * src["w"]
    sy = src["y"] + float(style.get("exitY", 0.5)) * src["h"]
    ex = dst["x"] + float(style.get("entryX", 0.5)) * dst["w"]
    ey = dst["y"] + float(style.get("entryY", 0.5)) * dst["h"]
    if abs(sx - ex) < 0.5:
        return [(sx, sy), (ex, ey)]
    if abs(sy - ey) < 0.5:
        return [(sx, sy), (ex, sy)]
    mid = (sy + ey) / 2.0
    return [(sx, sy), (sx, mid), (ex, mid), (ex, ey)]


def _rounded_path(x, y, w, h, r, steps=10):
    """Closed point list for a rounded rectangle (draw.io corner radius)."""
    pts = []
    corners = (
        (x + w - r, y + r, -90, 0), (x + w - r, y + h - r, 0, 90),
        (x + r, y + h - r, 90, 180), (x + r, y + r, 180, 270),
    )
    for cx, cy, a0, a1 in corners:
        for i in range(steps + 1):
            a = math.radians(a0 + (a1 - a0) * i / steps)
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    pts.append(pts[0])
    return pts


def _dashed(draw, pts, color, width, dash, gap):
    """Draw a dashed polyline by walking cumulative arclength."""
    on, left = True, dash
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        seg = math.hypot(x1 - x0, y1 - y0)
        if seg <= 1e-9:
            continue
        t = 0.0
        while t < seg:
            step = min(left, seg - t)
            if on:
                draw.line(
                    [(x0 + (x1 - x0) * t / seg, y0 + (y1 - y0) * t / seg),
                     (x0 + (x1 - x0) * (t + step) / seg,
                      y0 + (y1 - y0) * (t + step) / seg)],
                    fill=color, width=width,
                )
            t += step
            left -= step
            if left <= 1e-9:
                on = not on
                left = dash if on else gap
    return


def _arrow(draw, pts, color, width, filled, head_l, head_w):
    """Polyline with a draw.io-style arrowhead at ``pts[-1]``."""
    (x0, y0), (x1, y1) = pts[-2], pts[-1]
    seg = math.hypot(x1 - x0, y1 - y0) or 1.0
    ux, uy = (x1 - x0) / seg, (y1 - y0) / seg
    base = (x1 - ux * head_l, y1 - uy * head_l)
    if len(pts) > 2:
        draw.line(list(pts[:-1]) + [base], fill=color, width=width,
                  joint="curve")
    else:
        draw.line([pts[0], base], fill=color, width=width)

    px, py = -uy, ux
    if filled:
        draw.polygon(
            [(x1, y1),
             (base[0] + px * head_w / 2, base[1] + py * head_w / 2),
             (base[0] - px * head_w / 2, base[1] - py * head_w / 2)],
            fill=color,
        )
    else:
        for sign in (1, -1):
            draw.line(
                [(x1, y1),
                 (base[0] + sign * px * head_w / 2,
                  base[1] + sign * py * head_w / 2)],
                fill=color, width=width,
            )


def main() -> None:
    vertices, edges = _load(DRAWIO)
    img = Image.new("RGB", (int(PAGE_W * S * SS), int(PAGE_H * S * SS)), "white")
    draw = ImageDraw.Draw(img)

    def px(v: float) -> float:
        return v * S * SS

    # ── boxes ────────────────────────────────────────────────────────────
    for box in vertices.values():
        style = box["style"]
        stroke = style.get("strokeColor", "#666666")
        fill = style.get("fillColor", "#FFFFFF")
        lw = max(1, round(px(1.2)))
        x0, y0 = px(box["x"]), px(box["y"])
        x1, y1 = x0 + px(box["w"]), y0 + px(box["h"])

        if style.get("dashed") == "1":
            path = _rounded_path(x0, y0, px(box["w"]), px(box["h"]), px(8))
            _dashed(draw, path, stroke, lw, dash=px(5), gap=px(3))
        else:
            draw.rounded_rectangle([x0, y0, x1, y1], radius=px(8), fill=fill,
                                   outline=stroke, width=lw)

        if not box["text"]:
            continue
        size = float(style.get("fontSize", 13)) * S * SS
        font = _font(round(size), style.get("fontStyle") == "1", False)
        draw.multiline_text(
            ((x0 + x1) / 2, (y0 + y1) / 2), box["text"], font=font,
            fill="#000000", anchor="mm", align="center",
            spacing=round(size * 0.32),
        )

    # ── edges (after the boxes, so arrowheads sit on the borders) ────────
    for edge in edges:
        src, dst = vertices.get(edge["source"]), vertices.get(edge["target"])
        if src is None or dst is None:
            continue
        style = edge["style"]
        color = style.get("strokeColor", "#555555")
        width = max(1, round(px(1.4)))
        pts = [(px(x), px(y)) for x, y in _route(src, dst, style)]
        head_l, head_w = px(11), px(9)

        if style.get("dashed") == "1":
            _dashed(draw, pts, color, width, dash=px(4), gap=px(3))
            (x0, y0), (x1, y1) = pts[-2], pts[-1]
            seg = math.hypot(x1 - x0, y1 - y0) or 1.0
            ux, uy = (x1 - x0) / seg, (y1 - y0) / seg
            base = (x1 - ux * head_l, y1 - uy * head_l)
            nx, ny = -uy, ux
            for sign in (1, -1):
                draw.line([(x1, y1), (base[0] + sign * nx * head_w / 2,
                                      base[1] + sign * ny * head_w / 2)],
                          fill=color, width=width)
        else:
            _arrow(draw, pts, color, width,
                   style.get("endArrow") != "open", head_l, head_w)

        if edge["label"]:
            mx = (pts[0][0] + pts[-1][0]) / 2
            my = min(p[1] for p in pts) - px(6)
            draw.text((mx, my), edge["label"],
                      font=_font(round(13 * S * SS * 0.75), False, True),
                      fill=color, anchor="ms")

    # ── crop to the diagram bounds (the draw.io export is tightly cropped) ─
    xs0 = min(px(b["x"]) for b in vertices.values())
    ys0 = min(px(b["y"]) for b in vertices.values())
    xs1 = max(px(b["x"] + b["w"]) for b in vertices.values())
    ys1 = max(px(b["y"] + b["h"]) for b in vertices.values())
    pad = px(PAD)
    img = img.crop((int(xs0 - pad), int(ys0 - pad),
                    int(xs1 + pad), int(ys1 + pad)))
    img = img.resize((img.width // SS, img.height // SS), Image.LANCZOS)

    for out in OUTS:
        out.parent.mkdir(parents=True, exist_ok=True)
        img.save(out, dpi=(300, 300))
        print(f"saved: {out.relative_to(ROOT)} ({img.width}x{img.height})")


if __name__ == "__main__":
    main()
