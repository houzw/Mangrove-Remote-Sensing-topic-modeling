"""Render every page of the v5 and diff PDFs and measure ink coverage.

A page whose text layer is empty can still hold a figure; a page can also look blank while
carrying a few characters. Ink coverage settles it.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import tempfile

from PIL import Image

TMP = pathlib.Path(os.environ["TEMP"]) / "v5"
TARGETS = {
    "v5": TMP / "manuscript-v5.pdf",
    "diff": TMP / "manuscript-v5-diff.pdf",
    "appendix": TMP / "manuscript-v5-appendix.pdf",
}

for name, pdf in TARGETS.items():
    if not pdf.exists():
        print(f"{name}: missing")
        continue
    out = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, check=True)
    pages = int(out.stdout.decode("utf-8", "replace").split("Pages:")[1].split()[0])
    print(f"== {name}: {pages} pages")

    stem = pathlib.Path(tempfile.gettempdir()) / f"ink_{name}"
    for old in stem.parent.glob(f"{stem.name}-*.png"):
        old.unlink()
    subprocess.run(["pdftoppm", "-r", "50", "-gray", "-png", str(pdf), str(stem)], check=True)

    files = sorted(stem.parent.glob(f"{stem.name}-*.png"))
    blank = []
    for i, f in enumerate(files, 1):
        img = Image.open(f).convert("L")
        px = list(img.getdata())
        dark = sum(1 for v in px if v < 200) / len(px)
        tag = ""
        if dark < 0.002:
            tag = "  <== VISUALLY BLANK"
            blank.append(i)
        elif dark < 0.01:
            tag = "  <-- sparse"
        print(f"   p{i:>2}: ink {100 * dark:5.2f}%{tag}")
        f.unlink()
    print(f"   visually blank pages: {blank if blank else 'none'}")
    print()
