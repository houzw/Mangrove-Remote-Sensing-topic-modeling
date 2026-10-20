"""Shared utilities: logging, GPU detection, timing, I/O helpers."""

from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional


# =============================================================================
# Logging
# =============================================================================


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str | Path] = None,
    module_name: Optional[str] = None,
) -> logging.Logger:
    """Configure root logger with console and optional file handler.

    Args:
        level: Logging level name (DEBUG, INFO, WARNING, ERROR).
        log_file: Path to a log file (appended). If None, console-only.
        module_name: If given, return a named logger instead of root.

    Returns:
        Configured logger instance.
    """
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)-7s] %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.handlers.clear()

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    root.addHandler(console)

    # File handler (if requested)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(str(log_path), encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)
        root.info("Logging to %s", log_path)

    return logging.getLogger(module_name) if module_name else root


# =============================================================================
# GPU / device detection
# =============================================================================


def detect_device(prefer_gpu: bool = True) -> str:
    """Detect the best available PyTorch device.

    Args:
        prefer_gpu: If True, try CUDA then MPS before falling back to CPU.

    Returns:
        One of ``"cuda"``, ``"mps"``, or ``"cpu"``.
    """
    if not prefer_gpu:
        return "cpu"

    try:
        import torch

        if torch.cuda.is_available():
            device = "cuda"
            gpu_name = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "Unknown GPU"
            logging.getLogger(__name__).info("GPU detected: %s (%s)", gpu_name, device)
            return device

        if torch.backends.mps.is_available():
            logging.getLogger(__name__).info("MPS (Apple Silicon) detected")
            return "mps"

    except ImportError:
        logging.getLogger(__name__).warning("PyTorch not installed — GPU detection skipped")

    logging.getLogger(__name__).info("No GPU found — using CPU")
    return "cpu"


# =============================================================================
# File-system helpers
# =============================================================================


def ensure_dir(path: str | Path) -> Path:
    """Create directory if it does not exist; return resolved Path."""
    p = Path(path).resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_project_root() -> Path:
    """Walk up from this file to find the project root (contains config.yaml or pixi.toml)."""
    current = Path(__file__).resolve().parent
    for ancestor in [current] + list(current.parents):
        if (ancestor / "pixi.toml").exists() or (ancestor / "config.yaml").exists():
            return ancestor
    # Fallback: two levels up from src/
    return current.parent


# =============================================================================
# Country-space helper (single source of truth for CU parsing)
# =============================================================================


def canonical_countries(
    cu_value: object,
    fold: Optional[dict[str, str] | tuple[dict[str, str], ...] | list[dict[str, str]]] = None,
    drop: Optional[frozenset[str] | set[str]] = None,
) -> list[str]:
    """Distinct canonical countries named by one ``CU`` value.

    A ``CU`` cell is a semicolon-separated list and may name a metropole and one
    of its folded territories in the same record (``"China;Taiwan"``,
    ``"France;New Caledonia"``, ``"United States;Puerto Rico"``).  Callers must
    fold **and deduplicate** before computing anything per country: taking
    ``1/len(tokens)`` as a fractional weight, or counting one publication per
    token, silently double-counts the metropole.  That bug was live in the
    ``n_pubs`` column of ``bivariate_attr.csv``, in
    ``bibliometric_analysis.compute_country_stats``, in the deficit/LQ weights
    and in the citation-side indicator until 2026-09-21 (see CLAUDE.md
    "Process lessons"), which is why every per-country derivation must go
    through this helper.

    Args:
        cu_value: Raw ``CU`` cell (``"China;Taiwan"``); ``None``/NaN -> ``[]``.
        fold: Territory -> metropole mapping, or a sequence of mappings applied
            in order (the pipeline uses a two-step fold:
            ``_TERRITORY_TO_METROPOLE`` then ``_PUB_ONLY_FOLD``).
        drop: Names to exclude before folding (unknown sentinels and the
            ``*_CU_FALSE_POSITIVES`` sets of the consumer modules).

    Returns:
        Folded, de-duplicated country names, in first-seen order.
    """
    if cu_value is None:
        return []
    text = str(cu_value).strip()
    if not text or text.lower() in ("unknown", "nan"):
        return []

    folds = fold if isinstance(fold, (tuple, list)) else ([fold] if fold else [])

    out: list[str] = []
    seen: set[str] = set()
    for raw in text.split(";"):
        name = raw.strip()
        if not name or name.lower() in ("unknown", "nan"):
            continue
        if drop and name in drop:
            continue
        for mapping in folds:
            name = mapping.get(name, name)
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


# =============================================================================
# Timer context manager
# =============================================================================


class Timer:
    """Context manager that logs elapsed time.

    Usage::

        with Timer("Data preprocessing"):
            do_work()

        # Logs: "Data preprocessing completed in 12.34 s"
    """

    def __init__(self, label: str = "Operation"):
        self.label = label
        self._start: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        logging.getLogger(__name__).info("%s ...", self.label)
        return self

    def __exit__(self, *args: object) -> None:
        elapsed = time.perf_counter() - self._start
        logging.getLogger(__name__).info("%s completed in %.2f s", self.label, elapsed)


# =============================================================================
# DataFrame summarisers (used by multiple pipeline steps)
# =============================================================================


def log_df_stats(df: "pd.DataFrame", label: str = "DataFrame") -> None:  # noqa: F821
    """Log row/column counts and memory usage for a DataFrame."""
    import pandas as pd

    logger = logging.getLogger(__name__)
    logger.info(
        "%s: %d rows × %d columns, %.1f MB",
        label,
        len(df),
        len(df.columns),
        df.memory_usage(deep=True).sum() / (1024 * 1024),
    )
