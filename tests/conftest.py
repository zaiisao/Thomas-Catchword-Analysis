"""Shared pytest fixtures + path setup."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest


def have_data(*relative_paths: str) -> bool:
    """True iff every given path exists under repo root."""
    return all((ROOT / p).exists() for p in relative_paths)


def skip_if_no_data(*paths: str) -> pytest.MarkDecorator:
    return pytest.mark.skipif(
        not have_data(*paths),
        reason=f"missing data: {paths}",
    )
