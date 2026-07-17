"""Shared path bootstrap so scripts can import the src package."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def paths() -> dict:
    """Return the standard project directories, created if missing."""
    d = {
        "root": ROOT,
        "data": os.path.join(ROOT, "data"),
        "outputs": os.path.join(ROOT, "outputs"),
        "docs": os.path.join(ROOT, "docs"),
        "images": os.path.join(ROOT, "docs", "images"),
    }
    for key in ("data", "outputs", "docs", "images"):
        os.makedirs(d[key], exist_ok=True)
    return d
