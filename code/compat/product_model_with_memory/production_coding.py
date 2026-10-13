"""The production guard of the development package, for the public engine.

``configure_production_tables`` selects the sealed anchor store and fixes the three
numerical settings the paper's references were computed with. It refuses a directory
that is not a designed store (no ``anchors.json``), because a grow-on-demand cache
would silently change both the evaluator and its runtime.
"""
from __future__ import annotations
import os
from pathlib import Path

PRODUCTION_TABLE_DEFAULTS = {
    "PMM_UNIVERSAL_TABLES": os.path.expanduser("~/Projects/product_model_with_memory/tables/anchors_prod"),
    "PMM_PHI_LADDER_EVERY": "1",
    "PMM_PHI_LADDER_DEGREE": "11",
    "PMM_PHI_SADDLE_MIN_L": "54",
}


def configure_production_tables(universal_path: str | Path | None = None) -> Path:
    """Select and validate the sealed anchor store; set the PMM_* and LSA_* settings."""
    if universal_path is None:
        universal_path = os.environ.get("PMM_UNIVERSAL_TABLES") or os.environ.get("LSA_STORE") \
            or PRODUCTION_TABLE_DEFAULTS["PMM_UNIVERSAL_TABLES"]
    path = Path(os.path.expanduser(str(universal_path)))
    os.environ["PMM_UNIVERSAL_TABLES"] = str(path)
    os.environ["LSA_UNIVERSAL_TABLES"] = str(path)
    for key in ("PMM_PHI_LADDER_EVERY", "PMM_PHI_LADDER_DEGREE", "PMM_PHI_SADDLE_MIN_L"):
        required = PRODUCTION_TABLE_DEFAULTS[key]
        actual = os.environ.setdefault(key, required)
        if actual != required:
            raise RuntimeError(f"production requires {key}={required}, got {actual!r}")
        os.environ.setdefault("LSA_" + key[4:], required)
    if not (path / "anchors.json").is_file():
        raise RuntimeError(f"production table store {path} is not the sealed designed anchor store "
                           "(anchors.json is missing); refusing to grow ad-hoc exact columns")
    return path
