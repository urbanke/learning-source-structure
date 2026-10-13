"""Adapter: the public LSA engine under the name the experiment code was written against.

The Bayesian reference computations use the universal tables of the layered simplex
architecture. During the project they came from the private development package
``product_model_with_memory``. The same modules are published in the ``lsa`` package of
https://github.com/urbanke/layered-simplex-architecture (``universal_tables``, ``kernel``,
``mellin``): identical code, with the package name and the ``PMM_`` environment prefix
renamed to ``lsa`` and ``LSA_``.

This package maps the names back, so that every script runs unchanged against the public
engine. Point ``PMM_SRC`` at this directory (``code/compat``) and ``LSA_STORE`` at an anchor
store built with ``scripts/build_anchor_store.py`` of the public repository. If ``PMM_SRC``
points at the private package instead, nothing here is used.
"""
import os as _os

_KEYS = ("UNIVERSAL_TABLES", "PHI_LADDER", "PHI_LADDER_EVERY", "PHI_LADDER_DEGREE",
         "PHI_SADDLE_MIN_L", "SERIES_TAIL", "PHI_WAVE", "PHI_WAVE_SCALE", "PHI_BIAS")


def map_settings():
    """Copy PMM_* settings to the LSA_* names the public engine reads. Call before importing it."""
    for key in _KEYS:
        if "PMM_" + key in _os.environ and "LSA_" + key not in _os.environ:
            _os.environ["LSA_" + key] = _os.environ["PMM_" + key]


map_settings()
