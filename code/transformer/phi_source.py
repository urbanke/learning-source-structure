"""Where the rows ln phi_{L,r}(e^u) come from: either the table files phitab_*.npy of code/scripts/building_block
(through lsa_fast.py) or the sealed anchor store of product_model_with_memory, which serves any u grid directly.

The store is used when the environment variables LSA_STORE (the store directory, e.g.
~/Projects/product_model_with_memory/tables/anchors_prod) and PMM_SRC (the package source directory, e.g.
~/Projects/product_model_with_memory/src) are set; otherwise table_dir must hold the phitab files. The two agree to
1e-12 (checked on L = 2 and 20). Depth 1 is in closed form in both cases.
"""
from __future__ import annotations
import os, sys
import numpy as np
from scipy.special import gammaln

H = 1.0 / 64


class StoreTables:
    """Duck-type of lsa_fast.LSAfast for lsa_seq.SeqLSA and refs_extra.FullTables: .u (grid), .h, .rows(L, rs)."""

    def __init__(self, Lmax, store, src, u_lo=-45.0, u_hi=40.0):
        os.environ.setdefault("PMM_UNIVERSAL_TABLES", store); os.environ.setdefault("PMM_PHI_LADDER_EVERY", "1")
        os.environ.setdefault("PMM_PHI_LADDER_DEGREE", "11"); os.environ.setdefault("PMM_PHI_SADDLE_MIN_L", "54")
        if src not in sys.path:
            sys.path.insert(0, src)
        from product_model_with_memory import universal_tables as UT
        self.ut = UT.UniversalTables(store, read_only=True)
        self.Lmax = int(Lmax); self.h = H
        self.u = np.arange(u_lo, u_hi + H / 2, H)

    def rows(self, L, rs):
        rs = np.asarray(rs)
        if L == 0:                                            # W_0 = 1: phi_{0,r}(s) = e^{-s}, the uniform row
            return np.repeat(-np.exp(self.u)[None, :], len(rs), axis=0)
        if L == 1:                                            # ln phi_{1,r}(s) = ln r! - (r+1) ln(1+s)
            r = rs.astype(float)
            return gammaln(r + 1.0)[:, None] - (r + 1.0)[:, None] * np.log1p(np.exp(self.u))[None, :]
        return self.ut.log_phi_matrix(int(L), [int(r) for r in rs], self.u)


def open_tables(Lmax, table_dir=".", u_lo=-45.0, u_hi=40.0):
    """The row source: the store if LSA_STORE and PMM_SRC are set, else the phitab tables in table_dir."""
    store = os.environ.get("LSA_STORE"); src = os.environ.get("PMM_SRC")
    if store and src:
        return StoreTables(Lmax, os.path.expanduser(store), os.path.expanduser(src), u_lo=u_lo, u_hi=u_hi)
    if table_dir not in sys.path:
        sys.path.insert(0, table_dir)
    import lsa_fast

    class _Tables(lsa_fast.LSAfast):
        def rows(self, L, rs):
            if L == 0:
                return np.repeat(-np.exp(self.u)[None, :], len(np.asarray(rs)), axis=0)
            return super().rows(L, rs)
    return _Tables(max(Lmax, 1), table_dir=table_dir)
