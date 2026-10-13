"""Merge the r-blocks (u in [-45,15]) and the u-extension (u in (15,40]) into one file per depth:
phitab_L{L}_r0_1000_full.npy of shape (1001, 5441), columns = the full u grid."""
import sys, numpy as np
L0, L1 = int(sys.argv[1]), int(sys.argv[2])
for L in range(L0, L1 + 1):
    parts = [np.load(f"phitab_L{L}.npy", mmap_mode="r"), np.load(f"phitab_L{L}_r201_700.npy", mmap_mode="r"),
             np.load(f"phitab_L{L}_r701_1000.npy", mmap_mode="r")]
    base = np.vstack([np.asarray(p) for p in parts])
    ext = np.asarray(np.load(f"phitab_L{L}_u15_40.npy", mmap_mode="r"))
    assert base.shape[0] == 1001 and ext.shape[0] == 1001, (base.shape, ext.shape)
    full = np.hstack([base, ext])
    np.save(f"phitab_L{L}_r0_1000_full.npy", full)
    print(f"L={L} merged {full.shape}", flush=True)
