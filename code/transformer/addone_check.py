"""Depth-one (add-one) rungs of the ladder on the existing sequences: the add-one copier (one Laplace table per
context) and the informed add-one copier (the weighted code at depth one with the Zipf estimate). Compared per
window with the copier, the informed copier and the transformer. Usage: python3 tf/addone_check.py d alpha ls n_seq"""
import sys, time, numpy as np
sys.path.insert(0, "tf"); sys.path.insert(0, ".")
import lsa_seq
from refs_extra import FullTables, run_extra
from eval_sets import eval_set
from source import zipf
LN2 = np.log(2.0)
d, a, L, n_seq = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
run = {32: "d32", 64: "d64_seed1", 128: "d128"}[d]; tag = f"d{d}_a{a:g}_L{L}"
R = np.load(f"tf_refs/d{d}/{tag}.npz"); X = np.load(f"tf_refs3/d{d}/extra_{tag}.npz"); E = np.load(f"tf_eval/{run}/{tag}_step100000.npz")
assert str(R["md5"]) == str(X["md5"]) == str(E["md5"])
N = min(n_seq, int(X["n_seq"]))
x, meta = eval_set(d, a, L, 1024, 512, permute=True); q = zipf(d, a)
t0 = time.time()
S_uni = lsa_seq.SeqLSA(d, table_dir=".")                     # the unigram (for the Zipf estimate) and the copier's depths
S_one = lsa_seq.SeqLSA(d, table_dir=".", depths=[1])         # add-one per context
T_one = FullTables([1], S_uni.rmax, ".")                     # weighted code at depth one
print("tables", round(time.time() - t0), "s", flush=True)
lD1 = np.empty((N, 1024)); lZ1 = np.empty((N, 1024))
for i in range(N):
    perm = meta["perm"][i]; q_true = np.empty(d); q_true[perm] = q
    r = run_extra(x[i].astype(int), q_true, None, S_one, T_one, S_uni, which="DZ")
    lD1[i] = r["D"]; lZ1[i] = r["Z"]
    if i % 5 == 4: print(i + 1, round(time.time() - t0), "s", flush=True)
np.savez(f"tf_refs3/d{d}/addone_{tag}.npz", lD1=lD1, lZ1=lZ1, n_seq=N, md5=str(R["md5"]))
lF = R["lF"][:N].astype(float); lC = R["lC"][:N].astype(float); lZ = X["lZ"][:N].astype(float); lD = X["lD"][:N].astype(float); lO = X["lO"][:N].astype(float)
lT = -E["logp_true"][:N].astype(float) / LN2
print(f"({d},{a},{L}) N={N}: overall  copier {lC.mean():.3f}  add-one copier {lD1.mean():.3f}  true depth {lD.mean():.3f}  informed {lZ.mean():.3f}  informed add-one {lZ1.mean():.3f}  oracle {lO.mean():.3f}  transformer {lT.mean():.3f}")
for lo, hi in ((1, 8), (8, 32), (32, 128), (128, 512), (512, 1025)):
    s = slice(lo - 1, hi - 1)
    print(f"   {lo}-{hi-1}: copier {lC[:, s].mean():.3f}  add-one {lD1[:, s].mean():.3f}  true depth {lD[:, s].mean():.3f}  informed {lZ[:, s].mean():.3f}  informed add-one {lZ1[:, s].mean():.3f}  oracle {lO[:, s].mean():.3f}  transformer {lT[:, s].mean():.3f}")
