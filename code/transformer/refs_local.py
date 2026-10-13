"""Run all reference computations of the family experiments on one machine with several cores, reading the LSA
rows from the sealed anchor store of product_model_with_memory (no table files needed). Run from the project folder:

    export LSA_STORE=~/Projects/product_model_with_memory/tables/anchors_prod
    export PMM_SRC=~/Projects/product_model_with_memory/src
    python3 code/transformer/refs_local.py --jobs 12          # add --n_seq 256 for a quick first pass

Outputs (git-ignored): results/transformer/refs_fam/<set>.npz (refs_family.py) and results/transformer/refs2048/d<d>/<tag>.npz
(make_refs.py). Jobs whose output exists are skipped; interrupted jobs resume from their partial files.
"""
from __future__ import annotations
import argparse, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


def jobs(n_seq):
    J = []; ns = ["--n_seq", str(n_seq)] if n_seq else []
    out_f = "results/transformer/refs_fam"; out_m = "results/transformer/refs2048"
    tag = lambda spec: spec.replace(":", "_").replace("=", "").replace(",", "-")
    for spec in ("binary:twopoint", "binary:uniform", "binary:eightpoint", "binary:fixed=0.5", "binary:fixed=0.2"):
        J.append((f"{out_f}/{tag(spec)}.npz", [PY, f"{HERE}/refs_family.py", "--set", spec, "--n_ctx", "256", "--n_eval", "4096", "--out", out_f]))
    for spec in ("expfam:k=1", "expfam:k=2", "expfam:k=1:perm", "expfam:k=1:theta=1", "expfam:k=2:theta=1,0.1"):
        J.append((f"{out_f}/{tag(spec)}.npz", [PY, f"{HERE}/refs_family.py", "--set", spec, "--d", "32", "--n_ctx", "1024", "--n_eval", "2048", "--table_dir", HERE, "--out", out_f] + ns))
    for spec, n in (("product:8x8:L=2,2", 2048), ("product:8x8:L=4,4", 2048), ("product:8x8:L=2,6", 2048), ("product:16x16:L=2,2", 1024), ("product:16x16:L=4,4", 1024)):
        J.append((f"{out_f}/{tag(spec)}.npz", [PY, f"{HERE}/refs_family.py", "--set", spec, "--n_ctx", "1024", "--n_eval", str(n), "--table_dir", HERE, "--out", out_f] + ns))
    for d in (64, 128):
        for a, L in ((0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4)):
            J.append((f"{out_m}/d{d}/d{d}_a{a}_L{L}.npz", [PY, f"{HERE}/make_refs.py", "--d", str(d), "--settings", f"{a},{L}", "--n_eval", "2048", "--n_full", "32", "--table_dir", HERE, "--out", f"{out_m}/d{d}"] + ns))
    # the copiers with the depth prior of the training family (refs_famprior.py)
    for d in (64, 128):
        for a, L in ((0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4)):
            J.append((f"{out_m}/d{d}/d{d}_a{a}_L{L}_famprior.npz", [PY, f"{HERE}/refs_famprior.py", "--lsa", "--d", str(d), "--alpha", str(a), "--ls", str(L), "--n_eval", "2048", "--out", f"{out_m}/d{d}"] + ns))
    for spec, n in (("product:8x8:L=2,2", 2048), ("product:8x8:L=4,4", 2048), ("product:8x8:L=2,6", 2048), ("product:16x16:L=2,2", 1024), ("product:16x16:L=4,4", 1024)):
        J.append((f"{out_f}/{tag(spec)}_famprior.npz", [PY, f"{HERE}/refs_famprior.py", "--set", spec, "--n_eval", str(n), "--out", out_f] + ns))
    return J


def run(job):
    out, cmd = job
    if os.path.exists(out):
        return out, "exists", 0.0
    os.makedirs(os.path.dirname(out), exist_ok=True)
    t0 = time.time()
    with open(out + ".log", "w") as log:
        r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    return out, ("done" if r.returncode == 0 and os.path.exists(out) else f"failed (exit {r.returncode}, see {out}.log)"), time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=4, help="parallel processes")
    ap.add_argument("--n_seq", type=int, default=None, help="only the first n_seq sequences of every set (quick pass)")
    ap.add_argument("--only", default="", help="substring filter on the output path")
    args = ap.parse_args()
    if not (os.environ.get("LSA_STORE") and os.environ.get("PMM_SRC")):
        sys.exit("set LSA_STORE and PMM_SRC (see the docstring)")
    J = [j for j in jobs(args.n_seq) if args.only in j[0]]
    print(f"{len(J)} jobs, {args.jobs} at a time", flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        for out, status, dt in ex.map(run, J):
            print(f"{status:>10s}  {dt:6.0f} s  {out}", flush=True)
    print(f"all done in {time.time()-t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
