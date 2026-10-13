"""Historical ROW-LOCAL depth-mixture comparator, NOT the Bayes predictor of the training family.

The generators share one depth across all rows of a chain, whereas this
script mixes a separate depth in every row. Use refs_shared_depth.py for
the corrected reference. Retain this script and its old output names only
to reproduce historical comparisons.

The copier with the depth range of the training family, as a separate
reference file next to the ones of refs_family.py / make_refs.py:
  - for the paper's source at alpha = 0 (rows independent conditional on L_s, uniform in 0..8): the comparator mixes
    the depths 0..8 with equal weights (depth 0 is the uniform row), key 'copier_fam';
  - for the product chains (depths 1..8 per coordinate): the product of two such copiers with depths 1..8,
    key 'prod_copier_fam'.
The paper's copier mixes over the depths 1..L_max(d) (16, 20, 23, 26 at d = 32, 64, 128, 256).
Changing its depth range alone does not implement the shared-depth family prior.

    python3 refs_famprior.py --lsa --d 64 --alpha 0 --ls 2 --n_eval 2048 --out results/transformer/refs2048/d64
    python3 refs_famprior.py --set product:8x8:L=2,2 --n_eval 2048 --out results/transformer/refs_fam
Writes <tag>_famprior.npz (md5 of the tokens included). Chunked with resume like refs_family.py.
"""
from __future__ import annotations
import argparse, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from eval_sets import eval_set, md5_of, tag_of
from families import family_eval_set, tag_of_spec
import lsa_seq


def _depths(spec, default):
    a, b = (int(t) for t in spec.split("-")) if spec else default
    return list(range(a, b + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default=None, help="a product family spec")
    ap.add_argument("--lsa", action="store_true", help="the paper's source: use --d --alpha --ls")
    ap.add_argument("--d", type=int, default=None); ap.add_argument("--alpha", type=float, default=0.0); ap.add_argument("--ls", type=int, default=2)
    ap.add_argument("--n_ctx", type=int, default=1024); ap.add_argument("--n_eval", type=int, default=2048)
    ap.add_argument("--n_seq", type=int, default=None); ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--depths", default=None, help="explicit depth list, e.g. 0-8 (default: 0..8 for lsa, 1..8 for product)")
    ap.add_argument("--table_dir", default=HERE, help="directory with the phitab tables (ignored when the store is used)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.lsa:
        x, _ = eval_set(args.d, args.alpha, args.ls, args.n_ctx, args.n_eval); tag = tag_of(args.d, args.alpha, args.ls)
        depths = _depths(args.depths, (0, 8))
        S = lsa_seq.SeqLSA(args.d, table_dir=args.table_dir, depths=depths); key = "copier_fam"
        def losses(xc):
            return np.stack([S.run(xi.astype(int))["lC"][1:] for xi in xc])
    else:
        x, meta, fam = family_eval_set(args.set, args.n_ctx, args.n_eval); tag = tag_of_spec(args.set)
        depths = _depths(args.depths, (1, 8))
        S1 = lsa_seq.SeqLSA(fam.d1, table_dir=args.table_dir, depths=depths); S2 = S1 if fam.d2 == fam.d1 else lsa_seq.SeqLSA(fam.d2, table_dir=args.table_dir, depths=depths); key = "prod_copier_fam"
        def losses(xc):
            u, v = fam.split(xc.astype(int))
            return np.stack([S1.run(u[i])["lC"][1:] + S2.run(v[i])["lC"][1:] for i in range(len(xc))])
    if args.n_seq:
        x = x[: args.n_seq]
    N = x.shape[0]; md5 = md5_of(x); fn = os.path.join(args.out, f"{tag}_famprior.npz"); part = fn + ".part.npz"
    if os.path.exists(fn):
        print(f"{fn} exists"); return
    res = None; done = 0; t0 = time.time()
    if os.path.exists(part):
        P = np.load(part)
        if str(P["md5"]) == md5:
            done = int(P["done"]); res = P["loss"]; print(f"resuming at {done}", flush=True)
    while done < N:
        j = min(N, done + args.chunk); L = losses(x[done:j]).astype(np.float32)
        res = L if res is None else np.concatenate([res, L]); done = j
        np.savez(part, md5=md5, done=done, loss=res); print(f"  {tag}: {done}/{N}, {time.time()-t0:.0f} s", flush=True)
    np.savez_compressed(fn, md5=md5, depths=np.array(depths), **{key: res})
    os.remove(part)
    print(f"{tag}: {key} with depths {depths[0]}..{depths[-1]}: {res.mean():.4f} bits per token over {N} sequences ({time.time()-t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
