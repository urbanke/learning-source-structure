"""The mechanism references for the evaluation sets: per-position losses of the unigram and the copier and the
learner's weight on copying, for all sequences of a set, and their predictive distributions for the first n_full
sequences at every full_every-th position (the same sequences and positions as eval_model.py).

    python3 make_refs.py --d 64 --settings grid --out refs --table_dir <dir with phitab_*.npy and lsa_fast.py>
Needs the LSA tables; runs where they are (not on the training machine).
"""
from __future__ import annotations
import argparse, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from eval_sets import eval_set, md5_of, parse_settings, tag_of


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, required=True)
    ap.add_argument("--n_ctx", type=int, default=1024)
    ap.add_argument("--settings", default="grid")
    ap.add_argument("--n_eval", type=int, default=512)
    ap.add_argument("--n_full", type=int, default=64)
    ap.add_argument("--n_seq", type=int, default=None, help="compute only the first n_seq sequences of the set")
    ap.add_argument("--full_every", type=int, default=4)
    ap.add_argument("--no_permute", action="store_true")
    ap.add_argument("--table_dir", default=".")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sys.path.insert(0, args.table_dir)
    import lsa_seq
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    S = lsa_seq.SeqLSA(args.d, table_dir=args.table_dir)
    print(f"tables loaded in {time.time()-t0:.0f} s (d={args.d}, Lmax={S.Lmax})", flush=True)
    positions = np.arange(0, args.n_ctx, args.full_every)
    for alpha, ls in parse_settings(args.settings):
        tag = tag_of(args.d, alpha, ls)
        fn = os.path.join(args.out, f"{tag}.npz")
        if os.path.exists(fn):
            print(f"  {tag}: exists", flush=True); continue
        t0 = time.time()
        x, _ = eval_set(args.d, alpha, ls, args.n_ctx, args.n_eval, permute=not args.no_permute)
        if args.n_seq:
            x = x[: args.n_seq]
        N = x.shape[0]
        lF = np.empty((N, args.n_ctx), np.float32); lC = np.empty((N, args.n_ctx), np.float32); wC = np.empty((N, args.n_ctx), np.float32)
        pF = np.empty((args.n_full, len(positions), args.d), np.float16); pC = np.empty_like(pF)
        part = fn + ".part.npz"; start = 0
        if os.path.exists(part):                       # resume an interrupted setting
            P = np.load(part)
            if str(P["md5"]) == md5_of(x):
                start = int(P["done"]); lF[:start] = P["lF"][:start]; lC[:start] = P["lC"][:start]; wC[:start] = P["wC"][:start]
                pF[:min(start, args.n_full)] = P["pF"][:min(start, args.n_full)]; pC[:min(start, args.n_full)] = P["pC"][:min(start, args.n_full)]
                print(f"  {tag}: resuming at sequence {start}", flush=True)
        for i in range(start, N):
            if i > start and i % 128 == 0:
                np.savez(part, md5=md5_of(x), done=i, lF=lF, lC=lC, wC=wC, pF=pF, pC=pC)
            full = i < args.n_full
            # the mechanisms read the whole sequence; their prediction of token t+1 (t = 0..n_ctx-1) is compared with
            # the transformer's output at position t, which predicts the same token from the same context
            r = S.run(x[i].astype(int), full_probs=full, positions=positions + 1 if full else None)
            lF[i] = r["lF"][1:]; lC[i] = r["lC"][1:]; wC[i] = r["wC"][1:]
            if full:
                pF[i] = r["pF"]; pC[i] = r["pC"]
        np.savez_compressed(fn, tokens=x, lF=lF, lC=lC, wC=wC, pF=pF, pC=pC, positions=positions, alpha=alpha, ls=ls, d=args.d, md5=md5_of(x))
        if os.path.exists(part):
            os.remove(part)
        gap = (lF - lC).sum(axis=1)
        print(f"  {tag}: unigram {lF.mean():.3f}, copier {lC.mean():.3f} bits/token; mean gap at n_ctx {gap.mean():.0f} bits; {time.time()-t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
