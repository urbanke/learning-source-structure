"""Reference predictors for the evaluation sets of the families in families.py: per-position losses in bits,
aligned with the transformer's positions (position t predicts x_{t+1} from x_1..x_t, t = 1..n_ctx), plus h_true.

    python3 refs_family.py --set binary:twopoint --n_ctx 256 --n_eval 4096 --out refs_fam
    python3 refs_family.py --set expfam:k=1 --d 32 --n_ctx 1024 --n_eval 2048 --out refs_fam [--table_dir .]
    python3 refs_family.py --set product:8x8:L=2,2 --n_ctx 1024 --n_eval 512 --out refs_fam --table_dir .

Predictors per family (the family's Bayes predictor is marked *):
  binary:   laplace (add-one; * for the uniform prior), kt (add-half), twopoint (* for the two-point prior),
            eightpoint (* for the eight-point prior); the two finite-prior predictors are the posterior predictives.
  expfam:   bayes_k1 and bayes_k2 (* : the posterior predictive on a grid over theta with the labels known;
            for a relabelled set the oracle rung, since the labels are supplied), laplace_d (add-one over d symbols),
            zipf_plugin (the informed unigram predictor of Section 4: the Zipf estimate from ranked counts, k = 1),
            and, with --table_dir, the LSA unigram predictor of the paper.
  product:  with --table_dir: historical unigram and copier comparators on joint/coordinate symbols.
            These mix depth independently per row, unlike the source's shared depth per coordinate.
            Use refs_shared_depth.py for the corrected family Bayesian reference.
Writes <tag>.npz with one array per predictor, 'h_true', 'tokens', 'md5', 'spec'.
"""
from __future__ import annotations
import argparse, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from families import family_eval_set, tag_of_spec, parse_spec, BinaryFamily, ExpFamily, ProductFamily
from eval_sets import md5_of
LN2 = np.log(2.0)


# ---- binary --------------------------------------------------------------------------------------------------------
def binary_refs(x):
    """x: (N, n) in {0, 1}. Returns dict of (N, n-1) losses."""
    N, n = x.shape; out = {}
    n1 = np.cumsum(x, axis=1)[:, :-1].astype(float)                    # ones among x_1..x_t, t = 1..n-1
    t = np.arange(1, n, dtype=float)[None, :]; n0 = t - n1
    nxt = x[:, 1:]
    def loss(p1):
        p = np.where(nxt == 1, p1, 1 - p1); return -np.log2(np.clip(p, 1e-300, None))
    out["laplace"] = loss((n1 + 1) / (t + 2))
    out["kt"] = loss((n1 + 0.5) / (t + 1))
    for name, thetas in BinaryFamily.PRIORS.items():
        # posterior over the finite prior after x_1..x_t: w_j ~ theta_j^n1 (1-theta_j)^n0
        lw = n1[:, :, None] * np.log(thetas)[None, None, :] + n0[:, :, None] * np.log(1 - thetas)[None, None, :]
        lw -= lw.max(axis=2, keepdims=True); w = np.exp(lw); w /= w.sum(axis=2, keepdims=True)
        out[name] = loss((w * thetas[None, None, :]).sum(axis=2))
    return out


# ---- expfam (vectorised over the sequences) ---------------------------------------------------------------------------
def _grid(d, k, n1=201, n2=41):
    b = np.arange(1, d + 1, dtype=float); F = np.stack([-np.log(b), -np.log(b) ** 2])[:k]
    if k == 1:
        th = np.linspace(0.5, 1.5, n1)[:, None]
    else:
        g1, g2 = np.meshgrid(np.linspace(0.5, 1.5, n2), np.linspace(-0.15, 0.15, n2), indexing="ij")
        th = np.stack([g1.ravel(), g2.ravel()], 1)
    logits = th @ F; m = logits.max(1, keepdims=True)
    return logits - m - np.log(np.exp(logits - m).sum(1, keepdims=True))            # log p_theta(b): (G, d)


def grid_bayes(x, d, k):
    """Posterior predictive on a grid over theta (uniform prior on the grid), labels known. x: (N, n) int."""
    logp = _grid(d, k); p = np.exp(logp); G = logp.shape[0]
    N, n = x.shape; out = np.empty((N, n - 1)); ll = np.zeros((N, G)); rows = np.arange(N)
    for t in range(1, n):
        ll += logp[:, x[:, t - 1]].T                                                 # (N, G)
        w = np.exp(ll - ll.max(1, keepdims=True)); w /= w.sum(1, keepdims=True)
        pred = w @ p                                                                  # (N, d)
        out[:, t - 1] = -np.log2(pred[rows, x[:, t]])
    return out


def laplace_d(x, d):
    N, n = x.shape; out = np.empty((N, n - 1)); m = np.zeros((N, d)); rows = np.arange(N)
    for t in range(1, n):
        m[rows, x[:, t - 1]] += 1
        out[:, t - 1] = -np.log2((m[rows, x[:, t]] + 1) / (t + d))
    return out


def zipf_plugin(x, d):
    """The informed unigram predictor of Section 4 (refs_extra.zipf_estimate), vectorised: symbols ranked by their
    counts (ties share the mean Zipf mass of their block of ranks), exponents on the grid weighted by the multinomial
    likelihood with a uniform prior."""
    from refs_extra import ALPHAS
    ranks = np.arange(1, d + 1, dtype=float); Q = np.stack([ranks ** (-a) / (ranks ** (-a)).sum() for a in ALPHAS])   # (A, d)
    N, n = x.shape; out = np.empty((N, n - 1)); m = np.zeros((N, d)); rows = np.arange(N); base = (rows * d)[:, None]
    for t in range(1, n):
        m[rows, x[:, t - 1]] += 1
        order = np.argsort(-m, axis=1, kind="stable"); ms = np.take_along_axis(m, order, axis=1)          # sorted counts (N, d)
        gid = np.cumsum(np.concatenate([np.ones((N, 1), int), (ms[:, 1:] != ms[:, :-1]).astype(int)], 1), 1) - 1  # block ids per row
        ids = (base + gid).ravel(); cnt = np.bincount(ids, minlength=N * d)
        qsorted = np.empty((len(ALPHAS), N, d)); loglik = np.empty((N, len(ALPHAS)))
        for i in range(len(ALPHAS)):
            qs = np.bincount(ids, weights=np.broadcast_to(Q[i], (N, d)).ravel(), minlength=N * d)[ids].reshape(N, d) / cnt[ids].reshape(N, d)
            qsorted[i] = qs; loglik[:, i] = (ms * np.log(qs)).sum(1)
        w = np.exp(loglik - loglik.max(1, keepdims=True)); w /= w.sum(1, keepdims=True)
        qhat_sorted = np.einsum("na,and->nd", w, qsorted)
        qhat = np.empty((N, d)); np.put_along_axis(qhat, order, qhat_sorted, axis=1); qhat /= qhat.sum(1, keepdims=True)
        out[:, t - 1] = -np.log2(qhat[rows, x[:, t]])
    return out


def lsa_unigram(x, d, table_dir):
    """The LSA unigram predictor of the paper (one table per sequence)."""
    sys.path.insert(0, table_dir); import lsa_seq
    S = lsa_seq.SeqLSA(d, table_dir=table_dir); N, n = x.shape; out = np.empty((N, n - 1))
    for i in range(N):
        T = lsa_seq._Table(S); T.observe(int(x[i, 0]))
        for t in range(1, n):
            out[i, t - 1] = T.observe(int(x[i, t]))
    return out


# ---- product -----------------------------------------------------------------------------------------------------------
def product_refs(x, fam, table_dir):
    sys.path.insert(0, table_dir); import lsa_seq
    S = lsa_seq.SeqLSA(fam.d, table_dir=table_dir); S1 = lsa_seq.SeqLSA(fam.d1, table_dir=table_dir)
    S2 = S1 if fam.d2 == fam.d1 else lsa_seq.SeqLSA(fam.d2, table_dir=table_dir)
    u, v = fam.split(x.astype(int)); N, n = x.shape
    keys = ("unigram", "copier", "prod_unigram", "prod_copier"); out = {k: np.empty((N, n - 1)) for k in keys}
    for i in range(N):
        r = S.run(x[i].astype(int)); r1 = S1.run(u[i]); r2 = S2.run(v[i])
        out["unigram"][i] = r["lF"][1:]; out["copier"][i] = r["lC"][1:]
        out["prod_unigram"][i] = r1["lF"][1:] + r2["lF"][1:]; out["prod_copier"][i] = r1["lC"][1:] + r2["lC"][1:]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, help="family spec of the evaluation set")
    ap.add_argument("--d", type=int, default=None)
    ap.add_argument("--n_ctx", type=int, default=256)
    ap.add_argument("--n_eval", type=int, default=4096)
    ap.add_argument("--table_dir", default=None, help="directory with the LSA tables (enables the LSA predictors)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk", type=int, default=128, help="sequences per chunk (partial results are saved per chunk)")
    ap.add_argument("--n_seq", type=int, default=None, help="compute only the first n_seq sequences of the set")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    x, meta, fam = family_eval_set(args.set, args.n_ctx, args.n_eval, d=args.d)
    if args.n_seq:
        x = x[: args.n_seq]; meta = {k: (v[: args.n_seq] if isinstance(v, np.ndarray) and v.shape[:1] == (len(meta["h_true"]),) else v) for k, v in meta.items()}
    name, o = parse_spec(args.set); t0 = time.time(); res = {}
    tag = tag_of_spec(args.set); part = os.path.join(args.out, f"{tag}.part.npz"); md5 = md5_of(x)
    if name == "binary":
        res = binary_refs(x.astype(int))
    else:
        # the slow families are computed in chunks of sequences, with the partial result saved after every chunk, so
        # that an interrupted job resumes where it stopped
        N = x.shape[0]; done = 0
        if os.path.exists(part):
            P = np.load(part)
            if str(P["md5"]) == md5:
                done = int(P["done"]); res = {k: P[k] for k in P.files if k not in ("md5", "done")}
                print(f"resuming {tag} at sequence {done}", flush=True)
        S = None
        while done < N:
            j = min(N, done + args.chunk); xc = x[done:j]
            if name == "expfam":
                xi = xc.astype(int); xk = np.take_along_axis(np.argsort(meta["perm"][done:j], axis=1), xi, axis=1)   # labels known
                r = {"bayes_k1": grid_bayes(xk, fam.d, 1), "bayes_k2": grid_bayes(xk, fam.d, 2), "laplace_d": laplace_d(xi, fam.d), "zipf_plugin": zipf_plugin(xi, fam.d)}
                if args.table_dir:
                    r["lsa_unigram"] = lsa_unigram(xi, fam.d, args.table_dir)
            elif name == "product":
                assert args.table_dir, "the product family needs the LSA tables"
                r = product_refs(xc, fam, args.table_dir)
            else:
                raise SystemExit("use make_refs.py for the lsa family")
            for k, v in r.items():
                res[k] = v.astype(np.float32) if k not in res else np.concatenate([res[k], v.astype(np.float32)])
            done = j
            np.savez(part, md5=md5, done=done, **res)
            print(f"  {tag}: {done}/{N} sequences, {time.time()-t0:.0f} s", flush=True)
    res = {k: v.astype(np.float32) for k, v in res.items()}
    np.savez_compressed(os.path.join(args.out, f"{tag}.npz"), tokens=x, md5=md5, spec=args.set, h_true=meta["h_true"], **res)
    if os.path.exists(part):
        os.remove(part)
    h = meta["h_true"]
    print(f"{tag}: {x.shape[0]} sequences, {time.time()-t0:.0f} s; regret over the window (bits per sequence): " +
          "  ".join(f"{k} {(v - h).sum(1).mean():.2f}" for k, v in res.items()), flush=True)


if __name__ == "__main__":
    main()
