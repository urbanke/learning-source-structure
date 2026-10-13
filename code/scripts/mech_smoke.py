"""Project 1 smoke experiment: posterior trajectories of the two-mechanism menu.

Mechanisms (mechanisms.tex, Section 2):
  m0: one depth-averaged LSA code on the full sequence (memoryless).
  m1: per-context depth-averaged LSA codes on successor subsequences
      (the induction mechanism), plus log2(d) bits for the first token.

Sources (Section 3):
  iid    x_t iid Zipf(alpha) on {1..d}                      (control)
  table  per-sequence random association table: rows iid
         product-simplex draws at depth L_rows; Markov chain  (ICL world)

Everything is evaluated exactly from count profiles via the LSA package
(depth_averaged_codelength); Monte Carlo enters only through the sampled
trajectories, with fixed seeds.  One (source, seed) pair per invocation;
results land as JSON in --out.

Validation mode (--validate) checks, before anything is trusted:
  V1 chain rule: sequential accumulation of the depth-averaged predictive
     equals the batch profile codelength (tiny scale, to 1e-6 bits);
  V2 one-state collapse: m1 with a single merged context state equals m0
     run on x_2..x_n plus log2(d) for the start token (to 1e-9 bits);
  V3 the envelope sandwich of Proposition 1 holds pathwise.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from lsa.codelength import default_l_max, depth_averaged_codelength
from lsa.fast_tables import build_tables_fast
from lsa.product_simplex import sample_product_simplex_distribution

LOG2 = math.log(2.0)


# ---------------------------------------------------------------------------
# sources
# ---------------------------------------------------------------------------

def zipf(d: int, alpha: float) -> np.ndarray:
    p = np.arange(1, d + 1, dtype=float) ** (-alpha)
    return p / p.sum()


def draw_iid(n: int, d: int, alpha: float, rng: np.random.Generator) -> np.ndarray:
    return rng.choice(d, size=n, p=zipf(d, alpha))


def draw_table_chain(n: int, d: int, l_rows: int,
                     rng: np.random.Generator,
                     tilt: np.ndarray | None = None,
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Markov chain from a per-sequence random table T.

    Rows are iid product-simplex draws at depth ``l_rows``; with ``tilt``
    (a probability vector q), each row is multiplied by q coordinatewise and
    renormalized: a shared backbone layer (the "language") composed with
    per-sequence layers (the "document").  The marginal then follows q
    approximately, so the memoryless mechanism has something to learn first.
    """

    T = np.empty((d, d))
    for i in range(d):
        T[i] = sample_product_simplex_distribution(d, l_rows, rng)
        if tilt is not None:
            T[i] *= tilt
            T[i] /= T[i].sum()
    x = np.empty(n, dtype=np.int64)
    x[0] = rng.integers(d) if tilt is None else rng.choice(d, p=tilt)
    for t in range(1, n):
        x[t] = rng.choice(d, p=T[x[t - 1]])
    return x, T


# ---------------------------------------------------------------------------
# codelengths
# ---------------------------------------------------------------------------

def bits_of_counts(counts: np.ndarray, d: int, l_max: int, tables=None) -> float:
    """-log2 Q_avg of one count vector (0 tokens -> 0 bits)."""

    if counts.sum() == 0:
        return 0.0
    r = depth_averaged_codelength(counts, d=d, l_max=l_max, tables=tables)
    return -r.log2_q_avg


def trajectory(x: np.ndarray, d: int, checkpoints: list[int],
               state_of: np.ndarray | None = None,
               state_tables=None, l_max: int | None = None,
               verbose: bool = True) -> dict[int, tuple[float, float]]:
    """Cumulative (l0, l1) at each checkpoint.

    ``state_of`` maps a context symbol to its state (identity = full order-1
    mechanism; all-zeros = the V2 collapse check).  Per-state codelengths are
    memoized and recomputed only for states touched since the last checkpoint.
    """

    n = len(x)
    if l_max is None:
        l_max = default_l_max(d)
    if state_of is None:
        state_of = np.arange(d)
    n_states = int(state_of.max()) + 1

    uni = np.zeros(d, dtype=np.int64)
    succ = {}                    # state -> count vector of successors
    cl_state = np.zeros(n_states)  # memoized per-state codelengths
    dirty: set[int] = set()
    out: dict[int, tuple[float, float]] = {}
    cps = sorted(set(checkpoints))
    ci = 0
    t_start = time.time()

    for t in range(n):
        uni[x[t]] += 1
        if t >= 1:
            s = int(state_of[x[t - 1]])
            if s not in succ:
                succ[s] = np.zeros(d, dtype=np.int64)
            succ[s][x[t]] += 1
            dirty.add(s)
        while ci < len(cps) and t + 1 == cps[ci]:
            for s in dirty:
                cl_state[s] = bits_of_counts(succ[s], d, l_max, state_tables)
            dirty.clear()
            l0 = bits_of_counts(uni, d, l_max, tables=None)
            l1 = math.log2(d) + float(cl_state.sum())
            out[t + 1] = (l0, l1)
            if verbose:
                print(f"  checkpoint {t+1:>6}: l0={l0:9.1f}  l1={l1:9.1f}  "
                      f"odds={l0-l1:+8.2f} bits  [{time.time()-t_start:5.1f}s]",
                      flush=True)
            ci += 1
    return out


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------

def validate() -> None:
    from lsa.estimators import lsa_predictive_by_count

    rng = np.random.default_rng(1)
    d, n = 8, 40
    l_max = default_l_max(d)
    x = draw_iid(n, d, 1.0, rng)

    # V1: chain rule for m0.  (The empty profile is not accepted by the
    # families machinery; the first prediction is uniform by symmetry.)
    counts = np.zeros(d, dtype=np.int64)
    seq_bits = math.log2(d)
    counts[x[0]] += 1
    for t in range(1, n):
        pred = lsa_predictive_by_count(counts, d=d, l_max=l_max)
        q = pred.q_hat(counts)
        seq_bits -= math.log2(q[x[t]])
        counts[x[t]] += 1
    batch_bits = bits_of_counts(counts, d, l_max)
    assert abs(seq_bits - batch_bits) < 1e-6, (seq_bits, batch_bits)
    print(f"V1 chain rule: sequential {seq_bits:.9f} = batch "
          f"{batch_bits:.9f} bits  OK")

    # V2: one-state collapse.
    traj = trajectory(x, d, [n], state_of=np.zeros(d, dtype=np.int64),
                      verbose=False)
    l0_tail = bits_of_counts(np.bincount(x[1:], minlength=d), d, l_max)
    l1 = traj[n][1]
    assert abs(l1 - (math.log2(d) + l0_tail)) < 1e-9, (l1, l0_tail)
    print(f"V2 one-state collapse: l1 {l1:.9f} = log2(d) + m0(x_2..x_n) "
          f"{math.log2(d) + l0_tail:.9f} bits  OK")

    # V3: envelope sandwich.
    l0, l1 = trajectory(x, d, [n], verbose=False)[n]
    mix = -(np.logaddexp(-l0 * LOG2, -l1 * LOG2) / LOG2 - math.log2(2))
    assert min(l0, l1) <= mix <= min(l0, l1) + 1.0 + 1e-12, (l0, l1, mix)
    print(f"V3 sandwich: min {min(l0,l1):.3f} <= mixture {mix:.3f} <= "
          f"min + 1  OK")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--source", choices=["iid", "table", "tilted"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--l-rows", type=int, default=13)
    ap.add_argument("--n-checkpoints", type=int, default=20)
    ap.add_argument("--r-max-state", type=int, default=90)
    ap.add_argument("--out", type=Path, default=Path("out_mech"))
    args = ap.parse_args()

    if args.validate:
        validate()
        return
    if args.source is None:
        raise SystemExit("--source required unless --validate")

    d, n = args.d, args.n
    l_max = default_l_max(d)
    rng = np.random.default_rng([args.seed, hash(args.source) % 2**32])
    if args.source == "iid":
        x = draw_iid(n, d, args.alpha, rng)
    elif args.source == "table":
        x, _ = draw_table_chain(n, d, args.l_rows, rng)
    else:
        x, _ = draw_table_chain(n, d, args.l_rows, rng,
                                tilt=zipf(d, args.alpha))

    print(f"{args.source} seed={args.seed}: d={d} n={n} l_max={l_max}",
          flush=True)
    print("building shared moment tables for per-state evaluations...",
          flush=True)
    t0 = time.time()
    state_tables = build_tables_fast(
        max_L=l_max,
        r_values=tuple(range(args.r_max_state + 3)),
        cache_dir="/root/work/tables_fast",
    )
    print(f"tables ready in {time.time()-t0:.1f}s", flush=True)

    cps = sorted(set(np.unique(np.round(np.geomspace(8, n, args.n_checkpoints))
                               .astype(int)).tolist()))
    traj = trajectory(x, d, cps, state_tables=state_tables, l_max=l_max)

    args.out.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": args.source, "seed": args.seed, "d": d, "n": n,
        "alpha": args.alpha, "l_rows": args.l_rows, "l_max": l_max,
        "checkpoints": list(traj.keys()),
        "l0": [v[0] for v in traj.values()],
        "l1": [v[1] for v in traj.values()],
    }
    tag = "" if args.source == "iid" else f"_L{args.l_rows}"
    path = args.out / f"{args.source}{tag}_seed{args.seed}.json"
    path.write_text(json.dumps(payload, indent=1))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
