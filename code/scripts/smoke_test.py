#!/usr/bin/env python3
"""End-to-end check of the pipeline on a laptop, in a few minutes.

Trains a tiny transformer on coins with the two-point prior (the source of Figure 1a), evaluates it on
a fixed test set, computes the exact informed-Bayes reference on the same sequences, and compares the
two. These are the three programs the paper's experiments ran: train.py, eval_model.py and
refs_family.py. The model is tiny and briefly trained, so the numbers are not paper numbers; the gap
to informed Bayes should shrink with more steps.

    python3 code/scripts/smoke_test.py                 # about two minutes on a laptop CPU
    python3 code/scripts/smoke_test.py --steps 2000    # closer to Bayes

Writes output/smoke/summary.json and output/smoke/regret.png.
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TF = ROOT / "code/transformer"
SPEC = "binary:twopoint"
LN2 = float(np.log(2.0))


def run(cmd, log):
    print("$", " ".join(cmd), flush=True)
    with open(log, "ab") as handle:
        handle.write(("\n$ " + " ".join(cmd) + "\n").encode())
        code = subprocess.call(cmd, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
    if code:
        sys.exit(f"step failed with exit code {code}; see {log}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=300, help="training steps (batches of 32 sequences)")
    ap.add_argument("--n_ctx", type=int, default=64, help="tokens per sequence")
    ap.add_argument("--n_eval", type=int, default=256, help="test sequences")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, default=ROOT / "output/smoke")
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    log = out / "log.txt"
    py = sys.executable
    t0 = time.time()

    run([py, str(TF / "train.py"), "--d", "2", "--family", SPEC, "--n_ctx", str(args.n_ctx),
         "--n_layer", "1", "--n_head", "2", "--d_model", "32", "--batch", "32",
         "--steps", str(args.steps), "--warmup", str(max(1, args.steps // 10)),
         "--val_every", str(args.steps), "--ckpt_every", str(args.steps), "--val_n", "16",
         "--workers", "0", "--device", args.device, "--out", str(out / "run")], log)
    run([py, str(TF / "eval_model.py"), "--run", str(out / "run"), "--ckpt", "latest", "--settings", "none",
         "--family_sets", SPEC, "--n_eval", str(args.n_eval), "--n_full", "0", "--device", args.device,
         "--out", str(out / "eval")], log)
    run([py, str(TF / "refs_family.py"), "--set", SPEC, "--n_ctx", str(args.n_ctx),
         "--n_eval", str(args.n_eval), "--out", str(out / "refs")], log)

    sys.path.insert(0, str(TF))
    from families import tag_of_spec
    tag = tag_of_spec(SPEC)
    eval_file = sorted(glob.glob(str(out / "eval" / f"{tag}_step*.npz")))[-1]
    model = np.load(eval_file)
    refs = np.load(out / "refs" / f"{tag}.npz")
    if str(model["md5"]) != str(refs["md5"]):
        sys.exit("the model and the reference were evaluated on different sequences")
    h_true = refs["h_true"]
    tf_regret = (-model["logp_true"] / LN2 - h_true).mean(0)        # position t predicts token t+1
    bayes_regret = (refs["twopoint"] - h_true).mean(0)
    times = [t for t in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024) if t <= args.n_ctx]
    rows = [(t, float(bayes_regret[t - 1]), float(tf_regret[t - 1])) for t in times]
    print(f"\n{SPEC}: expected next-token regret in bits, {args.n_eval} test sequences, md5 {str(refs['md5'])[:8]}")
    print(f"{'t':>6} {'informed Bayes':>15} {'transformer':>12} {'gap':>8}")
    for t, b, m in rows:
        print(f"{t:>6} {b:>15.4f} {m:>12.4f} {m - b:>8.4f}")
    finite = bool(np.isfinite(tf_regret).all() and np.isfinite(bayes_regret).all())
    summary = dict(spec=SPEC, steps=args.steps, n_ctx=args.n_ctx, n_eval=args.n_eval, md5=str(refs["md5"]),
                   times=times, bayes_regret=[r[1] for r in rows], transformer_regret=[r[2] for r in rows],
                   mean_gap_bits=float((tf_regret - bayes_regret).mean()), all_finite=finite,
                   seconds=round(time.time() - t0, 1))
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4.5, 3))
        t = np.arange(1, args.n_ctx + 1)
        ax.plot(t, tf_regret, label=f"transformer, {args.steps} steps")
        ax.plot(t, bayes_regret, "o", ms=3, mfc="none", label="informed Bayes")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("observed tokens, t")
        ax.set_ylabel("regret (bits)")
        ax.set_title("coins, two-point prior")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out / "regret.png", dpi=150)
    except Exception as error:  # the plot is a convenience, not part of the check
        print("no plot:", error)
    print(f"\nmean gap over all positions: {summary['mean_gap_bits']:.4f} bits; {summary['seconds']} s;",
          "all values finite" if finite else "NON-FINITE VALUES")
    print(f"wrote {out / 'summary.json'} and {out / 'regret.png'}")
    if not finite:
        sys.exit(1)


if __name__ == "__main__":
    main()
