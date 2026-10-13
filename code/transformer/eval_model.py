"""Evaluate a trained transformer on the evaluation sets: the per-position loss of the true next token for all
sequences, and the full predictive distribution for the first n_full sequences at every full_every-th position.

    python3 eval_model.py --run runs/d64 --ckpt latest --settings grid --out eval/d64
Writes one file <tag>_step<step>.npz per setting (tokens as int16, logp_true as float32 in natural log,
logp_full as float16, positions, and the md5 of the tokens).
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import GPT
from eval_sets import eval_set, md5_of, parse_settings, tag_of
from families import family_eval_set, tag_of_spec
from train import pick_device


@torch.no_grad()
def run_model(model, x, device, n_full, positions, chunk=32):
    N, T1 = x.shape
    logp_true = np.empty((N, T1 - 1), dtype=np.float32)
    logp_full = np.empty((n_full, len(positions), model.cfg["vocab"]), dtype=np.float16)
    for i in range(0, N, chunk):
        xb = torch.from_numpy(x[i:i + chunk].astype(np.int64)).to(device)
        lp = F.log_softmax(model(xb[:, :-1]).float(), dim=-1)                 # (B, T, d)
        logp_true[i:i + chunk] = torch.gather(lp, 2, xb[:, 1:, None])[:, :, 0].cpu().numpy()
        if i < n_full:
            j = min(chunk, n_full - i)
            logp_full[i:i + j] = lp[:j][:, positions, :].cpu().numpy().astype(np.float16)
    return logp_true, logp_full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default="latest", help="'latest' or a step number")
    ap.add_argument("--settings", default="grid", help="alpha,L_s settings of the paper's source ('none' to skip)")
    ap.add_argument("--family_sets", default="", help="family specs (families.py) to evaluate on, separated by ';'")
    ap.add_argument("--n_eval", type=int, default=512)
    ap.add_argument("--n_full", type=int, default=64)
    ap.add_argument("--full_every", type=int, default=4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = json.load(open(os.path.join(args.run, "config.json")))
    device = pick_device(args.device)
    if args.ckpt == "latest":
        ckfile = os.path.join(args.run, "latest.pt")
    else:                                    # the full checkpoint if present, else the weights-only copy
        ckfile = os.path.join(args.run, f"ckpt_{args.ckpt}.pt")
        if not os.path.exists(ckfile):
            ckfile = os.path.join(args.run, f"model_{args.ckpt}.pt")
    ck = torch.load(ckfile, map_location=device)
    m = ck["cfg"]["model"]
    model = GPT(m["vocab"], m["n_ctx"], m["n_layer"], m["n_head"], m["d_model"], mlp=m["mlp"]).to(device)
    model.load_state_dict(ck["model"]); model.eval()
    step = ck["step"]; d = m["vocab"]; n_ctx = m["n_ctx"]; permute = not cfg.get("no_permute", False)
    os.makedirs(args.out, exist_ok=True)
    positions = np.arange(0, n_ctx, args.full_every)
    print(f"{ckfile}: step {step}, d={d}, n_ctx={n_ctx}, device {device}", flush=True)
    for alpha, ls in (parse_settings(args.settings) if args.settings != "none" else []):
        t0 = time.time()
        x, _ = eval_set(d, alpha, ls, n_ctx, args.n_eval, permute=permute)
        logp_true, logp_full = run_model(model, x, device, args.n_full, positions)
        tag = tag_of(d, alpha, ls)
        np.savez_compressed(os.path.join(args.out, f"{tag}_step{step}.npz"), tokens=x, logp_true=logp_true, logp_full=logp_full,
                            positions=positions, alpha=alpha, ls=ls, d=d, step=step, md5=md5_of(x), run=args.run)
        bits = -logp_true / np.log(2.0)
        print(f"  {tag}: mean loss {bits.mean():.3f} bits; positions 1-32: {bits[:, :32].mean():.3f}, last 128: {bits[:, -128:].mean():.3f}  ({time.time()-t0:.1f} s)", flush=True)
    for spec in [s for s in args.family_sets.split(";") if s]:
        t0 = time.time()
        x, meta, fam = family_eval_set(spec, n_ctx, args.n_eval, d=d)
        assert fam.d == d, f"family {spec} has d={fam.d}, the model has d={d}"
        logp_true, logp_full = run_model(model, x, device, args.n_full, positions)
        tag = tag_of_spec(spec)
        np.savez_compressed(os.path.join(args.out, f"{tag}_step{step}.npz"), tokens=x, logp_true=logp_true, logp_full=logp_full,
                            positions=positions, spec=spec, d=d, step=step, md5=md5_of(x), run=args.run, h_true=meta["h_true"])
        bits = -logp_true / np.log(2.0); reg = bits - meta["h_true"]
        print(f"  {tag}: mean loss {bits.mean():.4f} bits; regret over the window {reg.sum(1).mean():.2f} bits per sequence  ({time.time()-t0:.1f} s)", flush=True)


if __name__ == "__main__":
    main()
