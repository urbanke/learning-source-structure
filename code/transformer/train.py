"""Train a small transformer on the source of Section 2.1 (one fresh transition matrix per sequence).

Example (the main setup: alpha and L_s drawn per sequence, alphabet relabelled per sequence):
    python3 train.py --d 64 --out runs/d64 --steps 100000
Controls:  --alpha 1 1 --ls 2 2 (fixed source),  --no_permute (base law learnable in the weights),
           --n_layer 1,  --no_mlp (attention only).
Writes to the run directory: config.json, log.jsonl (training loss, and the validation losses by position window),
ckpt_<step>.pt at every checkpoint and latest.pt; restarts from latest.pt when --resume is given.
"""
from __future__ import annotations
import argparse, json, math, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from source import Source, seed_of
from families import make_family
from model import GPT

WINDOWS = [(1, 32), (33, 128), (129, 512), (513, 1024), (1025, 2048)]


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Stream(torch.utils.data.IterableDataset):
    """An endless stream of batches from the source; every worker gets its own seed."""
    def __init__(self, src: Source, batch: int, seed: int):
        super().__init__()
        self.src = src; self.batch = batch; self.seed = seed

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        wid = info.id if info is not None else 0
        rng = np.random.default_rng(seed_of(self.seed, "stream", wid))
        while True:
            x, _ = self.src.batch(self.batch, rng)
            yield torch.from_numpy(x)


def loss_by_window(logp_true, n_ctx):
    """Mean loss (bits) of the true next token in position windows; logp_true: (B, T) natural-log probabilities."""
    bits = -logp_true / math.log(2.0)
    out = {}
    for lo, hi in WINDOWS:
        if lo <= n_ctx:
            out[f"{lo}-{min(hi, n_ctx)}"] = float(bits[:, lo - 1:min(hi, n_ctx)].mean())
    return out


@torch.no_grad()
def evaluate(model, x, device, chunk=32):
    """Per-position log-probabilities (natural log) of the true next tokens for a tensor of sequences (N, n_ctx+1)."""
    model.eval()
    outs = []
    for i in range(0, x.shape[0], chunk):
        xb = x[i:i + chunk].to(device)
        logits = model(xb[:, :-1]).float()
        lp = F.log_softmax(logits, dim=-1)
        outs.append(torch.gather(lp, 2, xb[:, 1:, None])[:, :, 0].cpu())
    model.train()
    return torch.cat(outs, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, required=True)
    ap.add_argument("--n_ctx", type=int, default=1024)
    ap.add_argument("--alpha", type=float, nargs=2, default=(0.5, 1.5), help="range of alpha, drawn per sequence")
    ap.add_argument("--ls", type=int, nargs=2, default=(0, 8), help="range of L_s, drawn per sequence")
    ap.add_argument("--no_permute", action="store_true", help="do not relabel the alphabet per sequence")
    ap.add_argument("--family", default="lsa", help="source family spec (families.py); 'lsa' is the source of the paper")
    ap.add_argument("--n_layer", type=int, default=2)
    ap.add_argument("--n_head", type=int, default=4)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--no_mlp", action="store_true")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--steps", type=int, default=100000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--val_every", type=int, default=1000)
    ap.add_argument("--ckpt_every", type=int, default=5000)
    ap.add_argument("--val_settings", default="1,2;1,4;0.75,2;1.25,3", help="alpha,L_s pairs for the validation sets")
    ap.add_argument("--val_n", type=int, default=64, help="sequences per validation set")
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = pick_device(args.device)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    alpha = tuple(args.alpha) if args.alpha[0] != args.alpha[1] else float(args.alpha[0])
    ls = tuple(args.ls) if args.ls[0] != args.ls[1] else int(args.ls[0])
    if args.family == "lsa":
        src = Source(args.d, args.n_ctx + 1, alpha=alpha, ls=ls, permute=not args.no_permute)
    else:
        src = make_family(args.family, args.n_ctx + 1, d=args.d, alpha=alpha, ls=ls, permute=not args.no_permute)
        args.d = src.d
    model = GPT(args.d, args.n_ctx, args.n_layer, args.n_head, args.d_model, mlp=not args.no_mlp).to(device)
    cfg = dict(vars(args), device=str(device), n_params=model.n_params(), model=model.cfg)
    json.dump(cfg, open(os.path.join(args.out, "config.json"), "w"), indent=1)
    print(f"device {device}; {model.n_params()} parameters; family {args.family}; d={args.d} alpha={alpha} L_s={ls} permute={not args.no_permute}", flush=True)

    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([dict(params=decay, weight_decay=args.wd), dict(params=no_decay, weight_decay=0.0)],
                            lr=args.lr, betas=(0.9, 0.95))

    def lr_at(step):
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        t = (step - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, t))))

    # validation sets: fixed sequences from a few fixed settings (relabelled like the training data)
    vals = {}
    if args.family == "lsa":
        for pair in args.val_settings.split(";"):
            a, L = pair.split(","); a = float(a); L = int(L)
            vs = Source(args.d, args.n_ctx + 1, alpha=a, ls=L, permute=not args.no_permute)
            x, _ = vs.batch(args.val_n, np.random.default_rng(seed_of("val", args.d, a, L)))
            vals[f"a{a:g}_L{L}"] = torch.from_numpy(x)
    else:                                     # the family's own draws, fixed seed
        x, _ = src.batch(args.val_n, np.random.default_rng(seed_of("val", args.family, args.n_ctx)))
        vals["family"] = torch.from_numpy(x)

    step = 0
    latest = os.path.join(args.out, "latest.pt")
    if args.resume and os.path.exists(latest):
        ck = torch.load(latest, map_location=device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); step = ck["step"]
        print(f"resumed from step {step}", flush=True)

    loader = torch.utils.data.DataLoader(Stream(src, args.batch, args.seed + 1000 * step), batch_size=None,
                                         num_workers=args.workers, persistent_workers=args.workers > 0)
    log = open(os.path.join(args.out, "log.jsonl"), "a")
    t0 = time.time(); tokens = 0; run_loss = 0.0; run_n = 0
    model.train()
    for xb in loader:
        if step >= args.steps:
            break
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        xb = xb.to(device)
        logits = model(xb[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, args.d), xb[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        opt.step()
        step += 1; tokens += xb.shape[0] * args.n_ctx
        run_loss += loss.item(); run_n += 1
        if step % args.log_every == 0:
            rec = dict(step=step, loss_bits=run_loss / run_n / math.log(2.0), lr=lr_at(step), tok_per_s=tokens / (time.time() - t0), elapsed=time.time() - t0)
            log.write(json.dumps(rec) + "\n"); log.flush()
            print(f"step {step}: train loss {rec['loss_bits']:.4f} bits/token, {rec['tok_per_s']:.0f} tok/s", flush=True)
            run_loss = 0.0; run_n = 0
        if step % args.val_every == 0 or step == args.steps:
            rec = dict(step=step, val={})
            for name, x in vals.items():
                lp = evaluate(model, x, device)
                rec["val"][name] = loss_by_window(lp.numpy(), args.n_ctx)
            log.write(json.dumps(rec) + "\n"); log.flush()
            print("  val " + "; ".join(f"{k}: " + " ".join(f"{w}={v:.3f}" for w, v in r.items()) for k, r in rec["val"].items()), flush=True)
        if step % args.ckpt_every == 0 or step == args.steps:
            ck = dict(model=model.state_dict(), opt=opt.state_dict(), step=step, cfg=cfg)
            torch.save(ck, os.path.join(args.out, f"ckpt_{step}.pt")); torch.save(ck, latest)
            torch.save(dict(model=ck["model"], step=step, cfg=cfg), os.path.join(args.out, f"model_{step}.pt"))   # weights only, for git
    print(f"done: {step} steps in {time.time()-t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
