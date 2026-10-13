"""Write weights-only copies of checkpoints, small enough to travel through git (about 2 MB for the d = 64 model):
    python3 export_weights.py results/transformer/runs/d64 5000 20000      (steps; 'all' for every checkpoint)
Writes model_<step>.pt next to ckpt_<step>.pt (same keys, without the optimizer state)."""
import glob, os, re, sys
import torch

run = sys.argv[1]; steps = sys.argv[2:]
if steps == ["all"]:
    steps = sorted(int(re.search(r"ckpt_(\d+)\.pt$", f).group(1)) for f in glob.glob(os.path.join(run, "ckpt_*.pt")))
for s in steps:
    src = os.path.join(run, f"ckpt_{int(s)}.pt"); dst = os.path.join(run, f"model_{int(s)}.pt")
    ck = torch.load(src, map_location="cpu")
    torch.save(dict(model=ck["model"], step=ck["step"], cfg=ck["cfg"]), dst)
    print(dst, f"{os.path.getsize(dst)/1e6:.1f} MB")
