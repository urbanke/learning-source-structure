"""Reduce the raw run files to small committed results (results/building_block/)."""
import json, os, shutil, glob
import numpy as np

OUT = "repo_add/results/building_block"
os.makedirs(OUT, exist_ok=True)

# 1. single-distribution validation (truths with exact centring + formula)
for i in (3, 4, 5):
    if os.path.exists(f"single_truth{i}.json"):
        shutil.copy(f"single_truth{i}.json", os.path.join(OUT, f"single_truth_run{i}.json"))
if os.path.exists("single_truth_a125.json"):
    shutil.copy("single_truth_a125.json", os.path.join(OUT, "single_truth_a125.json"))
# 2. building-block grids
for f in sorted(glob.glob("bb_d*_a*_L*.json")):
    shutil.copy(f, os.path.join(OUT, f))
# 3. archived-run recomputation checks (the codelengths of the archived paths at d <= 256 recomputed with this implementation)
checks = {}
for f in sorted(glob.glob("archive_check_*.json")):
    checks[f[len("archive_check_"):-5]] = json.load(open(f))["summary"]
if checks:
    json.dump(checks, open(os.path.join(OUT, "archive_recomputation_check.json"), "w"), indent=1)
for f in ["exact_EH_archive.json"]:
    if os.path.exists(f):
        shutil.copy(f, os.path.join(OUT, f))
# summaries: the main one uses the exact small-k block (suffix _B); the formula-only and row-known variants are kept for the record
for src, dst in [("archive_summary_B.json", "archive_summary.json"), ("archive_stats_B.json", "archive_stats.json"),
                 ("archive_summary.json", "archive_summary_formula_only.json"), ("archive_summary_BR.json", "archive_summary_rowknown.json")]:
    if os.path.exists(src):
        shutil.copy(src, os.path.join(OUT, dst))
# exact small-k blocks, merged grids, row-known corrections
for f in sorted(glob.glob("bbx_d*_a*_L*.json")) + sorted(glob.glob("bb2_d*_a*_L*.json")):
    shutil.copy(f, os.path.join(OUT, f))
for f in sorted(glob.glob("tab_RK_*.json")):
    r = json.load(open(f)); rk = __import__("numpy").array(r["rkC"])
    small = {k: r[k] for k in ["d", "alpha", "Ls", "checkpoints", "NT", "seed", "theta0", "lengths_computed"]}
    small["mean_correction"] = rk.mean(axis=0).round(2).tolist(); small["se"] = (rk.std(axis=0, ddof=1) / rk.shape[0] ** 0.5).round(2).tolist()
    small["heavy_contexts_mean"] = float(__import__("numpy").mean(r["nrk"]))
    json.dump(small, open(os.path.join(OUT, f.replace("tab_RK_", "rowknown_")), "w"), indent=1)
# 3c. all thirty settings with the final predictions (tab_M: exact small-k block; extension settings with the added
#     sampled matrices): per-path codelengths and control variates, per-table predictions -> chains_<tag>.json
from compare_all import TAGS as ALL_TAGS
for tag in ALL_TAGS:
    simfile = f"sim_AX_{tag}.json" if os.path.exists(f"sim_AX_{tag}.json") else f"sim_A_{tag}.json"
    if not (os.path.exists(simfile) and os.path.exists(f"tab_M_{tag}.json")):
        continue
    s = json.load(open(simfile)); t = json.load(open(f"tab_M_{tag}.json"))
    keep_s = {k: s[k] for k in ["d", "alpha", "Ls", "Lmax", "R", "R_archive", "R_extra", "seed", "seeds", "source", "checkpoints",
                                "lF", "lC", "cvF", "cvC", "EcvF", "EcvC", "Hpi", "archive"] if k in s}
    keep_t = {k: t[k] for k in ["d", "alpha", "Ls", "Lmax", "checkpoints", "NT_pred", "NT_occ", "Hbar", "sources", "predF_U", "predF_eta", "predC", "covC",
                                "EcvF", "EcvC", "EcvF_occ", "EcvC_occ"] if k in t}
    def rnd(x):
        if isinstance(x, float): return round(x, 3)
        if isinstance(x, list): return [rnd(v) for v in x]
        if isinstance(x, dict): return {k: rnd(v) for k, v in x.items()}
        return x
    json.dump(rnd(dict(simulation=keep_s, tables=keep_t)), open(os.path.join(OUT, f"chains_{tag}.json"), "w"), separators=(",", ":"))
for f in ["summary_all.json", "stats_all.json"]:
    if os.path.exists(f):
        shutil.copy(f, os.path.join(OUT, f))
# 3d. the sampling-free predictions (sf_predict.py) and their comparison (sf_compare.py, make_sf_table.py)
for f in sorted(glob.glob("sfp_L*_a*_d*.json")) + ["sf_compare.json"]:
    if os.path.exists(f):
        shutil.copy(f, os.path.join(OUT, f))
os.makedirs("repo_add/figures", exist_ok=True)
for f in ["crossing_figure.pdf", "gap_figure_depth23.pdf", "gap_figure_deep.pdf"]:
    if os.path.exists(f):
        shutil.copy(f, os.path.join("repo_add/figures", f))
# 4. tables of the paper
for f in ["tables_out/single_table.tex", "tables_out/crossing_table_all.tex", "tables_out/regret_table_all.tex"]:
    if os.path.exists(f):
        shutil.copy(f, os.path.join(OUT, os.path.basename(f)))
print("exported:", sorted(os.listdir(OUT)))
os.system(f"du -sh {OUT}")
