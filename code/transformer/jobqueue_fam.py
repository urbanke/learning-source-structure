"""Work through the queue of reference jobs for the source families (refs_family.py, make_refs.py), two at a time, within a time budget; jobs whose output
exists are skipped, so the script can be called repeatedly. Usage: python3 jobqueue_fam.py <budget seconds>"""
import os, subprocess, sys, time

BUDGET = float(sys.argv[1]) if len(sys.argv) > 1 else 540.0
GRID = [(1.0, 2), (1.0, 3), (1.0, 4), (0.75, 2), (0.75, 3), (0.75, 4), (1.25, 2), (1.25, 3), (1.25, 4)]
RUNS = {32: ("d32", 100000, 64, 64), 64: ("d64_seed1", 100000, 48, 48), 128: ("d128", 100000, 24, 24)}   # run, step, n_seq, n_full
jobs = []
# F1: binary sets (closed forms, seconds)
for spec in ("binary:twopoint", "binary:uniform", "binary:eightpoint", "binary:fixed=0.5", "binary:fixed=0.2"):
    tag = spec.replace(":", "_").replace("=", "").replace(",", "-")
    jobs.append((f"refs_fam/{tag}.npz", ["python3", "tf/refs_family.py", "--set", spec, "--n_ctx", "256", "--n_eval", "4096", "--out", "refs_fam"], 30))
# F2: exponential family sets (grid posteriors, plug-ins and the LSA unigram; about an hour each)
for spec in ("expfam:k=1", "expfam:k=2", "expfam:k=1:perm", "expfam:k=1:theta=1", "expfam:k=2:theta=1,0.1"):
    tag = spec.replace(":", "_").replace("=", "").replace(",", "-")
    jobs.append((f"refs_fam/{tag}.npz", ["python3", "tf/refs_family.py", "--set", spec, "--d", "32", "--n_ctx", "1024", "--n_eval", "2048", "--table_dir", ".", "--out", "refs_fam"], 3600))
# F3: product sets
for spec, n in (("product:8x8:L=2,2", 2048), ("product:8x8:L=4,4", 2048), ("product:8x8:L=2,6", 2048), ("product:16x16:L=2,2", 1024), ("product:16x16:L=4,4", 1024)):
    tag = spec.replace(":", "_").replace("=", "").replace(",", "-")
    jobs.append((f"refs_fam/{tag}.npz", ["python3", "tf/refs_family.py", "--set", spec, "--n_ctx", "1024", "--n_eval", str(n), "--table_dir", ".", "--out", "refs_fam"], 3600 if "8x8" in spec else 6000))
# F4: the paper's source at alpha = 0 (and the alpha = 1 sets with 2048 sequences, for the cross-training comparisons)
for d in (64, 128):
    for a, L in ((0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4)):
        tag = f"d{d}_a{a}_L{L}"
        jobs.append((f"tf_refs_2048/d{d}/{tag}.npz", ["python3", "tf/make_refs.py", "--d", str(d), "--settings", f"{a},{L}", "--n_eval", "2048", "--n_full", "32", "--table_dir", ".", "--out", f"tf_refs_2048/d{d}"], 2400 if d == 64 else 4200))
# the paper's prediction of n*_d at alpha = 0 (building block, then the sampling-free crossing), for F4
LMAX = {64: 20, 128: 23}
for d in (64, 128):
    for L in (2, 3, 4):
        bbf = f"bb_d{d}_a0.0_L{L}.json"
        jobs.append((bbf, ["python3", "bb.py", str(d), "0.0", str(L), str(LMAX[d]), "300"], 1800))
        jobs.append((f"sf_pred_d{d}_a0_L{L}.json", ["python3", "sf_predict.py", str(d), "0.0", str(L), bbf, "16,32,64,128,256,512,1024,2048,4096", f"sf_pred_d{d}_a0_L{L}.json"], 900, bbf))


def alive(out):
    """A job started by an earlier call is still running (its pid file exists and the pid is alive)."""
    try:
        pid = int(open(out + ".pid").read())
        os.kill(pid, 0); return True
    except Exception:
        return False


# the full ladder (joint pairs, true depth, backoff, product model, oracle) at two settings per alphabet size
for d, n in ((32, 36), (64, 32), (128, 20)):
    for a, L in ((1.0, 2), (1.0, 4)):
        tag = f"d{d}_a{a:g}_L{L}"
        jobs.append((f"tf_refs3/d{d}/extra_{tag}.npz", ["python3", "tf/refs_extra.py", "--d", str(d), "--settings", f"{a},{L}", "--n_seq", str(n), "--n_full", "1", "--which", "JDBZO", "--table_dir", ".", "--out", f"tf_refs3/d{d}"], 1200))

t0 = time.time(); running = []
jobs = [j if len(j) == 4 else (j[0], j[1], j[2], None) for j in jobs]
pending = [j for j in jobs if not os.path.exists(j[0]) and not alive(j[0])]
old_alive = [j[0] for j in jobs if not os.path.exists(j[0]) and alive(j[0])]     # started by an earlier call, still running
print(f"{len(pending)} jobs pending", flush=True)
while pending or running:
    running = [(p, out, est) for p, out, est in running if p.poll() is None]
    old_alive = [o for o in old_alive if alive(o)]
    while pending and len(running) + len(old_alive) < 2 and time.time() - t0 < BUDGET:
        ready = [j for j in pending if j[3] is None or os.path.exists(j[3])]
        if not ready:
            break
        out, cmd, est, _ = ready[0]; pending.remove(ready[0])
        log = open(out + ".log", "w")
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        open(out + ".pid", "w").write(str(proc.pid))
        running.append((proc, out, est))
        print(f"started {out} (about {est} s)", flush=True)
    if not running and not old_alive:
        break
    if time.time() - t0 > BUDGET:
        print("budget used; leaving the running jobs to finish on their own", flush=True); break
    time.sleep(5)
done = [j[0] for j in jobs if os.path.exists(j[0])]
print(f"{len(done)} of {len(jobs)} outputs exist; {time.time()-t0:.0f} s", flush=True)
