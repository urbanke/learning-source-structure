"""Work through a queue of reference / evaluation jobs, two at a time, within a time budget; jobs whose output
exists are skipped, so the script can be called repeatedly. Usage: python3 queue.py <budget seconds>"""
import os, subprocess, sys, time

BUDGET = float(sys.argv[1]) if len(sys.argv) > 1 else 540.0
GRID = [(1.0, 2), (1.0, 3), (1.0, 4), (0.75, 2), (0.75, 3), (0.75, 4), (1.25, 2), (1.25, 3), (1.25, 4)]
RUNS = {32: ("d32", 100000, 64, 64), 64: ("d64_seed1", 100000, 48, 48), 128: ("d128", 100000, 24, 24)}   # run, step, n_seq, n_full
jobs = []
for d, (run, step, n_seq, n_full) in RUNS.items():
    for a, L in GRID:
        tag = f"d{d}_a{a:g}_L{L}"
        jobs.append((f"tf_refs/d{d}/{tag}.npz", ["python3", "tf/make_refs.py", "--d", str(d), "--settings", f"{a},{L}", "--table_dir", ".", "--out", f"tf_refs/d{d}"], 600 if d < 128 else 900))
        jobs.append((f"tf_eval/{run}/{tag}_step{step}.npz", ["python3", "tf/eval_model.py", "--run", f"tf_runs/{run}", "--ckpt", str(step), "--settings", f"{a},{L}", "--device", "cpu", "--out", f"tf_eval/{run}"], 120))
        jobs.append((f"tf_refs2/d{d}/extra_{tag}.npz", ["python3", "tf/refs_extra.py", "--d", str(d), "--settings", f"{a},{L}", "--n_seq", str(n_seq), "--n_full", str(n_full), "--table_dir", ".", "--out", f"tf_refs2/d{d}"], 480 if d == 32 else (900 if d == 64 else 1500)))

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
pending = [j for j in jobs if not os.path.exists(j[0]) and not alive(j[0])]
old_alive = [j[0] for j in jobs if not os.path.exists(j[0]) and alive(j[0])]     # started by an earlier call, still running
print(f"{len(pending)} jobs pending", flush=True)
while pending or running:
    running = [(p, out, est) for p, out, est in running if p.poll() is None]
    old_alive = [o for o in old_alive if alive(o)]
    while pending and len(running) + len(old_alive) < 2 and time.time() - t0 < BUDGET:
        out, cmd, est = pending.pop(0)
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
