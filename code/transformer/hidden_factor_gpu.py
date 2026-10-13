"""Chunked float64 exact hidden-factor Bayes and an explicit GPU sizing pilot.

No approximation or pruning of encodings is used. Benchmark results select
an alphabet only after full-sequence wall time has actually been measured.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

TIMES = (1, 4, 16, 64, 256, 1024)


def dimensions_arg(value):
    try:
        result = tuple(map(int, value.lower().split('x')))
    except (ValueError, AttributeError) as error:
        raise argparse.ArgumentTypeError('dimensions must be RxS, e.g. 2x6') from error
    if len(result) != 2 or min(result) < 2:
        raise argparse.ArgumentTypeError('both coordinate dimensions must be at least 2')
    return result


def encoding_count(dimensions):
    from hidden_factor import encoding_count as count
    return count(dimensions)


def encoding_chunks(dimensions, chunk_size=65536):
    from hidden_factor import iter_encodings
    yield from iter_encodings(dimensions, chunk_size)


def memory_estimate(dimensions, chunk_size):
    r, s = dimensions; d = r*s
    h = min(chunk_size, encoding_count(dimensions))
    # Conservative working peak: counts/totals, int64 maps/index vectors,
    # float64 log-likelihood, and multiple Hxd predictive intermediates.
    return h * (8*d*d + 32*(r*r+s*s+r+s) + 8*(3*d+12) + 8*(8*d+16))


def synchronize(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


def exact_streaming(tokens, dimensions=(2, 3), times=None, *, device='cpu',
                   chunk_size=65536, max_seconds=None):
    """Return exact probabilities at selected t and evidence at every prefix.

    ``predictive[k]`` predicts token t+1 after t observed tokens, where
    t=times[k]. ``log_evidence_bits[t]`` is log2 P(x[:t]), including the
    initial uniform token. All encodings are integrated with equal weights.
    """
    r, s = tuple(dimensions); d = r*s
    tokens = np.asarray(tokens)
    if (tokens.ndim != 1 or tokens.dtype.kind not in 'iu' or len(tokens) < 2
            or np.any(tokens < 0) or np.any(tokens >= d)):
        raise ValueError('tokens must be a nonempty valid integer sequence')
    times = np.asarray(tuple(range(1, len(tokens))) if times is None else times, dtype=np.int64)
    if (times.ndim != 1 or not len(times) or np.any(np.diff(times) <= 0)
            or times[0] < 1 or times[-1] >= len(tokens)):
        raise ValueError('times must be increasing observed-token counts before the final token')
    if min(r, s) < 2 or chunk_size < 1:
        raise ValueError('invalid dimensions or chunk size')
    device = torch.device(device)
    total = encoding_count((r, s))
    if device.type == 'cuda':
        free, _ = torch.cuda.mem_get_info(device)
        if memory_estimate((r, s), chunk_size) > .8*free:
            raise MemoryError('chunk memory estimate exceeds 80% of currently free GPU memory')
    start = time.perf_counter()
    cumulative = torch.full((len(tokens)+1,), -torch.inf, dtype=torch.float64, device=device)
    numerator = torch.full((len(times), d), -torch.inf, dtype=torch.float64, device=device)
    selected = {int(t): i for i, t in enumerate(times)}
    used = 0
    for grids in encoding_chunks((r, s), chunk_size):
        h = len(grids)
        inverse = np.argsort(grids, axis=1)
        u = torch.as_tensor(inverse // s, dtype=torch.long, device=device)
        v = torch.as_tensor(inverse % s, dtype=torch.long, device=device)
        idx = torch.arange(h, device=device)
        cu = torch.zeros((h, r, r), dtype=torch.int32, device=device)
        cv = torch.zeros((h, s, s), dtype=torch.int32, device=device)
        nu = torch.zeros((h, r), dtype=torch.int32, device=device)
        nv = torch.zeros((h, s), dtype=torch.int32, device=device)
        logw = torch.full((h,), -math.log(d), dtype=torch.float64, device=device)
        evidence = torch.empty(len(tokens)+1, dtype=torch.float64, device=device)
        evidence[0] = math.log(h)
        evidence[1] = math.log(h/d)
        for t in range(1, len(tokens)):
            a, b = u[:, int(tokens[t-1])], v[:, int(tokens[t-1])]
            denu, denv = nu[idx, a].to(torch.float64)+r, nv[idx, b].to(torch.float64)+s
            if t in selected:
                pu = (cu[idx[:, None], a[:, None], u].to(torch.float64)+1)/denu[:, None]
                pv = (cv[idx[:, None], b[:, None], v].to(torch.float64)+1)/denv[:, None]
                joint = torch.logsumexp(logw[:, None]+torch.log(pu)+torch.log(pv), dim=0)
                k = selected[t]
                numerator[k] = torch.logaddexp(numerator[k], joint)
            an, bn = u[:, int(tokens[t])], v[:, int(tokens[t])]
            logw = (logw + torch.log((cu[idx, a, an].to(torch.float64)+1)/denu)
                    + torch.log((cv[idx, b, bn].to(torch.float64)+1)/denv))
            cu[idx, a, an] += 1; cv[idx, b, bn] += 1
            nu[idx, a] += 1; nv[idx, b] += 1
            evidence[t+1] = torch.logsumexp(logw, dim=0)
            if max_seconds is not None and t % 64 == 0:
                synchronize(device)
                if time.perf_counter()-start > max_seconds:
                    raise TimeoutError(f'candidate exceeded {max_seconds}s before full sequence completed')
        cumulative = torch.logaddexp(cumulative, evidence)
        used += h
        synchronize(device)
        if max_seconds is not None and time.perf_counter()-start > max_seconds:
            raise TimeoutError(f'candidate exceeded {max_seconds}s before all {total} encodings completed')
    if used != total:
        raise RuntimeError(f'encoding enumeration incomplete: {used} != {total}')
    evidence = cumulative - math.log(total)
    predictive = torch.exp(numerator - cumulative[torch.as_tensor(times, device=device), None])
    predictive = predictive.cpu().numpy()
    evidence = (evidence / math.log(2)).cpu().numpy()
    if (not np.isfinite(predictive).all() or np.any(predictive <= 0)
            or not np.allclose(predictive.sum(-1), 1., rtol=0, atol=2e-10)):
        raise ArithmeticError('exact predictive normalization failed')
    realized = np.log2(predictive[np.arange(len(times)), tokens[times]])
    if not np.allclose(realized, evidence[times+1]-evidence[times], rtol=0, atol=3e-10):
        raise ArithmeticError('selected predictive/evidence identity failed')
    return dict(times=times, predictive=predictive, log_evidence_bits=evidence,
                encoding_count=used, elapsed_seconds=time.perf_counter()-start)


def exact_selected(tokens, dimensions=(2, 3), times=TIMES, *, device='cpu',
                   chunk_size=65536, max_seconds=None, encoding_cache=None):
    """Exact selected predictions and cumulative losses by Dirichlet integrals.

    Only the declared evidence_times are returned; unlike exact_streaming,
    this function does not imply all intermediate prefix evidences exist.
    Histograms contain transitions strictly before the prediction time.
    """
    r, s = tuple(dimensions); d = r*s
    tokens = np.asarray(tokens)
    times = np.asarray(times, dtype=np.int64)
    if (tokens.ndim != 1 or tokens.dtype.kind not in 'iu' or len(tokens) < 2
            or np.any(tokens < 0) or np.any(tokens >= d)):
        raise ValueError('invalid token sequence')
    if (times.ndim != 1 or not len(times) or np.any(np.diff(times) <= 0)
            or times[0] < 1 or times[-1] >= len(tokens)):
        raise ValueError('times must be increasing valid prediction times')
    if min(r, s) < 2 or chunk_size < 1:
        raise ValueError('invalid dimensions or chunk size')
    device = torch.device(device)
    total = encoding_count((r, s))
    if device.type == 'cuda':
        free, _ = torch.cuda.mem_get_info(device)
        if memory_estimate((r, s), chunk_size) > .8*free:
            raise MemoryError('chunk memory estimate exceeds 80% of currently free GPU memory')
    gamma=torch.as_tensor([0.]+[math.lgamma(j) for j in range(1,len(tokens)+max(r,s)+2)],dtype=torch.float64,device=device)
    histograms = []
    for t in times:
        counts = np.bincount(tokens[:t-1]*d + tokens[1:t], minlength=d*d).reshape(d,d)
        histograms.append(torch.as_tensor(counts, dtype=torch.int32, device=device))
    cumulative = torch.full((len(times),), -torch.inf, dtype=torch.float64, device=device)
    numerator = torch.full((len(times), d), -torch.inf, dtype=torch.float64, device=device)
    start = time.perf_counter(); used = 0
    if encoding_cache is not None:
        if encoding_cache.shape != (total,d) or encoding_cache.dtype != np.uint8:
            raise ValueError('encoding cache dimensions/dtype mismatch')
        chunks=(encoding_cache[start:start+chunk_size] for start in range(0,total,chunk_size))
    else:
        chunks=encoding_chunks((r,s),chunk_size)
    for grids in chunks:
        h = len(grids)
        encoding = torch.as_tensor(grids.astype(np.int64), device=device)
        inverse = torch.argsort(encoding, dim=1)
        u, v = inverse//s, inverse%s
        idx = torch.arange(h, device=device)
        for k,t in enumerate(times):
            joint_counts = histograms[k][encoding[:,:,None], encoding[:,None,:]].reshape(h,r,s,r,s)
            cu = joint_counts.sum(dim=(2,4))
            cv = joint_counts.sum(dim=(1,3))
            nu, nv = cu.sum(-1), cv.sum(-1)
            logw = (gamma[cu+1].sum(dim=(1,2))
                - gamma[nu+r].sum(-1) + r*math.lgamma(r)
                + gamma[cv+1].sum(dim=(1,2))
                - gamma[nv+s].sum(-1) + s*math.lgamma(s) - math.log(d))
            a, b = u[:,int(tokens[t-1])], v[:,int(tokens[t-1])]
            pu = (cu[idx[:,None], a[:,None], u].to(torch.float64)+1)/(nu[idx,a,None]+r)
            pv = (cv[idx[:,None], b[:,None], v].to(torch.float64)+1)/(nv[idx,b,None]+s)
            evidence = torch.logsumexp(logw, dim=0)
            child = torch.logsumexp(logw[:,None]+torch.log(pu)+torch.log(pv), dim=0)
            cumulative[k] = torch.logaddexp(cumulative[k], evidence)
            numerator[k] = torch.logaddexp(numerator[k], child)
        used += h
        synchronize(device)
        if max_seconds is not None and time.perf_counter()-start > max_seconds:
            raise TimeoutError(f'candidate exceeded {max_seconds}s after {used}/{total} encodings')
    if used != total:
        raise RuntimeError('incomplete encoding enumeration')
    predictive = torch.exp(numerator-cumulative[:,None]).cpu().numpy()
    evidence_selected = ((cumulative-math.log(total))/math.log(2)).cpu().numpy()
    evidence_next = ((numerator[torch.arange(len(times),device=device),
                         torch.as_tensor(tokens[times],device=device)]-math.log(total))/math.log(2)).cpu().numpy()
    evidence = {0:0., 1:-math.log2(d)}
    for t,before,after in zip(times,evidence_selected,evidence_next):
        for length,value in ((int(t),float(before)),(int(t+1),float(after))):
            if length in evidence and abs(evidence[length]-value) > 3e-10:
                raise ArithmeticError('overlapping prefix-evidence identities disagree')
            evidence[length] = value
    if (not np.isfinite(predictive).all() or np.any(predictive <= 0)
            or not np.allclose(predictive.sum(-1),1.,rtol=0,atol=3e-10)):
        raise ArithmeticError('exact predictive normalization failed')
    cumulative_losses = -evidence_next-math.log2(d)
    evidence_times = np.array(sorted(evidence),dtype=np.int64)
    return dict(times=times, predictive=predictive,
        evidence_times=evidence_times,
        log_evidence_bits=np.array([evidence[t] for t in evidence_times]),
        cumulative_loss_bits=cumulative_losses, encoding_count=used,
        elapsed_seconds=time.perf_counter()-start)


def benchmark(args):
    from hidden_factor import draw_batch, exact_curve
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('GPU sizing requires a working CUDA device')
    torch.set_num_threads(1)
    # Warm up kernels and validate this actual runtime before timing candidates.
    from hidden_factor_cache import prepare_encoding_cache, open_encoding_cache
    cache_root=args.cache_root or args.out.parent/'encoding_cache'
    validations=[]
    for dims in [(2,3),(3,3)]:
        validation_cache=cache_root/f'validation_{dims[0]}x{dims[1]}.npy'
        prepare_encoding_cache(validation_cache,dims,chunk_size=257,max_bytes=2**24)
        validation_maps=open_encoding_cache(validation_cache,dims)
        for seed in [2026091700,2026091701]:
            warmup=draw_batch(np.random.default_rng(seed),1,65,dimensions=dims)['xs'][0]
            expected=exact_curve(warmup,[1,4,16,64],dims)
            actual=exact_selected(warmup,dims,[1,4,16,64],device=args.device,chunk_size=257)
            cached_actual=exact_selected(warmup,dims,[1,4,16,64],device=args.device,chunk_size=193,encoding_cache=validation_maps)
            cache_error=max(float(np.max(np.abs(actual['predictive']-cached_actual['predictive']))),float(np.max(np.abs(actual['log_evidence_bits']-cached_actual['log_evidence_bits']))))
            probability_error=float(np.max(np.abs(expected['predictive']-actual['predictive'])))
            evidence_error=float(np.max(np.abs(expected['log_evidence_bits'][actual['evidence_times']]-actual['log_evidence_bits'])))
            if max(probability_error,evidence_error,cache_error)>1e-9:
                raise ArithmeticError('GPU/CPU exact-reference validation failed')
            validations.append(dict(dimensions=list(dims),seed=seed,probability_max_error=probability_error,evidence_max_error_bits=evidence_error,cache_chunk_max_error=cache_error,passed=True))
    report = dict(schema='hidden-factor-sizing-v1', device=args.device,
        torch_version=str(torch.__version__), numpy_version=np.__version__,
        gpu=torch.cuda.get_device_name() if args.device.startswith('cuda') else None,
        source_length=args.length, projected_n_sequences=args.n_eval,
        production_workload=(args.length==1025 and args.n_eval==4096),
        precision='float64', chunk_size=args.chunk_size,
        repeats=args.repeats, budget_gpu_hours=args.max_reference_hours,
        validation=dict(passed=True,tolerance=1e-9,cases=validations),
        note='Each completed timing uses an independent full sequence and all requested prediction times. Cumulative losses use exact evidence at those times. Encoding-cache build is measured once; reference timing includes source generation and bounded host-to-device cache transfers. Warmup is excluded. Cohort projection includes cache build and uses the slowest completed repeat.',
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).with_name('hidden_factor.py').read_bytes()).hexdigest(),
        cache_code_sha256=hashlib.sha256(Path(__file__).with_name('hidden_factor_cache.py').read_bytes()).hexdigest(),candidates=[])
    for dimensions in args.dimensions:
        times = [t for t in TIMES if t < args.length]
        if args.length-1 not in times: times.append(args.length-1)
        row = dict(dimensions=list(dimensions), alphabet=math.prod(dimensions),
            encoding_count=encoding_count(dimensions), memory_estimate_bytes=memory_estimate(dimensions, args.chunk_size),
            status='pending', measured_seconds=[], peak_allocated_bytes=[])
        before=time.perf_counter()
        try:
            cache_path=cache_root/f'{dimensions[0]}x{dimensions[1]}.npy'
            row['cache_reused']=cache_path.exists()
            cache_receipt=prepare_encoding_cache(cache_path,dimensions,chunk_size=args.chunk_size,max_bytes=int(args.max_cache_gib*1024**3),timeout_seconds=args.max_cache_build_seconds)
            cached=open_encoding_cache(cache_path,dimensions)
            row['cache_build_and_open_seconds']=time.perf_counter()-before
            row['cache_receipt']=cache_receipt
            for repeat in range(args.repeats):
                if args.device.startswith('cuda'):
                    torch.cuda.reset_peak_memory_stats()
                before = time.perf_counter()
                seed = np.random.SeedSequence([2026091701, *dimensions, repeat])
                source = draw_batch(np.random.default_rng(seed), 1, args.length, dimensions=dimensions)
                exact_selected(source['xs'][0], dimensions, times, device=args.device,
                               chunk_size=args.chunk_size, max_seconds=args.max_candidate_seconds, encoding_cache=cached)
                synchronize(args.device)
                row['measured_seconds'].append(time.perf_counter()-before)
                row['peak_allocated_bytes'].append(torch.cuda.max_memory_allocated() if args.device.startswith('cuda') else None)
            hours = (row['cache_build_and_open_seconds']+max(row['measured_seconds'])*args.n_eval)/3600
            row.update(status='complete', projected_reference_hours=hours,
                projected_reference_hours_mean=(row['cache_build_and_open_seconds']+float(np.mean(row['measured_seconds']))*args.n_eval)/3600,
                within_budget=hours <= args.max_reference_hours)
        except (TimeoutError, torch.cuda.OutOfMemoryError, MemoryError, OSError) as error:
            row.update(status='incomplete', reason=str(error), elapsed_incomplete_repeat_seconds=time.perf_counter()-before,
                       within_budget=False)
        report['candidates'].append(row)
        eligible = [r for r in report['candidates'] if r['status']=='complete' and r['within_budget'] and report['production_workload']]
        report['recommended_dimensions'] = (max(eligible, key=lambda r:(r['alphabet'], r['encoding_count']))['dimensions'] if eligible else None)
        report['selection_rule'] = 'Largest measured alphabet among declared candidates within reference-time budget; encoding count breaks alphabet-size ties. Not a proof that no other dimension pair is tractable.'
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        print(json.dumps(row), flush=True)
        if args.device.startswith('cuda'):
            torch.cuda.empty_cache()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('benchmark')
    p.add_argument('--device', default='cuda')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--dimensions', type=dimensions_arg, nargs='+', default=[(2,5),(2,6),(3,4),(2,7),(3,5),(2,8),(2,9)])
    p.add_argument('--length', type=int, default=1025)
    p.add_argument('--n-eval', type=int, default=4096)
    p.add_argument('--chunk-size', type=int, default=65536)
    p.add_argument('--max-candidate-seconds', type=float, default=60.)
    p.add_argument('--max-reference-hours', type=float, default=24.)
    p.add_argument('--repeats', type=int, default=2)
    p.add_argument('--cache-root',type=Path)
    p.add_argument('--max-cache-gib',type=float,default=8.)
    p.add_argument('--max-cache-build-seconds',type=float,default=300.)
    args = parser.parse_args()
    if args.length < 2 or args.n_eval < 1 or args.max_candidate_seconds <= 0 or args.repeats < 2:
        parser.error('positive benchmark sizes and time limit required')
    benchmark(args)


if __name__ == '__main__':
    main()
