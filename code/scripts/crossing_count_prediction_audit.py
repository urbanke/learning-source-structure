"""Independent audit and crossing bootstrap for the finite-count campaign.

OPENBLAS_NUM_THREADS=1 python code/scripts/crossing_count_prediction_audit.py
python code/scripts/crossing_count_prediction_audit.py --self-test

The default audit checks every saved source, count, path and summary. It
reevaluates a deterministic subset of code profiles per cell; pass
--all-code-profiles to reevaluate every profile. No new source or chain
experiments are run. Bootstrap units are the independent table indices.
Located-root intervals are explicitly conditional on a root being found;
null outcomes and their reasons are retained in the summaries.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.special import gammaln, logsumexp
from scipy.stats import binom

ROOT = Path(__file__).resolve().parents[2]
LN2 = math.log(2.)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(a, b, atol=2e-8, rtol=2e-11):
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape and np.allclose(a, b, atol=atol, rtol=rtol), (a, b)
    return float(np.max(np.abs(a-b))) if a.size else 0.


def root(ns, values):
    """Last nonpositive point followed by a positive final suffix."""
    y = np.asarray(values, dtype=float)
    up = int(np.sum((y[:-1] <= 0) & (y[1:] > 0)))
    if y[-1] <= 0:
        return None, "nonpositive_at_end", up
    nonpositive = np.flatnonzero(y <= 0)
    if not len(nonpositive):
        return None, "positive_at_every_check", up
    j = int(nonpositive[-1])
    fraction = -y[j]/(y[j+1]-y[j])
    return float(ns[j]*(ns[j+1]/ns[j])**fraction), "located", up


def l1_cost(z, d):
    z = np.asarray(z)
    return float((gammaln(z.sum()+d)-gammaln(d)-gammaln(z+1).sum())/LN2)


def depth_one_expected(T, q, t):
    """Direct iteration and marginal-binomial formula, in bits."""
    d = len(q)
    if not t:
        return np.array([math.log2(d), math.log2(d)])
    state, total = q.copy(), np.zeros(d)
    for _ in range(t):
        total += state
        state = state@T
    r = total/t
    J, v = r[:, None]*T, r@T
    k = np.arange(t+1)
    def expect(x, f):
        return float(binom.pmf(k, t, x)@f)
    factorial, plus, shifted = gammaln(k+1), np.log(k+1), gammaln(k+d)
    F = gammaln(t+1+d)-gammaln(d)
    F -= sum(expect(v[b], factorial)+q[b]*expect(v[b], plus) for b in range(d))
    C = d*(-gammaln(d))+sum(expect(x, shifted) for x in r)
    C -= sum(expect(x, factorial) for x in J.ravel())
    return np.array([F/LN2, math.log2(d)+C/LN2])


def self_test():
    T = np.array([[.83, .17], [.28, .72]])
    q = np.array([.67, .33])
    maximum = 0.
    for t in range(6):
        state, total = q.copy(), np.zeros(2)
        for _ in range(t):
            total += state
            state = state@T
        J = (total/t)[:, None]*T if t else np.full((2, 2), .25)
        value, mass = np.zeros(2), 0.
        for pairs in itertools.product(range(4), repeat=t):
            z = np.bincount(pairs, minlength=4).reshape(2, 2)
            probability = float(np.prod([J.flat[j] for j in pairs]))
            for initial in range(2):
                N = z.sum(0)
                N[initial] += 1
                weight = probability*q[initial]
                mass += weight
                value += weight*np.array([l1_cost(N, 2), 1+sum(l1_cost(row, 2) for row in z)])
        close(mass, 1., atol=1e-12)
        maximum = max(maximum, close(value, depth_one_expected(T, q, t), atol=1e-11))
    fixtures = [([-1., 1.], math.sqrt(200), "located"),
                ([1., 2.], None, "positive_at_every_check"),
                ([-1., -2.], None, "nonpositive_at_end")]
    for y, expected, status in fixtures:
        actual = root([10, 20], y)
        assert actual[1] == status
        if expected is not None:
            close(actual[0], expected)
    assert root([10, 20, 40, 80], [-1, 2, -2, 2])[0] == math.sqrt(3200)
    return dict(status="passed", maximum_depth_one_binomial_vs_enumeration_error_bits=maximum,
                scope="All auxiliary binary pair sequences through five transitions, nonuniform initial q, plus root/null fixtures.")


def independent_occupations(T, q, ts):
    """Direct iteration, completing numerically converged tails with pi.

    pi is independently solved from linear balance equations. The reported
    geometric remainder uses the positive-column minorization. This is a
    floating-point diagnostic, not a certified arithmetic error bound.
    """
    d = len(q)
    system = T.T-np.eye(d)
    system[-1] = 1.
    rhs = np.zeros(d)
    rhs[-1] = 1.
    pi = np.linalg.solve(system, rhs)
    assert pi.min() > 0
    residual = float(np.abs(pi@T-pi).sum())
    minorization = float(T.min(0).sum())
    state, total = q.copy(), np.zeros(d)
    result, j = [], 0
    for t in ts:
        while j < t:
            if np.abs(state-pi).sum() < 2e-14:
                break
            total += state
            state = state@T
            j += 1
        result.append((total+(t-j)*pi)/t if j < t else total/t)
    remainder = (float(np.abs(state-pi).sum())/max(minorization, 1e-300)/min(ts)
                 + residual/max(minorization, 1e-300))
    return np.asarray(result), pi, dict(direct_steps=j, stationary_residual_l1=residual,
        minorization_mass=minorization, geometric_tail_diagnostic=remainder)


def summarize_roots(values, statuses):
    finite = np.asarray([x for x in values if x is not None])
    return dict(draws=len(values), located=int(len(finite)), null=int(len(values)-len(finite)),
        status_counts=dict(Counter(statuses)),
        conditional_on_located_quantiles=([float(x) for x in np.quantile(finite, [.025, .5, .975])]
                                         if len(finite) else None),
        quantile_probabilities=[.025, .5, .975])


def bootstrap(ns, count, chain, archive_ns, archive_gaps, repetitions, seed):
    rng = np.random.default_rng(seed)
    number = len(count)
    weights = rng.multinomial(number, np.full(number, 1/number), size=repetitions)/number
    source_values, source_statuses = {}, {}
    for name, gaps in (("count", count), ("chain", chain)):
        results = [root(ns, y) for y in weights@gaps]
        source_values[name] = [x[0] for x in results]
        source_statuses[name] = [x[1] for x in results]
    number = len(archive_gaps)
    aw = rng.multinomial(number, np.full(number, 1/number), size=repetitions)/number
    ar = [root(archive_ns, y) for y in aw@archive_gaps]
    source_values["archive"] = [x[0] for x in ar]
    source_statuses["archive"] = [x[1] for x in ar]
    result = {name:summarize_roots(source_values[name], source_statuses[name]) for name in source_values}
    for other in ("chain", "archive"):
        values = [a/b-1 if a is not None and b is not None else None
                  for a, b in zip(source_values["count"], source_values[other])]
        statuses = ["both_located" if v is not None else "at_least_one_null" for v in values]
        result["count_over_"+other+"_minus_one"] = summarize_roots(values, statuses)
    return result


def archived_paths(cell):
    path = ROOT/cell["archive_path"]
    assert digest(path) == cell["archive_sha256"]
    metadata = json.loads(path.read_text())
    stem = path.stem.removeprefix("summary_")
    rows, hashes = [], {}
    for seed in metadata["seeds"]:
        source = path.parent/f"{stem}_seed{seed}.json"
        x = json.loads(source.read_text())
        assert (x["d"], x["alpha"], x["L"], x["seed"]) == (cell["d"],cell["alpha"],cell["L"],seed)
        assert x["l_max"] == cell["l_max"]
        close(np.asarray(x["l_F"])-x["l_C"], x["lam"])
        rows.append(x)
        hashes[str(source.relative_to(ROOT))] = digest(source)
    ns = sorted(set.intersection(*(set(x["checkpoints"]) for x in rows)))
    gaps = np.asarray([[dict(zip(x["checkpoints"], x["lam"]))[n] for n in ns] for x in rows])
    value, status, _ = root(ns, gaps.mean(0))
    assert status == "located"
    close(value, cell["archived_mean_curve_crossing"])
    return np.asarray(ns), gaps, hashes


def audit_profiles(arrays, cell, args):
    from lsa.codelength import default_l_max, depth_averaged_codelength_profiles
    d = cell["d"]
    assert cell["l_max"] == default_l_max(d) == round(2/(1-.5772156649015329)*math.log(d))
    profiles = {(), (1,), (2,), (1, 1)}
    positions = {"count":[], "chain":[]}
    def profile(z):
        return tuple(sorted((int(x) for x in z if x), reverse=True))
    for name in positions:
        for rep in range(cell["repetitions"]):
            rows = []
            for j in range(len(cell["tokens"])):
                fp = profile(arrays[name+"_frequency_counts"][rep,j])
                counts = Counter(profile(z) for z in arrays[name+"_transition_counts"][rep,j] if z.sum())
                rows.append((fp, counts))
                profiles.add(fp)
                profiles.update(counts)
            positions[name].append(rows)
    ordered = sorted(profiles, key=lambda p:(sum(p),len(p),p))
    assert len(profiles) == cell["profile_count"]
    if args.all_code_profiles:
        chosen = ordered
    else:
        indexes = np.unique(np.linspace(0, len(ordered)-1, min(args.code_profiles,len(ordered))).astype(int))
        chosen = sorted({ordered[i] for i in indexes}|{(), (1,), (2,), (1,1)},key=lambda p:(sum(p),len(p),p))
    evaluated = depth_averaged_codelength_profiles({p:p for p in chosen},d=d,l_max=cell["l_max"],
        jobs=1,tables_source="universal",universal_path=args.store)
    l1_error = mixture_error = 0.
    for p, value in evaluated.items():
        l1_error = max(l1_error, abs(-value.log2_q_by_depth[0]-l1_cost(p,d)))
        mixture_error = max(mixture_error,abs(value.log2_q_avg-(logsumexp(np.asarray(value.log2_q_by_depth)*LN2)-math.log(cell["l_max"]))/LN2))
    close(l1_error, 0., atol=1e-7)
    close(mixture_error, 0., atol=1e-8)
    mass = d*2**evaluated[(2,)].log2_q_avg+d*(d-1)*2**evaluated[(1,1)].log2_q_avg
    close(mass,1.,atol=1e-9)
    code_error = 0.
    if args.all_code_profiles:
        for name, repetitions in positions.items():
            for rep, rows in enumerate(repetitions):
                for j,(fp,multiplicities) in enumerate(rows):
                    F = -evaluated[fp].log2_q_avg
                    C = math.log2(d)-sum(v*evaluated[p].log2_q_avg for p,v in multiplicities.items())
                    code_error = max(code_error,close(F,arrays[name+"_frequency_codelength"][rep,j]),
                                     close(C,arrays[name+"_context_codelength"][rep,j]))
    return dict(all_profiles=len(profiles), reevaluated_profiles=len(chosen),
        all_code_costs_reassembled=args.all_code_profiles, maximum_code_cost_difference_bits=code_error if args.all_code_profiles else None,
        maximum_depth_one_gamma_difference_bits=l1_error, maximum_depth_mixture_difference_bits=mixture_error,
        two_token_total_probability=mass)


def audit_cell(cell,args):
    path = ROOT/cell["raw_path"]
    assert digest(path) == cell["raw_sha256"]
    # NPZ members are otherwise decompressed anew on every indexed access.
    # A cell fits in memory, so load each member once before the replay.
    with np.load(path) as packed:
        x = {key:packed[key] for key in packed.files}
    d,L,a,reps = (cell[k] for k in ("d","L","alpha","repetitions"))
    ns,ts,q = x["tokens"],x["transitions"],x["q"]
    close(ns,cell["tokens"]); close(ts,ns-1); close(ts,cell["transitions"])
    expected_q = np.arange(1,d+1,dtype=float)**(-a)
    expected_q /= expected_q.sum()
    close(q,expected_q,atol=0.,rtol=0.)
    maximum_occupation = maximum_pi = maximum_source = 0.
    occupation_checks=[]
    for rep in range(reps):
        prefix=[cell["seed"],d,L,round(a*100),rep]
        G=np.random.default_rng(prefix+[77]).exponential(size=(d,d,L)).prod(axis=2)
        S=G@q
        T=G*q[None,:]/S[:,None]
        maximum_source=max(maximum_source,close(T,x["transition_matrices"][rep],atol=0,rtol=0),
                           close(S,x["normalized_row_sums"][rep],atol=0,rtol=0))
        r,pi,diagnostic=independent_occupations(T,q,ts)
        maximum_occupation=max(maximum_occupation,close(r,x["expected_occupation_laws"][rep],atol=2e-10))
        maximum_pi=max(maximum_pi,close(pi,x["stationary_laws"][rep],atol=2e-10))
        occupation_checks.append(diagnostic)
        initial=int(np.random.default_rng(prefix+[78]).choice(d,p=q))
        assert initial==x["initial_states"][rep]
        path_states=x["chain_paths"][rep]
        assert path_states[0]==initial and len(path_states)==ns[-1]
        uniforms=np.random.default_rng(prefix+[79]).random(int(ts[-1]))
        cdf=np.cumsum(T,axis=1); cdf[:,-1]=1.
        before,after=path_states[:-1],path_states[1:]
        upper=cdf[before,after]
        lower=np.where(after>0,cdf[before,np.maximum(0,after-1)],0.)
        assert np.all((uniforms>=lower)&(uniforms<upper))
        for j,(n,t) in enumerate(zip(ns,ts)):
            stored_r=x["expected_occupation_laws"][rep,j]
            pair=(stored_r[:,None]*T).ravel(); pair/=pair.sum()
            generated=np.random.default_rng(prefix+[80,int(t)]).multinomial(int(t),pair).reshape(d,d)
            assert np.array_equal(generated,x["count_transition_counts"][rep,j])
            flat=path_states[:t].astype(np.int64)*d+path_states[1:n]
            genuine=np.bincount(flat,minlength=d*d).reshape(d,d)
            assert np.array_equal(genuine,x["chain_transition_counts"][rep,j])
            for name in ("count","chain"):
                counts=x[name+"_transition_counts"][rep,j]
                assert np.all(counts>=0) and counts.sum()==t
                frequency=counts.sum(0); frequency[initial]+=1
                assert np.array_equal(frequency,x[name+"_frequency_counts"][rep,j])
                assert frequency.sum()==n
            flow=np.bincount([initial],minlength=d)-np.bincount([path_states[t]],minlength=d)
            assert np.array_equal(genuine.sum(1)-genuine.sum(0),flow)
    for name in ("count","chain"):
        for field in ("frequency_codelength","context_codelength","gap"):
            data=x[name+"_"+field]
            close(data.mean(0),cell[name][field]["mean"])
            close(data.std(0,ddof=1)/math.sqrt(reps),cell[name][field]["SE"])
        close(x[name+"_frequency_codelength"]-x[name+"_context_codelength"],x[name+"_gap"])
        close(x[name+"_gap"],cell[name]["gap_by_repetition"])
        value,status,number=root(ns,x[name+"_gap"].mean(0))
        saved=cell[name]["mean_curve_crossing"]
        assert status==saved["status"] and number==saved["observed_upcrossings"]
        assert (value is None)==(saved["value"] is None)
        if value is not None:close(value,saved["value"])
    difference=x["count_gap"]-x["chain_gap"]
    close(difference.mean(0),cell["paired_gap_difference"]["mean"])
    close(difference.std(0,ddof=1)/math.sqrt(reps),cell["paired_gap_difference"]["SE"])
    cov=((x["count_gap"]-x["count_gap"].mean(0))*(x["chain_gap"]-x["chain_gap"].mean(0))).sum(0)/(reps-1)
    close(cov,cell["paired_gap_sample_covariance"])
    archive_ns,archive_gaps,hashes=archived_paths(cell)
    uncertainty=bootstrap(ns,x["count_gap"],x["chain_gap"],archive_ns,archive_gaps,args.bootstrap,
        [args.seed,d,L,round(a*100)])
    return dict(d=d,L=L,alpha=a,repetitions=reps,raw_path=cell["raw_path"],raw_sha256=cell["raw_sha256"],
        mean_curve_crossings={name:cell[name]["mean_curve_crossing"]["value"] for name in ("count","chain")}
                            | {"archive":cell["archived_mean_curve_crossing"]},
        maximum_source_replay_error=maximum_source,maximum_occupation_difference=maximum_occupation,
        maximum_stationary_difference=maximum_pi,maximum_direct_steps=max(z["direct_steps"] for z in occupation_checks),
        maximum_geometric_tail_diagnostic=max(z["geometric_tail_diagnostic"] for z in occupation_checks),
        profile_checks=audit_profiles(x,cell,args),bootstrap=uncertainty,
        archived_raw_source_hashes=hashes,status="passed")


def endpoint_slopes(source):
    index={(c["L"],c["alpha"],c["d"]):c for c in source["cells"]}
    answer=[]
    for L,a in sorted({(c["L"],c["alpha"]) for c in source["cells"]}):
        low,high=index.get((L,a,128)),index.get((L,a,512))
        if low is None or high is None:
            continue
        row=dict(L=L,alpha=a,alphabet_endpoints=[128,512])
        for name in ("count","chain","archive"):
            values=([c["archived_mean_curve_crossing"] for c in (low,high)] if name=="archive" else
                    [c[name]["mean_curve_crossing"]["value"] for c in (low,high)])
            row[name]=math.log(values[1]/values[0])/math.log(4) if all(x is not None for x in values) else None
        answer.append(row)
    return answer


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path,default=ROOT/"results/mechanism/crossing_finite_count_prediction.json")
    p.add_argument("--out",type=Path,default=ROOT/"results/mechanism/crossing_count_prediction_audit.json")
    p.add_argument("--bootstrap",type=int,default=2000)
    p.add_argument("--seed",type=int,default=202609074)
    p.add_argument("--code-profiles",type=int,default=96)
    p.add_argument("--all-code-profiles",action="store_true")
    p.add_argument("--self-test",action="store_true")
    p.add_argument("--lsa-src",type=Path,default=ROOT.parent/"layered-simplex-architecture/src")
    p.add_argument("--store",type=Path,default=None)
    args=p.parse_args()
    validation=self_test()
    if args.self_test:
        print(json.dumps(validation,indent=2)); return
    sys.path.insert(0,str(args.lsa_src))
    from tilted_row_reference import DEFAULT_STORE
    from takeover_experiment import engage_anchor_mode
    args.store=args.store or DEFAULT_STORE
    engage_anchor_mode(str(args.store))
    source_bytes=args.source.read_bytes()
    source=json.loads(source_bytes)
    source_hash=hashlib.sha256(source_bytes).hexdigest()
    evaluator_hash=digest(args.lsa_src/"lsa/codelength.py")
    options=dict(bootstrap=args.bootstrap,seed=args.seed,code_profiles=args.code_profiles,all_code_profiles=args.all_code_profiles)
    previous=json.loads(args.out.read_text()) if args.out.exists() else {}
    old={(c["d"],c["L"],c["alpha"],c["raw_sha256"]):c for c in previous.get("cells",[])} if previous.get("options")==options and previous.get("audit_sha256")==digest(Path(__file__)) and previous.get("evaluator_sha256")==evaluator_hash else {}
    cells=[]
    for c in source["cells"]:
        key=(c["d"],c["L"],c["alpha"],c["raw_sha256"])
        if key in old:
            # Resume only after verifying that all previously audited bytes
            # remain unchanged; an unchanged manifest is insufficient.
            assert digest(ROOT/c["raw_path"]) == c["raw_sha256"]
            assert digest(ROOT/c["archive_path"]) == c["archive_sha256"]
            for relative,expected in old[key]["archived_raw_source_hashes"].items():
                assert digest(ROOT/relative) == expected
        cells.append(old[key] if key in old else audit_cell(c,args))
        result=dict(schema_version=1,complete=source["complete"] and len(cells)==len(source["cells"]),
            source_path=str(args.source.relative_to(ROOT)),source_sha256=source_hash,audit_sha256=digest(Path(__file__)),
            evaluator_sha256=evaluator_hash,
            options=options,validation=validation,
            bootstrap_notes=["Independent table indices are resampled; count/chain use identical bootstrap weights; archived paths are resampled independently.",
                "Last upcrossings of the resampled mean gaps are interpolated in log token count on their original evaluation grids.",
                "Null outcomes retain their reasons and counts. Reported quantiles condition on a located root (or two located roots for ratios).",
                "Bootstrap describes sampling variation on the selected grid; it does not bound grid-interpolation bias, model error, or later reversals."],
            endpoint_slopes_128_to_512=endpoint_slopes(source),
            slope_note="Log crossing ratio divided by ln4 over the two alphabet endpoints; descriptive finite-window slopes, without an asymptotic claim.",
            cells=cells)
        args.out.parent.mkdir(parents=True,exist_ok=True)
        if not previous.get("complete") or result["complete"]:
            args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
        print(f"Audited L{c['L']} alpha{c['alpha']:g} d{c['d']}; {len(cells)} cells",flush=True)


if __name__=="__main__":
    main()
