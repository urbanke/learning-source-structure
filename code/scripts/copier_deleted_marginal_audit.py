"""Exploratory comparisons of saved copier-deleted predictions with unigram rules.

No model run, fitting, clipping or probability repair. The marginal counts
contain all tokens in the observed prefix, including its final query token.
"""
from pathlib import Path
import argparse
import io
import csv
import hashlib
import json

import numpy as np
from scipy.special import xlogy

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "output/copier_time/run_2026-09-15"
DEFAULT_OUTPUT = ROOT / "results/transformer/component_feasibility/marginal_unigram_audit"
TIMES = (64, 256, 1024)
N, D = 1024, 64


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def array_hash(value):
    value = np.ascontiguousarray(value)
    h = hashlib.sha256(value.dtype.str.encode())
    h.update(json.dumps(value.shape).encode())
    h.update(value.tobytes())
    return h.hexdigest()


def seed_of(*parts):
    h = 1469598103934665603
    for p in parts:
        for c in str(p).encode():
            h = ((h ^ c) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return int(h % (2**32 - 1))


def stats(values):
    values = np.asarray(values, dtype=float)
    good = np.isfinite(values)
    finite = values[good]
    return dict(n=int(len(values)), n_finite=int(good.sum()), n_positive_infinite=int(np.isposinf(values).sum()),
                mean=float(values.mean()) if good.all() else None,
                sequence_se=float(values.std(ddof=1)/np.sqrt(len(values))) if good.all() and len(values)>1 else None,
                mean_is_positive_infinite=bool(np.isposinf(values).any()),
                finite_subset_mean=float(finite.mean()) if len(finite) else None,
                finite_subset_quantiles={str(p): float(np.quantile(finite,p)) if len(finite) else None for p in (.5,.9,.95,.99)},
                maximum=float(values.max()) if good.all() else None)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,default=DEFAULT_INPUT)
    parser.add_argument("--out",type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument("--check-only",action="store_true",help="Recompute and verify saved outputs without writing")
    args=parser.parse_args(argv)
    RAW,OUT=args.input.resolve(),args.out.resolve()
    paths = sorted(RAW.glob("seq_*.npz"))
    if [p.name for p in paths] != [f"seq_{i:06d}.npz" for i in range(N)]:
        raise ValueError("Require all 1024 frozen source sequences")
    protocol = json.loads((RAW/"protocol.json").read_text())
    expected = dict(schema="natural-copier-sequence-v1", n_seq=1024, n_ctx=1024, d=D,
                    master_seed=20261004, candidate_head=[0,1], alpha=[0.,0.], source_depths=list(range(9)))
    if any(protocol.get(k)!=v for k,v in expected.items()):
        raise ValueError("Unexpected frozen source/protocol")
    spec_id = hashlib.sha256(canonical(protocol).encode()).hexdigest()
    names = ("uniform", "empirical_marginal", "laplace_marginal")
    metrics = {name: {key: np.empty((N,3)) for key in ("tv", "kl_deleted_to_baseline_bits", "kl_baseline_to_deleted_bits", "squared_error")}
               for name in names}
    cosine, magnitude_ratio, unseen = (np.empty((N,3)) for _ in range(3))
    depth = np.empty(N,dtype=int)
    inputs = []
    u = np.full(D,1/D)
    maxnorm = 0.
    for index,path in enumerate(paths):
        with np.load(path,allow_pickle=False) as a:
            meta=json.loads(str(a["metadata_json"]))
            if (meta["index"]!=index or meta["spec_id"]!=spec_id or
                    hashlib.sha256(canonical(meta["spec"]).encode()).hexdigest()!=spec_id or
                    meta["sequence_seed"]!=seed_of(protocol["schema"],protocol["master_seed"],index)):
                raise ValueError("Mismatched source identity/protocol")
            arrays={key:a[key] for key in ("tokens","source_depth","copier_deleted_probs")}
            for key,value in arrays.items():
                if array_hash(value)!=meta["array_hashes"][key]:
                    raise ValueError(f"Corrupt input {index}/{key}")
            x=arrays["tokens"];qfull=arrays["copier_deleted_probs"];depth[index]=int(arrays["source_depth"])
            if (x.shape!=(1025,) or x.dtype.kind not in "iu" or np.any(x<0) or np.any(x>=D)
                    or qfull.shape!=(1024,D) or qfull.dtype!=np.float64 or not np.isfinite(qfull).all() or np.any(qfull<=0)):
                raise ValueError("Invalid token/probability arrays")
            err=float(abs(qfull.sum(-1)-1).max());maxnorm=max(maxnorm,err)
            if err>1e-12:raise ValueError("Non-normalized raw deleted predictions")
            for j,t in enumerate(TIMES):
                counts=np.bincount(x[:t],minlength=D).astype(float)
                independently_counted=np.array([np.count_nonzero(x[:t]==b) for b in range(D)])
                if counts.sum()!=t or not np.array_equal(counts,independently_counted):
                    raise ValueError("Independent prefix counting failed")
                empirical=counts/t;laplace=(counts+1)/(t+D);q=qfull[t-1]
                if max(abs(empirical.sum()-1),abs(laplace.sum()-1))>1e-12:
                    raise ValueError("Marginal baseline normalization failed")
                if not np.allclose(laplace-u,(t/(t+D))*(empirical-u),rtol=0,atol=1e-15):
                    raise ValueError("Independent Laplace shrinkage identity failed")
                unseen[index,j]=np.count_nonzero(counts==0)
                a0=q-u;b0=empirical-u
                denominator=np.linalg.norm(a0)*np.linalg.norm(b0)
                cosine[index,j]=np.dot(a0,b0)/denominator if denominator>0 else np.nan
                magnitude_ratio[index,j]=np.linalg.norm(a0)/np.linalg.norm(b0) if np.linalg.norm(b0)>0 else np.nan
                for name,p in zip(names,(u,empirical,laplace)):
                    metrics[name]["tv"][index,j]=.5*np.abs(q-p).sum()
                    metrics[name]["squared_error"][index,j]=np.square(q-p).sum()
                    metrics[name]["kl_deleted_to_baseline_bits"][index,j]=np.sum(xlogy(q,q)-xlogy(q,p))/np.log(2)
                    metrics[name]["kl_baseline_to_deleted_bits"][index,j]=np.sum(xlogy(p,p)-xlogy(p,q))/np.log(2)
            inputs.append(dict(index=index, source_depth=int(depth[index]),
                               arrays={key:meta["array_hashes"][key] for key in arrays}))
    groups={};rows=[]
    for group,mask in [("all",np.ones(N,dtype=bool)),("depth_positive",depth>0),*((f"depth_{d}",depth==d) for d in range(9))]:
        entry=dict(n_sequences=int(mask.sum()),at_prefix={})
        for j,t in enumerate(TIMES):
            at={}
            uniform_sse=metrics["uniform"]["squared_error"][mask,j].sum()
            for name in names:
                at[name]={key:stats(value[mask,j]) for key,value in metrics[name].items()}
                at[name]["fraction_squared_error_reduction_relative_to_uniform"] = float(1-metrics[name]["squared_error"][mask,j].sum()/uniform_sse)
                at[name]["fraction_histories_with_smaller_TV_than_uniform"] = float(np.mean(metrics[name]["tv"][mask,j]<metrics["uniform"]["tv"][mask,j]))
                for key in metrics[name]:
                    row=at[name][key]
                    rows.append(dict(group=group, prefix=t, baseline=name, quantity=key,n=row["n"],
                                     mean=row["mean"],sequence_se=row["sequence_se"],
                                     positive_infinite=row["n_positive_infinite"],finite_mean=row["finite_subset_mean"],
                                     finite_median=row["finite_subset_quantiles"]["0.5"],finite_p95=row["finite_subset_quantiles"]["0.95"],
                                     maximum=row["maximum"]))
            at["residual_direction_cosine_with_empirical_minus_uniform"]=stats(cosine[mask,j])
            at["residual_to_empirical_deviation_magnitude_ratio"]=stats(magnitude_ratio[mask,j])
            at["unseen_marginal_symbol_count"]=stats(unseen[mask,j])
            entry["at_prefix"][str(t)]=at
        groups[group]=entry
    report=dict(schema="exploratory-disabled-marginal-unigram-comparison-v1",passed=True,spec_id=spec_id,
                n_sequences=N,positions=list(TIMES),source_depth_counts=np.bincount(depth,minlength=9).tolist(),
                definitions=dict(empirical="M_t(b)=count of token b in tokens[:t], divided by t",
                                 laplace="L_t(b)=[count of token b in tokens[:t]+1]/[t+64]",
                                 target="Saved intact-weights model with first-layer head2 deleted at all observed positions",
                                 tv="0.5 sum_b |Q_deleted(b)-baseline(b)|",
                                 kl_forward="sum_b Q_deleted(b) log2[Q_deleted(b)/baseline(b)]",
                                 residual_cosine="cosine between Q_deleted-U and empirical-U; descriptive directional alignment without fitting",
                                 magnitude_ratio="Euclidean norm of Q_deleted-U divided by norm of empirical-U; no coefficient is fitted or applied"),
                notes=["Neither marginal rule is the fixed informed-uniform component. They are separate count-based hypotheses.",
                       "Empirical probabilities may be zero: forward KL is then infinite. No pseudocount or clipping is applied to that baseline.",
                       "Laplace uses fixed pseudocount one, not fitted smoothing. Its deviation from U is t/(t+64) times empirical's deviation.",
                       "Means and standard errors use independent sequences within each depth. No token is counted as an independent replicate.",
                       "Directional alignment does not establish an internal unigram circuit or invariance to token ordering."],
                validation=dict(maximum_raw_probability_sum_error=maxnorm,probability_repair="none",
                                independent_count_enumeration=True,baseline_normalization=True,laplace_shrinkage_identity=True),
                script=dict(path=str(Path(__file__).relative_to(ROOT)),sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
                inputs=inputs,groups=groups)
    json_bytes=(json.dumps(report,indent=2,allow_nan=False)+"\n").encode()
    table=io.StringIO(newline="")
    writer=csv.DictWriter(table,fieldnames=list(rows[0]),delimiter="\t",lineterminator="\n")
    writer.writeheader()
    writer.writerows({k:"NA" if v is None else v for k,v in row.items()} for row in rows)
    artifacts={"summary.json":json_bytes,"summary.tsv":table.getvalue().encode()}
    receipt=dict(schema="disabled-marginal-audit-receipt-v1",passed=True,spec_id=spec_id,
                 script=report["script"],n_sequences=N,positions=list(TIMES),
                 artifacts={name:hashlib.sha256(data).hexdigest() for name,data in artifacts.items()})
    artifacts["receipt.json"]=(json.dumps(receipt,indent=2)+"\n").encode()
    if args.check_only:
        for name,data in artifacts.items():
            if not (OUT/name).is_file() or (OUT/name).read_bytes()!=data:
                raise ValueError("Saved audit differs from fresh recomputation: "+str(OUT/name))
    else:
        OUT.mkdir(parents=True,exist_ok=True)
        for name,data in artifacts.items():(OUT/name).write_bytes(data)
    print(json.dumps(dict(passed=True,check_only=args.check_only,n_sequences=N,positions=TIMES,
                          maximum_probability_sum_error=maxnorm)))



if __name__=="__main__":
    main()
