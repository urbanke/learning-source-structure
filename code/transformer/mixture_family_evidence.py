"""Normalized product/uniform-base evidence on frozen mixture cohorts.

Only count tables at the requested sequence lengths are needed. The existing
BatchEvidence implementation supplies every row integral from the read-only
LSA store; this module adds no numerical integration rule. A single latent
depth is summed AFTER multiplying all context-row evidences. Product sources
have two independent coordinate depths, each uniform on 1..8; the uniform-
base source has one depth uniform on 0..8.

The saved log marginal includes the first token. Cumulative loss omits it,
matching eval_mixture.py: time t scores tokens 2,...,t+1. These are family
evidences, not the three-family mixture (whose shared-base component is also
required). Selecting a shorter initial subset never redraws test sequences.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from scipy.special import logsumexp

from validate_shared_depth import BatchEvidence

LN2 = np.log(2.)
SCHEMA = "pooled-three-family-marginal-evidence-v1"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class CachedBatchEvidence(BatchEvidence):
    """Reuse the checked sparse integral, caching equal count profiles."""

    def __init__(self, depths, bounds=(-30., 30.), step=1 / 64,
                 max_cached_rows=100000):
        super().__init__(depths, bounds, step)
        self.evidence_cache = OrderedDict()
        self.max_cached_rows = int(max_cached_rows)
        if self.max_cached_rows < 1:
            raise ValueError("max_cached_rows must be positive")
        self.row_calls = 0
        self.row_hits = 0

    def row(self, counts, depth, exact_cases=True):
        self.row_calls += 1
        key = (int(depth), bool(exact_cases), tuple(sorted(map(int, counts))))
        if key in self.evidence_cache:
            self.row_hits += 1
            self.evidence_cache.move_to_end(key)
            return self.evidence_cache[key]
        value = super().row(key[2], depth, exact_cases=exact_cases)
        self.evidence_cache[key] = value
        if len(self.evidence_cache) > self.max_cached_rows:
            self.evidence_cache.popitem(last=False)
        return value

    def cache_info(self):
        return dict(row_calls=self.row_calls, row_hits=self.row_hits,
                    cached_row_evidences=len(self.evidence_cache),
                    cached_phi_rows=len(self.cache),
                    phi_cache_bytes=sum(row.nbytes for row in self.cache.values()))


def _tokens(values, d):
    values = np.asarray(values)
    if (values.ndim != 1 or values.dtype.kind not in "iu"
            or np.any(values < 0) or np.any(values >= d)):
        raise ValueError(f"tokens must be an integer vector in 0..{d - 1}")
    return values.astype(np.int64, copy=False)


def chain_log_marginal(tokens, d, kernel):
    """Log normalized sequence probability, averaged over kernel.depths."""
    tokens = _tokens(tokens, d)
    if not len(tokens):
        return 0.
    counts = np.zeros((d, d), dtype=np.int64)
    np.add.at(counts, (tokens[:-1], tokens[1:]), 1)
    occupied = counts[counts.sum(axis=1) > 0]
    by_depth = np.array([sum(kernel.row(row, depth) for row in occupied)
                         for depth in kernel.depths])
    return float(-np.log(d) + logsumexp(by_depth) - np.log(len(kernel.depths)))


def product_log_marginal(tokens, d1, d2, kernel1, kernel2=None):
    """Independent coordinate evidence, with a separate depth per factor."""
    tokens = _tokens(tokens, d1 * d2)
    kernel2 = kernel1 if kernel2 is None else kernel2
    return (chain_log_marginal(tokens // d2, d1, kernel1)
            + chain_log_marginal(tokens % d2, d2, kernel2))


class FamilyEvidence:
    """Reusable full-prior evidence provider; observed tokens are its only data."""

    def __init__(self, family, *, bounds=(-30., 30.), step=1 / 64):
        if family not in ("product", "uniform_base"):
            raise ValueError("only product and uniform_base are implemented")
        self.family = family
        depths = range(1, 9) if family == "product" else range(9)
        self.kernel = CachedBatchEvidence(depths, bounds, step)

    def log_joint(self, tokens):
        if self.family == "product":
            return product_log_marginal(tokens, 8, 8, self.kernel)
        return chain_log_marginal(tokens, 64, self.kernel)

    def cumulative_losses(self, tokens, prefixes):
        return family_evidence(tokens, prefixes, self.family, self.kernel)


def family_evidence(tokens, prefixes, family, kernel):
    tokens = _tokens(tokens, 64)
    prefixes = np.asarray(prefixes)
    if (prefixes.ndim != 1 or not len(prefixes) or prefixes.dtype.kind not in "iu"
            or np.any(prefixes < 1) or np.any(prefixes >= len(tokens))
            or np.any(np.diff(prefixes) <= 0)):
        raise ValueError("prefixes must be increasing integer prediction times")
    if family == "uniform_base":
        expected_depths = tuple(range(9))
        calculate = lambda x: chain_log_marginal(x, 64, kernel)
    elif family == "product":
        expected_depths = tuple(range(1, 9))
        calculate = lambda x: product_log_marginal(x, 8, 8, kernel)
    else:
        raise ValueError("only product and uniform_base are implemented")
    if tuple(kernel.depths) != expected_depths:
        raise ValueError("kernel depths do not match the family's training prior")
    log_marginals = np.array([calculate(tokens[:int(t) + 1]) for t in prefixes])
    losses = -(log_marginals + np.log(64)) / LN2
    if not np.isfinite(log_marginals).all() or np.any(losses < -1e-8):
        raise FloatingPointError("invalid cumulative evidence")
    return log_marginals, losses


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--family", choices=("product", "uniform_base"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-seq", type=int)
    parser.add_argument("--bounds", nargs=2, type=float, default=(-30., 30.))
    parser.add_argument("--step", type=float, default=1 / 64)
    parser.add_argument("--chunk", type=int, default=16)
    args = parser.parse_args()
    if (args.chunk < 1 or args.step <= 0 or not np.isfinite(args.step)
            or not np.isfinite(args.bounds).all() or args.bounds[0] >= args.bounds[1]):
        parser.error("invalid chunk, grid bounds, or grid spacing")
    if not os.environ.get("LSA_STORE") or not os.environ.get("PMM_SRC"):
        parser.error("set LSA_STORE and PMM_SRC to the read-only anchor store")
    with np.load(args.tokens, allow_pickle=False) as saved:
        tokens, prefixes = saved["tokens"], saved["prefixes"]
        identities = saved["sequence_seed"]
    if tokens.ndim != 2 or not 1 <= (args.n_seq or len(tokens)) <= len(tokens):
        parser.error("n-seq must select a nonempty initial subset of frozen sequences")
    if args.n_seq is not None and args.n_seq < 1:
        parser.error("n-seq must be positive")
    count = len(tokens) if args.n_seq is None else args.n_seq
    tokens, identities = tokens[:count], identities[:count]
    depths = list(range(1, 9)) if args.family == "product" else list(range(9))
    backend = Path(os.environ["PMM_SRC"]) / "product_model_with_memory/universal_tables.py"
    config = dict(schema=SCHEMA, family=args.family, depths=depths, prior="uniform",
                  n_seq=count, n_tokens=tokens.shape[1], prefixes=prefixes.tolist(),
                  input_path=str(args.tokens.resolve()), input_sha256=sha256(args.tokens),
                  bounds=list(args.bounds), step=args.step,
                  evidence_kind="normalized_sequence_marginal",
                  loss_convention="tokens[1:t+1]; first token unscored",
                  store=str(Path(os.environ["LSA_STORE"]).resolve()),
                  code_sha256={name: sha256(Path(__file__).with_name(name)) for name in
                               ("mixture_family_evidence.py", "validate_shared_depth.py", "phi_source.py")},
                  backend_sha256=sha256(backend))
    config_json = json.dumps(config, sort_keys=True, allow_nan=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_name(args.out.name + ".part.npz")
    done = 0
    marginals = np.empty((count, len(prefixes)))
    losses = np.empty_like(marginals)
    for existing in (args.out, partial):
        if existing.exists():
            with np.load(existing, allow_pickle=False) as saved:
                if str(saved["config_json"]) != config_json:
                    raise ValueError("output provenance changed; select a new output path")
                done = int(saved["done"])
                if not 0 <= done <= count:
                    raise ValueError("invalid completed sequence count")
                if (saved["log_marginals"].shape != (done, len(prefixes))
                        or saved["cumulative_loss_bits"].shape != (done, len(prefixes))
                        or not np.array_equal(saved["sequence_seed"], identities[:done])):
                    raise ValueError("inconsistent saved evidence arrays")
                marginals[:done] = saved["log_marginals"]
                losses[:done] = saved["cumulative_loss_bits"]
            if existing == args.out:
                if done != count:
                    raise ValueError("final output is incomplete")
                print(f"Verified existing {args.out}", flush=True)
                return
            break
    kernel = CachedBatchEvidence(depths, args.bounds, args.step)
    started = time.monotonic()
    while done < count:
        stop = min(done + args.chunk, count)
        for i in range(done, stop):
            marginals[i], losses[i] = family_evidence(tokens[i], prefixes, args.family, kernel)
        done = stop
        temporary = partial.with_name(partial.name + ".tmp.npz")
        np.savez_compressed(temporary, config_json=config_json, done=done,
                            log_marginals=marginals[:done], cumulative_loss_bits=losses[:done],
                            prefixes=prefixes, sequence_seed=identities[:done])
        temporary.replace(partial)
        print(json.dumps(dict(done=done, n_seq=count, elapsed_seconds=time.monotonic() - started,
                              cache=kernel.cache_info())), flush=True)
    partial.replace(args.out)


if __name__ == "__main__":
    main()
