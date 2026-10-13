"""Conditional-source B versus U curves, retaining every next-token vector.

The source has a fixed uniform base and fixed source depth. B retains the
training depth prior 0..8; U retains the original coding prior 1..lmax_of(d).
X1 is supplied free and updates U. Prefix t predicts token X(t+1).

One immutable NPZ per independent source draw contains the observed-label
transition matrix, complete path, full float64 predictions, conditional
oracle regrets, and observed losses. Increasing --n-seq reuses prior draws;
changing numerical or source inputs requires a new output directory.
This driver does not locate crossings or aggregate a selected subset.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
SCHEMA = "conditional-crossing-sequence-v1"
SEED_TAG = "uniform-base-conditional-crossing-v1"
ARRAYS = {
    "tokens", "initial_probs", "transition_probs", "permutation", "prefix_lengths",
    "cumulative_prefix_lengths", "targets", "oracle_probs", "bayes_probs", "unigram_probs",
    "bayes_regret_bits", "unigram_regret_bits", "instantaneous_gap_bits", "cumulative_gap_bits",
    "bayes_loss_bits", "unigram_loss_bits", "oracle_loss_bits", "observed_gap_bits",
    "observed_cumulative_gap_bits", "bayes_sum_error", "unigram_sum_error", "oracle_sum_error",
    "bayes_depth_posterior", "bayes_joint_log_evidence",
    "conditioned_bayes_probs", "conditioned_to_unigram_kl_bits", "conditioned_to_bayes_kl_bits",
    "conditioned_gap_bits", "conditioned_cumulative_gap_bits", "conditioned_loss_bits", "conditioned_sum_error",
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_hash(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256(value.dtype.str.encode())
    digest.update(json.dumps(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def configure_backend(store, src):
    """Reuse the settled production guard before importing moment tables."""
    if not store or not src:
        raise ValueError("provide LSA_STORE and PMM_SRC or explicit store/src paths")
    store, src = Path(store).expanduser().resolve(), Path(src).expanduser().resolve()
    if not (store / "anchors.json").is_file() or not (store / "manifest.json").is_file():
        raise RuntimeError("a sealed production anchor store with its manifest is required")

    def imported_mode():
        module = sys.modules.get("product_model_with_memory.universal_tables")
        if module is not None:
            for key, expected in (("_LADDER_EVERY", 1), ("_LADDER_DEGREE", 11), ("_SADDLE_MIN_L", 54)):
                if getattr(module, key, None) != expected:
                    raise RuntimeError("universal_tables was already imported outside production mode")

    imported_mode()
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from product_model_with_memory.production_coding import configure_production_tables
    configure_production_tables(store)
    imported_mode()
    os.environ["LSA_STORE"], os.environ["PMM_SRC"] = str(store), str(src)
    return store, src


def build_kernels(d, n_ctx, store, src, u_lo=-30.0, u_hi=30.0):
    """Return (SharedDepthLSA B, SeqLSA U), sharing one immutable allocation."""
    from lsa_seq import SeqLSA, lmax_of
    from shared_depth import SharedDepthLSA
    d, n_ctx = _integer(d, "d", 2), _integer(n_ctx, "n_ctx", 1)
    if not np.isfinite([u_lo, u_hi]).all() or u_lo >= u_hi:
        raise ValueError("invalid integration bounds")
    configure_backend(store, src)
    maximum = max(8, lmax_of(d))
    common = SeqLSA(d, depths=range(maximum + 1), rmax=n_ctx + 1, u_lo=u_lo, u_hi=u_hi)
    for value in (common.u, common.D, common.LN0, common.lgam):
        value.setflags(write=False)

    def view(first, last):
        kernel = copy.copy(common)
        kernel.depths = list(range(first, last + 1))
        kernel.Lmax = last
        kernel.D = common.D[:, first:last + 1, :]
        kernel.LN0 = common.LN0[first:last + 1]
        return kernel

    return SharedDepthLSA(view(0, 8)), view(1, lmax_of(d))


def build_conditioned_bayes(bayes, source_depth):
    """Diagnostic Bayes rule conditional on source depth, sharing B's tables."""
    from shared_depth import SharedDepthLSA
    source_depth = _integer(source_depth, "source_depth")
    matches = np.flatnonzero(np.asarray(bayes.kernel.depths) == source_depth)
    if len(matches) != 1:
        raise ValueError("source depth must be present in the Bayes kernel")
    index = int(matches[0])
    kernel = copy.copy(bayes.kernel)
    kernel.depths = [source_depth]
    kernel.Lmax = source_depth
    kernel.D = bayes.kernel.D[:, index:index + 1, :]
    kernel.LN0 = bayes.kernel.LN0[index:index + 1]
    return SharedDepthLSA(kernel)


def draw_source(d, n_ctx, source_depth, sequence_seed):
    """Mirror Source.batch, retaining the matrix after the same relabeling."""
    from source import Source
    d, n_ctx = _integer(d, "d", 2), _integer(n_ctx, "n_ctx", 1)
    source_depth = _integer(source_depth, "source_depth")
    sequence_seed = _integer(sequence_seed, "sequence_seed")
    source = Source(d, n_ctx + 1, alpha=0.0, ls=source_depth, permute=True)
    rng = np.random.default_rng(sequence_seed)
    alpha, depth = source.draw_params(1, rng)
    q, transition = source.matrices(alpha, depth, rng)
    original = source.sequences(q, transition, rng)[0]
    permutation = rng.permutation(d)
    inverse = np.argsort(permutation)
    return dict(tokens=np.asarray(permutation[original], dtype="<i8"),
                initial_probs=np.asarray(q[0, inverse], dtype="<f8"),
                transition_probs=np.asarray(transition[0][inverse][:, inverse], dtype="<f8"),
                permutation=np.asarray(permutation, dtype="<i8"))


def _probabilities(value, name, tolerance, *, positive):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim < 2 or value.shape[-1] < 2 or value.shape[-2] < 1 or not np.isfinite(value).all():
        raise ValueError(f"{name} must have shape (..., K>=1, d>=2) and finite entries")
    if (np.any(value <= 0) if positive else np.any(value < 0)):
        raise ValueError(f"{name} has invalid probability entries")
    error = value.sum(axis=-1) - 1.0
    if np.max(np.abs(error)) > tolerance:
        raise ValueError(f"{name} is not normalized; raw maximum error {np.max(np.abs(error)):.3g}")
    return value, error


def conditional_metrics(bayes_probs, unigram_probs, oracle_probs, targets, *, prob_tol=1e-7):
    """Per-sequence quantities, never averaging positions or independent draws.

    Inputs have shape (..., K, d), targets (..., K). g=r_U-r_B and its
    cumulative sum G includes a leading zero, corresponding to observed
    prefix length 1. Regrets are KL(true current row || predictive vector).
    No probability normalization or clipping is performed.
    """
    if not np.isfinite(prob_tol) or prob_tol <= 0:
        raise ValueError("prob_tol must be finite and positive")
    b, b_error = _probabilities(bayes_probs, "Bayes", prob_tol, positive=True)
    u, u_error = _probabilities(unigram_probs, "unigram", prob_tol, positive=True)
    oracle, oracle_error = _probabilities(oracle_probs, "oracle", prob_tol, positive=False)
    target = np.asarray(targets)
    if b.shape != u.shape or b.shape != oracle.shape:
        raise ValueError("prediction and oracle shapes must agree")
    if target.shape != b.shape[:-1] or target.dtype.kind not in "iu" or np.any(target < 0) or np.any(target >= b.shape[-1]):
        raise ValueError("targets must be integer indices aligned with the predictions")
    log_b, log_u = np.log2(b), np.log2(u)
    log_oracle = np.zeros_like(oracle)
    np.log2(oracle, out=log_oracle, where=oracle > 0)
    r_b = np.sum(oracle * (log_oracle - log_b), axis=-1)
    r_u = np.sum(oracle * (log_oracle - log_u), axis=-1)
    g = r_u - r_b

    def selected(prob):
        value = np.take_along_axis(prob, target[..., None], axis=-1)[..., 0]
        if np.any(value <= 0):
            raise ValueError("observed target has zero oracle probability")
        return -np.log2(value)

    b_loss, u_loss, oracle_loss = selected(b), selected(u), selected(oracle)
    observed_gap = u_loss - b_loss

    def cumulative(value):
        return np.concatenate((np.zeros(value.shape[:-1] + (1,)), np.cumsum(value, axis=-1)), axis=-1)

    return dict(bayes_regret_bits=r_b, unigram_regret_bits=r_u, instantaneous_gap_bits=g,
                cumulative_gap_bits=cumulative(g), bayes_loss_bits=b_loss, unigram_loss_bits=u_loss,
                oracle_loss_bits=oracle_loss, observed_gap_bits=observed_gap,
                observed_cumulative_gap_bits=cumulative(observed_gap), bayes_sum_error=b_error,
                unigram_sum_error=u_error, oracle_sum_error=oracle_error)


def evaluate_sequence(tokens, transition_probs, bayes, unigram_kernel, *, prob_tol=1e-7, loss_tol=2e-6,
                      conditioned_bayes=None):
    """Return (arrays, checks) for one path using fresh predictor states."""
    x = np.asarray(tokens)
    d = bayes.d
    if x.ndim != 1 or x.size < 2 or x.dtype.kind not in "iu" or np.any(x < 0) or np.any(x >= d):
        raise ValueError("tokens must be a nonempty integer path with at least two tokens")
    if unigram_kernel.d != d:
        raise ValueError("Bayes and unigram alphabets differ")
    matrix, _ = _probabilities(transition_probs, "transition matrix", 1e-12, positive=True)
    if matrix.shape != (d, d):
        raise ValueError("transition matrix must have shape (d,d)")
    if not np.isfinite(loss_tol) or loss_tol <= 0:
        raise ValueError("loss_tol must be finite and positive")
    positions = np.arange(1, len(x), dtype=np.int64)
    reference = bayes.run(x, full_probs=True, positions=positions, depth_posteriors=True)
    if not np.array_equal(reference["positions"], positions):
        raise ValueError("Bayes prediction positions are misaligned")
    unigram = unigram_kernel.new_table()
    first_unigram_loss = unigram.observe(int(x[0]))
    unigram_probs, unigram_loss = np.empty((len(positions), d)), np.empty(len(positions))
    for i, position in enumerate(positions):
        unigram_probs[i] = unigram.predictive()
        unigram_loss[i] = unigram.observe(int(x[position]))
    oracle = matrix[x[:-1]]
    metrics = conditional_metrics(reference["probs"], unigram_probs, oracle, x[1:], prob_tol=prob_tol)
    b_error = float(np.max(np.abs(metrics["bayes_loss_bits"] - reference["loss"][1:])))
    u_error = float(np.max(np.abs(metrics["unigram_loss_bits"] - unigram_loss)))
    # This source has a uniform first-token law. It supplies no depth evidence.
    first_b_error = float(np.max(np.abs(reference["probs"][0] - 1.0 / d)))
    first_depth_error = float(np.max(np.abs(reference["depth_posterior"][1] - bayes.prior)))
    evidence_error = float(np.max(np.abs(np.cumsum(reference["loss"][1:])
                            + (reference["log_evidence"][1:] - reference["log_evidence"][0]) / np.log(2))))
    checks = dict(bayes_observed_loss_error_bits=b_error, unigram_observed_loss_error_bits=u_error,
                  first_bayes_uniform_error=first_b_error, first_bayes_depth_prior_error=first_depth_error,
                  free_first_unigram_loss_error_bits=abs(float(first_unigram_loss) - np.log2(d)),
                  bayes_evidence_telescope_error_bits=evidence_error,
                  max_bayes_normalization_error=float(np.max(np.abs(metrics["bayes_sum_error"]))),
                  max_unigram_normalization_error=float(np.max(np.abs(metrics["unigram_sum_error"]))),
                  source_averaged_first_gap_bits=float(np.mean(np.log2((1.0 / d) / unigram_probs[0]))),
                  probability_tolerance=prob_tol, loss_tolerance_bits=loss_tol)
    if (max(b_error, u_error, evidence_error) > loss_tol or max(first_b_error, first_depth_error) > prob_tol
            or checks["free_first_unigram_loss_error_bits"] > loss_tol):
        raise ValueError(f"prediction alignment or free-first-token checks failed: {checks}")
    arrays = dict(prefix_lengths=positions, cumulative_prefix_lengths=np.arange(1, len(x) + 1),
                  targets=np.asarray(x[1:], dtype="<i8"), oracle_probs=oracle,
                  bayes_probs=np.asarray(reference["probs"], dtype="<f8"), unigram_probs=unigram_probs,
                  bayes_depth_posterior=np.asarray(reference["depth_posterior"][1:], dtype="<f8"),
                  bayes_joint_log_evidence=np.asarray(reference["log_evidence"], dtype="<f8"), **metrics)
    if conditioned_bayes is not None:
        if conditioned_bayes.d != d or len(conditioned_bayes.depths) != 1:
            raise ValueError("diagnostic requires one fixed source depth and the same alphabet")
        conditional = conditioned_bayes.run(x, full_probs=True, positions=positions)
        if not np.array_equal(conditional["positions"], positions):
            raise ValueError("conditional Bayes prediction positions are misaligned")
        averaged = conditional_metrics(reference["probs"], unigram_probs, conditional["probs"], x[1:], prob_tol=prob_tol)
        alignment = float(np.max(np.abs(averaged["oracle_loss_bits"] - conditional["loss"][1:])))
        if alignment > loss_tol:
            raise ValueError("conditional Bayes observed-token loss alignment failed")
        arrays.update(conditioned_bayes_probs=np.asarray(conditional["probs"], dtype="<f8"),
                      conditioned_to_unigram_kl_bits=averaged["unigram_regret_bits"],
                      conditioned_to_bayes_kl_bits=averaged["bayes_regret_bits"],
                      conditioned_gap_bits=averaged["instantaneous_gap_bits"],
                      conditioned_cumulative_gap_bits=averaged["cumulative_gap_bits"],
                      conditioned_loss_bits=averaged["oracle_loss_bits"],
                      conditioned_sum_error=averaged["oracle_sum_error"])
        checks.update(conditioned_observed_loss_error_bits=alignment,
                      max_conditioned_normalization_error=float(np.max(np.abs(averaged["oracle_sum_error"]))))
    return arrays, checks


def make_identity(d, n_ctx, source_depth, master_seed, index, store, src,
                  u_lo=-30.0, u_hi=30.0, prob_tol=1e-7, loss_tol=2e-6):
    from lsa_seq import lmax_of
    from phi_source import H
    from source import seed_of
    d, n_ctx = _integer(d, "d", 2), _integer(n_ctx, "n_ctx", 1)
    source_depth, master_seed, index = (_integer(source_depth, "source_depth"),
                                      _integer(master_seed, "master_seed"), _integer(index, "index"))
    if source_depth > 8:
        raise ValueError("initial conditional-source protocol requires depth in 0..8")
    store, src = configure_backend(store, src)
    code = {name: sha_file(HERE / name) for name in
            ("eval_conditional_crossing.py", "source.py", "lsa_seq.py", "shared_depth.py", "phi_source.py")}
    for name in ("universal_tables.py", "production_coding.py", "mellin.py", "kernel.py"):
        code["pmm/" + name] = sha_file(src / "product_model_with_memory" / name)
    return dict(schema=SCHEMA, sequence_index=index, master_seed=master_seed,
                sequence_seed=seed_of(SEED_TAG, master_seed, index), seed_tag=SEED_TAG,
                source=dict(family="uniform-base LSA Markov", d=d, alpha=0.0, source_depth=source_depth,
                            n_tokens=n_ctx + 1, permute=True, initial_law="uniform, not stationary",
                            draw_rule="Source.draw_params, matrices, sequences, then permutation"),
                prediction=dict(bayes_depths=list(range(9)), bayes_depth_prior=[1 / 9] * 9,
                                unigram_depths=list(range(1, lmax_of(d) + 1)),
                                unigram_depth_prior=[1 / lmax_of(d)] * lmax_of(d),
                                source_condition_not_given_to_predictors=True, first_token_scored=False,
                                diagnostic_conditioned_depth=source_depth,
                                diagnostic_role="fixed-source-depth Bayes; does not replace B's training prior",
                                u_lo=float(u_lo), u_hi=float(u_hi), step=float(H), rmax=n_ctx + 1,
                                prob_tol=prob_tol, loss_tol=loss_tol),
                backend=dict(store=str(store), src=str(src), read_only=True,
                             anchors_sha256=sha_file(store / "anchors.json"),
                             manifest_sha256=sha_file(store / "manifest.json"),
                             production_settings={name: os.environ[name] for name in
                               ("PMM_UNIVERSAL_TABLES", "PMM_PHI_LADDER_EVERY", "PMM_PHI_LADDER_DEGREE", "PMM_PHI_SADDLE_MIN_L")}),
                code_sha256=code, runtime=dict(python=list(sys.version_info[:3]), numpy=np.__version__,
                                               scipy=importlib.metadata.version("scipy")))


def validate_record(arrays, metadata, expected_identity=None):
    """Check identity, hashes, indexing, and recompute metrics from full vectors."""
    identity = metadata["identity"]
    if metadata["cache_id"] != fingerprint(identity) or (expected_identity is not None and identity != expected_identity):
        raise ValueError("cache inputs changed; use a new output directory")
    if set(arrays) != ARRAYS or set(metadata["array_sha256"]) != ARRAYS:
        raise ValueError("cache array schema is incomplete or unexpected")
    for key, array in arrays.items():
        if array_hash(array) != metadata["array_sha256"][key]:
            raise ValueError(f"cache array checksum failed: {key}")
    source, prediction = identity["source"], identity["prediction"]
    n, d = source["n_tokens"], source["d"]
    x = np.asarray(arrays["tokens"])
    if x.shape != (n,) or x.dtype.kind not in "iu" or np.any(x < 0) or np.any(x >= d):
        raise ValueError("invalid cached token path")
    if (not np.array_equal(arrays["prefix_lengths"], np.arange(1, n))
            or not np.array_equal(arrays["cumulative_prefix_lengths"], np.arange(1, n + 1))
            or not np.array_equal(arrays["targets"], x[1:])):
        raise ValueError("cached prediction indexing mismatch")
    if not np.array_equal(np.sort(arrays["permutation"]), np.arange(d)):
        raise ValueError("invalid cached relabeling")
    matrix, _ = _probabilities(arrays["transition_probs"], "cached transition matrix", 1e-12, positive=True)
    if (matrix.shape != (d, d) or arrays["initial_probs"].shape != (d,)
            or not np.allclose(arrays["initial_probs"], 1 / d, atol=1e-15, rtol=0)
            or not np.array_equal(arrays["oracle_probs"], matrix[x[:-1]])):
        raise ValueError("cached source law or current-row alignment mismatch")
    if arrays["bayes_probs"].shape != (n - 1, d) or arrays["unigram_probs"].shape != (n - 1, d):
        raise ValueError("cached predictive vectors have incorrect shape")
    metrics = conditional_metrics(arrays["bayes_probs"], arrays["unigram_probs"], arrays["oracle_probs"],
                                  arrays["targets"], prob_tol=prediction["prob_tol"])
    for key, value in metrics.items():
        if not np.array_equal(value, arrays[key]):
            raise ValueError(f"cached metric readback mismatch: {key}")
    averaged = conditional_metrics(arrays["bayes_probs"], arrays["unigram_probs"], arrays["conditioned_bayes_probs"],
                                   arrays["targets"], prob_tol=prediction["prob_tol"])
    for output, calculated in (("conditioned_to_unigram_kl_bits", "unigram_regret_bits"),
                               ("conditioned_to_bayes_kl_bits", "bayes_regret_bits"),
                               ("conditioned_gap_bits", "instantaneous_gap_bits"),
                               ("conditioned_cumulative_gap_bits", "cumulative_gap_bits"),
                               ("conditioned_loss_bits", "oracle_loss_bits"),
                               ("conditioned_sum_error", "oracle_sum_error")):
        if not np.array_equal(arrays[output], averaged[calculated]):
            raise ValueError(f"cached conditional diagnostic mismatch: {output}")
    posterior = arrays["bayes_depth_posterior"]
    if (posterior.shape != (n - 1, len(prediction["bayes_depths"])) or not np.isfinite(posterior).all()
            or np.any(posterior < 0) or np.max(np.abs(posterior.sum(axis=1) - 1)) > prediction["prob_tol"]
            or not np.allclose(posterior[0], prediction["bayes_depth_prior"], atol=1e-14, rtol=0)):
        raise ValueError("cached Bayes depth posterior invalid")
    evidence = arrays["bayes_joint_log_evidence"]
    if (evidence.shape != (n,) or not np.isfinite(evidence).all()
            or np.max(np.abs(np.cumsum(arrays["bayes_loss_bits"])
                            + (evidence[1:] - evidence[0]) / np.log(2))) > prediction["loss_tol"]):
        raise ValueError("cached Bayesian evidence/loss mismatch")
    return True


def save_record(path, arrays, metadata):
    """Publish once, atomically; an existing result is never overwritten."""
    path = Path(path)
    validate_record(arrays, metadata)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    payload = canonical(metadata)
    try:
        with temporary.open("xb") as stream:
            np.savez_compressed(stream, **arrays, metadata=np.asarray(payload),
                                metadata_sha256=np.asarray(hashlib.sha256(payload.encode()).hexdigest()))
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_record(path, expected_identity=None):
    with np.load(path, allow_pickle=False) as stored:
        if set(stored.files) != ARRAYS | {"metadata", "metadata_sha256"}:
            raise ValueError("invalid cache schema")
        payload = str(stored["metadata"].item())
        if hashlib.sha256(payload.encode()).hexdigest() != str(stored["metadata_sha256"].item()):
            raise ValueError("cache metadata checksum failed")
        metadata = json.loads(payload)
        arrays = {key: stored[key].copy() for key in ARRAYS}
    validate_record(arrays, metadata, expected_identity)
    return arrays, metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("output/conditional_crossing/uniform_d64_L2"))
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--source-depth", type=int, default=2)
    parser.add_argument("--n-seq", type=int, default=128)
    parser.add_argument("--n-ctx", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--store", type=Path, default=os.environ.get("LSA_STORE"))
    parser.add_argument("--pmm-src", type=Path, default=os.environ.get("PMM_SRC"))
    parser.add_argument("--u-lo", type=float, default=-30.0)
    parser.add_argument("--u-hi", type=float, default=30.0)
    args = parser.parse_args(argv)
    _integer(args.n_seq, "n_seq", 1)
    _integer(args.start_index, "start_index")
    kernels = None
    started = time.monotonic()
    for index in range(args.start_index, args.start_index + args.n_seq):
        identity = make_identity(args.d, args.n_ctx, args.source_depth, args.seed, index,
                                 args.store, args.pmm_src, args.u_lo, args.u_hi)
        path = args.out / f"seq_{index:06d}.npz"
        if path.exists():
            load_record(path, identity)
            print(json.dumps(dict(index=index, reused=True, path=str(path))), flush=True)
            continue
        if kernels is None:
            before = time.monotonic()
            kernels = build_kernels(args.d, args.n_ctx, args.store, args.pmm_src, args.u_lo, args.u_hi)
            print(json.dumps(dict(kernel_setup_seconds=time.monotonic() - before,
                                  shared_difference_table_bytes=kernels[1].D.base.nbytes)), flush=True)
        before = time.monotonic()
        source = draw_source(args.d, args.n_ctx, args.source_depth, identity["sequence_seed"])
        result, checks = evaluate_sequence(source["tokens"], source["transition_probs"], *kernels,
                                          conditioned_bayes=build_conditioned_bayes(kernels[0], args.source_depth))
        arrays = dict(**source, **result)
        metadata = dict(identity=identity, cache_id=fingerprint(identity), checks=checks,
                        array_sha256={key: array_hash(value) for key, value in arrays.items()},
                        elapsed_seconds=time.monotonic() - before)
        save_record(path, arrays, metadata)
        load_record(path, identity)
        print(json.dumps(dict(index=index, reused=False, path=str(path), checks=checks,
                              elapsed_seconds=metadata["elapsed_seconds"])), flush=True)
    print(json.dumps(dict(completed=args.n_seq, elapsed_seconds=time.monotonic() - started)), flush=True)


if __name__ == "__main__":
    main()
