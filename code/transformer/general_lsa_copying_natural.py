"""Paired head interventions on the general shared-base LSA source.

Uses the existing source/cohort generator and generic causal head instrument.
The source exponent and depth specify test conditions, never model inputs.
All heads are reported; selection must come from independent controlled probes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/scripts'))
from circuit_probe import run_probe
from model import GPT
from shared_base_conditional_cohort import load_source, prepare_setting


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, default=ROOT / 'output/general_lsa_copying_2026-09-17/natural')
    ap.add_argument('--n-seq', type=int, default=256)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--threads', type=int, default=4)
    args = ap.parse_args()
    if args.n_seq < 2 or args.n_seq > 256 or args.batch < 1:
        ap.error('Use 2..256 independent sequences and a positive batch size')
    torch.set_num_threads(args.threads)
    checkpoint = ROOT / 'results/transformer/runs/d64/model_100000.pt'
    config_path = checkpoint.parent / 'config.json'
    config = json.loads(config_path.read_text())
    assert config['alpha'] == [.5, 1.5] and config['ls'] == [0, 8]
    assert config['d'] == 64 and config['n_ctx'] == 1024 and not config['no_permute']
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    assert saved['step'] == 100000 and saved['cfg']['model'] == config['model']
    model = GPT(**config['model']).eval().double()
    model.load_state_dict(saved['model'])
    heads = [(l, h) for l in range(config['n_layer']) for h in range(config['n_head'])]
    names = ['intact'] + [f'off_L{l+1}H{h+1}' for l, h in heads]
    times = np.array([1, 4, 16, 32, 64, 128, 256, 512, 1024])
    protocol = dict(schema='general-lsa-natural-head-screen-v1', alpha=1., depths=[0, 2, 3],
                    d=64, n_sequences=args.n_seq, n_tokens=1025, names=names,
                    selection='Head selection uses independent controlled probes; this file reports all heads.',
                    model_prior=dict(alpha=[.5, 1.5], depths=list(range(9)), permutation='uniform'),
                    precision='float64 evaluation of frozen float32-trained parameter values',
                    observed_tokens=times.tolist(), first_token='supplied, not scored',
                    checkpoint_sha256=sha(checkpoint), config_sha256=sha(config_path),
                    code_sha256={str(p.relative_to(ROOT)): sha(p) for p in
                                 [Path(__file__), ROOT / 'code/transformer/circuit_probe.py',
                                  ROOT / 'code/transformer/model.py', ROOT / 'code/transformer/source.py',
                                  ROOT / 'code/scripts/shared_base_conditional_cohort.py']})
    args.out.mkdir(parents=True, exist_ok=True)
    pp = args.out / 'protocol.json'
    if pp.exists():
        assert json.loads(pp.read_text()) == protocol, 'Existing output has different provenance'
    else:
        pp.write_text(json.dumps(protocol, indent=2) + '\n')
    start = time.monotonic()
    for depth in (0, 2, 3):
        destination = args.out / f'a1_L{depth}.npz'
        if destination.exists():
            with np.load(destination, allow_pickle=False) as old:
                assert str(old['protocol_sha256']) == sha(pp)
            print(json.dumps(dict(depth=depth, status='verified_existing')), flush=True)
            continue
        if depth == 0:
            folder = ROOT / 'output/shared_base_conditional_curves_2026-09-15/a1_L0'
        else:
            folder = prepare_setting(args.out / 'sources', 1., depth, 20260930,
                                     [1, 4, 16, 64, 256, 1024], args.n_seq)
        records = [load_source(folder, i) for i in range(args.n_seq)]
        tokens = np.array([r['tokens'] for r in records], dtype=np.int64)
        base = np.array([r['oracle']['initial_probs'] for r in records])
        transition = np.array([r['oracle']['transition_probs'] for r in records])
        assert tokens.shape == (args.n_seq, 1025)
        assert np.max(np.abs(transition.sum(-1) - 1)) < 1e-12
        if depth == 0:
            assert np.array_equal(transition, np.broadcast_to(base[:, None], transition.shape))
        shape = (args.n_seq, len(names), 1024)
        ce, kl, actual = [np.empty(shape) for _ in range(3)]
        probabilities = np.empty((args.n_seq, len(names), len(times), 64))
        max_forward_error = max_causal_error = 0.
        for left in range(0, args.n_seq, args.batch):
            right = min(args.n_seq, left + args.batch)
            x = torch.from_numpy(tokens[left:right, :-1])
            truth = transition[left:right][np.arange(right-left)[:, None], tokens[left:right, :-1]]
            entropy = -np.sum(np.where(truth > 0, truth * np.log2(np.maximum(truth, 1e-300)), 0), -1)
            for j, name in enumerate(names):
                scales = None if j == 0 else {heads[j-1]: 0.}
                with torch.inference_mode():
                    logits = run_probe(model, x, head_scales=scales, return_cache=False)['logits']
                    logp = logits.log_softmax(-1).numpy()
                    if left == 0 and j == 0:
                        for block in model.blocks:
                            block.attn.manual = True
                        ref = model(x)
                        max_forward_error = float((ref - logits).abs().max())
                    if left == 0:
                        short = run_probe(model, x[:, :32], head_scales=scales, return_cache=False)['logits']
                        max_causal_error = max(max_causal_error, float((short[:, -1] - logits[:, 31]).abs().max()))
                ce[left:right, j] = -np.sum(truth * logp, -1) / np.log(2)
                kl[left:right, j] = ce[left:right, j] - entropy
                actual[left:right, j] = -np.take_along_axis(logp, tokens[left:right, 1:, None], axis=-1)[..., 0] / np.log(2)
                probabilities[left:right, j] = np.exp(logp[:, times-1])
            if right % 32 == 0 or right == args.n_seq:
                print(json.dumps(dict(depth=depth, completed=right, total=args.n_seq,
                                      elapsed_seconds=time.monotonic()-start)), flush=True)
        assert max_forward_error < 1e-10 and max_causal_error < 1e-10
        assert np.isfinite(ce).all() and np.min(kl) > -1e-10
        assert np.max(np.abs(probabilities.sum(-1) - 1)) < 1e-12
        temporary = destination.with_suffix('.part.npz')
        np.savez_compressed(temporary, tokens=tokens, ce_bits=ce, kl_bits=kl,
                            actual_loss_bits=actual, probabilities=probabilities,
                            times=times, names=np.array(names), base=base,
                            source_ids=np.array([r['source_id'] for r in records]),
                            source_files_sha256=np.array([sha(folder / 'data' / f'seq_{i:06d}.json') for i in range(args.n_seq)]),
                            source_folder=str(folder), protocol_sha256=sha(pp),
                            max_forward_error=max_forward_error, max_causal_error=max_causal_error)
        temporary.replace(destination)
        print(json.dumps(dict(depth=depth, status='complete', seconds=time.monotonic()-start,
                              forward_error=max_forward_error, causal_error=max_causal_error)), flush=True)


if __name__ == '__main__':
    main()
