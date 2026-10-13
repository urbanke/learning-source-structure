"""Measure the candidate retrieval-key path on the frozen general-source cohort."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from circuit_probe import run_probe
from general_lsa_key_path import run_key_path
from model import GPT

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / 'output/general_lsa_copying_2026-09-17'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    torch.set_num_threads(4)
    selection = json.loads((RAW / 'discovery/selection.json').read_text())
    for path, expected in selection['provenance'].items():
        assert sha(ROOT / path) == expected
    checkpoint = ROOT / 'results/transformer/runs/d64/model_100000.pt'
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model = GPT(**saved['cfg']['model']).eval().double()
    model.load_state_dict(saved['model'])
    heads = [tuple(selection[k]) for k in ('candidate_head', 'control_head')]
    out = RAW / 'key_path_natural'
    out.mkdir(parents=True, exist_ok=True)
    protocol = dict(schema='general-lsa-key-natural-v1', checkpoint_sha256=sha(checkpoint),
                    code_sha256={str(p.relative_to(ROOT)):sha(p) for p in
                                 (Path(__file__), ROOT / 'code/transformer/general_lsa_key_path.py',
                                  ROOT / 'code/transformer/circuit_probe.py')},
                    selection_sha256=sha(RAW / 'discovery/selection.json'),
                    controlled_summary_sha256=sha(RAW / 'key_path_controlled/summary.json'),
                    candidate_head=list(heads[0]), control_head=list(heads[1]),
                    source=dict(d=64, alpha=1, depths=[0, 2, 3], n_sequences=256, n_tokens=1025),
                    precision='float64', intervention='Replace only second-layer keys by same-input donor keys after deleting one first-layer head; queries, values, residual and MLP retain intact inputs.')
    pp = out / 'protocol.json'
    if pp.exists():
        assert json.loads(pp.read_text()) == protocol
    else:
        pp.write_text(json.dumps(protocol, indent=2) + '\n')
    started = time.monotonic()
    for depth in (0, 2, 3):
        dest = out / f'a1_L{depth}.npz'
        if dest.exists():
            with np.load(dest) as a:
                assert str(a['protocol_sha256']) == sha(pp)
            continue
        source_path = RAW / 'natural' / f'a1_L{depth}.npz'
        with np.load(source_path, allow_pickle=False) as raw:
            tokens = raw['tokens']; times = raw['times']; base = raw['base']
            folder = Path(str(raw['source_folder']))
            rows = []
            for i, expected in enumerate(raw['source_files_sha256']):
                file = folder / 'data' / f'seq_{i:06d}.json'
                assert sha(file) == str(expected)
                rows.append(json.loads(file.read_text())['oracle']['transition_probs'])
            transition = np.asarray(rows)
            intact_ce = raw['ce_bits'][:, 0].copy()
            intact_probs = raw['probabilities'][:, 0].copy()
        ce = np.empty((256, 2, 1024)); kl = np.empty_like(ce)
        probability_tv = np.empty_like(ce)
        probs = np.empty((256, 2, len(times), 64))
        eligible = np.zeros((256, 1024), dtype=bool)
        seen = np.zeros((256, 64), dtype=bool)
        for t in range(1024):
            eligible[:, t] = seen[np.arange(256), tokens[:, t]]
            seen[np.arange(256), tokens[:, t]] = True
        for left in range(0, 256, 4):
            right = left+4
            x = torch.from_numpy(tokens[left:right, :-1])
            truth = transition[left:right][np.arange(4)[:, None], tokens[left:right, :-1]]
            entropy = -np.sum(truth * np.log2(np.maximum(truth, 1e-300)), -1)
            intact = run_probe(model, x, return_cache=False)['logits']
            p0 = intact.softmax(-1).numpy()
            np.testing.assert_allclose(p0[:, times-1], intact_probs[left:right], atol=1e-12, rtol=1e-12)
            for j, head in enumerate(heads):
                logits = run_key_path(model, x, donor_head=head)['logits']
                assert torch.equal(logits[:, 0], intact[:, 0])
                logp = logits.log_softmax(-1).numpy()
                p = np.exp(logp)
                ce[left:right, j] = -np.sum(truth*logp, -1) / np.log(2)
                kl[left:right, j] = ce[left:right, j] - entropy
                probability_tv[left:right, j] = .5*np.abs(p-p0).sum(-1)
                probs[left:right, j] = p[:, times-1]
            if right % 64 == 0:
                print(json.dumps(dict(depth=depth, completed=right, elapsed_seconds=time.monotonic()-started)), flush=True)
        assert np.isfinite(ce).all() and kl.min() > -1e-10
        assert np.max(np.abs(probs.sum(-1)-1)) < 1e-12
        np.savez_compressed(dest, ce_bits=ce, kl_bits=kl, probability_tv=probability_tv,
                            probabilities=probs, base=base, times=times, eligible=eligible,
                            intact_ce_bits=intact_ce, names=np.array(['candidate_keys_off', 'control_keys_off']),
                            source_sha256=sha(source_path), protocol_sha256=sha(pp))
        print(json.dumps(dict(depth=depth, status='complete', elapsed_seconds=time.monotonic()-started)), flush=True)


if __name__ == '__main__':
    main()
