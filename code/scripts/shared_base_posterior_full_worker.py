"""Run the established posterior worker with periodic lossless checkpoints.

Only checkpoint scheduling changes. The sampler, RNG sequence, draw budget,
numerical diagnostics and report schema remain in shared_base_curve_reference.
The full-cohort controller independently audits each completed draw budget.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from pathlib import Path
import sys
import time

import shared_base_curve_reference as reference


@contextmanager
def periodic_checkpoints(path, interval_seconds=300., clock=time.monotonic):
    """Save committed draw blocks, preserving the original checkpoint format."""
    if interval_seconds <= 0:
        raise ValueError('Checkpoint interval must be positive')
    cls = reference.CurveChainRun
    inherited = 'extend' not in cls.__dict__
    original = cls.extend
    last_saved = [clock()]

    def extend(self, draws, *, block_draws=2048, progress=None):
        def report(event):
            if progress is not None:
                progress(event)
            # The established worker already saves at burn/chain completion.
            standard_save = (event['phase'] == 'burn' or
                             event.get('draws') == event.get('requested_draws'))
            if standard_save and progress is not None:
                last_saved[0] = clock()
            elif event['phase'] == 'draws' and clock() - last_saved[0] >= interval_seconds:
                self.save(path)
                last_saved[0] = clock()
        return original(self, draws, block_draws=block_draws, progress=report)

    cls.extend = extend
    try:
        yield
    finally:
        if inherited:
            del cls.extend
        else:
            cls.extend = original


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if '-h' in argv or '--help' in argv:
        reference.main()
        return
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--index', type=int, required=True)
    parser.add_argument('--prefix', type=int, required=True)
    parser.add_argument('--alpha-order', type=int, default=16)
    args, _ = parser.parse_known_args(argv)
    checkpoint = (args.out / args.input.name / f'alpha{args.alpha_order}' /
                  f'seq_{args.index:06d}_t{args.prefix:04d}_chains.npz')
    previous = sys.argv
    try:
        sys.argv = [str(Path(reference.__file__)), *argv]
        with periodic_checkpoints(checkpoint):
            reference.main()
    finally:
        sys.argv = previous


if __name__ == '__main__':
    main()
