"""Combine the three frozen validated endpoints with the separately repaired fourth.

All four prespecified source draws and both models are required. The
repaired reference must pass its full independent raw-data audit first.
This is an endpoint pilot, not a representative family learning curve.
"""
from copy import deepcopy
import argparse
import json
from pathlib import Path

from shared_base_full_context_summary import aggregate_sources, sha_file


def combine(old_path, repair_path):
    old = json.loads(old_path.read_text())
    repair = json.loads(repair_path.read_text())
    if not (repair['passed'] and repair['error'] is None
            and repair['sequence_index'] == 3 and repair['prefix_length'] == 1024
            and old['n_sources'] == 4 and old['prefix_length'] == 1024):
        raise ValueError('the complete fourth-source audit must pass before merging')
    if any(not c['passed'] and not c['name'].startswith(('reference3:', 'source3:'))
           for c in old['checks']):
        raise ValueError('a frozen first-three-source check failed')
    if repair['inputs']['cohort'] != old['provenance']['inputs']['cohort']:
        raise ValueError('the repaired reference belongs to a different cohort')
    for inputs in (old['provenance']['inputs'], repair['inputs']):
        for item in inputs.values():
            if sha_file(Path(item['path'])) != item['sha256']:
                raise ValueError(f"input changed: {item['path']}")
    rows = deepcopy([r for r in old['rows'] if r['sequence_index'] < 3])
    if len(rows) != 6 or not all(r['resolved'] for r in rows):
        raise ValueError('need both validated models for each of the first three sources')
    rows.extend(deepcopy(repair['rows']))
    aggregate = aggregate_sources(rows)
    if not aggregate['reported']:
        raise ValueError('no partial-cohort average is allowed')
    return dict(schema='shared-base-repaired-endpoint-pilot-v1', passed=True,
                n_sources=4, prefix_length=1024, rows=rows, aggregate=aggregate,
                provenance=dict(original_audit=dict(path=str(old_path.resolve()), sha256=sha_file(old_path)),
                                repair_audit=dict(path=str(repair_path.resolve()), sha256=sha_file(repair_path)),
                                script_sha256=sha_file(Path(__file__))),
                limitation='Four prespecified source draws at one time; numerical repair does not establish typical family performance.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path('results/transformer/shared_base_full_context')
    parser.add_argument('--old', type=Path, default=root/'pilot_2026-09-15.json')
    parser.add_argument('--repair', type=Path, default=root/'source3_surrogate_refined_audit_2026-09-15.json')
    parser.add_argument('--out', type=Path, default=root/'pilot_repaired_2026-09-15.json')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    result = combine(args.old, args.repair)
    if args.check_only:
        if json.loads(args.out.read_text()) != result:
            raise ValueError('saved combined pilot differs')
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(passed=result['passed'], aggregate=result['aggregate'])))


if __name__ == '__main__':
    main()
