"""Check the archived appendix figure release without numerical recomputation."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
RELEASE = ROOT / 'results/transformer/appendix_style_2026-09-24'


def main():
    manifest = json.loads((RELEASE / 'manifest.json').read_text())
    for group in ('outputs', 'builders'):
        for name, expected in manifest[group].items():
            path = ROOT / name
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != expected:
                raise SystemExit(f'Changed {group[:-1]}: {name}')
    expected_pdfs = {f'figure{i}.pdf' for i in range(3, 13)}
    assert {Path(p).name for p in manifest['outputs'] if p.endswith('.pdf')} == expected_pdfs
    print('Appendix Figures3–12: output and builder hashes match the release.')


if __name__ == '__main__':
    main()
