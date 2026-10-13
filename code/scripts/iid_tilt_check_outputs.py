"""Reproduce or collect the tilted L3 campaign and export its reviewed RTF.

python code/scripts/iid_tilt_check_outputs.py --run --collect --report
python code/scripts/iid_tilt_check_figure.py
The --run option uses three workers; --collect alone reuses ignored raw data.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys


ALPHAS = (0, .5, .75, .9, 1, 1.25, 1.5, 2)


def commands(raw):
    for d in (256, 1024, 4096):
        for a in ALPHAS:
            out = raw / f'd{d}_a{a:g}_L3.json'
            yield [sys.executable, 'code/scripts/iid_layered_tilt_check.py',
                   '--d', str(d), '--alpha', str(a), '--reps',
                   '256' if d == 4096 else '512',
                   '--ks', '16,64,256,1024,4096,16384', '--out', str(out)]


def run(cmd):
    subprocess.run(cmd, check=True)


def collect(raw, out):
    cells = []
    for cmd in commands(raw):
        path = Path(cmd[-1])
        z = json.loads(path.read_text())
        assert z['complete'] and z['reps'] == z['requested_reps']
        assert len(z['k']) == 6 and z['L'] == 3
        cells.append(z)
    result = dict(date='2026-09-06',
                  target='Single iid normalized tilted distribution, matched Bayes code with q and L known',
                  cells=cells,
                  raw_store=f'{raw}/; ignored by git; commands in each cell reproduce draws')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2)+'\n')


def rich_text(source):
    def escape(text):
        parts = []
        for ch in text:
            if ch in '\\{}':
                parts.append('\\'+ch)
            elif ord(ch) > 127:
                n = ord(ch)
                parts.append('\\u'+str(n if n < 32768 else n-65536)+'?')
            else:
                parts.append(ch)
        return ''.join(parts)
    out = [r'{\rtf1\ansi\deff0{\fonttbl{\f0 Helvetica;}}',
           r'\paperw11906\paperh16838\margl1080\margr1080\margt1080\margb1080',
           r'\widowctrl\f0\fs22']
    for line in source.read_text().splitlines():
        if not line.strip() or line.startswith('| ---'):
            continue
        if line.startswith('|'):
            cells = [s.strip() for s in line.strip('|').split('|')]
            width = 9746//len(cells)
            out.append(r'\trowd\trgaph100\trleft0')
            for i in range(len(cells)):
                out.append(r'\clbrdrb\brdrs\brdrw5\cellx'+str(width*(i+1)))
            for cell in cells:
                out.append(r'\pard\intbl\fs20 '+escape(cell)+r'\cell')
            out.append(r'\row\pard\fs22')
        elif line.startswith('# '):
            out.append(r'\pard\keepn\sb0\sa220\fs34\b '+escape(line[2:])+r'\b0\fs22\par')
        elif line.startswith('## '):
            out.append(r'\pard\keepn\sb240\sa100\fs26\b '+escape(line[3:])+r'\b0\fs22\par')
        else:
            out.append(r'\pard\sb0\sa130\sl275\slmult1 '+escape(line)+r'\par')
    out.append('}')
    source.with_suffix('.rtf').write_text('\n'.join(out), encoding='ascii')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', action='store_true')
    ap.add_argument('--collect', action='store_true')
    ap.add_argument('--report', action='store_true')
    ap.add_argument('--raw-dir', type=Path, default=Path('output/iid_tilt_L3'))
    ap.add_argument('--out', type=Path, default=Path('results/mechanism/iid_layered_tilt_check.json'))
    args = ap.parse_args()
    if args.run:
        with ThreadPoolExecutor(max_workers=3) as pool:
            list(pool.map(run, commands(args.raw_dir)))
    if args.collect:
        collect(args.raw_dir, args.out)
    if args.report:
        rich_text(Path('notes/reviews/mechanisms_iid_coverage_check_2026-09-06.md'))


if __name__ == '__main__':
    main()
