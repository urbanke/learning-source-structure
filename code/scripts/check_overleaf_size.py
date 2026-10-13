#!/usr/bin/env python3
"""Check versioned editable content before pushing the Overleaf mirror.

Default extensions from Overleaf's public text_file_defaults.js, checked
2026-09-19. Hosted classification can differ; keep headroom below its 7 MB
limit. This checks Git blobs, so ignored local reports never count.
"""
import argparse
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
EXTENSIONS = set("tex latex sty cls bst bib bibtex txt tikz mtx rtex md asy lbx "
                 "bbx cbx m lco dtx ins ist def clo ldf rmd qmd lua py gv mf "
                 "yml yaml lhs lean lean4 hs mk xmpdata cfg rnw ltx inc".split())
FILENAMES = {"latexmkrc", ".latexmkrc", "makefile", "gnumakefile"}


def git(*args, **kwargs):
    return subprocess.check_output(["git", *args], cwd=ROOT, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", default="HEAD")
    parser.add_argument("--index", action="store_true", help="Check staged files")
    parser.add_argument("--budget-bytes", type=int, default=6_700_000)
    args = parser.parse_args()
    entries = []
    if args.index:
        for row in git("ls-files", "--stage", "-z").decode().split("\0"):
            if not row:
                continue
            info, name = row.split("\t", 1)
            _, oid, stage = info.split()
            if stage != "0":
                raise SystemExit("Resolve unmerged files before checking the index")
            entries.append((oid, name))
    else:
        for row in git("ls-tree", "-rz", args.revision).decode().split("\0"):
            if row:
                info, name = row.split("\t", 1)
                _, kind, oid = info.split()
                if kind == "blob":
                    entries.append((oid, name))
    editable = [(oid, name) for oid, name in entries if
                Path(name).suffix.lower().lstrip(".") in EXTENSIONS or
                Path(name).name.lower() in FILENAMES]
    rows = git("cat-file", "--batch-check=%(objectsize)",
               input="".join(oid + "\n" for oid, _ in editable).encode()).decode().splitlines()
    sizes = [(int(size), name) for size, (_, name) in zip(rows, editable, strict=True)]
    total = sum(size for size, _ in sizes)
    print(f"Overleaf editable content: {total:,} bytes in {len(sizes)} files; "
          f"repository budget: {args.budget_bytes:,} bytes")
    if total > args.budget_bytes:
        print("Archive historical notes or remove generated reports from tracking before pushing.")
        for size, name in sorted(sizes, reverse=True)[:10]:
            print(f"  {size:>10,}  {name}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
