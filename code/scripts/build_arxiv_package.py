#!/usr/bin/env python3
"""Build the arXiv upload package from arxiv.tex.

Steps: compile arxiv.tex with latexmk in a scratch folder; collect the
source, the generated .bbl, the style file and every included figure into
arxiv_upload/ (figure paths unchanged); write arxiv_upload.tar.gz; unpack
the tarball in a clean scratch folder and compile it with pdflatex alone,
as arXiv does (arXiv does not run BibTeX; it uses the .bbl).
"""
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MAIN = "arxiv"
STYLE_FILES = ["iclr2027_conference.sty"]
OUT_DIR = ROOT / "arxiv_upload"
TARBALL = ROOT / "arxiv_upload.tar.gz"

GRAPHICS_RE = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}")


def run(cmd, cwd, log):
    with open(log, "ab") as f:
        f.write(("\n$ " + " ".join(cmd) + "\n").encode())
        return subprocess.call(cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT)


def page_count(pdf):
    out = subprocess.check_output(["pdfinfo", str(pdf)], text=True)
    return int(re.search(r"Pages:\s+(\d+)", out).group(1))


def figure_paths(tex_text):
    paths = []
    for name in GRAPHICS_RE.findall(tex_text):
        p = ROOT / name
        if not p.suffix:
            for ext in (".pdf", ".png", ".jpg"):
                if (ROOT / (name + ext)).is_file():
                    p = ROOT / (name + ext)
                    break
        if not p.is_file():
            sys.exit(f"figure not found: {name}")
        rel = p.relative_to(ROOT)
        if rel not in paths:
            paths.append(rel)
    return paths


def main():
    src = ROOT / f"{MAIN}.tex"
    text = src.read_text(encoding="utf-8")
    if "\\input{" in text:
        sys.exit("arxiv.tex must be self-contained (no \\input)")
    if re.search(r"(?<!\\)%", text):
        print("note: arxiv.tex still contains % characters (comments or line ends)")

    scratch = Path(tempfile.mkdtemp(prefix="arxiv_build_"))
    log = scratch / "build.log"
    print("compiling", src.name, "in", scratch)
    rc = run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error",
              f"-outdir={scratch}", src.name], cwd=ROOT, log=log)
    if rc != 0:
        sys.exit(f"latexmk failed, see {log}")
    pages = page_count(scratch / f"{MAIN}.pdf")
    bbl = scratch / f"{MAIN}.bbl"
    if not bbl.is_file():
        sys.exit("no .bbl produced")

    figs = figure_paths(text)
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir()
    shutil.copy2(src, OUT_DIR / src.name)
    shutil.copy2(bbl, OUT_DIR / bbl.name)
    for s in STYLE_FILES:
        shutil.copy2(ROOT / s, OUT_DIR / s)
    for rel in figs:
        dst = OUT_DIR / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, dst)

    if TARBALL.exists():
        TARBALL.unlink()
    with tarfile.open(TARBALL, "w:gz") as tar:
        for p in sorted(OUT_DIR.rglob("*")):
            if p.is_file():
                tar.add(p, arcname=str(p.relative_to(OUT_DIR)))

    # clean-room test: unpack and compile with pdflatex only
    test = Path(tempfile.mkdtemp(prefix="arxiv_test_"))
    with tarfile.open(TARBALL) as tar:
        tar.extractall(test)
    tlog = test / "test.log"
    for _ in range(3):
        rc = run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", f"{MAIN}.tex"],
                 cwd=test, log=tlog)
        if rc != 0:
            sys.exit(f"clean-room pdflatex failed, see {tlog}")
    tpages = page_count(test / f"{MAIN}.pdf")
    texlog = (test / f"{MAIN}.log").read_text(errors="replace")
    bad = [l for l in texlog.splitlines()
           if "undefined" in l.lower() or "multiply" in l.lower()]

    n_files = sum(1 for p in OUT_DIR.rglob("*") if p.is_file())
    print(f"package: {OUT_DIR.name}/ ({n_files} files), {TARBALL.name} "
          f"({TARBALL.stat().st_size / 1e6:.1f} MB)")
    print(f"figures: {len(figs)}")
    print(f"pages: {pages} (repo build), {tpages} (clean-room build)")
    if bad:
        print("warnings in the clean-room build:")
        for l in bad:
            print("  ", l)
    if pages != tpages or bad:
        sys.exit("check the package before uploading")
    print("ok")
    shutil.rmtree(scratch, ignore_errors=True)
    shutil.rmtree(test, ignore_errors=True)


if __name__ == "__main__":
    main()
