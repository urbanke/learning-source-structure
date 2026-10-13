"""Compact Section 4 figure from audited, reused development results only."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from section4_product_accuracy import draw_required_context
from section4_unseen_row_transfer import draw_panel

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/transformer/section4_partial"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    bases = [ROOT / "results/transformer" / name for name in
             ("section4_product_accuracy", "section4_unseen_row_transfer")]
    sources = [base / "summary.json" for base in bases]
    receipts = [bases[0] / "receipt.json", bases[1] / "validation.json"]
    data = [json.loads(p.read_text()) for p in sources]
    checks = [json.loads(p.read_text()) for p in receipts]
    assert checks[0]["verified"] and checks[1]["passed"]
    assert checks[0]["outputs_sha256"]["summary.json"] == sha(sources[0])
    assert checks[1]["files"]["summary.json"] == sha(sources[1])
    inputs = {str(p.relative_to(ROOT)): sha(p) for p in sources + receipts}
    inputs.update({str(p.relative_to(ROOT)): sha(p) for p in
                   (Path(__file__), Path(__file__).with_name("section4_product_accuracy.py"),
                    Path(__file__).with_name("section4_unseen_row_transfer.py"))})
    if args.check_only:
        receipt = json.loads((OUT / "receipt.json").read_text())
        assert receipt["inputs"] == inputs
        for name, expected in receipt["outputs"].items():
            assert sha(OUT / name) == expected
        print("Verified compact application figure and its input receipts.")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 7,
                         "axes.titlesize": 8, "axes.labelsize": 7,
                         "axes.linewidth": .6, "pdf.fonttype": 42}):
        fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.85))
        draw_required_context(axes[0], data[0], data[0]["compact_case_ids"])
        axes[0].set_title("(a) Product: required context", loc="left")
        axes[0].set_xlabel("Allowed next-token regret (bits)")
        axes[0].set_ylabel("Observed tokens needed")
        for text in axes[0].get_legend().get_texts():
            text.set_fontsize(7)
        draw_panel(axes[1], data[1], 16, legend=False)
        axes[1].set_title("")
        axes[1].set_title("(b) Shared base: new context", loc="left")
        axes[1].set_ylim(-.12, .86)
        handles, _ = axes[1].get_legend_handles_labels()
        axes[1].legend(handles, ["Bayes", "TF seed 0", "TF seed 1", "Token counts"],
                       fontsize=7, frameon=False, loc="upper right", ncol=2,
                       columnspacing=.5, handletextpad=.25)
        axes[1].set_xlabel("Source depth; 16 observed tokens")
        fig.tight_layout(pad=.65, w_pad=1.4)
        fig.savefig(OUT / "overview.pdf", metadata={"CreationDate": None, "ModDate": None})
        fig.savefig(OUT / "overview.png", dpi=240)
        plt.close(fig)
    receipt = {"purpose": "Exploratory Section 4 applications; no new inference or samples.",
               "inputs": inputs,
               "outputs": {name: sha(OUT / name) for name in ("overview.pdf", "overview.png")}}
    (OUT / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print("Wrote compact application figure from validated summaries.")


if __name__ == "__main__":
    main()
