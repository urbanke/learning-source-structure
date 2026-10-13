"""Verify and bind the completed fixed-depth product comparison artifacts.

This reads existing records only. Run after conditional_product_accuracy.py.
It writes receipt.json beside the curated summary and figures.
"""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "results/transformer/conditional_accuracy/product"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(test, message):
    if not test:
        raise ValueError(message)


def main():
    summary = json.loads((RESULT / "summary.json").read_text())
    validation = json.loads((RESULT / "validation.json").read_text())
    require(validation["passed"], "Numerical validation failed")
    protocol = summary["protocol"]
    require(protocol["depth_pairs"] == [[1, 1], [1, 8], [4, 4], [8, 8]], "Changed selected depths")
    require(protocol["n_sequences_per_pair"] == 256 and protocol["n_tokens"] == 1025, "Incomplete cohort")
    require(protocol["times"] == [2**i for i in range(11)], "Changed observation times")
    require(protocol["reference_depths_revealed"] is False, "Test parameter leakage")
    files = {}
    for name in ("conditional_product_accuracy.py", "conditional_product_accuracy_figures.py", "conditional_product_accuracy_receipt.py"):
        path = ROOT / "code/scripts" / name
        files[str(path.relative_to(ROOT))] = sha(path)
    for name, expected in protocol["source_code_sha256"].items():
        path = ROOT / "code/transformer" / name
        require(sha(path) == expected, "Changed source implementation")
        files[str(path.relative_to(ROOT))] = sha(path)
    checkpoint = Path(summary["checkpoint"])
    require(sha(checkpoint) == protocol["checkpoint_sha256"], "Changed frozen checkpoint")
    files[str(checkpoint.relative_to(ROOT))] = sha(checkpoint)
    for case in summary["cases"]:
        base = Path(case["raw_directory"])
        digest = hashlib.sha256()
        for part in ("data", "references", "model"):
            require(len(list((base / part).glob("seq_*.npz"))) == 256, "Incomplete or enlarged cohort")
        for index in range(256):
            name = f"seq_{index:06d}.npz"
            for part in ("data", "references", "model"):
                digest.update(f"{part}/{name}:{sha(base / part / name)}".encode())
        require(digest.hexdigest() == case["raw_records_sha256"], "Raw record digest differs from plotted summary")
    for name in ("summary.json", "summary.tsv", "validation.json", "curves.pdf", "curves.png", "gaps.pdf", "gaps.png", "README.md"):
        path = RESULT / name
        files[str(path.relative_to(ROOT))] = sha(path)
    receipt = dict(passed=True, schema="conditional-product-accuracy-receipt-v1",
                   files_sha256=files,
                   raw_cohort_digests={str(case["depth_pair"]): case["raw_records_sha256"] for case in summary["cases"]},
                   raw_digest_rule="SHA256 of data/name:SHA256, references/name:SHA256, model/name:SHA256 in sequence-index order",
                   command="python code/scripts/conditional_product_accuracy.py --stage all --device cpu --batch 8",
                   figure_command="python code/scripts/conditional_product_accuracy_figures.py",
                   receipt_command="python code/scripts/conditional_product_accuracy_receipt.py",
                   required_environment={"LSA_STORE": "/Users/ruediger/Projects/product_model_with_memory/tables/anchors_prod",
                                         "PMM_SRC": "/Users/ruediger/Projects/product_model_with_memory/src",
                                         "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"},
                   validation=validation)
    (RESULT / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print("Product artifact receipt passed")


if __name__ == "__main__":
    main()
