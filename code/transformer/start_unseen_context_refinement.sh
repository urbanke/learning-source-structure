#!/usr/bin/env bash
# Continue the original laptop checkpoints in their matching local runtime.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="$ROOT/output/unseen_context_refinement_runtime_2026-09-19/bin/python"
PLAN="$ROOT/code/transformer/configs/unseen_context_refinement_plan_2026-09-19.json"
OUT="$ROOT/output/unseen_context_refinement_2026-09-19"
DRIVER="$ROOT/code/scripts/unseen_context_refinement.py"
ACTION=check
RUN_ARGS=(run --plan "$PLAN" --out "$OUT")
for argument in "$@"; do
    case "$argument" in
        --check) ACTION=check ;;
        --run) ACTION=run ;;
        --retry-incomplete) RUN_ARGS+=(--retry-incomplete) ;;
        *) echo "Usage: bash $0 [--check | --run] [--retry-incomplete]" >&2; exit 2 ;;
    esac
done
if [[ ! -x "$PYTHON" ]]; then
    echo "The matching local Python runtime is missing: $PYTHON" >&2
    exit 1
fi
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
"$PYTHON" - "$ROOT" "$PLAN" <<'PY'
import json, sys
from pathlib import Path
import numpy as np
import scipy
root, path = map(Path, sys.argv[1:])
sys.path.insert(0, str(root/'code/scripts'))
from prepare_unseen_context_refinement import read_plan
plan, digest = read_plan(path)
actual = dict(python=list(sys.version_info[:3]), numpy=np.__version__, scipy=scipy.__version__)
if actual != plan['runtime'] or np.finfo(np.longdouble).nmant+1 != plan['accumulator_mantissa_bits']:
    raise SystemExit('The runtime differs from the saved checkpoints; refusing to resume.')
print(json.dumps(dict(preflight_passed=True, plan_sha256=digest, histories=123,
    references=246, max_workers=plan['resources']['max_workers'], sampling_started=False)), flush=True)
PY
if [[ "$ACTION" == check ]]; then
    exit 0
fi
if command -v caffeinate >/dev/null 2>&1; then
    caffeinate -i "$PYTHON" "$DRIVER" "${RUN_ARGS[@]}"
    caffeinate -i "$PYTHON" "$DRIVER" merge --plan "$PLAN" --out "$OUT"
else
    "$PYTHON" "$DRIVER" "${RUN_ARGS[@]}"
    "$PYTHON" "$DRIVER" merge --plan "$PLAN" --out "$OUT"
fi
