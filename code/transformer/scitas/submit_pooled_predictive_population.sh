#!/bin/bash
# Login side: small plan/receipt reads and scheduler queries/submissions only.
set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
exec python3 "$repo_root/code/scripts/pooled_predictive_population_submit.py" "$repo_root" "$@"
