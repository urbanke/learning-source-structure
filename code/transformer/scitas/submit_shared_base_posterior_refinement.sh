#!/bin/bash
# Nine fixed posterior refinements; dry-run default, explicit user submission.
set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
exec "${SHARED_BASE_LAUNCH_PYTHON:-python3}" "$repo_root/code/scripts/refine_shared_base_posterior.py" launch "$@"
