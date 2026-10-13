#!/bin/bash
# Read-only saved-chain diagnostic; dry run by default.
set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
exec "${SHARED_BASE_LAUNCH_PYTHON:-python3}" "$repo_root/code/scripts/inspect_shared_base_depth_diagnostics.py" launch "$@"
