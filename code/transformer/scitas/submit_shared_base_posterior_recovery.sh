#!/bin/bash
# Audit saved chains only; dry run by default. Original failed records are retained.
set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
exec "${SHARED_BASE_LAUNCH_PYTHON:-python3}" "$repo_root/code/scripts/recover_shared_base_posterior_audit.py" launch "$@"
