#!/bin/bash
# Complete remaining cohort, with bounded reservations and persistent workers.
set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
exec "${SHARED_BASE_LAUNCH_PYTHON:-python3}" "$repo_root/code/scripts/shared_base_posterior_full.py" launch "$@"
