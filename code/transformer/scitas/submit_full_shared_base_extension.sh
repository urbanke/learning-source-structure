#!/bin/bash
# Three frozen terminal records only. Default is a lightweight dry run.
set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
exec "${SHARED_BASE_LAUNCH_PYTHON:-python3}" "$repo_root/code/scripts/full_shared_base_extension.py" launch "$@"
