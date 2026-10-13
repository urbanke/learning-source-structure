#!/bin/bash
# Local validation and printed command by default; --submit contacts Slurm.
set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
exec "${SHARED_BASE_LAUNCH_PYTHON:-python3}" "$repo_root/code/transformer/scitas/launch_shared_base.py" "$@"
