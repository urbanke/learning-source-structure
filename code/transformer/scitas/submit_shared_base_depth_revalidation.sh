#!/bin/bash
# Revalidate saved depth statistics uniformly; dry-run default, no sampling.
set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
exec "${SHARED_BASE_LAUNCH_PYTHON:-python3}" "$repo_root/code/scripts/revalidate_shared_base_depth.py" launch "$@"
