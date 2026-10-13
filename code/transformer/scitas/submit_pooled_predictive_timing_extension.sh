#!/bin/bash
# Default is a read-only preview. --submit explicitly queues the bounded chain.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
exec python3 "$ROOT/code/scripts/pooled_predictive_timing_extension.py" submit "$@"
