#!/bin/bash
# User-led posterior pilot. Dry-run by default; never submits successors.
set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
if [[ "${1:-}" == --merge ]]; then
  shift
  execute=false
  action_seen=false
  options=()
  while (($#)); do
    case "$1" in
      --submit|--dry-run)
        [[ "$action_seen" == false ]] || { echo 'Choose one action flag.' >&2; exit 2; }
        action_seen=true
        if [[ "$1" == --submit ]]; then execute=true; fi
        shift ;;
      --plan|--out)
        (($# >= 2)) || { echo "Missing value for $1" >&2; exit 2; }
        options+=("$1" "$2"); shift 2 ;;
      *) echo "Unknown merge option: $1" >&2; exit 2 ;;
    esac
  done
  command=("${SHARED_BASE_VENV:-$HOME/venv-tf}/bin/python" "$repo_root/code/scripts/shared_base_posterior_cluster.py" summary "${options[@]}")
  printf '%q ' "${command[@]}"; printf '\n'
  if [[ "$execute" == true ]]; then exec "${command[@]}"; fi
  exit 0
fi
exec "${SHARED_BASE_LAUNCH_PYTHON:-python3}" "$repo_root/code/scripts/shared_base_posterior_cluster.py" launch "$@"
