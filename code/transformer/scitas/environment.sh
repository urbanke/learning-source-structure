#!/bin/bash
# Source only inside a user-submitted allocation. Never installs software.
if type module >/dev/null 2>&1; then
  module load gcc python
fi
shared_base_venv=${SHARED_BASE_VENV:-"$HOME/venv-tf"}
if [[ ! -f "$shared_base_venv/bin/activate" ]]; then
  echo "Missing Python environment: $shared_base_venv. Set SHARED_BASE_VENV to the prepared environment." >&2
  exit 2
fi
source "$shared_base_venv/bin/activate"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
