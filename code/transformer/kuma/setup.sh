#!/bin/bash
# One-time setup on the SCITAS Kuma cluster, run from the project folder after cloning it:
#     bash code/transformer/kuma/setup.sh
# Creates the Python environment ~/venv-tf with PyTorch (CUDA build) and checks that a GPU build was installed.
set -e
module load gcc python 2>/dev/null || true
python3 --version
python3 -m venv ~/venv-tf
source ~/venv-tf/bin/activate
pip install --upgrade pip
pip install -r code/transformer/requirements.txt
python3 -c "import torch; print('torch', torch.__version__, 'cuda build:', torch.version.cuda)"
