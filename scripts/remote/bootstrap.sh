#!/usr/bin/env bash
# One-time setup of a Linux server: environment, data, references, self-test.
#
#   bash scripts/remote/bootstrap.sh              # everything
#   bash scripts/remote/bootstrap.sh --skip-data  # environment + tests only
#
# Safe to re-run: every stage skips what is already done and verified.
#   1. Miniforge (conda + mamba) in ~/miniforge3, if no conda is present
#   2. the `utrscan` environment from environment.yml
#   3. PyTorch: CUDA build if an NVIDIA GPU is visible, CPU build otherwise
#   4. tool check: bowtie2, cutadapt, ViennaRNA, torch/CUDA
#   5. FASTQs (ENA, MD5-verified) + references   -> scripts/remote/fetch_data.py
#   6. unit tests, including the GPU-vs-CPU backend agreement test
set -euo pipefail

SKIP_DATA=0
SKIP_TESTS=0
for a in "$@"; do
  case "$a" in
    --skip-data) SKIP_DATA=1 ;;
    --skip-tests) SKIP_TESTS=1 ;;
    *) echo "unknown option $a"; exit 2 ;;
  esac
done

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
ENV_NAME=utrscan
say() { printf '\n\033[1m[bootstrap] %s\033[0m\n' "$*"; }

[ "$(uname -s)" = "Linux" ] || { echo "This script targets Linux."; exit 1; }

# --- 1. conda ---------------------------------------------------------------
if ! command -v conda >/dev/null 2>&1; then
  if [ -x "$HOME/miniforge3/bin/conda" ]; then
    :
  else
    say "installing Miniforge into ~/miniforge3"
    ARCH="$(uname -m)"
    curl -fsSL -o /tmp/miniforge.sh \
      "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-${ARCH}.sh"
    bash /tmp/miniforge.sh -b -p "$HOME/miniforge3"
    rm -f /tmp/miniforge.sh
  fi
  export PATH="$HOME/miniforge3/bin:$PATH"
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
SOLVER=conda
command -v mamba >/dev/null 2>&1 && SOLVER=mamba

# --- 2. environment ---------------------------------------------------------
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  say "updating environment $ENV_NAME"
  $SOLVER env update -n "$ENV_NAME" -f environment.yml --prune
else
  say "creating environment $ENV_NAME (a few minutes)"
  $SOLVER env create -n "$ENV_NAME" -f environment.yml
fi
conda activate "$ENV_NAME"

# --- 3. PyTorch -------------------------------------------------------------
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  say "NVIDIA GPU found: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -n1)"
  if ! python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    say "installing PyTorch (CUDA build)"
    # Linux wheels on PyPI ship with CUDA 12.x runtime libraries bundled.
    pip install --upgrade "torch>=2.5"
  fi
  python - <<'EOF'
import torch, sys
if not torch.cuda.is_available():
    sys.exit("\nPyTorch cannot use the GPU (driver too old for this CUDA build?).\n"
             "The pipeline will still run, on CPU.  `nvidia-smi` shows the driver version;\n"
             "PyTorch needs a driver supporting CUDA 12.x.\n")
print("[bootstrap] torch", torch.__version__, "CUDA", torch.version.cuda, "->",
      torch.cuda.get_device_name(0))
EOF
else
  say "no NVIDIA GPU visible: installing the CPU build of PyTorch"
  python -c "import torch" 2>/dev/null || \
    pip install "torch>=2.5" --index-url https://download.pytorch.org/whl/cpu
fi

# --- 4. tools ---------------------------------------------------------------
say "checking tools"
bowtie2-align-s --version | head -n1
bowtie2-build-s --version | head -n1
echo "cutadapt $(cutadapt --version)"
python -c "import RNA; print('ViennaRNA', RNA.__version__)"
python -c "import numpy, scipy, sklearn, matplotlib; print('numpy', numpy.__version__, 'scipy', scipy.__version__)"
python -c "
from analysis.recoverability import config as C
r = C.resources(); print('[bootstrap] resources', r, '-> backend', C.backend(), 'workers', C.default_workers())"

# --- 5. data ----------------------------------------------------------------
if [ "$SKIP_DATA" = 0 ]; then
  say "downloading and verifying data and references (~22 GB FASTQ + ~12 GB references)"
  python scripts/remote/fetch_data.py
fi

# --- 6. tests ---------------------------------------------------------------
if [ "$SKIP_TESTS" = 0 ]; then
  say "unit tests"
  python -m analysis.recoverability.tests.test_likelihood
  python -m analysis.recoverability.tests.test_nulls
  python -m analysis.recoverability.tests.test_chainfast
  python -m analysis.recoverability.tests.test_backends
fi

touch "$REPO/.bootstrap_ok"
say "done.  Start the pipeline with:   bash scripts/remote/run_pipeline.sh"
