#!/bin/bash
#SBATCH --job-name=table-image-icl
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=240G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/%x-%j.out
# One training run on a GPU node, the full ImageNet-1k array read into RAM. Submit from the repository root:
#   RUN_NAME=baseline sbatch -p <gpu partition> slurm/train.sbatch
#   RUN_NAME=distractor sbatch -p <gpu partition> slurm/train.sbatch --importance-sampling --distractor-share 0.15
# Environment variables: TIICL_ENV, the command that activates the Python environment (e.g. "source venv/bin/activate");
# IMAGE_CACHE, the decoded ImageNet-1k folder (images.npy, labels.npy, names.json, done.json, hierarchy.json), symlinked
# as cache/imagenet1k-1024; RUN_NAME, the run's folder under runs/. Extra arguments go to scripts/train.py.
set -euo pipefail
: "${TIICL_ENV:?set TIICL_ENV to the command that activates the environment}"
: "${RUN_NAME:?set RUN_NAME}"
IMAGE_CACHE=${IMAGE_CACHE:-/scratch-shared/$USER/image-cache/imagenet1k}
eval "$TIICL_ENV"
cd "$SLURM_SUBMIT_DIR"
mkdir -p cache slurm/logs
ln -sfn "$IMAGE_CACHE" cache/imagenet1k-1024
export PYTHONPATH=.
python -c "import torch; print(torch.cuda.get_device_name(0))"
python scripts/train.py --out "runs/$RUN_NAME" --photos imagenet1k --in-memory --max-steps 10000 --workers 16 "$@"
