#!/bin/bash
#SBATCH --job-name=tiicl-benchmark
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/%x-%j.out
# Score one run on the four validation datasets (scripts/benchmark.py --grouped: patients, fruits and listings kept
# on one side of each split; HAM10000 has no group key and keeps the benchmark's row-wise split) and save the
# per-row predictions for scripts/bootstrap.py. Submit from the repository root:
#   sbatch -p <gpu partition> slurm/benchmark.sbatch runs/ablation-baseline results/ablation/baseline/multabench-grouped.json
# TIICL_ENV as in slurm/train.sbatch. The MulTaBench datasets come from Kaggle (table_image_icl/multabench.py), so
# KAGGLEHUB_CACHE must already hold them when compute nodes are offline.
set -euo pipefail
: "${TIICL_ENV:?set TIICL_ENV to the command that activates the environment}"
run=${1:?run folder}
out=${2:?output json}
eval "$TIICL_ENV"
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH=.
python scripts/benchmark.py --run "$run" --out "$out" --set validation --grouped
