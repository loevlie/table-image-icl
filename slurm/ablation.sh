#!/bin/bash
# The prior ablation: six 10,000-step runs on the full ImageNet-1k array, one flag group each, same seed, submitted
# from the repository root on a SLURM cluster (TIICL_ENV and IMAGE_CACHE as in slurm/train.sbatch; PARTITION, e.g. gpu_a100).
#   bash slurm/ablation.sh                      # all six
#   bash slurm/ablation.sh baseline distractor  # some of them
# Score each run afterwards with scripts/benchmark.py --grouped and compare with scripts/bootstrap.py (results/ablation/).
set -euo pipefail
: "${PARTITION:?set PARTITION}"
declare -A FLAGS=(
  [baseline]=""
  [distractor]="--importance-sampling --distractor-share 0.15"
  [fine-grained]="--fine-grained-share 0.5"
  [sibling-ordinal]="--sibling-noise 0.5 --ordinal-confusion 0.3"
  [hygiene]="--without-replacement --augment-test"
  [learnable]="--skip-unlearnable"
)
ORDER=(baseline distractor fine-grained sibling-ordinal hygiene learnable)
for name in "${@:-${ORDER[@]}}"; do
  # shellcheck disable=SC2086
  job=$(RUN_NAME=ablation-$name sbatch --parsable -p "$PARTITION" ${DEPENDENCY:+--dependency=afterok:$DEPENDENCY} slurm/train.sbatch ${FLAGS[$name]})
  echo "$name $job"
done
