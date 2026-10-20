#!/bin/bash
# The prior ablation: six 10,000-step runs on the full ImageNet-1k array, one flag group each, same seed, each
# scored on the validation datasets as soon as it finishes (slurm/benchmark.sbatch). Submitted from the repository
# root on a SLURM cluster (TIICL_ENV and IMAGE_CACHE as in slurm/train.sbatch; PARTITION, e.g. gpu_a100).
#   bash slurm/ablation.sh                      # all six
#   bash slurm/ablation.sh baseline distractor  # some of them
#   BEGIN=2026-10-21T00:05 bash slurm/ablation.sh    # queued now, eligible to start then
# 10,000 steps take about 2.3 h on one A100 with the array in RAM; the 3 h limit bounds a slow node.
# Compare the runs afterwards with scripts/bootstrap.py on results/ablation/*/multabench-grouped-predictions.npz.
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
  job=$(RUN_NAME=ablation-$name sbatch --parsable -p "$PARTITION" --time=03:00:00 ${BEGIN:+--begin=$BEGIN} \
        ${DEPENDENCY:+--dependency=afterok:$DEPENDENCY} slurm/train.sbatch ${FLAGS[$name]})
  score=$(sbatch --parsable -p "$PARTITION" --dependency=afterok:"$job" slurm/benchmark.sbatch \
          "runs/ablation-$name" "results/ablation/$name/multabench-grouped.json")
  echo "$name train $job score $score"
done
