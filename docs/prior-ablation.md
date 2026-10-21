# The prior ablation: five additions to how photos enter the generated tables

Six training runs, launched 2026-10-21 on Snellius (`slurm/ablation.sh`), each 10,000 steps from the same seed on
the full ImageNet-1k array, each scored on the four validation datasets as soon as it finishes. One run is the
baseline; the other five switch on one addition each. The question for each addition is the same: does it move
the paired bootstrap difference against the baseline on real data?

## What the prior does without the additions

Every training batch is a TabICL v2 synthetic table (64 to 256 rows, 3 to 33 columns, up to 10 classes) in which
one hidden variable is shown through a photo per row instead of as a number (`table_image_icl/tasks.py`). Two roles,
chosen per batch with equal probability:

- **Photo as noisy label.** The label stays in the table. Each row's photo shows the row's class with a per-table
  probability p drawn from 0.3 to 1, otherwise a uniformly random class of the task. The model has to learn, from the
  context rows, how much to trust the photo.
- **Photo as hidden feature.** A random forest on the context rows names the most important column. That column is
  removed from the table and cut into 2 to 10 quantile bins. Each bin is given an ImageNet class, and each row's photo
  is a random photo of its bin's class. The model has to recover the hidden column from the photos.

The classes of a task are a random draw from the 900 training classes (100 are held out for the synthetic
validation tasks). Photos are drawn with replacement and shown as raw pixels.

## The five additions and why

Each one is a `TaskConfig` field, off by default, with a `scripts/train.py` flag. The reasons come from the
tabular-prior literature (TabICL v2's ablations, TabPFN v2's uninformative features, TabICL v2's learnability
filter) and from few-shot image learning (fine-grained task sampling, paired label noise, query-side augmentation).

| run | flags | what changes | why it might help |
|---|---|---|---|
| baseline | none | | the reference |
| distractor | `--importance-sampling --distractor-share 0.15` | feature role: the hidden column is drawn in proportion to its importance instead of always the top one. Both roles: in 15% of tables the photos are drawn independently of the rows | the photo ranges from decisive to marginal to useless, so the model learns from the context when a photo is worth reading (every strong tabular prior has uninformative features; on CheXpert no method's photo helps and ours must not hurt) |
| fine-grained | `--fine-grained-share 0.5` | in half the tables the task's classes are neighbours under one WordNet ancestor (dog breeds, kinds of boat) instead of random classes from all of ImageNet | every benchmark task is within one domain (all mammograms, all lesions, all mangoes); fine-grained few-shot tasks are much harder than coarse ones, and coarse training teaches coarse features |
| sibling-ordinal | `--sibling-noise 0.5 --ordinal-confusion 0.3` | label role: half the wrong photos come from the nearest other class of the task rather than a uniform one. Feature role: a row shows its neighbouring bin's class at a per-table rate from 0 to 0.3, so adjacent bins are confusable and the bin order is visible in the photos | confusable-class noise is the noise real data has; ordinal and continuous targets (CBIS-DDSM density, Mango Mass) look like this through a camera |
| hygiene | `--without-replacement --augment-test` | a bin's photos are distinct when the class has enough; test-row photos get a horizontal flip and a mild crop (85 to 100%) during training only | query-side augmentation helps few-shot learners while support-side augmentation hurts; distinct photos stop the model from matching identical pixels |
| distinct | `--without-replacement` | the first half of hygiene on its own | the hygiene run stepped four times slower than the others (the per-step augmentation) and could not finish inside its limit, so it was stopped; the augmentation waits for a cheaper implementation |
| learnable | `--skip-unlearnable` | a table is skipped when its hidden variable collapsed to one bin, or when the photo agrees with the label less than half the time (p + (1 - p) / k below 0.5) | TabICL v2 drops about a third of its tables as unlearnable and converges faster; such tables only add loss |

## How the runs are compared

- Each run is scored on CBIS-DDSM, Mango Mass and CheXpert with grouped splits (a patient, fruit or listing on one side
  only) and on HAM10000 with the benchmark's row-wise split, 256 context rows, 3 splits (`scripts/benchmark.py
  --grouped`). AUROC for classification, RMSE for regression. The per-row predictions are saved.
- `scripts/bootstrap.py` then resamples the test rows (within class, the same rows for every method) and reports the
  paired difference of each run against the baseline, with an interval and the probability of being better, next to
  the frozen reference (TabICL v2 on the table plus PCA of frozen DINOv2 features).
- A run's own synthetic validation loss is not comparable across runs, because each run's validation tasks are
  generated under its own flags. The ImageNet-21k pilot also showed that the synthetic loss kept improving while the
  Mango Mass error got worse from step 5,000 on, so real-data scores decide.
- Additions that help are combined into one full-length run; the ones that do not are left off.

## Cost and schedule

A 10,000-step run takes about 2.3 h on one A100 with the array in RAM (240 GB, billed as two GPU shares) and the
scoring job about 20 minutes. The six runs were queued with `BEGIN=2026-10-21T00:05` so that their cost lands on
2026-10-21, about 3,600 SBU in total; the day's cap is 4,200.

## Results (2026-10-21, 10,000 steps each, 256 context rows, paired bootstrap against the baseline run)

Full tables: `results/ablation/bootstrap-vs-baseline.md` and `bootstrap-vs-frozen.md`; raw scores in `results/ablation/<run>/`.

| run | CBIS-DDSM AUROC | HAM10000 AUROC | Mango Mass RMSE | CheXpert AUROC |
|---|---|---|---|---|
| baseline, seed 0 | 0.728 | 0.874 | 0.044 | 0.722 |
| baseline, seed 1 | 0.683 | 0.878 | 0.041 | 0.722 |
| distractor | 0.704 | 0.872 | 0.045 | 0.722 |
| fine-grained | 0.683 | 0.879 | 0.047 | 0.721 |
| sibling-ordinal | 0.708 | 0.872 | 0.041 | 0.721 |
| distinct photos | 0.702 | 0.869 | 0.045 | 0.722 |
| learnable | 0.711 | 0.876 | 0.040 | 0.719 |
| frozen reference | 0.799 | 0.863 | 0.047 | 0.729 |

What the two baseline seeds say: the same prior with another seed moves CBIS-DDSM by 0.045 and Mango Mass by 0.004.
Every difference between an addition and the seed-0 baseline is inside that range: the CBIS-DDSM losses of 0.017 to
0.045 and the Mango Mass gains of 0.003 to 0.004 for the learnability guard and the ordinal confusion are the size of
seed noise. HAM10000 and CheXpert move by at most 0.005. So at 10,000 steps and one seed per arm, none of the five
additions has an effect that can be told from training variance, and none is adopted on this evidence. The paired
bootstrap intervals are tight because they resample test rows only; the seed pair is the honest error bar for a
training change, and any future claim about the prior needs at least two seeds per arm.

Two side findings: the baseline at 10,000 steps beats the frozen reference on HAM10000 (0.874 against 0.863) and
Mango Mass (0.044 against 0.047, with the second seed at 0.041), and trails it on CBIS-DDSM by 0.07 to 0.12 depending
on the seed. The hygiene run (test-row augmentation) stepped four times slower than the others and was stopped; the
distinct-photos half ran alone.
