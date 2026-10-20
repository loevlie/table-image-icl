# table-image-icl

In-context learning on tables whose rows carry a photo. TabICL v2's pretrained table stages, plus a small image side trained from scratch that reads each row's photo against the photos of its 32 nearest labelled context rows. Hand it a labelled table with photos as context and it predicts the test rows in one forward pass, with no training on the data.

<p align="center"><img src="assets/architecture.png" width="720"></p>

## Install

```bash
pip install -r requirements.txt && pip install -e .
```

TabICL v2 and DINOv2 download themselves on first use.

## Use a published model

```python
from table_image_icl.pretrained import load_pretrained
from table_image_icl.inputs import inputs_for, letterbox
from table_image_icl.predict import predict_classes

model, record = load_pretrained("imagenet1k")           # fetched once into checkpoints/, checked by sha256
to_tokens = inputs_for(model.cfg.encoder, "cpu")
photos = (np.stack([letterbox(p) for p in context_paths]), np.stack([letterbox(p) for p in test_paths]))
probabilities, classes = predict_classes(model, x_context, y_context, x_test, photos=photos, to_tokens=to_tokens)
```

`x_context`, `x_test` are float arrays with NaN for missing cells; `predict_values` does regression; without `photos` the model is TabICL v2. The published models are listed in `table_image_icl/pretrained.py`.

To look at a model on one benchmark dataset:

```bash
python scripts/probe.py --dataset "Mango Mass"                      # 256 context rows, 3 splits, against the frozen-feature reference, with bootstrap intervals
python scripts/probe.py --dataset HAM10000 --row 17                 # one test row: its cells, its photo, its prediction with and without the photo
```

## Results

Four MulTaBench datasets, 256 labelled rows as context, three splits, no training on them. The reference is MulTaBench's frozen-feature pipeline with our own table model as the learner (TabICL v2 on the table plus 30 PCA columns of frozen DINOv2 features), so the only difference is how the photo comes in. AUROC for classification, RMSE for regression; 95% intervals from a paired bootstrap of the test rows (`scripts/bootstrap.py`). CBIS-DDSM, CheXpert and Mango Mass are split with each patient or fruit on one side (`--grouped`); HAM10000 has one photo per lesion.

| | CBIS-DDSM AUROC | HAM10000 AUROC | Mango Mass RMSE | CheXpert AUROC |
|---|---|---|---|---|
| TabICL v2, table + frozen DINOv2 PCA | 0.799 [0.785, 0.813] | 0.863 [0.854, 0.872] | 0.047 [0.044, 0.050] | 0.729 [0.723, 0.735] |
| ours, from scratch | 0.765 [0.751, 0.780] | **0.895** [0.887, 0.903] | **0.040** [0.037, 0.043] | 0.723 [0.717, 0.729] |
| ours, frozen DINOv2 blocks 1-4 | **0.808** [0.794, 0.823] | 0.879 [0.871, 0.887] | 0.070 [0.065, 0.074] | 0.721 [0.715, 0.727] |

Under the benchmark's own protocol (up to 10,000 context rows) the from-scratch model is first of the 27 leaderboard models on Mango Mass and behind the frozen pipeline on the three classification sets. All records are in [`results/`](results/) and plotted in [`notebooks/results.ipynb`](notebooks/results.ipynb).

## Reproduce

Photos for training come from one source, decoded once (`table_image_icl/photos.py`): the full ImageNet-1k training set (1.28M photos of 1,000 classes, from timm's webdataset release on the Hub) or the full ImageNet-21k Winter 2021 set (13.2M photos of 19,167 classes). Both need the dataset's terms accepted on the Hub and `hf auth login`. A tenth of the classes is held out: validation tasks show photos of classes training never sees. A laptop can decode a few shards of either for a short test.

```bash
python scripts/photos.py imagenet1k --cache cache --shards 16                                           # a laptop: 20,000 photos, a few minutes
python scripts/train.py --out runs/try --photos imagenet1k --shards 16 --max-steps 500
python scripts/photos.py imagenet1k --cache cache --workers 64                                          # the full set, 181 GB decoded
python scripts/train.py --out runs/imagenet1k --photos imagenet1k --in-memory --max-steps 25000         # one A100, 240 GB of RAM, about 3.5 hours
python scripts/evaluate.py --run runs/imagenet1k --tasks-file evaluation-tasks.pt --out results/imagenet1k/evaluation.json
python scripts/benchmark.py --run runs/imagenet1k --out results/imagenet1k/multabench.json --grouped
python scripts/bootstrap.py --predictions "ours=results/imagenet1k/multabench-predictions.npz" --out results/bootstrap/validation.json
```

`--encoder block4` trains the frozen-DINOv2 variant; `--max-context 4096 --max-sources 4096` on `benchmark.py` scores with more context rows. `python scripts/release.py --run runs/imagenet1k --name imagenet1k --tag v0.2` prints the checksums and the upload command that publish a run for `load_pretrained`. `PYTHONPATH=. python scripts/cost.py --sizes 64,256,1024` writes the compute (FLOPs) and the time per table of ours and of the benchmark's frozen baseline to `docs/figures/cost.csv`.

## Code tour

`python tour/app.py` serves http://127.0.0.1:8899: the model's files section by section (Code Hike scrollycoding), with what each section does, links to the upstream file at the pinned commit with the installed copy shown inline, tensor shapes recorded from a real run, hover definitions, and equivalence cards for every piece of code that is a port of someone else's (`tour/provenance.py`): our function against the upstream one in a Monaco diff, both outputs on one fixed input, a Hypothesis differential test over random inputs, and the pytest proof runnable from the page. `/classic` is the same tour as one plain page.

Build the page once with `cd tour/web && npm install && npm run build` (the server needs `pip install django pygments jedi`); regenerate the data under `tour/data/` with `python tour/trace.py`, `python tour/definitions.py` and `python tour/equivalence.py`.

## Notebooks

- [`walkthrough`](notebooks/walkthrough.ipynb): one small table through the model, stage by stage, with the code of each stage next to a run of it.
- [`results`](notebooks/results.ipynb): the figures behind the numbers above.

Tests: `pytest` (CPU, under two minutes, no downloads).

## Acknowledgements

Table stages and prior from [TabICL v2](https://github.com/soda-inria/tabicl) (BSD-3); frozen features from [DINOv2](https://github.com/facebookresearch/dinov2); benchmark from [MulTaBench](https://github.com/alanarazi7/MulTaBench); reference learner [TabPFN v2](https://github.com/PriorLabs/TabPFN); photos from [ImageNet](https://www.image-net.org/) via timm's webdataset releases. MIT license.
