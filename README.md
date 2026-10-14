# table-image-icl

In-context learning on tables whose rows carry a photo. TabICL v2's pretrained table stages, plus a small image side that reads each row's photo against the photos of its 32 nearest labelled context rows. Hand it a labelled table with photos as context and it predicts the test rows in one forward pass, with no training on the data.

<p align="center"><img src="assets/architecture.png" width="720"></p>

## Install

```bash
pip install -r requirements.txt && pip install -e .
```

Download the released weights from the [releases page](https://github.com/loevlie/table-image-icl/releases) to `checkpoints/block4/model.pt`. TabICL v2 and DINOv2 download themselves on first use.

## Usage

```python
from table_image_icl import load_run
from table_image_icl.inputs import inputs_for, letterbox
from table_image_icl.predict import predict_classes

model, record = load_run("checkpoints/block4")
to_tokens = inputs_for(record["arm"], "cpu", block=record["dino_block"])

photos = (np.stack([letterbox(p) for p in context_paths]), np.stack([letterbox(p) for p in test_paths]))
probabilities, classes = predict_classes(model, x_context, y_context, x_test, photos=photos, to_tokens=to_tokens)
```

`x_context`, `x_test` are float arrays with NaN for missing cells; `predict_values` does regression; without `photos` the model is TabICL v2.

## Results

MulTaBench's validation datasets, 256 labelled rows as context, three splits. The reference is MulTaBench's frozen-encoder baseline.

| | CBIS-DDSM | HAM10000 | CheXpert | Mango Mass (R²) |
|---|---|---|---|---|
| TabPFN v2, table | 0.665 | 0.745 | 0.715 | 0.205 |
| TabPFN v2, table + frozen DINOv2 PCA | 0.797 | 0.852 | 0.720 | **0.736** |
| ours, table + photo | **0.817** | **0.879** | 0.708 | 0.323 |

Trained on tables of at most 256 rows, the model keeps improving when given up to 10,000 context rows at inference ([figure](assets/results/context.png)). All records are in [`results/`](results/) and plotted in [`notebooks/results.ipynb`](notebooks/results.ipynb).

## Reproduce

One A100. Photo collections download from Hugging Face on the first run, MulTaBench from Kaggle.

```bash
python scripts/train.py --out runs/block4 --arm dino --dino-block 4 --max-steps 25000 --hours 5 --image-cache cache
python scripts/evaluate.py --run runs/block4 --tasks-file evaluation-tasks.pt --out results/block4/evaluation.json --image-cache cache
python scripts/benchmark.py --run runs/block4 --out results/block4/multabench.json
```

`--arm pixels` or `--dino-block 8` trains the other arms of the ablation; `--max-context 4096 --max-sources 4096` on `benchmark.py` scores with more context rows.

## Notebooks

- [`walkthrough`](notebooks/walkthrough.ipynb): one small table through the model, stage by stage, with the code of each stage next to a run of it.
- [`top_k_read`](notebooks/top_k_read.ipynb): the top-k read as a gather, a dense mask and a block-sparse kernel, with identical outputs and gradients.
- [`results`](notebooks/results.ipynb): the figures behind the numbers above.

Tests: `pytest` (CPU, under a minute, no downloads).

## Acknowledgements

Table stages and prior from [TabICL v2](https://github.com/soda-inria/tabicl) (BSD-3); frozen features from [DINOv2](https://github.com/facebookresearch/dinov2); benchmark from [MulTaBench](https://github.com/alanarazi7/MulTaBench); reference learner [TabPFN v2](https://github.com/PriorLabs/TabPFN). MIT license.
