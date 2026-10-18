# The image in-context read, in detail

A verbose explainer of one stage of the model: the image in-context read (class `LabelledRead` in
`table_image_icl/model.py`, with `LabelRead`, `Attention`, `nearest_photos` and the patch marker added in
`TableImageICL.photo_tokens`). Open `preview.html` (or `preview.png`) to see all four figures.

| File | Shows |
|------|-------|
| `full-dataflow.svg` | 900 × 1400: every tensor and operation of one table's forward pass, top to bottom, with shapes: inputs, `pick_sources`, the source tokens and label embeddings, retrieval (mean tokens, cosine matrix, top-32), layer 1 in full (the two LayerNorms, the per-reader gather of keys and values, the four linears, the 4 heads of 32, scores, softmax, the weighted sum of label embeddings, the residual, the MLP), layer 2 with the same sources and neighbours, the output and the marker. Tensors that involve labels are yellow, content-only ones purple; the chunk loop is marked as an implementation detail. Parameter counts are read from the module. |
| `example-1.svg` | a test photo whose two nearest context photos both carry label 1: cosines, N(t), the per-patch attention weights over the 4 neighbour patches, the read r = e1 exactly, and the tokens the photo carries out |
| `example-2.svg` | a test photo between a label-1 and a label-0 photo: one patch reads mostly e1, the other mostly e0; the photo average is about half and half, the vote is soft and per patch |
| `example-3.svg` | a context photo with label 0: its nearest is itself, it reads back mostly e0; why that is allowed, and that the test photo's label is never an input of the stage |
| `examples.json` | every number drawn in the example panels, written by the generator from the module's own output (plus the full-size parameter counts) |

The examples run the real `LabelledRead` with `width 4, heads 1, layers 1, n_slots 3, max_sources 8, read "top_k",
top_k 2`, its weights set by hand (identity Wq/Wk/Wv/Wo with zero bias, LayerNorms at weight 1 and bias 0, one-hot
label rows, the MLP's second linear zeroed), on photos of 2 patches whose tokens are the zero-mean unit-variance
patterns A = (+1 +1 −1 −1), B = (+1 −1 +1 −1), C = (+1 −1 −1 +1). The generator recomputes each intermediate from the
module's parts under `torch.no_grad`, asserts that the read matches `LabelRead`'s own output and that the stage's
`forward` gives tokens + r, and only then draws the numbers; nothing in the SVGs is typed by hand.

Regenerate everything with

    PYTHONPATH=. python3 docs/figures/image-icl/image_icl.py

from the repository root (it imports `table_image_icl.model`, so torch and the package must be importable).
`preview.png` is rendered with headless Chrome when `/Applications/Google Chrome.app` exists. Style and palette are
the ones of `docs/figures/image-side` (2 px rounded strokes, pastel fill + darker stroke per module type, Helvetica
text, transparent background, a `<title>` per file, no filters, images or external fonts): purple = content only,
yellow = involves labels, orange = trained attention and norms, green = MLP, blue = parameter-free operations.

## Explorable

`explorable.html` is the same stage as six short interactive steps: the table, who looks like whom (hover, slider for k), the read (click a patch, see the attention lines and the label mix), make it soft (slide the test photo between two patterns), no cheating, then what. One file, no server needed: double-click it. Every number is recomputed in the page from the patch vectors and checked against examples.json in the footer.
