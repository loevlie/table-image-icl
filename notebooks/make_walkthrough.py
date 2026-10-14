"""Builds notebooks/walkthrough.ipynb: the model one stage at a time, each stage's code next to a run of it on a
table small enough to read. Run `python notebooks/make_walkthrough.py`, then execute the notebook
(`jupyter nbconvert --to notebook --execute --inplace notebooks/walkthrough.ipynb`). CPU, under a minute."""
import os

import nbformat as nbf

HERE = os.path.dirname(os.path.abspath(__file__))

CELLS = [
    ("md", """# The model, one stage at a time

The model is TabICL v2 with photos in its rows: a column stage, a row stage and an in-context stage, pretrained on synthetic tables, plus an image side that turns each row's photo into tokens that join the row stage next to the row's cells. This notebook takes one small table through it. Every section prints the code of the stage as it is in `table_image_icl/` and then runs it, so the shapes and a few values can be checked against the prose. The table stages carry TabICL's released weights; the image side is untrained and tiny, since what matters here is what each stage is allowed to see, not what it predicts."""),
    ("code", '''import os, sys, inspect
os.chdir(next(p for p in [os.getcwd(), os.path.dirname(os.getcwd())] if os.path.exists(os.path.join(p, "table_image_icl"))))
sys.path.insert(0, os.getcwd())
import torch
torch.manual_seed(0); torch.set_printoptions(precision=2, sci_mode=False, linewidth=120)

def show(*objects):
    for o in objects:
        print(inspect.getsource(o))

def changed(a, b, dims=(-1,)):
    """1 where a differs from b by more than float noise, after collapsing the given dims"""
    d = (a - b).abs()
    for dim in sorted((x % a.dim() for x in dims), reverse=True):
        d = d.amax(dim)
    return (d > 1e-4).int()'''),

    # ------------------------------------------------------------------ 1. inputs
    ("md", """## 1. The inputs

One task is a table whose first rows are the context (values and label known) and whose last rows are the test rows (values known, label to predict). Here: 3 columns, 5 context rows, 3 test rows, one task in the batch. The first two columns are the same numbers in different units, the third has a missing value.

Labels are label slots, not class ids: a task's classes sit on any of 10 slots, so the model cannot learn that slot 0 means anything in particular. This task uses slots 3 and 5. The photos are 28 px, so arm `pixels` with 14 px patches gives a 2 x 2 grid of 4 patches per photo."""),
    ("code", '''from table_image_icl.model import TableImageICL, ModelConfig

nan = float("nan")
x = torch.tensor([[[1.0, 10.0, 0.5],      # context rows: values and label known
                   [2.0, 20.0, 0.1],
                   [3.0, 30.0, nan],
                   [4.0, 40.0, 0.9],
                   [5.0, 50.0, 0.3],
                   [1.5, 15.0, 0.4],      # test rows: label to predict
                   [4.5, 45.0, 0.2],
                   [6.0, 60.0, 0.8]]])
y_context = torch.tensor([[3, 5, 3, 5, 3]])
y_test = torch.tensor([[3, 5, 5]])
n_context = y_context.shape[1]
images = torch.randn(1, x.shape[1], 3, 28, 28)                        # one random 28 px photo per row

cfg = ModelConfig(image_size=28, patch_size=14, encoder_layers=1, patch_layers=1)
model = TableImageICL(cfg, "pixels").eval()                               # TabICL's weights, a fresh image side
print("x        ", tuple(x.shape), "(batch, rows, columns)")
print("y_context", tuple(y_context.shape), "label slots of the context rows:", y_context[0].tolist())
print("images   ", tuple(images.shape), "(batch, rows, channels, height, width)")
for name, module in model.named_children():
    print(f"{name:13s} {sum(p.numel() for p in module.parameters()) / 1e6:6.2f}M parameters", "(TabICL v2, pretrained)" if name == "table" else "(ours)")'''),

    # ------------------------------------------------------------------ 2. columns
    ("md", """## 2. The column stage

TabICL first standardises every column with the mean and spread of its context rows, fills missing cells with the mean, and softens outliers, so units do not matter and a test row cannot shift the scale. Then each column is handled on its own: its cells attend to each other through a set of inducing tokens, with the context cells' labels added, so a cell comes out knowing where it sits in its column's distribution and how that relates to the label. The output is one vector per cell, plus four CLS tokens per row that the row stage will pool. TabICL wants classes numbered 0..k-1, so the slots are renumbered first."""),
    ("code", '''from table_image_icl.tables import prepare_table, contiguous_codes

show(prepare_table, contiguous_codes)'''),
    ("code", '''z = prepare_table(x, n_context)
codes, slots = contiguous_codes(y_context)
print("standardised table (columns 0 and 1 agree; the missing cell is 0):"); print(z[0])
print("codes:", codes[0].tolist(), "| slot of each code:", slots[0, :2].tolist())
with torch.no_grad():
    cells = model.table.col_embedder._train_forward(z, codes, None, False)
print("cells", tuple(cells.shape), "(batch, rows, 4 CLS + columns, width)")'''),
    ("md", """What the stage lets through: change one cell of test row 7 and see which rows' tokens move."""),
    ("code", '''x_moved = x.clone(); x_moved[0, 7, 0] = 100.0
with torch.no_grad():
    cells_moved = model.table.col_embedder._train_forward(prepare_table(x_moved, n_context), codes, None, False)
print("rows whose tokens changed when test row 7, column 0 was moved:", changed(cells_moved, cells, dims=(-1, -2))[0].tolist())'''),

    # ------------------------------------------------------------------ 3. image side
    ("md", """## 3. The image side

With photos, every row carries one. Three steps turn it into tokens that join the row: a patch embedder, a small encoder over the patches of one photo, and the labelled read, where the patches of every photo read labels from the context photos.

The patch embedder is the one thing that differs between the arms. Arm `pixels` is ViT's linear patch embedding trained from scratch, one token per `patch_size` x `patch_size` square. Arm `dino` takes the patch tokens of a frozen DINOv2 ViT-B/14 block, computed outside the model, through a layer norm and one trained linear layer: 768 wide, 16 x 16 = 256 tokens for a 224 px photo. Arm `dino_cls` does the same with DINOv2's final CLS and register tokens."""),
    ("code", '''from table_image_icl.images import PixelPatches, DinoPatches, ImageEncoder

show(PixelPatches, DinoPatches)'''),
    ("code", '''flat = images.flatten(0, 1)
with torch.no_grad():
    tokens = model.embed(flat)
    moved = flat.clone(); moved[0, :, :14, :14] += 1.0                  # touch the top-left patch of photo 0
    tokens_moved = model.embed(moved)
print("images", tuple(images.shape), "-> flattened", tuple(flat.shape), "-> patch tokens", tuple(tokens.shape), "(photos, patches, width)")
print("tokens of photo 0 changed by touching its top-left patch:", changed(tokens_moved[0], tokens[0]).tolist())
dino_input = torch.randn(x.shape[1], 256, 768)
print("arm dino: input per photo", tuple(dino_input.shape), "->", tuple(DinoPatches(128, 768)(dino_input).shape))'''),
    ("md", """The encoder adds a learned position to each patch and runs a few transformer layers over the patches of one photo. No labels and no other photos are involved; photos in the batch do not see each other."""),
    ("code", '''show(ImageEncoder.forward)
with torch.no_grad():
    encoded = model.encoder(tokens)
    encoded_moved = model.encoder(tokens_moved)
print("encoder", tuple(tokens.shape), "->", tuple(encoded.shape))
print("photos whose tokens changed when photo 0 was touched:", changed(encoded_moved, encoded, dims=(-1, -2)).tolist())'''),
    ("md", """The labelled read is this model's own. It is a cross-attention from every patch of every photo to the patches of the context photos: the query and the key are patch content, but the value is the label of the photo the key patch belongs to. So a test patch that looks like the patches of context photos labelled 5 reads back "5". A context photo is among its own sources, since the context is given. Relabelling one context photo changes what every photo reads back, while changing a test photo changes that photo only."""),
    ("code", '''from table_image_icl.reading import LabelRead, LabelledRead

show(LabelRead.forward, LabelledRead.forward)'''),
    ("code", '''encoded_rows = encoded.unflatten(0, (1, x.shape[1]))                   # back to (batch, rows, patches, width)
y_other = y_context.clone(); y_other[0, 1] = 3                       # relabel context photo 1
with torch.no_grad():
    read = model.read(encoded_rows, y_context)
    read_other = model.read(encoded_rows, y_other)
    test_moved = encoded_rows.clone(); test_moved[0, 7, 0, 0] += 1.0  # move one value of test photo 7
    read_moved = model.read(test_moved, y_context)
print("labelled read", tuple(encoded_rows.shape), "->", tuple(read.shape))
print("photos whose patches changed when context photo 1 was relabelled:", changed(read_other, read, dims=(-1, -2))[0].tolist())
print("photos whose patches changed when test photo 7 was changed:      ", changed(read_moved, read, dims=(-1, -2))[0].tolist())'''),

    # ------------------------------------------------------------------ 4. top-k
    ("md", """## 4. The top-k read: retrieve, then read

The trained setting is `patch_read = "top_k"` with `patch_top_k = 32`: before reading, every photo picks the 32 context photos that look most like it and reads labels from their patches only. With 5 context photos, as above, that is every one of them, so the read so far was the plain one. Two steps, both in `table_image_icl/reading.py`.

Retrieve: each photo is summarised by the mean of its patch tokens, the cosine similarity between its summary and every context photo's summary is taken, and the most similar context photos are kept. A nearest-neighbour lookup on content, with no labels involved. A context photo is its own nearest, so it reads its own label among others."""),
    ("code", '''from table_image_icl.reading import nearest_photos

show(nearest_photos)
with torch.no_grad():
    sources = encoded_rows[:, :n_context]
    summary = torch.nn.functional.normalize(encoded_rows.mean(2), dim=-1)
    similarity = summary @ torch.nn.functional.normalize(sources.mean(2), dim=-1).transpose(1, 2)
    nearest = nearest_photos(encoded_rows, sources, k=2)
print("cosine similarity of each photo's summary to each context photo's (photos x context):")
print(similarity[0])
print("the 2 nearest context photos of each photo:", nearest[0].tolist())
print("every context photo is among its own nearest:", all(i in nearest[0, i].tolist() for i in range(n_context)))'''),
    ("md", """Read: the `nearest` branch of `LabelRead.forward`, printed in section 3. For each photo, the patches of its neighbours are gathered as the keys and the neighbours' labels as the values, and the photo's patches attend over them. So a photo's query sees `top_k` x patches keys whatever the size of the context, which is why a model trained on at most 256 rows can be scored with 10,000 context rows unchanged: the attention problem it faces is the same.

The setting is an attribute, not a weight: the read built above is switched to 2 neighbours here. Relabelling a context photo now changes only the photos that retrieved it."""),
    ("code", '''model.read.top_k = 2
far = next(i for i in range(n_context) if i not in nearest[0, 7].tolist())    # a context photo test photo 7 did not retrieve
near = nearest[0, 7, 0].item()                                                   # one it did
y_far, y_near = y_context.clone(), y_context.clone()
y_far[0, far], y_near[0, near] = 7, 7
with torch.no_grad():
    base = model.read(encoded_rows, y_context)
    read_far, read_near = model.read(encoded_rows, y_far), model.read(encoded_rows, y_near)
print("photos that retrieved context photo", far, ":", [i for i in range(x.shape[1]) if far in nearest[0, i].tolist()])
print("photos whose patches changed when it was relabelled:  ", changed(read_far, base, dims=(-1, -2))[0].tolist())
print("photos that retrieved context photo", near, ":", [i for i in range(x.shape[1]) if near in nearest[0, i].tolist()])
print("photos whose patches changed when it was relabelled:  ", changed(read_near, base, dims=(-1, -2))[0].tolist())
model.read.top_k = 32                                                            # back to the trained setting'''),

    # ------------------------------------------------------------------ 5. rows
    ("md", """## 5. The row stage

`TableImageICL.photo_tokens` strings the three image steps together and adds a learned marker that tells the row stage a token is a patch and not a cell. The row stage is TabICL's: a row's tokens, its 4 CLS tokens, its cells and now its patches, attend to each other, and the CLS tokens are read out and concatenated, so a row becomes one vector of width 4 x 128. Each row is handled on its own."""),
    ("code", '''show(TableImageICL.photo_tokens)
with torch.no_grad():
    patches = model.photo_tokens(images, y_context)
    rows = model.table.row_interactor._train_forward(torch.cat([cells, patches], 2), None)
    images_moved = images.clone(); images_moved[0, 7, :, :14, :14] += 1.0                     # touch test row 7's photo
    rows_moved = model.table.row_interactor._train_forward(torch.cat([cells, model.photo_tokens(images_moved, y_context)], 2), None)
print("photo tokens", tuple(patches.shape), "-> rows with photos", tuple(rows.shape),
      "| tokens per row: 4 CLS +", x.shape[2], "cells +", patches.shape[2], "patches")
print("rows changed by touching test row 7's photo:", changed(rows_moved, rows)[0].tolist())'''),

    # ------------------------------------------------------------------ 6. in-context
    ("md", """## 6. The in-context stage and the output

Now rows attend to rows. The context rows get their label added; then every row attends to the context rows, and the test rows' vectors are scored over the k classes of the task. A test row reads which labelled rows it resembles; because the keys are the context rows, a test row never influences another row. The scores over the codes are put back on the task's slots, and slots absent from the context get no probability. That is `TableImageICL.forward`: the stages above in sequence."""),
    ("code", '''show(TableImageICL.forward)
with torch.no_grad():
    logits = model.table.icl_predictor._icl_predictions(rows, codes)[:, n_context:]
    log_probs = model(x, y_context, images=images)
    rows_test_moved = rows.clone(); rows_test_moved[0, 6, 0] += 1.0                           # move one value of test row 6
    rows_context_moved = rows.clone(); rows_context_moved[0, 0, 0] += 1.0                     # move one value of context row 0
    out = model.table.icl_predictor._icl_predictions(rows, codes)
    out_test_moved = model.table.icl_predictor._icl_predictions(rows_test_moved, codes)
    out_context_moved = model.table.icl_predictor._icl_predictions(rows_context_moved, codes)
print("in-context stage", tuple(rows.shape), "-> logits over the codes", tuple(logits.shape))
print("rows whose scores changed by changing test row 6:   ", changed(out_test_moved, out)[0].tolist())
print("rows whose scores changed by changing context row 0:", changed(out_context_moved, out)[0].tolist())
print("forward", tuple(log_probs.shape), "(batch, test rows, label slots); probabilities per test row, slots 3 and 5 only:")
print(log_probs[0].exp())
print("largest difference from the stage-by-stage logits:", f"{(log_probs[0][:, [3, 5]] - logits[0][:, :2].log_softmax(-1)).abs().max():.1e}")'''),
    ("md", """The training loss is the negative log-likelihood of the true slot over the test rows whose slot appears in the context; a test row whose class the context never shows is left out, since nothing in the task could predict it."""),
    ("code", '''from table_image_icl.training import task_loss

show(task_loss)
loss, accuracy = task_loss(log_probs, y_context, y_test)
print(f"loss {loss:.3f} (chance for two classes is log 2 = 0.693), accuracy {accuracy:.2f}")
y_unseen = torch.tensor([[3, 5, 7]])                                   # slot 7 is in no context row
loss_unseen, _ = task_loss(log_probs, y_context, y_unseen)
print(f"with test row 7 labelled with an unseen slot, the loss is over two rows only: {loss_unseen:.3f}")'''),
]


def build(path=os.path.join(HERE, "walkthrough.ipynb")):
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src) for kind, src in CELLS]
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nbf.write(nb, path)
    return path


if __name__ == "__main__":
    print("wrote", build())
