"""Builds notebooks/image_icl.ipynb: ImageICL, the image side's in-context read, built up from a dot product, in the style of Raschka's
chapter 3 (one small step per cell, every tensor printed). Run `python notebooks/make_image_icl.py`, then execute
the notebook (`jupyter nbconvert --to notebook --execute --inplace notebooks/image_icl.ipynb`). CPU, seconds."""
import os

import nbformat as nbf

HERE = os.path.dirname(os.path.abspath(__file__))

CELLS = [
    ("md", """# ImageICL, from scratch

The image side of the model has one stage that involves labels: **ImageICL**, the image in-context read (the class is `LabelledRead` in `table_image_icl/model.py`). Every patch of every photo attends to the patches of the labelled context photos. The queries and keys are patch tokens, so content decides where to look. The values are **label embeddings**, so what comes back is a mix of the labels of the photos it looked at. What a patch reads is added to its token, and the tokens then join their row in TabICL's row stage; the prediction itself is made later, by TabICL's in-context stage.

This notebook builds that up from nothing, one step per cell:

- **Steps 1–5:** a dot product, a softmax, labels as the values, and why that is in-context learning.
- **Steps 6–13:** adding the vote into the photo's own token, one small step at a time.
- **Step 14:** how those tokens enter TabICL's row stage.

An appendix shows how the query and key projections learn *what* to compare across many small tasks.

Each photo starts as one small vector. Nothing here needs real images."""),
    ("code", '''import os, sys
os.chdir(next(p for p in [os.getcwd(), os.path.dirname(os.getcwd())] if os.path.exists(os.path.join(p, "table_image_icl"))))
sys.path.insert(0, os.getcwd())
import torch
import torch.nn.functional as F
torch.manual_seed(0); torch.set_printoptions(precision=2, sci_mode=False, linewidth=120)'''),

    # ------------------------------------------------------------------ 1
    ("md", """## 1. A tiny table of photos

Six context photos of animals with labels, and one test photo without. Each photo is a 4-number vector, standing in for what the encoder makes of it. To keep it readable, each number is how strongly the photo shows one thing:

| | stripes | spots | wings | four legs |
|---|---|---|---|---|
| zebra (label 0) | 2 or 1 | 0 | 0 | 2 |
| bird (label 1) | 0 | 0 or 1 | 2 | 0 |
| leopard (label 2) | 0 | 2 or 1 | 0 | 2 or 1 |

The test photo is a bird whose legs show a little: wings 2, four legs 1. Its label should be 1, but the model never sees it."""),
    ("code", '''#                        stripes spots wings legs
context = torch.tensor([[2.,     0.,   0.,   2.],      # zebra    label 0
                        [1.,     0.,   0.,   2.],      # zebra    label 0
                        [0.,     0.,   2.,   0.],      # bird     label 1
                        [0.,     1.,   2.,   0.],      # bird     label 1
                        [0.,     2.,   0.,   2.],      # leopard  label 2
                        [0.,     1.,   0.,   1.]])     # leopard  label 2
labels = torch.tensor([0, 0, 1, 1, 2, 2])
test = torch.tensor([0., 0., 2., 1.])                  # a bird, legs showing

print("context photos", tuple(context.shape), "(photos, width)")
print(context)
print("labels        ", labels.tolist())
print("test photo    ", test)'''),

    # ------------------------------------------------------------------ 2
    ("md", """## 2. Scores: how much does the test photo look like each context photo?

The simplest similarity is a dot product. One score per context photo:

$$s_j = x_{\\text{test}} \\cdot x_j$$

The two birds share the wings and score 4. Because the test bird's legs show, every four-legged animal also gets a little score (1 or 2): the dot product counts every feature the two photos share."""),
    ("code", '''scores = context @ test                     # (6,)
for j, (s, y) in enumerate(zip(scores, labels)):
    print(f"photo {j}  label {y.item()}  score {s.item():4.1f}")'''),

    # ------------------------------------------------------------------ 3
    ("md", """## 3. Weights: a softmax over the scores

Attention turns scores into weights that are positive and add up to 1. Before the softmax the scores are divided by $\\sqrt{d}$, with $d$ the width (4 here, so by 2), as in every transformer; it keeps the softmax from saturating when $d$ is large. The birds get about 0.30 each; the rest share what is left.

$$w_j = \\frac{\\exp(s_j / \\sqrt{d})}{\\sum_{j'} \\exp(s_{j'} / \\sqrt{d})}$$"""),
    ("code", '''d = context.shape[1]
weights = torch.softmax(scores / d ** 0.5, dim=0)
print("weights", weights)
print("sum    ", weights.sum().item())'''),

    # ------------------------------------------------------------------ 4
    ("md", """## 4. The values: ordinary attention, then labels

In ordinary self-attention the output is a weighted average of the *same* vectors you compared with. Here that gives back something that looks mostly like a bird (lots of wings), but nothing about the label itself:

$$o = \\sum_j w_j \\, x_j$$"""),
    ("code", '''ordinary = weights @ context
print("weighted average of the photos:", ordinary)
print("   (stripes, spots, wings, legs)")'''),
    ("md", """ImageICL swaps the values. Keys stay the photos, so the weights are the same, but each context photo passes on **its label** instead of its content. With one-hot labels, $e_y$ has a 1 at position $y$:

$$r = \\sum_j w_j \\, e_{y_j}$$

Every context photo votes for its own label, as strongly as the test photo attends to it. The result is a distribution over labels."""),
    ("code", '''n_labels = 3
E = F.one_hot(labels, n_labels).float()       # (6, 3): one row per context photo
print("label embeddings E (values)")
print(E)
read = weights @ E
print("read r =", read, " -> predicted label", read.argmax().item())'''),
    ("md", """`weights @ E` is the same as adding up the weights per label, which you can check by hand:"""),
    ("code", '''by_hand = torch.zeros(n_labels)
for w, y in zip(weights, labels):
    by_hand[y] += w
print(by_hand)'''),

    # ------------------------------------------------------------------ 5
    ("md", """## 5. Why this is in-context learning

Nothing in steps 2–4 depends on which label means what. Relabel the context, so the birds are now label 2 and the leopards label 1, and the same computation predicts 2 for the same test photo. No weight was trained or changed; the answer comes from the labels that came in with the context.

This is also why the model's labels are *slots* rather than class names: a task's classes land on random slots, so the model cannot memorise that slot 1 means "cat", and has to read the answer off the context."""),
    ("code", '''new_labels = torch.tensor([0, 0, 2, 2, 1, 1])
new_read = weights @ F.one_hot(new_labels, n_labels).float()
print("old labels", labels.tolist(), "-> read", read, "-> label", read.argmax().item())
print("new labels", new_labels.tolist(), "-> read", new_read, "-> label", new_read.argmax().item())'''),

    # ------------------------------------------------------------------ 6
    ("md", """## 6. A token has a fixed width

Step 4 gave the test photo a vote, $(0.22, 0.60, 0.18)$. The rest of the model only takes tokens, so the vote has to go *into* the photo's token. The token does not get longer for it: in the model a patch token is 128 numbers when it enters the read and 128 when it leaves.

So give the toy a fixed width too: 7 numbers per photo. The first 4 are the appearance numbers from step 1. The other 3 we **set to zero ourselves**, to keep space free where the vote can be read at a glance; nothing in steps 1–5 changes, since zeros add nothing to a dot product.

That is a choice for this toy, not something the model does. The real encoder uses all 128 numbers, and the vote is added on top of them. Step 12 shows that case."""),
    ("code", '''WIDTH = 7                                   # stripes spots wings legs, then 3 zeros we keep free for the vote
def token(appearance):
    t = torch.zeros(WIDTH); t[:4] = appearance
    return t

test_token = token(test)
print("test token:", test_token)'''),

    # ------------------------------------------------------------------ 7
    ("md", """## 7. Each label becomes a vector of the same width

Each label's embedding is a vector as wide as a token, 7 numbers. Here, zebra is a 1 in place 5, bird in place 6, leopard in place 7: the three places we kept free. These are the values now, instead of the one-hot rows of step 4."""),
    ("code", '''label_vectors = torch.zeros(n_labels, WIDTH)
label_vectors[0, 4] = 1                     # zebra
label_vectors[1, 5] = 1                     # bird
label_vectors[2, 6] = 1                     # leopard
print(label_vectors)'''),

    # ------------------------------------------------------------------ 8
    ("md", """## 8. The read is the same vote, now token-shaped

Same weights as step 3, same weighted sum, new values. The read is 7 numbers wide, like a token, and the vote sits in the last three:

$$r = \\sum_j w_j \\, e_{y_j}$$"""),
    ("code", '''r = weights @ label_vectors[labels]
print("r =", r)'''),

    # ------------------------------------------------------------------ 9
    ("md", """## 9. Add the read to the token

The token keeps what it had and gains what it read: $h = x + r$, 7 numbers plus 7 numbers, still 7. This is the residual connection of any transformer layer, and the same way transformers add position embeddings to word embeddings, or TabICL adds its label embedding to a context row: by adding, not by appending."""),
    ("code", '''h_test = test_token + r
print("x     ", test_token, " width", len(test_token))
print("r     ", r, " width", len(r))
print("x + r ", h_test, " width", len(h_test))'''),

    # ------------------------------------------------------------------ 10
    ("md", """## 10. The context photos do the same

Every photo reads, the context photos too. They read from the context photos, themselves included, so each mostly reads back its own label. That is allowed: the context labels are inputs. The test photo is never read from, so its label is never used."""),
    ("code", '''tokens = torch.stack([token(x) for x in context] + [test_token])      # 6 context photos, then the test photo
w = torch.softmax(tokens @ tokens[:6].T / 4 ** 0.5, dim=-1)           # every photo against the 6 context photos
h = tokens + w @ label_vectors[labels]

print("          stripes spots wings legs  zebra bird leopard")
for name, t in zip(["zebra", "zebra", "bird", "bird", "leopard", "leopard", "TEST"], h):
    print(f"{name:8s}", t)'''),

    # ------------------------------------------------------------------ 11
    ("md", """## 11. Change the labels, and only the vote changes

As in step 5, make the birds label 2 and the leopards label 1. The test photo looks the same, so its first four numbers stay put; what the context says about it has moved."""),
    ("code", '''h_new = tokens + w @ label_vectors[new_labels]
print("old labels:", h[6])
print("new labels:", h_new[6])'''),

    # ------------------------------------------------------------------ 12
    ("md", """## 12. In the model, nothing is set aside

The tidy layout above, appearance in the first 4 numbers and the vote in the last 3, was our choice, for reading. In the model nothing is kept free: the image encoder uses all 128 numbers, and the label embeddings are 10 learned directions in that same space, not places of their own.

To see that the read works just as well that way, turn the whole toy by a random rotation of the 7-dimensional space: every photo token and every label vector. Now no single number means "wings" or "bird"; every number is a mix."""),
    ("code", '''torch.manual_seed(1)
turn, _ = torch.linalg.qr(torch.randn(WIDTH, WIDTH))                 # a random rotation: turn @ turn.T = I
tokens_mixed, labels_mixed = tokens @ turn, label_vectors @ turn

w_mixed = torch.softmax(tokens_mixed @ tokens_mixed[:6].T / 4 ** 0.5, dim=-1)
h_mixed = tokens_mixed + w_mixed @ labels_mixed[labels]
print("test token, mixed:", h_mixed[6], " width", len(h_mixed[6]))
print("same weights as before:", torch.allclose(w_mixed, w, atol=1e-6))'''),
    ("md", """The numbers look like noise, but nothing is lost. A rotation keeps every dot product, so the weights are the same as in step 10. And the vote can be read back by a dot product with each label's direction, which is what a later layer can learn to do:

$$h \\cdot e_c \\;=\\; \\underbrace{x \\cdot e_c}_{0} \\;+\\; r \\cdot e_c \\;=\\; p_c$$

The first term is 0 only because the toy made the appearance exactly perpendicular to the label directions. In the model nothing guarantees that: a photo's content has some part along every label direction (and the directions the vote really lands on are $e_c W_V W_O$, after the value and output projections). That part shows up as crosstalk when a later layer reads the vote back. Training can push the content away from those directions, since crosstalk that looks like a vote would hurt the prediction, but nothing forces it to zero."""),
    ("code", '''print("vote read back, h . e_c:", h_mixed[6] @ labels_mixed.T)
print("vote from step 4:      ", read)'''),

    # ------------------------------------------------------------------ 13
    ("md", """## 13. Patches: every patch reads for itself

In the model a photo is not one token but 256 patch tokens, and each patch reads on its own: its keys are the patches of the context photos, and a key patch's value is the label of the photo it belongs to. So two patches of one photo can read different things.

A bird in front of a zebra, in two patches (the context photos stay one patch each to keep it small):"""),
    ("code", '''patches = torch.stack([token(torch.tensor([0., 0., 2., 0.])),       # the bird part
                       token(torch.tensor([2., 0., 0., 1.]))])      # the zebra behind it
w_patches = torch.softmax(patches @ tokens[:6].T / 4 ** 0.5, dim=-1)
patch_tokens = patches + w_patches @ label_vectors[labels]
print("          stripes spots wings legs  zebra bird leopard")
print("bird part ", patch_tokens[0])
print("zebra part", patch_tokens[1])'''),
    ("md", """Both tokens move on as they are; nothing averages them into one vote."""),

    # ------------------------------------------------------------------ 14
    ("md", """## 14. Into TabICL's row stage

TabICL turns each table cell into a token, and keeps 4 places at the front of the row for its CLS tokens. Our patch tokens are put **after the cell tokens, in the same row**:

```
row = [CLS × 4 | one token per table column | 256 patch tokens]
```

Before that, a learned **patch marker** is added to every patch token so the row stage can tell patches from cells. Here are the shapes at the model's real sizes, with the real stage: 40 photos (36 context, 4 test), width 128, and a stand-in for TabICL's tokens of a 3-column table."""),
    ("code", '''from table_image_icl.model import LabelledRead
torch.manual_seed(0)
stage = LabelledRead(width=128, heads=4, layers=2, n_slots=10, max_sources=256, read="top_k", top_k=32).eval()
patch_marker = torch.randn(128) * 0.02

image_tokens = torch.randn(1, 40, 256, 128)          # from the image encoder: (batch, photos, patches, width)
y_context = torch.randint(10, (1, 36))               # the 36 context photos' label slots
with torch.no_grad():
    patch_tokens = stage(image_tokens, y_context) + patch_marker
cell_tokens = torch.zeros(1, 40, 4 + 3, 128)         # TabICL's: (batch, rows, CLS + columns, width)
rows = torch.cat([cell_tokens, patch_tokens], dim=2)

print("patch tokens in    ", tuple(image_tokens.shape))
print("patch tokens out   ", tuple(patch_tokens.shape), "  same shape, label vote added")
print("into the row stage ", tuple(rows.shape), "  4 CLS + 3 cells + 256 patches per row")'''),
    ("md", """From here it is TabICL's own machinery: 3 transformer blocks over each row (cells and patches attend to each other), the 4 CLS outputs joined into the row's vector (4 × 128 = 512), and the in-context stage, which adds TabICL's own label embedding to the context rows and predicts the test rows. Nothing in the read has a loss of its own; it is all trained end to end through that prediction.

## What the real stage adds to the toy

The toy is the core. The real stage, `LabelledRead`, adds, in order:

- **Learned label vectors.** 10 label slots, each a learned direction in the token's 128 dimensions, as in step 12; they start orthogonal to each other.
- **Projections.** Queries and keys go through $W_Q$ and $W_K$ (learned: *what* to compare, see the appendix); the label vectors through $W_V$ and $W_O$. Those are linear, so the read is still the vote, over 10 learned label directions.
- **4 heads**, each with its own weights and its own vote.
- **LayerNorms** before each read.
- **An MLP, then a second read.** The MLP mixes appearance and vote in one token. In the second read the queries carry the vote; the keys are still the label-free tokens.
- **The 32 nearest context photos.** Each photo reads only the patches of the 32 context photos most like it (mean patch token, cosine).

## Check: the real code, set to the toy

Set `LabelledRead` to one head, one read, the identity for every projection, no LayerNorm, the MLP at zero, and our label vectors. It gives back the tokens of step 10. (PyTorch divides by $\\sqrt{7}$, the token width; scaling $W_Q$ by $\\sqrt{7}/2$ turns that back into the $\\sqrt{4}$ of step 3.)"""),
    ("code", '''check = LabelledRead(WIDTH, heads=1, layers=1, n_slots=n_labels, max_sources=256, read="all").eval()
with torch.no_grad():
    check.label.weight.copy_(label_vectors)
    att = check.reads[0].attention
    for lin in (att.q, att.k, att.v, att.o):
        lin.weight.copy_(torch.eye(WIDTH)); lin.bias.zero_()
    att.q.weight.mul_(WIDTH ** 0.5 / 2)
    check.reads[0].norm_readers = check.reads[0].norm_sources = torch.nn.Identity()
    check.mlps[0][-1].weight.zero_(); check.mlps[0][-1].bias.zero_()
    out = check(tokens[None, :, None], labels[None])[0, :, 0]       # (batch 1, photos 7, patches 1, width)
print("test photo:", out[6])
print("matches step 10:", torch.allclose(out, h, atol=1e-5))'''),

    # ------------------------------------------------------------------ appendix
    ("md", """## Appendix: learning *what* to compare

So far the similarity was a plain dot product of the raw vectors. That only works if the vectors are already arranged so that "similar" means "same class". Real photo tokens also carry things that have nothing to do with the label: lighting, background, framing.

The fix is the one every attention layer uses. Each photo gets a query and a key through two learned matrices, and the score compares those:

$$q = x_{\\text{test}} W_Q, \\qquad k_j = x_j W_K, \\qquad s_j = q \\cdot k_j$$

The values stay the label embeddings. The matrices are not trained on any one task. They are trained across **many** small tasks, each with its own classes and its own random label slots, so the only thing they can learn is *which parts of a photo to compare*. At test time they are fixed, and a new task's answer comes from its context labels.

The toy below trains $W_Q$ and $W_K$ with the read itself as the prediction, to show the idea in isolation. In the model there is no loss on the read: it is trained end to end, through the row stage, the in-context stage and the output head, together with everything else on the image side.

Photos here get 4 more numbers for the **background**: grass, sky, snow, indoors. Backgrounds are loud (3, bigger than any animal feature) and say nothing about the label. Take the animals of step 1 photographed where you would expect them, and a test bird that happens to stand on grass. A plain dot product sees "grass" and votes zebra."""),
    ("code", '''#                        stripes spots wings legs | grass sky snow indoors
farm = torch.tensor([[[2.,     0.,   0.,   2.,     3.,   0.,  0.,  0.],     # zebra on grass    label 0
                      [1.,     0.,   0.,   2.,     3.,   0.,  0.,  0.],     # zebra on grass    label 0
                      [0.,     0.,   2.,   0.,     0.,   3.,  0.,  0.],     # bird in the sky   label 1
                      [0.,     1.,   2.,   0.,     0.,   3.,  0.,  0.],     # bird in the sky   label 1
                      [0.,     2.,   0.,   2.,     0.,   0.,  3.,  0.],     # leopard in snow   label 2
                      [0.,     1.,   0.,   1.,     0.,   0.,  3.,  0.]]])   # leopard in snow   label 2
farm_labels = torch.tensor([[0, 0, 1, 1, 2, 2]])
bird_on_grass = torch.tensor([[[0., 0., 2., 1., 3., 0., 0., 0.]]])
print("scores, plain dot product:", (farm @ bird_on_grass.transpose(1, 2))[0, :, 0].tolist())'''),
    ("md", """So the model has to learn *what* to compare, and it cannot learn it from this one task. It learns it from many: below, every training task gets fresh random classes (a random prototype in the 4 animal dims each) and fresh random backgrounds (noise of size 2 in the other 4), and its classes land on random label slots. These random tasks are the training data, like the synthetic tables of the real model; the hand-made farm is only for looking at."""),
    ("code", '''WIDTH, N_CLASSES, PER_CLASS, N_TEST = 8, 3, 3, 6    # 8 appearance dims: 4 animal, 4 background

def make_tasks(batch):
    """Tasks of N_CLASSES classes. Each class has a random prototype in the first 4 dims; every photo adds
    nuisance noise (std 2) in the last 4. Labels are random slots per task."""
    protos = torch.randn(batch, N_CLASSES, 4)
    y = torch.arange(N_CLASSES).repeat_interleave(PER_CLASS).expand(batch, -1)          # context classes
    y_test = torch.randint(N_CLASSES, (batch, N_TEST))
    def photos(cls):
        content = protos.gather(1, cls[..., None].expand(-1, -1, 4)) + 0.3 * torch.randn(*cls.shape, 4)
        return torch.cat([content, 2.0 * torch.randn(*cls.shape, 4)], -1)
    slots = torch.stack([torch.randperm(N_CLASSES) for _ in range(batch)])               # class -> label slot
    return photos(y), slots.gather(1, y), photos(y_test), slots.gather(1, y_test)

def labelled_read(test, context, labels, Wq, Wk):
    """test (batch, m, width), context (batch, n, width), labels (batch, n) -> label distribution (batch, m, N_CLASSES)"""
    q, k = test @ Wq, context @ Wk
    w = torch.softmax(q @ k.transpose(1, 2) / WIDTH ** 0.5, dim=-1)
    return w @ F.one_hot(labels, N_CLASSES).float()

def accuracy(Wq, Wk, batch=2000):
    with torch.no_grad():
        xc, yc, xt, yt = make_tasks(batch)
        return (labelled_read(xt, xc, yc, Wq, Wk).argmax(-1) == yt).float().mean().item()

I = torch.eye(WIDTH)
print(f"plain dot product (Wq = Wk = I): accuracy {accuracy(I, I):.2f} on random tasks  (chance is {1 / N_CLASSES:.2f})")
print("the farm, bird on grass, read:", labelled_read(bird_on_grass, farm, farm_labels, I, I)[0, 0])'''),
    ("md", """Now train $W_Q$ and $W_K$. The loss is the cross-entropy of the read: the log of the weight the read puts on the test photo's true label slot. Every step is a fresh batch of tasks."""),
    ("code", '''Wq = (torch.eye(WIDTH) + 0.1 * torch.randn(WIDTH, WIDTH)).requires_grad_()
Wk = (torch.eye(WIDTH) + 0.1 * torch.randn(WIDTH, WIDTH)).requires_grad_()
opt = torch.optim.Adam([Wq, Wk], lr=0.02)
for step in range(1501):
    xc, yc, xt, yt = make_tasks(128)
    r = labelled_read(xt, xc, yc, Wq, Wk)
    loss = F.nll_loss(torch.log(r + 1e-9).flatten(0, 1), yt.flatten())
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 300 == 0:
        print(f"step {step:4d}  loss {loss.item():.3f}  accuracy on new tasks {accuracy(Wq.detach(), Wk.detach()):.2f}")'''),
    ("md", """What did it learn? The score is $x_{\\text{test}} W_Q W_K^\\top x_j^\\top$, so $W_Q W_K^\\top$ says which dimensions of one photo get compared with which of the other. The top-left block (animal dims) should be large, the bottom-right (background) near zero: the model has learned to ignore the background and compare the animals, without ever being told which dims were which."""),
    ("code", '''M = (Wq @ Wk.T).detach()
print("W_Q W_K^T  (rows: test photo dims, columns: context photo dims; 0-3 animal, 4-7 background)")
print(M)
print(f"mean |entry|, animal block    : {M[:4, :4].abs().mean():.2f}")
print(f"mean |entry|, background block: {M[4:, 4:].abs().mean():.2f}")'''),
    ("md", """Back to the farm, which the model never trained on. The bird on grass is now read as a bird. And it is still in-context: swap the bird and zebra labels and the answer follows, with $W_Q$ and $W_K$ frozen."""),
    ("code", '''with torch.no_grad():
    r = labelled_read(bird_on_grass, farm, farm_labels, Wq, Wk)[0, 0]
    swapped = labelled_read(bird_on_grass, farm, torch.tensor([1, 0, 2])[farm_labels], Wq, Wk)[0, 0]
print("bird on grass, trained read          :", r, "-> label", r.argmax().item())
print("same, zebra and bird labels swapped  :", swapped, "-> label", swapped.argmax().item())'''),

]


def main():
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src) for kind, src in CELLS]
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    path = os.path.join(HERE, "image_icl.ipynb")
    nbf.write(nb, path)
    print("wrote", path)


if __name__ == "__main__":
    main()
