# The image side of the model, in seven panels

SVG panels of the image side of `table_image_icl/model.py`: from a row's photo to the point where its patch tokens
join TabICL's row transformer. Open `preview.html` (or `preview.png`) to see them all at full size. Every number is
the one in `model.py`, `inputs.py` and `training.py` (ModelConfig: 224 px photos, 14 px patches, a 4-layer per-photo
encoder, 2 label reads, top-32 of up to 256 sources, width 128, 4 heads; lr 1e-5 for the table stages, 1e-4 for the
image side). Shapes are written as (rows, patches, width).

| File | Shows |
|------|-------|
| `0-overview.svg` | the whole strip: photo → frozen DINOv2 blocks 1–4 → 256 × 768 → LayerNorm + linear → 256 × 128 → per-photo encoder → retrieval → image in-context read → + marker → appended after the row's cells → TabICL row transformer → in-context transformer, with the tensor shape under every block and the learning rate of each part |
| `1-tokens.svg` | one photo cut into 16 × 16 = 256 patches of 14 px; DINOv2 adds CLS + 4 registers (261 tokens), runs blocks 1–4 only, CLS and registers are dropped; LayerNorm then a linear map 768 → 128 |
| `2-per-photo-encoder.svg` | a learned position embedding is added, then 4 pre-norm transformer layers; attention arcs stay inside one photo, none cross to the other photo |
| `3-retrieval.svg` | every photo, context or test, is summarised by the mean of its 256 patch tokens; cosine similarity to the context photos' mean tokens picks its k = 32 nearest (3 drawn); a context photo's nearest is itself |
| `4-labelled-read.svg` | the image in-context read (the labelled read inside it): a reading photo's patches are the queries, the patches of its k neighbours are the keys, and the values are each neighbour's label embedding repeated for every one of its patches; what flows back is a weighted sum of labels |
| `5-join.svg` | the row as the row transformer sees it, [4 CLS \| cells \| 256 patch tokens + m], pooled into the row embedding (4 × 128 = 512); the in-context transformer then attends row to row and never sees a photo |
| `6-who-attends-to-what.svg` | a table of the queries, keys and values of the five stages and whether each looks across photos or uses labels |

Each panel is 900 × 420 px with a title at the top left, a one-line caption at the bottom and a short-form math line in
a monospace box. `image_side.py` draws the whole set (`python3 docs/figures/image-side/image_side.py`); the palette,
fonts and panel frame are set once at the top of it, and `preview.png` is rendered with headless Chrome when it is
installed. The style and glyphs (photo, patch grid, label chip, token, flame, snowflake) follow the component library
in the companion figures: 2 px rounded strokes, pastel fill and darker stroke per module type, Helvetica text as real
`<text>` elements, transparent background, a `<title>` per file, no filters, images or external fonts.

| Role | Fill | Stroke / text |
|------|------|---------------|
| data, parameter-free (retrieval, the photo) | `#DBEAFE` | `#2563EB` |
| attention, trained from scratch (lr 1e-4) | `#FFEDD5` | `#EA580C` |
| MLP | `#DCFCE7` | `#16A34A` |
| TabICL's pretrained stages (lr 1e-5) and the frozen DINOv2 | `#E5E7EB` | `#6B7280` |
| tokens | `#EDE9FE` | `#7C3AED` |
| labels | `#FEF3C7` | `#D97706` |

Snowflake = frozen, flame = trained. To use a panel in a slide, drag the `.svg` onto it (Keynote and PowerPoint keep it
as a vector); for LaTeX, `rsvg-convert -f pdf 4-labelled-read.svg -o labelled-read.pdf`.
