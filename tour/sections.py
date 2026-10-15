"""The annotated tour of the model's code (tour/app.py serves it). Every section is a line range of one of our files
with a one-sentence lead that says what the lines do, a note that holds the rest (the why, the detail, the upstream
functions) for the reader who expands it, where the code comes from when it comes from elsewhere, and which test shows
that a re-implementation matches its source. Links: `url` is the upstream file at the commit we pin; `local` is the
same file as installed here, (package, relative path, first line, last line), shown next to ours."""
from dataclasses import dataclass, field

TABICL = "https://github.com/soda-inria/tabicl/blob/0dbff3e/tabicl/"
DINO = "https://github.com/facebookresearch/dinov2/blob/7764ea0f912e53c92e82eb78a2a1631e92725fc8/"
TABPFN = "https://github.com/PriorLabs/TabPFN/blob/main/src/tabpfn/"
MULTA = "https://github.com/alanarazi7/MulTaBench/blob/eebb15e/"
TORCH = "https://github.com/pytorch/pytorch/blob/v2.6.0/torch/"


@dataclass
class Link:
    label: str
    url: str
    local: tuple | None = None          # ("tabicl" | "dinov2" | "torch", "relative/path.py", first_line, last_line)


@dataclass
class Section:
    start: int
    end: int
    note: str                           # HTML: what the lead does not say, shown on demand
    lead: str = ""                      # one plain sentence, at most 18 words: what the lines do
    fold: bool = False                  # folded by default: a docstring, imports, constants or an argument list
    links: list = field(default_factory=list)
    checks: list = field(default_factory=list)   # pytest node ids that show the claim


@dataclass
class File:
    path: str
    title: str
    blurb: str                          # HTML
    sections: list


def L(label, url, local=None):
    return Link(label, url, local)


FILES = [
    File("table_image_icl/model.py", "The model: TabICL v2's table stages with the image side in the rows",
         """<p>One file holds the whole model. The three table stages are TabICL v2's, loaded with its released weights and
         called on their training path. The image side (ours) turns each row's photo into patch tokens that read labels from
         the context photos; those tokens are concatenated to the row's cells between the column stage and the row stage.
         Without photos the model is TabICL v2 on the table.</p>""",
         [
             Section(1, 23, fold=True,
                 lead="Module docstring: the four stages in order, the five encoder names, and the label slots",
                 note="""<p>The encoder setting names where the patch tokens come from: "block4" (the released run), "block8",
                 "block12" and "cls" take a frozen DINOv2 ViT-B/14's tokens from inputs.py; "pixels" is a patch embedding
                 trained from scratch. The imports are torch and the standard library.</p>"""),
             Section(25, 38, fold=True,
                 lead="Names the released checkpoint on the Hub and spells out TabICL v2's constructor arguments",
                 note="""<p><code>TABICL_V2</code> repeats, argument by argument, the values stored inside the checkpoint;
                 <code>load_tabicl</code> asserts the two agree, so a new checkpoint cannot silently change the architecture.
                 <code>WIDTH</code> 128 and <code>N_SLOTS</code> 10 are what TabICL v2 fixes; the image side is built at the same
                 width so its tokens can sit next to cells. <code>OUTLIER_SIGMAS</code> is TabICL's default
                 <code>outlier_threshold</code>.</p>""",
                 links=[L("TabICL class and its arguments", TABICL + "_model/tabicl.py#L15", ("tabicl", "_model/tabicl.py", 15, 60)),
                        L("the checkpoint on the Hub", "https://huggingface.co/jingang/TabICL/tree/main"),
                        L("outlier_threshold default 4.0", TABICL + "_sklearn/classifier.py#L70", ("tabicl", "_sklearn/classifier.py", 66, 74))]),
             Section(41, 50,
                 lead="Builds TabICL from TABICL_V2 and, when pretrained, downloads the checkpoint and loads its weights",
                 note="""<p><code>hf_hub_download</code> caches the file; the config stored inside is asserted to match before
                 the weights are loaded. Tests build it fresh (<code>pretrained=False</code>) so they need no download. The
                 parameter names under <code>table.</code> are TabICL's own, which is why the released weights load
                 strictly.</p>""",
                 links=[L("TabICL.__init__", TABICL + "_model/tabicl.py#L153", ("tabicl", "_model/tabicl.py", 153, 200))],
                 checks=["tests/test_model.py::test_the_parameter_names_are_the_ones_the_released_run_was_trained_with",
                         "tests/test_model.py::test_the_released_checkpoint_loads_and_gives_the_recorded_probabilities"]),
             Section(53, 65,
                 lead="Standardises each table on its context rows, fills NaN with the mean, clips, then softens outliers",
                 note="""<p>TabICL's own preprocessing, in torch so it runs on the GPU inside the forward pass. Its sklearn
                 classifier fits <code>CustomStandardScaler</code> (z-score with 1e-6 added to the spread, clipped to ±100) and
                 <code>OutlierRemover</code> on the training rows. Line by line: <code>valid</code> marks cells that are not NaN;
                 <code>mean</code> and <code>std</code> use valid context cells only; NaN → 0 is the context mean, TabICL's
                 imputation; the last three lines are the remover's transform, <code>max(-log(1+|z|) + lower, z)</code> then
                 <code>min(log(1+|z|) + upper, z)</code>. Test rows never enter the statistics, so they cannot change the
                 scale.</p>
                 <p>The equivalence test fits TabICL's two sklearn classes on the same context rows and compares the output cell
                 by cell.</p>""",
                 links=[L("CustomStandardScaler", TABICL + "_sklearn/preprocessing.py#L397", ("tabicl", "_sklearn/preprocessing.py", 397, 470)),
                        L("OutlierRemover.transform", TABICL + "_sklearn/preprocessing.py#L366", ("tabicl", "_sklearn/preprocessing.py", 366, 395))],
                 checks=["tests/test_equivalence.py::test_prepare_table_is_tabicl_s_standard_scaler_then_its_outlier_remover_fitted_on_the_context",
                         "tests/test_model.py::test_prepare_table_standardises_on_the_context_fills_missing_cells_and_softens_outliers",
                         "tests/test_model.py::test_test_rows_do_not_change_the_scale"]),
             Section(68, 80,
                 lead="Computes the two-stage outlier bounds: mean ± 4 spreads of the context after dropping first-stage outliers",
                 note="""<p><code>OutlierRemover.fit</code> in torch. <code>stats</code> uses the sample spread (divisor n−1, as
                 numpy's <code>nanstd(ddof=1)</code>) over the cells marked <code>keep</code>; <code>inlier</code> keeps the cells
                 within 4 first-stage spreads of the first-stage mean, and the bounds come from the second pass.</p>""",
                 links=[L("OutlierRemover.fit (two stages)", TABICL + "_sklearn/preprocessing.py#L315", ("tabicl", "_sklearn/preprocessing.py", 315, 364))],
                 checks=["tests/test_equivalence.py::test_prepare_table_is_tabicl_s_standard_scaler_then_its_outlier_remover_fitted_on_the_context"]),
             Section(83, 92,
                 lead="Renumbers each table's label slots to codes 0..k−1 and remembers which slot each code stands for",
                 note="""<p>Ours. TabICL's in-context stage (<code>_icl_predictions</code>) wants classes numbered 0..k−1 per
                 table; our tasks put classes on random slots out of 10 so every slot embedding is trained. <code>torch.unique</code>
                 with <code>return_inverse</code> gives the sorted slots present and each row's index into them; <code>slots</code>
                 is −1 past the k classes, so <code>forward</code> can scatter the predictions back.</p>""",
                 links=[L("_icl_predictions expects codes 0..k-1", TABICL + "_model/learning.py#L241", ("tabicl", "_model/learning.py", 241, 283))],
                 checks=["tests/test_model.py::test_contiguous_codes_number_the_classes_of_each_table_from_zero_and_remember_the_slots",
                         "tests/test_model.py::test_moving_the_classes_to_other_slots_does_not_change_the_prediction"]),
             Section(95, 116,
                 lead="Multi-head attention with separate query, key and value inputs, on torch's scaled-dot-product kernel",
                 note="""<p>Separate inputs because the labelled read sends queries from one set of tokens and keys and values
                 from another. <code>split</code> turns (batch, tokens, width) into (batch, heads, tokens, width/heads); the
                 output projection joins the heads again. The test writes softmax(qkᵀ/√d)v out by hand and gets the same
                 numbers.</p>""",
                 links=[L("torch scaled_dot_product_attention", TORCH + "nn/functional.py")],
                 checks=["tests/test_model.py::test_attention_equals_the_hand_written_reference",
                         "tests/test_model.py::test_queries_do_not_interact"]),
             Section(119, 133,
                 lead="Pre-norm transformer layer: LayerNorm, self-attention, residual; LayerNorm, GELU MLP, residual",
                 note="""<p>The layer the within-photo encoder stacks. Normalising before each sublayer (not after, as in the
                 original transformer) is what ViT and DINOv2 do too.</p>""",
                 links=[L("On layer normalization in the transformer architecture (Xiong et al. 2020)", "https://arxiv.org/abs/2002.04745")],
                 checks=["tests/test_model.py::test_a_layer_mixes_its_tokens_and_keeps_the_shape"]),
             Section(136, 141, fold=True,
                 lead="The patch size, DINOv2 ViT-B/14's token width and summary token count, and the five encoder names",
                 note="""<p>14 px patches: DINOv2's, kept for the from-scratch embedding so every encoder gives 256 tokens per
                 224 px photo. The 5 summary tokens are the CLS token and the 4 registers of the <code>_reg</code> models.</p>""",
                 links=[L("hub entry dinov2_vitb14_reg (4 registers)", DINO + "dinov2/hub/backbones.py#L129", ("dinov2", "dinov2/hub/backbones.py", 129, 143)),
                        L("Vision transformers need registers (Darcet et al. 2024)", "https://arxiv.org/abs/2309.16588")]),
             Section(144, 155,
                 lead="ViT patch embedding: one convolution with kernel and stride equal to the patch size",
                 note="""<p>Encoder "pixels", trained from scratch. Each 14×14 patch goes through one linear map;
                 <code>flatten(2).transpose(1, 2)</code> orders the patches row-major as (n, patches, width).
                 <code>n_tokens</code> is None: the count follows the image size.</p>""",
                 links=[L("ViT, section 3.1 (Dosovitskiy et al. 2021)", "https://arxiv.org/abs/2010.11929")],
                 checks=["tests/test_model.py::test_pixel_patches_give_one_token_per_14_px_patch",
                         "tests/test_model.py::test_a_pixel_token_sees_only_its_own_patch"]),
             Section(158, 171,
                 lead="Projects precomputed frozen DINOv2 tokens to the model's width with a LayerNorm and one linear map",
                 note="""<p>Encoders "block4", "block8", "block12" and "cls"; the tokens arrive from inputs.py, 768 wide.
                 <code>n_tokens</code> is set only for "cls", whose 5 summary tokens do not follow the image size.</p>""",
                 checks=["tests/test_model.py::test_dino_patches_project_each_token"]),
             Section(174, 188,
                 lead="Adds a learned position embedding and runs transformer layers over the patches of one photo",
                 note="""<p>Photos never see each other here, and no label is involved: this is where the model learns what a
                 patch is before the read asks what it means.</p>""",
                 checks=["tests/test_model.py::test_encoder_keeps_the_shape_and_takes_no_labels",
                         "tests/test_model.py::test_encoder_treats_photos_independently",
                         "tests/test_model.py::test_encoder_knows_where_a_patch_is"]),
             Section(191, 205,
                 lead="The labelled read explained, then nearest_photos: each photo's k most similar sources by mean token",
                 note="""<p>The comment is the specification of the stage below. <code>nearest_photos</code> is the "retrieve"
                 of retrieve-then-read: every photo is summarised by its mean patch token, cosine similarity is one matrix
                 product after normalising, and <code>topk</code> picks the k sources. Reading from a fixed k is what keeps the
                 attention the same size whatever the context, so a model trained on 256-row tables can be scored on
                 10,000-row ones.</p>""",
                 checks=["tests/test_model.py::test_nearest_photos_are_the_most_similar_by_mean_token"]),
             Section(208, 216,
                 lead="LabelRead's constructor: a LayerNorm for readers, one for sources, the attention, the chunk size",
                 note="""<p>This model's own operation. Queries come from the reading photo's patches, keys from the labelled
                 context patches; what comes back is the <em>label</em> embedding of the photo a key patch belongs to.
                 <code>chunk</code> only bounds how many reading photos one attention call handles in the top-k branch.</p>"""),
             Section(218, 240,
                 lead="The read: one attention over all source patches, or per photo over the patches of its k neighbours",
                 note="""<p>Without <code>nearest</code>, readers and sources are both flattened to (batch, tokens, width) and
                 every key's value is its photo's label, broadcast over the patches. With <code>nearest</code> (batch, photos,
                 k), for a chunk of reading photos <code>keys[rows, idx]</code> gathers their neighbours' patches into one key set
                 per reading photo, (batch·c, k·patches, width), and the attention runs once per chunk. The chunk size only bounds
                 memory; the result does not depend on it, and with k equal to the number of sources it is the plain
                 read.</p>""",
                 checks=["tests/test_model.py::test_what_is_read_back_is_labels_not_content",
                         "tests/test_model.py::test_top_k_reads_only_from_the_nearest_context_photos",
                         "tests/test_model.py::test_top_k_with_every_context_photo_is_the_plain_read",
                         "tests/test_model.py::test_top_k_reads_do_not_depend_on_the_chunk_of_photos",
                         "tests/test_model.py::test_the_top_k_read_is_the_same_whether_photos_come_in_chunks_or_one_by_one"]),
             Section(243, 256,
                 lead="The stage's constructor: the label embedding, then per layer one read, a LayerNorm and an MLP",
                 note="""<p>The label embedding has one vector per slot, initialised orthogonal so the ten labels start maximally
                 distinct. The keys of every read are the label-free tokens the stage received, so matching is by photo content,
                 and a context photo is among its own sources: the context is given, only test rows are predicted.</p>""",
                 checks=["tests/test_model.py::test_the_stage_keeps_the_shape_and_takes_only_context_labels"]),
             Section(258, 263,
                 lead="Picks which context photos are sources: all up to max_sources, else a random subset while training",
                 note="""<p>At evaluation the first <code>max_sources</code>, so scoring is deterministic. It is a setting, not a
                 weight: benchmark.py raises it for the context sweeps without retraining.</p>""",
                 checks=["tests/test_model.py::test_at_most_max_sources_context_photos_are_read",
                         "tests/test_model.py::test_the_source_cap_is_a_setting_not_a_weight"]),
             Section(265, 279,
                 lead="Per forward: pick sources, embed their labels, retrieve neighbours for top_k, then read and MLP per layer",
                 note="""<p>Retrieval runs only when there are more sources than k. Output shape equals input shape: the tokens
                 with what they read added. The tests show a test photo changes only its own tokens, every context label can
                 reach the test photos, and a short training run learns to look at the labelled context.</p>""",
                 checks=["tests/test_model.py::test_a_test_photo_influences_only_itself",
                         "tests/test_model.py::test_every_context_label_reaches_the_test_photos",
                         "tests/test_model.py::test_a_context_photo_reads_its_own_label_too",
                         "tests/test_model.py::test_the_stage_learns_where_to_look_from_the_labelled_context",
                         "tests/test_model.py::test_gradients_reach_every_parameter"]),
             Section(282, 293, fold=True,
                 lead="ModelConfig: the encoder, the image size, the layer counts, heads, the source cap, the read and its k",
                 note="""<p>Saved into every run's config.json and rebuilt from it by <code>config_from_record</code>. 224 px photos
                 cut into 14 px patches (256 per photo), a 4-layer encoder, 2 label reads, up to 256 source photos, and the
                 top-32 read. The table side has no settings: TabICL v2 fixes them.</p>""",
                 checks=["tests/test_model.py::test_a_new_record_round_trips_and_the_shipped_records_are_read"]),
             Section(296, 307,
                 lead="Reads the DINOv2 block number off an encoder name, and builds the embedder the encoder needs",
                 note="""<p>"blockN" → N, else None. "pixels" gets the from-scratch embedding, "cls" the projection of 5 summary
                 tokens, the block encoders the projection of 256 patch tokens.</p>""",
                 checks=["tests/test_model.py::test_the_encoders_pick_their_embedder_and_their_dino_block"]),
             Section(310, 322,
                 lead="The module: TabICL's stages, the embedder, the within-photo encoder, the labelled read, the patch marker",
                 note="""<p><code>n_patches</code> comes from the embedder's fixed token count or from the image size.
                 <code>patch_marker</code> is a learned vector added to every patch token so the row stage can tell patches from
                 cells. The attribute names are frozen by a test against the released run's state dict, so a rename would be
                 caught before it made the released weights unloadable.</p>""",
                 checks=["tests/test_model.py::test_the_parameter_names_are_the_ones_the_released_run_was_trained_with"]),
             Section(324, 325,
                 lead="Loads a state dict strictly: every parameter must be present",
                 note="",
                 checks=["tests/test_model.py::test_weights_round_trip_strictly"]),
             Section(327, 333,
                 lead="Turns each row's photo into patch tokens: embed, encode, labelled read, then the marker",
                 note="""<p>Photos of all tables and rows are flattened for the embedder and the encoder (neither looks across
                 photos) and unflattened for the read, which works per table. Shapes: images (batch, rows, …) → tokens (batch,
                 rows, patches, width).</p>""",
                 checks=["tests/test_model.py::test_photos_join_the_rows_and_change_the_prediction",
                         "tests/test_model.py::test_the_dino_encoders_take_frozen_tokens"]),
             Section(335, 351,
                 lead="Runs TabICL's three stages in order, with the patch tokens concatenated after the columns",
                 note="""<p>TabICL's <code>_train_forward</code>, stage by stage. <code>codes</code> and <code>slots</code> come from
                 <code>contiguous_codes</code>. The column stage (<code>col_embedder._train_forward</code>) gives (batch, rows,
                 cls + columns, width); the patch tokens join on the token axis; the row stage
                 (<code>row_interactor._train_forward</code>) attends across each row's tokens and reads it out with CLS tokens;
                 the in-context stage (<code>icl_predictor._icl_predictions</code>) gives logits over the codes, of which the test
                 rows are kept. The loop scatters each table's k logits back onto its slots, −inf elsewhere; the log-softmax over
                 the slots is what the loss, the scorers and predict.py take. The test runs TabICL's own forward on the same
                 tables with no photos and gets the same probabilities.</p>""",
                 links=[L("TabICL._train_forward (the three calls in order)", TABICL + "_model/tabicl.py#L305", ("tabicl", "_model/tabicl.py", 305, 370)),
                        L("ColEmbedding._train_forward", TABICL + "_model/embedding.py#L423", ("tabicl", "_model/embedding.py", 423, 463)),
                        L("RowInteraction._train_forward", TABICL + "_model/interaction.py#L170", ("tabicl", "_model/interaction.py", 170, 215)),
                        L("ICLearning._icl_predictions", TABICL + "_model/learning.py#L241", ("tabicl", "_model/learning.py", 241, 283))],
                 checks=["tests/test_model.py::test_without_photos_the_model_is_tabicl_on_the_standardised_table",
                         "tests/test_model.py::test_photos_join_the_rows_and_change_the_prediction",
                         "tests/test_model.py::test_log_probabilities_cover_only_the_slots_in_the_context_and_sum_to_one",
                         "tests/test_model.py::test_moving_the_classes_to_other_slots_does_not_change_the_prediction"]),
             Section(354, 354, fold=True,
                 lead="The parameter prefixes of the image side: everything trained from scratch",
                 note="""<p>training.py puts these in their own optimizer group at a higher learning rate; everything else is
                 under <code>table.</code>, TabICL's pretrained weights.</p>""",
                 checks=["tests/test_model.py::test_the_image_side_is_exactly_what_is_not_tabicl"]),
             Section(357, 364,
                 lead="Rebuilds a ModelConfig from a run record, translating older records' arm and block fields",
                 note="""<p>Records under results/ predate the <code>encoder</code> setting and name the encoder as an arm
                 ("pixels", "dino" with "dino_block", "dino_cls"); the mapping turns those into the five encoder names. Unknown
                 fields in a record are ignored, so an old record still loads.</p>""",
                 checks=["tests/test_model.py::test_an_old_record_names_its_encoder_as_an_arm_and_a_block",
                         "tests/test_model.py::test_a_new_record_round_trips_and_the_shipped_records_are_read"]),
             Section(367, 373,
                 lead="Loads a trained run from its folder: config.json for the settings, model.pt for the weights",
                 note="""<p>The entry point of the README, evaluate.py and benchmark.py. TabICL is built unpretrained because the
                 run's weights cover it. The test compares the released weights' probabilities on a fixed table with the ones
                 recorded when the checkpoint was made.</p>""",
                 checks=["tests/test_model.py::test_the_released_checkpoint_loads_and_gives_the_recorded_probabilities"]),
         ]),

    File("table_image_icl/inputs.py", "Photo inputs: letterboxed pixels, or frozen DINOv2 tokens",
         """<p>What turns stored uint8 photos into what an encoder's embedder takes. Nothing here is trained. The DINOv2 encoders
         run DINOv2's own code: its token preparation and its transformer blocks, stopped early.</p>""",
         [
             Section(1, 8, fold=True,
                 lead="Docstring and imports; the encoder names and dino_block come from model.py",
                 note=""),
             Section(10, 14, fold=True,
                 lead="ImageNet normalisation constants, the pinned DINOv2 hub commit and weights, and the padding colour",
                 note="""<p><code>IMAGENET_MEAN</code> and <code>IMAGENET_STD</code> are torchvision's, the normalisation DINOv2 was
                 trained with. Pinning <code>DINOV2</code> to one commit makes <code>torch.hub.load</code> always fetch the same
                 code. <code>PAD_COLOUR</code> is the mean colour in uint8, so letterbox padding is zero after
                 normalisation.</p>""",
                 links=[L("torchvision normalisation constants", "https://pytorch.org/vision/stable/models.html"),
                        L("hub entry dinov2_vitb14_reg", DINO + "dinov2/hub/backbones.py#L129", ("dinov2", "dinov2/hub/backbones.py", 129, 143))]),
             Section(17, 27,
                 lead="Pads a photo to a square without stretching and resizes it to 224 px uint8",
                 note="""<p>A path is opened with PIL's <code>draft</code>, which decodes a JPEG at reduced size and is much faster
                 for the large benchmark photos. The longer side is scaled to <code>size</code>, the photo is centred on a canvas of
                 the padding colour.</p>""",
                 links=[L("PIL Image.draft", "https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.draft")],
                 checks=["tests/test_inputs.py::test_letterbox_pads_to_a_square_without_stretching"]),
             Section(30, 40,
                 lead="Converts uint8 (n, H, W, 3) photos to float channel-first tensors with ImageNet normalisation",
                 note="""<p>Encoder "pixels" takes this directly; the DINOv2 encoders run it first. The mean and spread tensors
                 are made once on the device.</p>""",
                 checks=["tests/test_inputs.py::test_pixels_are_channel_first_and_normalised"]),
             Section(43, 52,
                 lead="DinoTokens: loads frozen DINOv2 ViT-B/14 with registers from the hub at the pinned commit",
                 note="""<p>The one place that calls the hub; tests pass a tiny stand-in <code>model</code> instead.
                 <code>eval()</code> and <code>requires_grad_(False)</code> freeze it.</p>""",
                 links=[L("hubconf.py", DINO + "hubconf.py")],
                 checks=["tests/test_inputs.py::test_tokens_do_not_depend_on_the_chunk_size_and_carry_no_gradient"]),
             Section(54, 66,
                 lead="Runs DINOv2's token preparation and its first blocks, keeping the patch tokens or the summary tokens",
                 note="""<p>DINOv2's <code>forward_features</code> cut short. It calls <code>prepare_tokens_with_masks</code> (patch
                 embedding, CLS and register tokens prepended, position embedding added), then every block, then the final norm;
                 here only <code>blocks[:block]</code> run and the first <code>1 + num_register_tokens</code> tokens (CLS and
                 registers) are dropped. With <code>summary</code> every block and the norm run and only those tokens are kept:
                 <code>forward_features</code>' <code>x_norm_clstoken</code> and <code>x_norm_regtokens</code> side by side.
                 <code>no_grad</code> and bfloat16 autocast keep memory low; <code>chunk</code> bounds the photos per call. The
                 equivalence test runs all 12 blocks this way, applies the norm, and gets <code>x_norm_patchtokens</code>.</p>""",
                 links=[L("prepare_tokens_with_masks", DINO + "dinov2/models/vision_transformer.py#L216", ("dinov2", "dinov2/models/vision_transformer.py", 216, 236)),
                        L("forward_features (the full version)", DINO + "dinov2/models/vision_transformer.py#L257", ("dinov2", "dinov2/models/vision_transformer.py", 257, 275)),
                        L("Block.forward", DINO + "dinov2/layers/block.py#L89", ("dinov2", "dinov2/layers/block.py", 89, 115))],
                 checks=["tests/test_equivalence.py::test_running_every_block_ourselves_gives_dinov2_s_own_patch_features",
                         "tests/test_inputs.py::test_block_tokens_are_the_patch_tokens_after_the_chosen_block",
                         "tests/test_inputs.py::test_summary_tokens_are_the_normed_cls_and_registers_after_the_last_block",
                         "tests/test_inputs.py::test_tokens_do_not_depend_on_the_chunk_size_and_carry_no_gradient",
                         "tests/test_inputs.py::test_real_dino_block_4_gives_256_patch_tokens"]),
             Section(69, 76,
                 lead="Picks the photo input by encoder name: normalised pixels, or frozen DINOv2 tokens of the right block",
                 note="""<p>train.py, evaluate.py and benchmark.py call this with the run's encoder; "pixels" gets pixels
                 because its embedding lives inside the model.</p>""",
                 checks=["tests/test_inputs.py::test_inputs_for_picks_the_input_by_encoder"]),
         ]),

    File("table_image_icl/tasks.py", "Training tasks: synthetic tables where a photo stands for a hidden variable",
         """<p>Ours. A table is drawn from TabICL v2's prior, used unmodified as a library; the row's photo stands for a hidden
         variable, either a column removed from the table (photo as feature) or the row's class (photo as label). A batch
         carries photo indices into in-memory collections, never pixels, so DataLoader workers stay light.</p>""",
         [
             Section(1, 23, fold=True,
                 lead="Docstring: the two photo roles and how photos are assigned to bins; imports",
                 note="""<p>Every bin (or shown class) gets one class of a photo collection, each row a photo of its bin's class.
                 <code>bin_targets</code> is the same quantile cut predict.py uses for regression.</p>"""),
             Section(26, 39, fold=True,
                 lead="TaskConfig: the prior's settings, the bin cap, the photo-as-label share, and from_record",
                 note="""<p>4 tables per batch and 64–256 rows (memory at 256 patches per row), 3–33 features before one is
                 hidden, up to 10 classes and bins, 30–90% of rows as context, half the batches with the photo as label, TabICL
                 v2's graph SCM prior. <code>from_record</code> ignores fields it does not know, so old records load.</p>""",
                 links=[L("PriorDataset's arguments", TABICL + "prior/_dataset.py#L1255", ("tabicl", "prior/_dataset.py", 1255, 1330))],
                 checks=["tests/test_tasks.py::test_a_task_config_reads_a_record_and_ignores_what_it_does_not_know",
                         "tests/test_tasks.py::test_the_share_of_photo_as_label_batches_is_a_setting"]),
             Section(42, 51, fold=True,
                 lead="Batch: the table without the hidden column, labels, photo indices, hidden values, bins, the role",
                 note="""<p>Also the real column count per table and the name of the collection its photos come from. A
                 NamedTuple so a DataLoader passes it through unchanged.</p>"""),
             Section(54, 60,
                 lead="Maps classes 0..K−1 onto K distinct random slots out of 10, per task",
                 note="""<p>Every slot embedding is then trained and no slot comes to mean anything; <code>contiguous_codes</code>
                 in the model undoes it.</p>""",
                 checks=["tests/test_tasks.py::test_classes_land_on_distinct_random_slots_and_every_slot_gets_used"]),
             Section(63, 79,
                 lead="Seeds the three random sources, builds PriorDataset with our settings, and yields tables forever",
                 note="""<p>The prior draws from the global torch, numpy and random states, hence the three seeds.
                 <code>batch_size_per_gp</code> = batch size, so a whole batch shares one setting of the prior and one context
                 size; the batches come from <code>get_batch</code>. Columns are cut to the widest table; labels go onto
                 slots.</p>""",
                 links=[L("PriorDataset", TABICL + "prior/_dataset.py#L1255", ("tabicl", "prior/_dataset.py", 1255, 1330)),
                        L("PriorDataset.get_batch", TABICL + "prior/_dataset.py#L1444", ("tabicl", "prior/_dataset.py", 1444, 1480))]),
             Section(82, 88,
                 lead="Fits a random forest on the context rows and returns the column it leans on most",
                 note="""<p>So the hidden column is one the table's labels depend on: a photo that stands for a column nobody
                 needs would teach nothing.</p>""",
                 links=[L("sklearn: random forest feature importances", "https://scikit-learn.org/stable/modules/ensemble.html#feature-importance-evaluation")],
                 checks=["tests/test_tasks.py::test_the_most_important_column_is_the_one_the_label_depends_on"]),
             Section(91, 97,
                 lead="Removes one column from every table with a boolean mask, returning its values and the new counts",
                 note="""<p>The mask runs over the column axis after a transpose; the column counts are decremented.</p>""",
                 checks=["tests/test_tasks.py::test_hiding_removes_the_given_column_of_each_table"]),
             Section(100, 105,
                 lead="Photo as label: the row's own class for a per-task share of rows, a random class otherwise",
                 note="""<p>The share is drawn from 0.3 to 1; below 0.3 a 10-class photo would say little. The random classes
                 come from those the context shows.</p>""",
                 checks=["tests/test_tasks.py::test_a_noisy_label_shows_the_class_on_a_share_of_the_rows"]),
             Section(108, 110,
                 lead="Cuts a column into bins at the quantiles of its context rows",
                 note="""<p>predict.py's <code>bin_targets</code>: a column with few distinct values gets one bin per value, and
                 no bin is empty.</p>""",
                 checks=["tests/test_tasks.py::test_bins_come_from_the_context_rows_and_keep_the_order",
                         "tests/test_tasks.py::test_a_column_with_few_distinct_values_gets_one_bin_per_value"]),
             Section(113, 129,
                 lead="Draws one photo per row: a random collection class per bin, or an attribute's two values",
                 note="""<p>For an attribute collection (CelebA) one random attribute is chosen and possibly flipped: false for one
                 bin, true for the other. Rows of a bin draw with replacement from that membership.</p>""",
                 checks=["tests/test_tasks.py::test_each_bin_shows_one_class_of_the_collection",
                         "tests/test_tasks.py::test_two_bins_can_show_a_face_attribute"]),
             Section(132, 135,
                 lead="Whether a collection can show that many bins: exactly 2 for attributes, enough classes otherwise",
                 note=""),
             Section(138, 161,
                 lead="The stream: draws the role, hides and bins a column or noises the labels, then draws photos",
                 note="""<p>The tables come from <code>prior_tables</code>. For "feature", the most important column per table is
                 hidden and cut into 2 to 10 bins drawn per table; a batch with a 1-column table is skipped. For "label", the
                 noisy labels are the bins and the hidden value. Per table, a collection that can show that many bins is picked
                 at random. The batch is yielded with columns cut to the widest real table.</p>""",
                 checks=["tests/test_tasks.py::test_batches_tie_the_hidden_variable_to_the_photos",
                         "tests/test_tasks.py::test_the_share_of_photo_as_label_batches_is_a_setting"]),
             Section(164, 173,
                 lead="Wraps the stream as an IterableDataset; each DataLoader worker shifts the seed by its id",
                 note="""<p>For a <code>DataLoader</code> with <code>batch_size=None</code>, so workers produce different tables. Only
                 the collections' label arrays are copied to the workers, never the photos.</p>"""),
             Section(176, 182,
                 lead="Puts a column back as each table's next real column, for the column-visible view",
                 note="""<p>The hidden column returns after the table's real columns, which differ per table, hence the fancy
                 index.</p>""",
                 checks=["tests/test_tasks.py::test_a_column_is_put_back_after_each_tables_real_columns"]),
             Section(185, 202,
                 lead="Tasks: the four views, the collections, the photo input, and the no-photo share",
                 note="""<p>The views: image; no image, the floor; column visible, the ceiling; images shuffled, another row's
                 photo. <code>to_tokens</code> is inputs.py's photo input; <code>p_no_image</code> keeps the table-only path
                 trained; <code>coin</code> is a private random state so validation does not disturb it.</p>"""),
             Section(204, 215,
                 lead="The training stream, the validation stream on other collections, the no-photo coin, and pixel lookup",
                 note="""<p>Training and validation draw from disjoint collection sets. <code>images</code> looks the pixels up per
                 table in its collection and turns them into the model's input with <code>to_tokens</code>.</p>""",
                 checks=["tests/test_tasks.py::test_training_and_validation_tasks_use_different_collections",
                         "tests/test_tasks.py::test_training_drops_the_photo_in_the_given_share_of_steps",
                         "tests/test_tasks.py::test_the_image_view_looks_up_each_rows_photo_in_its_tables_collection"]),
             Section(217, 229,
                 lead="Builds the model's keyword arguments for a batch in a given view",
                 note="""<p>The hidden column is put back for "column visible"; the "image" view adds the photos, and "images
                 shuffled" the photos of a fixed permutation of the rows.</p>""",
                 checks=["tests/test_tasks.py::test_the_no_image_view_is_the_table_alone",
                         "tests/test_tasks.py::test_the_column_visible_view_gives_the_hidden_column_back_instead_of_the_photo",
                         "tests/test_tasks.py::test_the_shuffled_view_gives_rows_the_photos_of_other_rows"]),
         ]),

    File("table_image_icl/collections.py", "Photo collections: labelled photos from Hugging Face, cached as arrays",
         """<p>The photos the training tasks borrow. Eleven public collections are decoded once to 224 px uint8 arrays; CelebA
         is split by attribute so that some attributes are never trained on. The validation collections are never trained
         on either.</p>""",
         [
             Section(1, 10, fold=True,
                 lead="Docstring and imports; letterbox comes from inputs.py",
                 note=""),
             Section(12, 27, fold=True,
                 lead="The eleven collections with their Hub ids, the training and validation sets, the held-out attribute count",
                 note="""<p>Kind "class" has one label column, "attr" (CelebA) boolean attributes. Training: celeba, cars, flowers,
                 gtsrb, food101, mini-imagenet, cifar100, eurosat. Validation: dtd, resisc45, pets, and CelebA's last 10
                 attributes.</p>""",
                 checks=["tests/test_collections.py::test_training_and_validation_collections_do_not_overlap"]),
             Section(30, 35,
                 lead="Loads every collection and splits CelebA into training and held-out attributes",
                 note=""),
             Section(38, 45,
                 lead="Splits an attribute collection in two: the first attributes for training, the last n for validation",
                 note="""<p>Both halves share the photos; what differs is which attributes a task may show.</p>""",
                 checks=["tests/test_collections.py::test_split_attributes_shares_photos_but_no_attribute"]),
             Section(48, 50,
                 lead="One collection, decoded or read from the cache file named by collection and size",
                 note=""),
             Section(53, 67,
                 lead="Caches the arrays build() returns at path, and reads them back with the file's sha256",
                 note="""<p>Written through a temporary file and renamed, so a crash never leaves a half-written cache. The
                 sha256 goes into the run record (train.py), so a run says exactly which photos it saw.</p>""",
                 checks=["tests/test_collections.py::test_a_cached_collection_is_decoded_once_and_read_back_unchanged"]),
             Section(70, 75,
                 lead="The sha256 of a file, read in 16 MB blocks",
                 note=""),
             Section(78, 95,
                 lead="Downloads a collection, shuffles it at seed 0, keeps max_images, and letterboxes the photos in threads",
                 note="""<p>The image column is found by type. For "class", the label column is the first <code>ClassLabel</code>
                 feature or a conventional name, and labels are renumbered 0..K−1 with their names kept; for "attr", every
                 boolean feature is an attribute.</p>""",
                 links=[L("datasets.load_dataset", "https://huggingface.co/docs/datasets/package_reference/loading_methods#datasets.load_dataset")]),
         ]),

    File("table_image_icl/training.py", "Training: schedule, optimizer groups, loss, the loop",
         """<p>Ours, with one borrowed piece: the cosine schedule of Loshchilov & Hutter, which the equivalence test checks against
         torch's <code>CosineAnnealingLR</code>. The pretrained table stages and the image side get separate learning
         rates.</p>""",
         [
             Section(1, 32, fold=True,
                 lead="Docstring listing what a run folder holds; imports; the validation and evaluation seeds",
                 note="""<p>Validation tasks and the fixed evaluation tasks every run is scored on (evaluate.py) each have their
                 own seed, far from any training seed.</p>"""),
             Section(35, 51, fold=True,
                 lead="TrainConfig: budget, the two learning rates, clip, weight decay, warm-up, validation cadence, seed, autocast",
                 note="""<p>The budget is hours or steps, whichever comes first. 1e-5 for the pretrained table stages, 1e-4 for the
                 image side trained from scratch.</p>"""),
             Section(54, 60,
                 lead="Mean negative log-likelihood and accuracy over the test rows whose class appears in the context",
                 note="""<p>A class the context never shows cannot be predicted from it, so those rows count for neither loss nor
                 accuracy.</p>""",
                 checks=["tests/test_training.py::test_loss_ignores_test_rows_whose_class_is_not_in_the_context",
                         "tests/test_training.py::test_accuracy_counts_only_rows_that_can_be_predicted"]),
             Section(63, 66,
                 lead="Linear warm-up, then cosine decay from the group's base rate to lr_end over the budget",
                 note="""<p>Progress is time or steps, whichever is further along. After the warm-up this is exactly torch's
                 <code>CosineAnnealingLR</code> with <code>T_max = max_steps</code> and <code>eta_min = lr_end</code>.</p>""",
                 links=[L("SGDR: cosine annealing (Loshchilov & Hutter 2017), eq. 5", "https://arxiv.org/abs/1608.03983"),
                        L("torch CosineAnnealingLR", TORCH + "optim/lr_scheduler.py")],
                 checks=["tests/test_equivalence.py::test_learning_rate_is_cosine_annealing_after_the_warm_up",
                         "tests/test_training.py::test_learning_rate_warms_up_then_decays_and_is_a_factor_of_each_groups_base_rate"]),
             Section(69, 74,
                 lead="AdamW with two groups: the table stages at lr, the image side at image_lr",
                 note="""<p>The split is by parameter name on <code>IMAGE_SIDE</code> (model.py); each group remembers its base
                 rate for the schedule.</p>""",
                 links=[L("torch AdamW", TORCH + "optim/adamw.py")],
                 checks=["tests/test_model.py::test_the_image_side_is_exactly_what_is_not_tabicl"]),
             Section(77, 84,
                 lead="The share of the budget used: the larger of time used and steps done",
                 note=""),
             Section(87, 98,
                 lead="Saves and restores checkpoints: weights, optimizer state, step, seconds trained, best loss",
                 note="""<p>Written through a temporary file and renamed, so a crash never leaves a half-written checkpoint.</p>""",
                 checks=["tests/test_training.py::test_checkpoint_round_trip"]),
             Section(101, 119,
                 lead="Records what produced a result: the git commit (dirty or not), the command, the machine, library versions",
                 note="""<p>"unknown" where the code is not a git checkout. The same dict goes into every evaluation and benchmark
                 report.</p>"""),
             Section(122, 133,
                 lead="Refuses to resume a run under other settings than its record's; only the budget and folders may change",
                 note="""<p>Otherwise the run would train under a config.json that no longer describes it. Settings are compared
                 as sorted JSON.</p>""",
                 checks=["tests/test_training.py::test_a_resume_with_other_settings_is_refused"]),
             Section(136, 141,
                 lead="Validation batches, made once and saved, read back after: the same tasks for the whole run",
                 note="""<p>A run and its resume, or two machines, validate on the same tasks; evaluate.py uses it for the
                 evaluation tasks every run shares.</p>""",
                 checks=["tests/test_evaluation.py::test_evaluation_tasks_are_made_once_and_read_back_the_same"]),
             Section(144, 149,
                 lead="One batch's loss and accuracy in a given view, under bfloat16 autocast on a GPU",
                 note="""<p>The tasks give the model's keyword arguments for the view.</p>"""),
             Section(152, 161,
                 lead="Validation: mean loss and accuracy per view over the fixed batches, model left in training mode",
                 note="",
                 checks=["tests/test_training.py::test_validation_leaves_the_model_and_the_random_state_as_they_were"]),
             Section(164, 186,
                 lead="Setup: device, folders, optimizer, resume from the checkpoint or write the record, the validation batches",
                 note="""<p>The optimizer is built before a resume so its state can be restored; a resume loads the checkpoint, runs
                 <code>check_resumed_record</code> and notes the step in metrics.jsonl. A fresh run writes config.json: the
                 settings, the parameter count and the provenance.</p>""",
                 checks=["tests/test_training.py::test_a_short_run_writes_its_record_logs_every_view_and_resumes",
                         "tests/test_training.py::test_a_resumed_run_continues_from_its_checkpoint_not_from_the_model_it_is_given"]),
             Section(188, 193,
                 lead="Seeds the training stream past the current step and runs it in DataLoader workers",
                 note="""<p>So a resume sees new tasks. <code>losses</code> and <code>norms</code> collect since the last validation;
                 <code>elapsed</code> adds this session's time to the time trained before.</p>"""),
             Section(195, 210,
                 lead="Each validation: score, keep the best weights, append a metrics line, save the checkpoint",
                 note="""<p>The best is by the first view's loss ("image"). The line holds the step, mean training loss, validation
                 loss and accuracy, every view, gradient norms, the learning rate and hours trained.</p>"""),
             Section(212, 230,
                 lead="The loop: set the learning rates, one batch, clip the gradient norm, step; validate periodically",
                 note="""<p>Stops when the budget is used up. Each step trains in the view the no-photo coin picks; a last
                 validation covers the steps since the previous one, and the final weights are saved beside the best ones.</p>""",
                 links=[L("torch clip_grad_norm_", TORCH + "nn/utils/clip_grad.py")],
                 checks=["tests/test_training.py::test_a_short_run_writes_its_record_logs_every_view_and_resumes"]),
         ]),

    File("table_image_icl/predict.py", "Using the model on one real table: classes, or values through bins",
         """<p>Numpy in, probabilities or values out. Regression goes through the classification head: the context targets are
         cut into quantile bins, the bins are predicted as classes, and the prediction is the expected bin mean. That is
         TabPFN's recipe for regression, with 10 bins instead of its thousands.</p>""",
         [
             Section(1, 10, fold=True,
                 lead="Docstring and imports; N_SLOTS bounds the number of bins",
                 note=""),
             Section(13, 23,
                 lead="Cuts targets into at most n_bins quantile bins of the context, returning each row's bin and bin means",
                 note="""<p>Edges are quantiles of the context targets at actual values (<code>method="lower"</code>), de-duplicated,
                 so a column with few distinct values gets one bin per value; <code>used</code> drops bins no context row falls in
                 and <code>nearest</code> moves every row to the closest used bin, so every bin holds at least one context row and a
                 test row beyond the context's range lands in an end bin. TabPFN's <code>BarDistribution</code> is the same idea
                 with borders from the training targets' quantiles.</p>""",
                 links=[L("TabPFN BarDistribution", TABPFN + "architectures/shared/bar_distribution.py#L19", ("tabpfn", "architectures/shared/bar_distribution.py", 19, 40)),
                        L("TabPFN (Hollmann et al. 2025), regression as classification over bins", "https://www.nature.com/articles/s41586-024-08328-6")],
                 checks=["tests/test_predict.py::test_every_bin_holds_a_context_row_and_order_is_kept",
                         "tests/test_predict.py::test_bins_come_from_the_context_only",
                         "tests/test_predict.py::test_repeated_values_leave_no_empty_bin",
                         "tests/test_predict.py::test_bin_means_are_the_mean_context_target_of_each_bin"]),
             Section(26, 28,
                 lead="Point prediction from a distribution over bins: probabilities times bin means",
                 note="""<p>TabPFN's <code>BarDistribution.mean</code> with the bin mean in place of the bar's midpoint.</p>""",
                 links=[L("BarDistribution.mean", TABPFN + "architectures/shared/bar_distribution.py#L255", ("tabpfn", "architectures/shared/bar_distribution.py", 255, 275))],
                 checks=["tests/test_predict.py::test_the_expected_value_of_a_certain_bin_is_that_bins_mean"]),
             Section(31, 49,
                 lead="Class probabilities for the test rows in chunks, each chunk seeing the whole context, photos optional",
                 note="""<p>The context labels are renumbered to slots 0..K−1 (<code>np.unique</code>), so any label values work;
                 the model's log-probabilities over the slots present are cut to the first K and exponentiated. With photos, the
                 context photos and the chunk's test photos go through <code>to_tokens</code> together.</p>""",
                 checks=["tests/test_predict.py::test_one_probability_row_per_test_row_over_the_context_classes",
                         "tests/test_predict.py::test_label_names_do_not_matter",
                         "tests/test_predict.py::test_chunking_the_test_rows_changes_nothing",
                         "tests/test_predict.py::test_missing_values_are_accepted",
                         "tests/test_predict.py::test_photos_go_with_their_rows"]),
             Section(52, 57,
                 lead="Regression through the classifier: bin the context targets, predict the bins, take the expected mean",
                 note="""<p>Predictions stay within the range of the context targets by construction.</p>""",
                 checks=["tests/test_predict.py::test_regression_predictions_stay_within_the_context_targets"]),
         ]),

    File("table_image_icl/multabench.py", "MulTaBench: loading the datasets and the benchmark's own splits",
         """<p>Real-data scoring uses MulTaBench's datasets from Kaggle and re-implements its <code>get_split</code>, so that our
         scores are on exactly the benchmark's rows. The grouped split is ours, for the datasets whose rows share a patient,
         listing, product or fruit.</p>""",
         [
             Section(1, 11, fold=True,
                 lead="Imports, sklearn's splitters, and the row caps of MulTaBench's 10k size",
                 note="""<p>The same splitters the benchmark uses.</p>""",
                 links=[L("MulTaBench splits.py", MULTA + "multabench/benchmark/splits.py")]),
             Section(13, 29, fold=True,
                 lead="The validation, no-harm and held-out test dataset sets, with the reasoning in the comment above",
                 note="""<p>Validation = core datasets where encoder tuning helps, plus HAM10000; no-harm = extended datasets
                 where frozen features suffice; test = eight core datasets fixed before the first long run and untouched
                 since.</p>""",
                 checks=["tests/test_multabench.py::test_the_validation_no_harm_and_test_sets_are_four_three_and_eight"]),
             Section(32, 39, fold=True,
                 lead="Dataset: table, column names, targets, photo paths, task type, Kaggle version, optional group ids",
                 note="""<p>Categories are codes, NaN is missing; <code>groups</code> is set only when loaded with
                 <code>grouped</code> and a group key is known.</p>"""),
             Section(42, 70,
                 lead="Loads a dataset from Kaggle or a local folder and turns it into a numeric table",
                 note="""<p>The slug's owner is chico89; a local folder has the same layout (data.csv, metadata.json, images). The
                 target becomes class codes or floats; the table drops the target and image columns, drops free-text columns
                 (over 50 distinct values averaging over 20 characters) and turns the remaining categories into codes with NaN
                 for missing.</p>""",
                 checks=["tests/test_multabench.py::test_a_classification_table",
                         "tests/test_multabench.py::test_a_regression_table",
                         "tests/test_multabench.py::test_load_grouped_uses_the_dataset_s_group_key"]),
             Section(73, 81,
                 lead="The file name of a photo path, and CBIS-DDSM's patient per file from a shipped lookup table",
                 note="""<p>CBIS-DDSM names its files by DICOM series UID; data/cbis_ddsm_patients.csv maps each file to its
                 patient, built from the collection's description tables.</p>"""),
             Section(84, 97,
                 lead="Group keys: how rows of one patient, listing, product or fruit are recognised per dataset",
                 note="""<p>Ours. MulTaBench splits rows i.i.d. and its README says the splits are not audited for group structure,
                 so a patient's other lesion or a listing's duplicate can sit in the context of a test row. Each key reads the
                 group off the file name or the listing's columns.</p>""",
                 checks=["tests/test_multabench.py::test_group_keys_read_patients_listings_and_fruits_off_the_benchmark_files"]),
             Section(100, 108,
                 lead="One group id per row when the dataset has a known key, found by slug or by path",
                 note="""<p>None without a key: every row its own group. A row the key cannot place is an error, not a silent
                 singleton.</p>""",
                 checks=["tests/test_multabench.py::test_group_ids_are_none_without_a_key_and_fail_on_an_unmatched_row"]),
             Section(111, 129,
                 lead="Re-implements MulTaBench's get_split: repeated stratified 3-fold at seed 42, then capped train and test",
                 note="""<p>2 repeats, stratified for classification when every class has 3+ rows; split k is fold k % 3 of repeat
                 k // 3; train and test are capped at 10,000 and 5,000 rows with a stratified <code>train_test_split</code> at
                 seed 42 + k when every class has 2+ rows. With <code>groups</code>, a <code>StratifiedGroupKFold</code> keeps
                 each group on one side instead. The equivalence test carries a verbatim copy of the benchmark's function and
                 compares every split, both sizes, classification and regression.</p>""",
                 links=[L("MulTaBench get_split", MULTA + "multabench/benchmark/splits.py#L28"),
                        L("sklearn: repeated k-fold", "https://scikit-learn.org/stable/modules/cross_validation.html#repeated-k-fold"),
                        L("sklearn: stratified group k-fold", "https://scikit-learn.org/stable/modules/cross_validation.html#stratified-group-k-fold")],
                 checks=["tests/test_equivalence.py::test_split_gives_the_benchmark_s_rows_for_every_split_and_both_sizes",
                         "tests/test_multabench.py::test_splits_are_disjoint_capped_and_repeatable",
                         "tests/test_multabench.py::test_the_three_folds_of_a_repeat_cover_every_row_once",
                         "tests/test_multabench.py::test_regression_splits_need_no_classes",
                         "tests/test_multabench.py::test_a_grouped_split_keeps_every_group_on_one_side"]),
         ]),

    File("scripts/train.py", "The launcher",
         """<p>The command every run was started with: it builds the model for the chosen encoder, loads the photo collections,
         sets up the tasks with the encoder's photo input, and hands everything to <code>train</code> with the settings that
         go into the run's record.</p>""",
         [
             Section(1, 15, fold=True,
                 lead="Docstring with the command of the released run; imports",
                 note="""<p>Running the same command again with the same folders resumes the run.</p>"""),
             Section(18, 41, fold=True,
                 lead="Every flag of a run, with its default",
                 note="""<p>The released run is <code>--encoder block4 --max-steps 25000 --hours 5</code> with every other flag at
                 its default; the ablations set <code>--encoder</code> and <code>--patch-read all</code>.</p>"""),
             Section(43, 61,
                 lead="Builds the model, loads collections and tasks, records the settings, and trains",
                 note="""<p>The model is seeded; collections are decoded or read from the cache; <code>inputs_for</code> gives the
                 photo input for the encoder. The settings dict is what config.json holds beside the training settings: the
                 model and task settings, the TabICL checkpoint name, the collection names and their sha256.</p>"""),
         ]),

    File("scripts/evaluate.py", "Scoring a run on held-out synthetic tasks, by photo role and view",
         """<p>Every run is scored on the same fixed tasks, with photos from collections never trained on. The report is per role
         (photo as feature, photo as label) and per view, with the share of the floor-to-ceiling gap the photo closes.</p>""",
         [
             Section(1, 20, fold=True,
                 lead="Docstring with the command and the four views; imports",
                 note=""),
             Section(23, 42,
                 lead="Scores every batch in every view, grouped by role, with the paired image gain and its standard error",
                 note="""<p>Every view sees the same tasks, so the per-batch difference between "no image" and "image" loss is
                 paired; its mean, standard error and the share of batches where the photo helped go into the report beside the
                 per-view means.</p>""",
                 checks=["tests/test_evaluation.py::test_scores_come_per_role_and_per_view"]),
             Section(45, 54,
                 lead="The share of the floor-to-ceiling gap the photo closes, in loss and in accuracy",
                 note="""<p>0 = the photo adds nothing over the table alone, 1 = as good as seeing the hidden variable; NaN when
                 floor and ceiling coincide.</p>""",
                 checks=["tests/test_evaluation.py::test_gap_closed_is_zero_at_the_floor_and_one_at_the_ceiling"]),
             Section(57, 85,
                 lead="Loads the run, rebuilds its tasks from the record, reads the shared tasks file, scores, and saves",
                 note="""<p>The tasks come from the run's own record (settings, collections) at <code>EVALUATION_SEED</code>, made
                 on first use by <code>fixed_batches</code> and shared by every run; the report carries the file's sha256 and
                 the provenance.</p>""",
                 checks=["tests/test_evaluation.py::test_evaluation_tasks_are_made_once_and_read_back_the_same"]),
         ]),

    File("scripts/benchmark.py", "Scoring on MulTaBench against TabPFN v2 and the benchmark's frozen baseline",
         """<p>Real data, no training on it. On every dataset and split the model is scored three ways on the same rows (table
         only, table + photo, table + another row's photo) next to two references: TabPFN v2 on the table, and MulTaBench's
         frozen baseline, TabPFN v2 on the table plus 30 PCA components of frozen DINOv2 features.</p>""",
         [
             Section(1, 26, fold=True,
                 lead="Docstring with the command and the five methods; imports; the dataset names",
                 note=""),
             Section(29, 50,
                 lead="DINOv2 ViT-B/14's usual frozen features: the final CLS token and the mean patch token, 1536 wide",
                 note="""<p>The features MulTaBench's frozen-encoder baseline feeds its learners. The same loop as
                 <code>DinoTokens</code> (inputs.py) run through every block and the final norm; CLS is token 0, the patches follow
                 the registers.</p>""",
                 links=[L("forward_features", DINO + "dinov2/models/vision_transformer.py#L257", ("dinov2", "dinov2/models/vision_transformer.py", 257, 275)),
                        L("MulTaBench", MULTA + "README.md")]),
             Section(53, 62,
                 lead="Standardises on the context rows, then PCA with at most 30 components fitted on the context",
                 note="""<p>The baseline's image columns: never more components than context rows minus one, and standardised
                 again after the projection.</p>""",
                 links=[L("sklearn: PCA", "https://scikit-learn.org/stable/modules/decomposition.html#pca")]),
             Section(65, 69,
                 lead="TabPFN v2 as the reference learner, 8 estimators, pretraining limits ignored",
                 note="""<p>Classifier or regressor by task; <code>ignore_pretraining_limits</code> lets it take the 10k-row context
                 sweeps.</p>""",
                 links=[L("TabPFNRegressor.create_default_for_version", TABPFN + "regressor.py#L621", ("tabpfn", "regressor.py", 621, 640)),
                        L("TabPFN v2 (Hollmann et al. 2025)", "https://www.nature.com/articles/s41586-024-08328-6")]),
             Section(72, 84,
                 lead="The scores: AUROC, one-vs-rest beyond two classes, over the classes present; R² for regression",
                 note="""<p>MulTaBench's metrics. Classes no test row has are dropped and the probabilities renormalised, since
                 <code>roc_auc_score</code> wants every label present.</p>""",
                 links=[L("sklearn: ROC metrics", "https://scikit-learn.org/stable/modules/model_evaluation.html#roc-metrics")]),
             Section(87, 91,
                 lead="Letterboxes a list of photo paths in threads",
                 note=""),
             Section(94, 109,
                 lead="Every method's prediction for one split: ours three ways, TabPFN on the table, TabPFN with PCA columns",
                 note="""<p>Our three: <code>predict_classes</code> or <code>predict_values</code> without photos, with them, and with
                 the test photos permuted at seed 0. The references fit on the context rows and predict the test rows, on the
                 plain table and on the table with the frozen PCA columns appended.</p>"""),
             Section(112, 114,
                 lead="Mean, standard error and count of a list of scores",
                 note=""),
             Section(117, 162,
                 lead="Per dataset and split: cap the context, drop unseen test classes, score every method, save the report",
                 note="""<p><code>--max-sources</code> raises the model's source cap at inference (a setting, see
                 <code>pick_sources</code>). The context is a stratified subset of the benchmark's train rows when it exceeds
                 <code>--max-context</code>; test rows whose class the context lacks are dropped, since no in-context learner can
                 predict them. <code>--grouped</code> uses the group keys of multabench.py. The report holds per dataset the metric,
                 the Kaggle version, the row counts, and every method's mean and standard error over the splits.</p>"""),
         ]),
]
