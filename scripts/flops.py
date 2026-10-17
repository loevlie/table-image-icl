"""FLOPs per table, counted with torch's FlopCounterMode, for our pipeline (DINOv2 blocks 1-4, then our model) and
MulTaBench's frozen baseline (all 12 DINOv2 blocks, PCA-30, TabPFN v2 with 8 estimators) as the context grows. Hardware
independent, unlike scripts/timing.py. One table, 10 columns, 3 classes, test rows = 10% of context.

    PYTHONPATH=. python3 scripts/flops.py cpu 64,256,1024,4096 docs/figures/flops.json"""
import importlib.util
import json
import sys

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.flop_counter import FlopCounterMode

sys.path.insert(0, ".")
from table_image_icl.inputs import DinoTokens
from table_image_icl.model import load_run

spec = importlib.util.spec_from_file_location("benchmark", "scripts/benchmark.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def flops(fn, grad: bool = False) -> float:
    """Forward FLOPs of fn() as torch counts them (matrix products and convolutions). The counter's module tracker
    trips on tensors that require a gradient but have none, so a module is counted either with every parameter
    frozen under no_grad (our model, DINOv2), or with autograd enabled and nothing backpropagated (TabPFN, whose
    parameters are not ours to freeze)."""
    grad_mode = torch.enable_grad() if grad else torch.no_grad()
    with grad_mode, sdpa_kernel([SDPBackend.MATH]), FlopCounterMode(display=False) as counter:   # MATH: attention as matrix products the counter sees
        fn()
    return float(counter.get_total_flops())


def main():
    device, sizes, out = sys.argv[1], [int(s) for s in sys.argv[2].split(",")], sys.argv[3]
    torch.manual_seed(0)
    model, _ = load_run("checkpoints/block4", device)
    model.read.max_sources = 256
    model.requires_grad_(False)
    dino4, dino12 = DinoTokens(device, block=4), bench.FrozenDinoFeatures(device)
    results = []
    for n_context in sizes:
        n_test = max(8, n_context // 10)
        rows = n_context + n_test
        rng = np.random.default_rng(0)
        x = rng.standard_normal((rows, 10)).astype(np.float32)
        y = rng.integers(0, 3, rows)
        photos = rng.integers(0, 255, (rows, 224, 224, 3), dtype=np.uint8)
        r = {"device": device, "context": n_context, "test": n_test}
        sample = 8                                                             # DINOv2's FLOPs are per photo: count 8, scale
        r["ours_dino4"] = flops(lambda: dino4(photos[:sample])) * rows / sample
        r["ref_dino12"] = flops(lambda: dino12(photos[:sample])) * rows / sample
        with torch.no_grad():
            tokens = dino4(photos)[None]
            features = dino12(photos)
        xt, yt = torch.tensor(x, device=device)[None], torch.tensor(y[:n_context], device=device)[None]
        r["ours_model"] = flops(lambda: model(xt, yt, images=tokens))
        tokens = None
        cols = bench.pca_columns(features, n_context)
        both = np.concatenate([x, cols], 1)

        def tabpfn_run(table):
            clf = bench.tabpfn_v2(True, device)
            clf.fit(table[:n_context], y[:n_context])
            return clf.predict_proba(table[n_context:])
        r["ref_tabpfn"] = flops(lambda: tabpfn_run(both), grad=True)
        r["tabpfn_table_only"] = flops(lambda: tabpfn_run(x), grad=True)
        r["ours_total"] = r["ours_dino4"] + r["ours_model"]
        r["ref_total"] = r["ref_dino12"] + r["ref_tabpfn"]                  # PCA is negligible and not a torch op
        print(json.dumps(r), flush=True)
        results.append(r)
        json.dump(results, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
