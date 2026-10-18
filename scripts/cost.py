"""Cost of predicting one table with a photo per row, ours against MulTaBench's frozen baseline, written to a CSV:
forward FLOPs (hardware independent) and seconds on this machine, at each context size.

    PYTHONPATH=. python3 scripts/cost.py --sizes 64,256,1024 --device mps --out docs/figures/cost.csv

ours: DINOv2 blocks 1-4 -> our model.  baseline: all 12 DINOv2 blocks -> PCA-30 fitted on the context -> TabPFN v2
with 8 estimators, as the benchmark runs it. One table, 10 columns, 3 classes, test rows = 10% of context."""
import argparse
import csv
import importlib.util
import time

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.flop_counter import FlopCounterMode

from table_image_icl.inputs import DinoTokens
from table_image_icl.model import load_run

spec = importlib.util.spec_from_file_location("benchmark", "scripts/benchmark.py")
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def tflops(fn, grad: bool = False) -> float:
    """Forward FLOPs of fn(), from torch's counter with attention on the plain matrix-product path so it is seen.
    Counted on the CPU: the fused attention kernels of other devices hide their products from the counter. Autograd
    is on only for TabPFN, whose parameters are not ours to freeze; nothing is backpropagated."""
    with (torch.enable_grad() if grad else torch.no_grad()), sdpa_kernel([SDPBackend.MATH]), FlopCounterMode(display=False) as counter:
        fn()
    return counter.get_total_flops() / 1e12


def seconds(fn, device: str) -> float:
    with torch.no_grad():
        fn()                                                               # warm-up
        if device == "mps":
            torch.mps.synchronize()
        start = time.perf_counter()
        fn()
        if device == "mps":
            torch.mps.synchronize()
    return time.perf_counter() - start


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="64,256,1024", help="context rows, comma separated")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--out", default="docs/figures/cost.csv")
    ap.add_argument("--run", default="checkpoints/block4")
    args = ap.parse_args()
    device = args.device
    on = {d: load_run(args.run, d)[0].requires_grad_(False) for d in {"cpu", device}}          # FLOPs on the CPU, seconds on the device
    for m in on.values():
        m.read.max_sources = 256
    dino4 = {d: DinoTokens(d, block=4) for d in on}
    dino12 = {d: benchmark.FrozenDinoFeatures(d) for d in on}
    fields = ["context", "test", "ours_dino4_tflops", "ours_model_tflops", "ours_tflops", "ref_dino12_tflops", "ref_tabpfn_tflops", "ref_tflops", "ratio_tflops",
              "ours_dino4_s", "ours_model_s", "ours_s", "ref_dino12_s", "ref_tabpfn_s", "ref_s", "ratio_s", "device"]
    with open(args.out, "w", newline="") as f:
        out = csv.DictWriter(f, fieldnames=fields)
        out.writeheader()
        for n_context in [int(s) for s in args.sizes.split(",")]:
            n_test = max(8, n_context // 10)
            rows = n_context + n_test
            rng = np.random.default_rng(0)
            x, y = rng.standard_normal((rows, 10)).astype(np.float32), rng.integers(0, 3, rows)
            photos = rng.integers(0, 255, (rows, 224, 224, 3), dtype=np.uint8)
            r = {"context": n_context, "test": n_test, "device": device}
            sample = 8                                                     # DINOv2's FLOPs are per photo: count 8, scale by the rows
            r["ours_dino4_tflops"] = tflops(lambda: dino4["cpu"](photos[:sample])) * rows / sample
            r["ref_dino12_tflops"] = tflops(lambda: dino12["cpu"](photos[:sample])) * rows / sample
            r["ours_dino4_s"], r["ref_dino12_s"] = seconds(lambda: dino4[device](photos), device), seconds(lambda: dino12[device](photos), device)
            with torch.no_grad():
                tokens = {d: dino4[d](photos)[None] for d in on}
                features = dino12["cpu"](photos)
            xt = {d: torch.tensor(x, device=d)[None] for d in on}
            yt = {d: torch.tensor(y[:n_context], device=d)[None] for d in on}
            r["ours_model_tflops"] = tflops(lambda: on["cpu"](xt["cpu"], yt["cpu"], images=tokens["cpu"]))
            r["ours_model_s"] = seconds(lambda: on[device](xt[device], yt[device], images=tokens[device]), device)
            table = np.concatenate([x, benchmark.pca_columns(features, n_context)], 1)

            def tabpfn(d):
                clf = benchmark.tabpfn_v2(True, d)
                clf.fit(table[:n_context], y[:n_context])
                return clf.predict_proba(table[n_context:])
            r["ref_tabpfn_tflops"] = tflops(lambda: tabpfn("cpu"), grad=True)
            r["ref_tabpfn_s"] = seconds(lambda: tabpfn(device), device)
            for unit in ("tflops", "s"):
                r[f"ours_{unit}"] = r[f"ours_dino4_{unit}"] + r[f"ours_model_{unit}"]
                r[f"ref_{unit}"] = r[f"ref_dino12_{unit}"] + r[f"ref_tabpfn_{unit}"]
                r[f"ratio_{unit}"] = r[f"ref_{unit}"] / r[f"ours_{unit}"]
            out.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
            f.flush()
            print(f"{n_context:>5} rows: ours {r['ours_tflops']:.2f} TFLOPs / {r['ours_s']:.1f} s, baseline {r['ref_tflops']:.2f} TFLOPs / {r['ref_s']:.1f} s", flush=True)


if __name__ == "__main__":
    main()
