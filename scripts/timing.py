"""End-to-end inference time on this machine as the context grows: ours (DINOv2 block-4 tokens + our model) against
MulTaBench's frozen baseline (full DINOv2 features -> PCA-30 fitted on the context -> TabPFN v2 on table + PCA
columns), and TabPFN v2 on the table alone. One table, 10 columns, 3 classes, test rows = 10% of context.

    PYTHONPATH=. python3 scripts/timing.py mps 64,256,1024 docs/figures/inference-time-m1.json   (device: cpu, mps or cuda)"""
import importlib.util
import json
import sys
import time

import numpy as np
import torch

sys.path.insert(0, ".")
from table_image_icl.inputs import DinoTokens
from table_image_icl.model import load_run

spec = importlib.util.spec_from_file_location("benchmark", "scripts/benchmark.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)

device = sys.argv[1]
sizes = [int(s) for s in sys.argv[2].split(",")]
out_path = sys.argv[3]
torch.manual_seed(0)
model, _ = load_run("checkpoints/block4", device)
model.read.max_sources = 256
dino4 = DinoTokens(device, block=4)
dino12 = bench.FrozenDinoFeatures(device)


def timed(fn, repeats):
    fn()
    if device == "mps":
        torch.mps.synchronize()
    times = []
    for _ in range(repeats):
        t = time.perf_counter()
        fn()
        if device == "mps":
            torch.mps.synchronize()
        times.append(time.perf_counter() - t)
    return float(np.median(times))


results = []
for n_context in sizes:
    n_test = max(8, n_context // 10)
    rows = n_context + n_test
    reps = 3 if rows <= 300 else 1
    rng = np.random.default_rng(0)
    x = rng.standard_normal((rows, 10)).astype(np.float32)
    y = rng.integers(0, 3, rows)
    photos = rng.integers(0, 255, (rows, 224, 224, 3), dtype=np.uint8)
    r = {"device": device, "context": n_context, "test": n_test}
    with torch.no_grad():
        # ours
        r["ours_dino4"] = timed(lambda: dino4(photos), reps)
        tokens = dino4(photos)[None]
        xt = torch.tensor(x, device=device)[None]
        yt = torch.tensor(y[:n_context], device=device)[None]
        r["ours_model"] = timed(lambda: model(xt, yt, images=tokens), reps)
        # the frozen baseline
        r["ref_dino12"] = timed(lambda: dino12(photos), reps)
        features = dino12(photos)
        r["ref_pca"] = timed(lambda: bench.pca_columns(features, n_context), reps)
        cols = bench.pca_columns(features, n_context)
        both = np.concatenate([x, cols], 1)
        tabpfn_device = device if device != "mps" else "mps"
        def tabpfn_run(table):
            clf = bench.tabpfn_v2(True, tabpfn_device)
            clf.fit(table[:n_context], y[:n_context])
            return clf.predict_proba(table[n_context:])
        try:
            r["ref_tabpfn"] = timed(lambda: tabpfn_run(both), reps)
            r["tabpfn_table_only"] = timed(lambda: tabpfn_run(x), reps)
            r["tabpfn_device"] = tabpfn_device
        except Exception as e:
            print("tabpfn on", tabpfn_device, "failed:", str(e)[:120], "-> cpu", flush=True)
            tabpfn_device = "cpu"
            r["ref_tabpfn"] = timed(lambda: tabpfn_run(both), reps)
            r["tabpfn_table_only"] = timed(lambda: tabpfn_run(x), reps)
            r["tabpfn_device"] = "cpu"
    r["ours_total"] = r["ours_dino4"] + r["ours_model"]
    r["ref_total"] = r["ref_dino12"] + r["ref_pca"] + r["ref_tabpfn"]
    print(json.dumps(r), flush=True)
    results.append(r)
    json.dump(results, open(out_path, "w"), indent=1)
