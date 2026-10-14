"""The notebooks show code that is actually there, and their executed copies ran without errors."""
import importlib
import importlib.util
import os
import re

import nbformat as nbf
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTEBOOKS = ["walkthrough", "top_k_read", "results"]


def builder(name):
    spec = importlib.util.spec_from_file_location(f"make_{name}", os.path.join(ROOT, "notebooks", f"make_{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def code_of(cells):
    return "\n".join(src for kind, src in cells if kind == "code")


def imported_names(code):
    """Every name a notebook imports from table_image_icl, resolved against the package as it is."""
    names = {}
    for module, imports in re.findall(r"from (table_image_icl(?:\.\w+)?) import ([\w, ]+)", code):
        mod = importlib.import_module(module)
        for name in imports.replace(" ", "").split(","):
            assert hasattr(mod, name), f"{module} has no {name}"
            names[name] = getattr(mod, name)
    return names


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_every_name_the_notebook_imports_and_shows_exists(name):
    cells = builder(name).CELLS
    code = code_of(cells)
    names = imported_names(code)
    assert names
    for dotted in [n.strip() for call in re.findall(r"^show\((.+)\)$", code, flags=re.M) for n in call.split(",")]:
        obj = names[dotted.split(".")[0]]
        for attr in dotted.split(".")[1:]:
            obj = getattr(obj, attr)


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_the_cells_alternate_and_the_sections_are_numbered(name, tmp_path):
    module = builder(name)
    kinds = [kind for kind, _ in module.CELLS]
    assert kinds[0] == "md"
    if name != "results":                                        # the results notebook comments on a figure before the next section
        assert all(not (a == "md" and b == "md") for a, b in zip(kinds, kinds[1:])), "two markdown cells in a row"
    headers = [int(n) for n in re.findall(r"^## (\d+)\. ", "\n".join(s for k, s in module.CELLS if k == "md"), flags=re.M)]
    assert headers == list(range(1, len(headers) + 1))
    nb = nbf.read(module.build(str(tmp_path / "nb.ipynb")), as_version=4)
    assert len(nb.cells) == len(module.CELLS)


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_the_executed_notebook_has_no_errors(name):
    path = os.path.join(ROOT, "notebooks", f"{name}.ipynb")
    if not os.path.exists(path):
        pytest.skip("not executed")
    for cell in nbf.read(path, as_version=4).cells:
        for out in cell.get("outputs", []):
            assert out.output_type != "error", out.get("ename")


def test_the_walkthrough_follows_the_data_flow_and_prints_what_its_prose_claims():
    cells = builder("walkthrough").CELLS
    titles = " | ".join(t.lower() for t in re.findall(r"^## \d+\. (.+)$", "\n".join(s for k, s in cells if k == "md"), flags=re.M))
    positions = [titles.index(word) for word in ["input", "column", "image", "top-k", "row", "in-context"]]
    assert positions == sorted(positions), titles
    path = os.path.join(ROOT, "notebooks", "walkthrough.ipynb")
    if not os.path.exists(path):
        pytest.skip("not executed")
    text = "\n".join(out.get("text", "") for cell in nbf.read(path, as_version=4).cells for out in cell.get("outputs", []))
    expected = {
        "rows whose tokens changed when test row 7, column 0 was moved": "[0, 0, 0, 0, 0, 0, 0, 1]",
        "photos whose tokens changed when photo 0 was touched": "[1, 0, 0, 0, 0, 0, 0, 0]",
        "photos whose patches changed when context photo 1 was relabelled": "[1, 1, 1, 1, 1, 1, 1, 1]",
        "photos whose patches changed when test photo 7 was changed": "[0, 0, 0, 0, 0, 0, 0, 1]",
        "rows changed by touching test row 7's photo": "[0, 0, 0, 0, 0, 0, 0, 1]",
        "rows whose scores changed by changing test row 6": "[0, 0, 0, 0, 0, 0, 1, 0]",
        "rows whose scores changed by changing context row 0": "[1, 1, 1, 1, 1, 1, 1, 1]",
    }
    for label, pattern in expected.items():
        match = re.search(re.escape(label) + r":\s*(\[[\d, ]+\])", text)
        assert match, label
        assert match.group(1) == pattern, f"{label}: {match.group(1)}"


def test_the_top_k_notebook_compares_the_three_implementations_at_the_trained_top_k():
    code = code_of(builder("top_k_read").CELLS)
    assert "TOP_K = 32" in code and "forward_masked" in code and "flex_attention(" in code
    assert code.count("torch.allclose(gathered") == 2 and "backward()" in code


def test_every_run_the_results_notebook_names_has_its_records():
    code = code_of(builder("results").CELLS)
    runs = set(re.findall(r'"(block\d+|cls|pixels)"', code))
    assert runs == {"block4", "block8", "block12", "cls", "pixels"}
    for run in runs:
        for name in ("config.json", "evaluation.json", "multabench.json"):
            assert os.path.exists(os.path.join(ROOT, "results", run, name)), f"{run}/{name}"
