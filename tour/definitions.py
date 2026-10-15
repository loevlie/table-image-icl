"""Where every identifier of the tour's files is defined, for the web page's hover links (tour/sections.py lists the files).

    python3 tour/definitions.py      -> tour/data/definitions.json

jedi resolves each name occurrence (definitions and references, all scopes) with goto, or infer when goto gives
nothing, and keeps those defined in this repository, in torch, in tabicl or in the pinned DINOv2 hub checkout. Builtins
and names that resolve nowhere else are dropped. Output per file: a list of
{"line", "col", "end", "name", "def_path", "def_line", "kind", "doc", "external"}, def_path repo-relative inside the
repository and relative to the package's root elsewhere ("torch/nn/functional.py", "tabicl/_model/tabicl.py",
"dinov2/models/vision_transformer.py"; never a path of this machine), doc the definition's signature or first
docstring line (at most 140 characters)."""
import json
import os
import sys
import time

import jedi
import tabicl
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "tour", "data", "definitions.json")
DINOV2_CACHE = os.path.expanduser("~/.cache/torch/hub/facebookresearch_dinov2_7764ea0f912e53c92e82eb78a2a1631e92725fc8")
PACKAGES = {"torch": os.path.dirname(os.path.abspath(torch.__file__)), "tabicl": os.path.dirname(os.path.abspath(tabicl.__file__))}
ROOTS = {"repo": REPO, **PACKAGES, "dinov2": DINOV2_CACHE}      # the hub checkout is a repository root: its files are already dinov2/...
KINDS = {"module": "module", "class": "class", "function": "function", "param": "param", "statement": "statement", "property": "function", "instance": "class"}
DOC_LENGTH = 140

sys.path.insert(0, os.path.join(REPO, "tour"))
from sections import FILES  # noqa: E402


def inside(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip(os.sep) + os.sep)


def shown_path(path: str) -> str:
    """A definition's path as the page shows it: relative to this repository, or to the package root with the
    package's name in front (the DINOv2 checkout already starts with dinov2/)."""
    if inside(path, REPO):
        return os.path.relpath(path, REPO)
    for name, root in PACKAGES.items():
        if inside(path, root):
            return os.path.join(name, os.path.relpath(path, root))
    return os.path.relpath(path, DINOV2_CACHE)


def doc_of(definition) -> str:
    """The signature line of a function or class, else the first docstring line, else the defining line of code.
    A parameter or an assignment shows its line of code (its docstring would be the inferred value's)."""
    text = ""
    if definition.type in ("param", "statement"):
        try:
            text = definition.get_line_code().strip()
        except Exception:
            pass
    if not text:
        try:
            text = next((line.strip() for line in definition.docstring(raw=False).splitlines() if line.strip()), "")
        except Exception:
            pass
    if not text and definition.type == "module":
        text = f"module {definition.full_name or definition.name}"
    if not text:
        try:
            text = definition.get_line_code().strip()
        except Exception:
            text = ""
    return text if len(text) <= DOC_LENGTH else text[: DOC_LENGTH - 3] + "..."


def resolve(name):
    """The definition a name occurrence stands for, or None when it is a builtin or resolves nowhere we show."""
    definitions = name.goto(follow_imports=True, follow_builtin_imports=False) or name.infer()
    candidates = []
    for definition in definitions:
        if definition.in_builtin_module() or definition.module_path is None or definition.line is None:
            continue
        path = os.path.abspath(str(definition.module_path))
        if any(inside(path, root) for root in ROOTS.values()):
            candidates.append((definition, path))
    if not candidates:
        return None
    return min(candidates, key=lambda c: (c[1], c[0].line, c[0].column or 0))     # several (an attribute set in sibling classes): a fixed choice


def entries_for(path: str, project: jedi.Project) -> list[dict]:
    with open(path) as f:
        code = f.read()
    script = jedi.Script(code=code, path=path, project=project)
    names = script.get_names(all_scopes=True, definitions=False, references=True) + script.get_names(all_scopes=True, definitions=True, references=False)
    seen, entries = set(), []
    for name in names:
        try:
            found = resolve(name)
            if found is None:
                continue
            definition, def_path = found
            entry = {"line": name.line, "col": name.column, "end": name.column + len(name.name), "name": name.name,
                     "def_path": shown_path(def_path), "def_line": definition.line,
                     "kind": KINDS.get(definition.type, "statement"), "doc": doc_of(definition), "external": not inside(def_path, REPO)}
        except Exception as e:                                       # jedi gives up on some names; keep the rest
            print(f"  {os.path.relpath(path, REPO)}:{name.line}:{name.column} {name.name}: {type(e).__name__}: {e}", file=sys.stderr)
            continue
        key = tuple(entry.items())
        if key not in seen:
            seen.add(key)
            entries.append(entry)
    entries.sort(key=lambda e: (e["line"], e["col"], e["name"], e["def_path"], e["def_line"]))
    return entries


def main() -> None:
    started = time.time()
    project = jedi.Project(path=REPO, sys_path=[REPO] + [p for p in sys.path if p])
    out = {}
    for file in FILES:
        out[file.path] = entries_for(os.path.join(REPO, file.path), project)
        externals = sum(e["external"] for e in out[file.path])
        print(f"{len(out[file.path]):5d} names  {file.path}  ({externals} outside the repo)", flush=True)
    assert not any(os.path.isabs(e["def_path"]) for entries in out.values() for e in entries), "a definition path is a path of this machine"
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    print(f"{sum(len(v) for v in out.values())} names in {len(out)} files -> {os.path.relpath(OUT, REPO)} ({time.time() - started:.0f}s)")


if __name__ == "__main__":
    main()
