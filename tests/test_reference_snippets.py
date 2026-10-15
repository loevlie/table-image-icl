"""The verbatim copies under tests/reference/ are pinned to the upstream files installed here: a copy that drifts
from what the equivalence tests actually run against fails here. Ports whose upstream is not installed are skipped."""
import os
import re

import pytest

from tour.provenance import PORTS, installed_lines

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COPIED = [p for p in PORTS if p["reference"]]


@pytest.mark.parametrize("port", COPIED, ids=[p["id"] for p in COPIED])
def test_the_reference_copy_is_the_installed_upstream_lines(port):
    local = port["upstream"]["local"]
    lines = installed_lines(local)
    if lines is None:
        pytest.skip(f"{local[0]} is not installed here")
    with open(os.path.join(ROOT, port["reference"])) as f:
        header, *body = f.read().splitlines(keepends=True)
    assert header.startswith("# Reference copy of "), "the first line names the source, its license and the date"
    assert f"lines {local[2]}-{local[3]}," in header, "the header names the line range the registry points at"
    assert re.search(r"\(license: [A-Za-z0-9.-]+\)", header) and re.search(r"copied \d{4}-\d{2}-\d{2}\.", header)
    assert body == lines, f"{port['reference']} is not lines {local[2]}-{local[3]} of the installed {local[1]}"


def test_every_port_names_lines_of_a_file_in_this_repository():
    for port in PORTS:
        path = os.path.join(ROOT, port["ours"]["path"])
        with open(path) as f:
            n = len(f.read().splitlines())
        assert 1 <= port["ours"]["start"] <= port["ours"]["end"] <= n, port["id"]
