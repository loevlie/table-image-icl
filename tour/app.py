"""The code tour: our model's files section by section, in the style of Code Hike's scrollycoding, with what each
section does, where its code comes from (links to the pinned upstream files, the installed copies shown inline), tensor
shapes recorded from a real run, hover definitions, and equivalence cards (our code against the upstream code in a
Monaco diff, both implementations' outputs on one input, a Hypothesis differential test, and a button that runs the
pytest proof here and streams its output).

    python tour/app.py            then open http://127.0.0.1:8899
    (cd tour/web && npm run build) rebuilds the page after a change to tour/web/src

Content: tour/sections.py (notes), tour/data/*.json (traces from tour/trace.py, definitions from tour/definitions.py,
equivalence cards from tour/equivalence.py). /classic is the plain two-column rendering."""
import html
import importlib.util
import json
import mimetypes
import os
import subprocess
import sys

import django
from django.conf import settings
from django.http import HttpResponse, HttpResponseNotAllowed, JsonResponse, StreamingHttpResponse
from django.urls import path
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import PythonLexer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tour.sections import FILES  # noqa: E402

HUB = os.path.expanduser("~/.cache/torch/hub/facebookresearch_dinov2_7764ea0f912e53c92e82eb78a2a1631e92725fc8")
DATA = os.path.join(ROOT, "tour", "data")
DIST = os.path.join(ROOT, "tour", "web", "dist")
PORT = 8899


def read_json(name: str, default):
    p = os.path.join(DATA, name)
    if not os.path.exists(p):
        return default
    with open(p) as f:
        return json.load(f)


def cards() -> list[dict]:
    """The equivalence cards (tour/equivalence.py writes {"ports": [...]} with a little metadata beside them)."""
    data = read_json("equivalence.json", [])
    return data.get("ports", []) if isinstance(data, dict) else data


def checks() -> list[str]:
    """Every pytest node id the page may run: the sections' proofs and the equivalence cards' tests."""
    ids = {check for f in FILES for s in f.sections for check in s.checks}
    for card in cards():
        ids.update(card.get("tests", []))
        if (card.get("hypothesis") or {}).get("test"):
            ids.add(card["hypothesis"]["test"])
    return sorted(ids)


def package_root(name: str) -> str | None:
    if name == "dinov2":
        return HUB if os.path.isdir(HUB) else None
    spec = importlib.util.find_spec(name)
    return os.path.dirname(spec.origin) if spec and spec.origin else None


def file_lines(path: str) -> list[str]:
    with open(path) as f:
        return f.readlines()


# ---------- the React page and its data ----------

def tour_data(request):
    ids = checks()
    files = []
    for f in FILES:
        files.append({"path": f.path, "title": f.title, "blurb": f.blurb, "code": "".join(file_lines(os.path.join(ROOT, f.path))),
                      "sections": [{"start": s.start, "end": s.end, "lead": s.lead, "note": s.note, "fold": s.fold,
                                    "links": [{"label": link.label, "url": link.url, "local": bool(link.local and package_root(link.local[0]))} for link in s.links],
                                    "checks": [{"id": ids.index(c), "node": c} for c in s.checks]} for s in f.sections]})
    return JsonResponse({"files": files, "traces": read_json("traces.json", {}), "definitions": read_json("definitions.json", {}),
                         "equivalence": cards(), "checks": ids})


def upstream_data(request, fi: int, si: int, li: int):
    link = FILES[fi].sections[si].links[li]
    package, rel, first, last = link.local
    root = package_root(package)
    if root is None:
        return JsonResponse({"error": "not installed here"})
    lines = file_lines(os.path.join(root, rel))
    return JsonResponse({"text": "".join(lines[first - 1:last]), "first": first, "last": min(last, len(lines)),
                         "where": f"{package}/{rel}, as installed at {root}", "url": link.url})


def run(request, i: int):
    """Runs one of the registered tests and streams pytest's output as it comes."""
    if request.method != "POST":
        return HttpResponseNotAllowed(["POST"])
    check = checks()[i]
    command = [sys.executable, "-m", "pytest", check, "-q", "-p", "no:cacheprovider", "-rA", "--no-header", "-W", "ignore"]

    def lines():
        yield f"$ python -m pytest {check} -q -rA\n\n"
        process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        yield from process.stdout
        process.wait()
        yield f"\n[exit {process.returncode}]\n"

    response = StreamingHttpResponse(lines(), content_type="text/plain; charset=utf-8")
    response["Cache-Control"] = "no-cache"
    response["X-Accel-Buffering"] = "no"
    return response


def page(request):
    index = os.path.join(DIST, "index.html")
    if not os.path.exists(index):
        return HttpResponse("<p>The page is not built: run <code>cd tour/web && npm install && npm run build</code>, then reload. "
                            "The plain rendering is at <a href='/classic'>/classic</a>.</p>")
    return HttpResponse(open(index).read())


def asset(request, name: str):
    p = os.path.normpath(os.path.join(DIST, "assets", name))
    if not p.startswith(DIST) or not os.path.exists(p):
        return HttpResponse(status=404)
    with open(p, "rb") as f:
        return HttpResponse(f.read(), content_type=mimetypes.guess_type(p)[0] or "application/octet-stream")


# ---------- the plain two-column rendering (/classic) ----------

def code_html(lines: list[str], first: int) -> str:
    return highlight("".join(lines), PythonLexer(), HtmlFormatter(linenos="table", linenostart=first, cssclass="hl"))


def render_file(fi: int, f) -> str:
    lines = file_lines(os.path.join(ROOT, f.path))
    rows = []
    for s in f.sections:
        links = "".join(f'<li><a href="{html.escape(link.url)}" target="_blank" rel="noopener">{html.escape(link.label)}</a></li>' for link in s.links)
        tests = "".join(f'<li><code>{html.escape(c.split("::")[-1])}</code></li>' for c in s.checks)
        extra = (f'<div class="links"><span class="label">Source</span><ul>{links}</ul></div>' if links else "") + (f'<div class="checks"><span class="label">Shown by</span><ul>{tests}</ul></div>' if tests else "")
        rows.append(f'<div class="row"><div class="note"><div class="lines">{s.start}–{s.end}</div><p><b>{html.escape(s.lead)}</b></p>{s.note}{extra}</div><div class="code">{code_html(lines[s.start - 1:s.end], s.start)}</div></div>')
    return f'<section class="file" id="f{fi}"><h2><span class="path">{html.escape(f.path)}</span>{html.escape(f.title)}</h2><div class="blurb">{f.blurb}</div>{"".join(rows)}</section>'


CLASSIC_STYLE = """
body { margin: 0; background: #fbfaf7; color: #1d1d1b; font: 15px/1.5 Georgia, serif; }
header { position: sticky; top: 0; background: #fbfaf7; border-bottom: 1px solid #e4e1da; padding: 10px 24px; }
header a { color: #6b6a66; margin-right: 12px; font-size: 13px; text-decoration: none; }
section.file h2 { font-size: 20px; font-weight: normal; margin: 48px 24px 6px; } section.file h2 .path { display: block; font: 13px Menlo, monospace; color: #6b6a66; }
.blurb { margin: 0 24px 18px; max-width: 70ch; color: #6b6a66; }
.row { display: grid; grid-template-columns: minmax(280px, 38fr) minmax(0, 62fr); border-top: 1px solid #e4e1da; }
.note { padding: 14px 24px 16px; font-size: 14.5px; } .note p { margin: 0 0 8px; } .note code { font: 12.5px Menlo, monospace; background: #f3f1ec; }
.lines { font: 11px Menlo, monospace; color: #6b6a66; } .code { background: #f3f1ec; padding: 8px 12px; overflow-x: auto; }
.hl table { border-spacing: 0; font: 12.5px/1.45 Menlo, monospace; } .hl .linenos { color: #b4b0a6; padding-right: 12px; text-align: right; } .hl pre { margin: 0; }
.label { display: block; font: 11px Menlo, monospace; letter-spacing: .06em; text-transform: uppercase; color: #6b6a66; margin-top: 10px; }
.links ul, .checks ul { list-style: none; margin: 2px 0 0; padding: 0; font-size: 13.5px; } .links a { color: #8a3b12; }
"""


def classic(request):
    nav = "".join(f'<a href="#f{i}">{html.escape(os.path.basename(f.path))}</a>' for i, f in enumerate(FILES))
    body = "".join(render_file(i, f) for i, f in enumerate(FILES))
    css = HtmlFormatter(cssclass="hl").get_style_defs(".hl")
    return HttpResponse(f"<!doctype html><html><head><meta charset='utf-8'><title>Code tour (classic)</title><style>{css}{CLASSIC_STYLE}</style></head>"
                        f"<body><header>{nav} <a href='/'>interactive version</a></header><main>{body}</main></body></html>")


urlpatterns = [path("", page), path("assets/<path:name>", asset), path("api/tour", tour_data),
               path("api/upstream/<int:fi>/<int:si>/<int:li>", upstream_data), path("run/<int:i>", run), path("classic", classic)]

if __name__ == "__main__":
    settings.configure(DEBUG=True, SECRET_KEY="tour", ROOT_URLCONF=__name__, ALLOWED_HOSTS=["127.0.0.1", "localhost"], MIDDLEWARE=[])
    django.setup()
    from django.core.management import execute_from_command_line
    execute_from_command_line([sys.argv[0], "runserver", f"127.0.0.1:{PORT}", "--noreload"])
