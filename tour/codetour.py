"""Exports the tour's sections as VS Code CodeTour files (.tours/<file>.tour), one tour per source file, so the
same notes can be stepped through inside the editor: `python tour/codetour.py`, then open the CodeTour panel."""
import html
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tour.sections import FILES  # noqa: E402


def markdown(note: str) -> str:
    """The HTML note as Markdown: paragraphs separated, <code> as backticks, links kept, other tags dropped."""
    text = re.sub(r"</p>\s*<p>", "\n\n", note.strip())
    text = re.sub(r"<code>(.*?)</code>", r"`\1`", text, flags=re.S)
    text = re.sub(r'<a href="([^"]+)"[^>]*>(.*?)</a>', r"[\2](\1)", text, flags=re.S)
    text = re.sub(r"<[^>]+>", "", text)
    return html.unescape(re.sub(r"[ \t]*\n[ \t]*", " ", text)).replace("  ", " ").strip()


def tour_for(index: int, f) -> dict:
    steps = []
    for s in f.sections:
        parts = [f"**{s.lead}**"]
        if s.note and s.note.strip():
            parts.append(markdown(s.note))
        if s.links:
            parts.append("Source: " + ", ".join(f"[{link.label}]({link.url})" for link in s.links))
        if s.checks:
            parts.append("Shown by: " + ", ".join(f"`{c.split('::')[-1]}`" for c in s.checks))
        steps.append({"file": f.path, "line": s.start, "title": f"{f.path.split('/')[-1]}:{s.start}-{s.end}", "description": "\n\n".join(parts)})
    tour = {"$schema": "https://aka.ms/codetour-schema", "title": f"{index + 1:02d} {f.path}", "description": markdown(f.blurb), "steps": steps}
    if index + 1 < len(FILES):
        tour["nextTour"] = f"{index + 2:02d} {FILES[index + 1].path}"
    return tour


def main():
    folder = os.path.join(ROOT, ".tours")
    os.makedirs(folder, exist_ok=True)
    for i, f in enumerate(FILES):
        name = f.path.replace("/", "_").removesuffix(".py")
        with open(os.path.join(folder, f"{i + 1:02d}_{name}.tour"), "w") as out:
            json.dump(tour_for(i, f), out, indent=1)
        print(f"{name}: {len(f.sections)} steps")


if __name__ == "__main__":
    main()
