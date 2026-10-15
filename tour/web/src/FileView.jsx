// One file as scrollycoding: the steps (sections) scroll on the left, the whole file sits once on the right with the
// current step's lines focused and the rest dimmed (Code Hike's SelectionProvider / Selectable / Selection).
import { useEffect, useMemo, useState } from "react"
import { Pre, highlight } from "codehike/code"
import { Selectable, Selection, SelectionProvider, useSelectedIndex } from "codehike/utils/selection"
import { handlers } from "./handlers.jsx"
import { loadUpstream } from "./api.js"
import RunButton from "./RunButton.jsx"
import Modal from "./Modal.jsx"
import { Card } from "./Equivalence.jsx"

const THEME = "github-light"
const DEF_KINDS = new Set(["function", "class", "statement"])

function defAnnotations(file, defs) {
  const seen = new Set()
  const out = []
  for (const d of defs) {
    if (!DEF_KINDS.has(d.kind)) continue
    if (d.external && d.kind === "statement") continue
    if (d.def_path === file.path && d.def_line === d.line) continue           // the definition itself
    const key = `${d.line}:${d.col}`
    if (seen.has(key)) continue
    seen.add(key)
    out.push({ name: "def", query: d.name, lineNumber: d.line, fromColumn: d.col + 1, toColumn: d.end, data: d })
  }
  return out
}

function shapeAnnotations(trace) {
  return Object.entries(trace).map(([line, text]) => ({ name: "shape", query: text, fromLineNumber: Number(line), toLineNumber: Number(line) }))
}

function UpstreamSnippet({ fi, si, li }) {
  const [snippet, setSnippet] = useState(null)
  const [hl, setHl] = useState(null)
  useEffect(() => { loadUpstream(fi, si, li).then(setSnippet) }, [fi, si, li])
  useEffect(() => { if (snippet && snippet.text) highlight({ value: snippet.text, lang: "python", meta: "" }, THEME).then(setHl) }, [snippet])
  if (!snippet) return <div className="muted">loading…</div>
  if (snippet.error) return <div className="muted">{snippet.error}</div>
  return (
    <div className="upstream">
      <div className="where">{snippet.where}, lines {snippet.first}–{snippet.last}</div>
      {hl && <Pre code={hl} className="snippet" style={{ ...hl.style }} />}
    </div>
  )
}

function Step({ fi, si, section, file, cards, checks, onCompare }) {
  const [shown, setShown] = useState({})
  const card = cards.find((c) => c.ours && c.ours.path === file.path && c.ours.start <= section.end && c.ours.end >= section.start)
  return (
    <>
      <div className="lines">{section.start}–{section.end}</div>
      <h3>{section.lead}</h3>
      {section.note && section.note.trim() && (
        <details className="note">
          <summary>why, and line by line</summary>
          <div dangerouslySetInnerHTML={{ __html: section.note }} />
        </details>
      )}
      {section.links.length > 0 && (
        <div className="links">
          <span className="label">source</span>
          <ul>
            {section.links.map((link, li) => (
              <li key={li}>
                <a href={link.url} target="_blank" rel="noopener">{link.label}</a>
                {link.local && <button className="small" onClick={() => setShown((s) => ({ ...s, [li]: !s[li] }))}>{shown[li] ? "hide" : "show here"}</button>}
                {shown[li] && <UpstreamSnippet fi={fi} si={si} li={li} />}
              </li>
            ))}
          </ul>
        </div>
      )}
      {card && <button className="compare" onClick={() => onCompare(card)}>compare with upstream: {card.title}</button>}
      {section.checks.length > 0 && (
        <div className="checks">
          <span className="label">shown by</span>
          {section.checks.map((c) => <RunButton key={c.node} check={c} />)}
        </div>
      )}
    </>
  )
}

/** Selects the step that holds `line` when the hash asks for one. */
function GoToLine({ sections, line }) {
  const [, select] = useSelectedIndex()
  useEffect(() => {
    if (line == null) return
    const i = sections.findIndex((s) => s.start <= line && line <= s.end)
    if (i >= 0) {
      select(i)
      const scroll = () => {
        const el = document.getElementById("step-" + i)
        if (el) window.scrollTo({ top: el.getBoundingClientRect().top + window.scrollY - window.innerHeight * 0.3, behavior: "smooth" })
      }
      const timer = setTimeout(scroll, 60)                                   // after the panel has laid out
      return () => clearTimeout(timer)
    }
  }, [line, sections, select])
  return null
}

export default function FileView({ fi, file, traces, definitions, cards, checks, line }) {
  const [hl, setHl] = useState(null)
  const [compare, setCompare] = useState(null)
  useEffect(() => {
    setHl(null)
    highlight({ value: file.code, lang: "python", meta: "" }, THEME).then(setHl)
  }, [file.path, file.code])
  const base = useMemo(() => [...defAnnotations(file, definitions[file.path] || []), ...shapeAnnotations(traces[file.path] || {})], [file, definitions, traces])
  if (!hl) return <div className="empty">highlighting {file.path}…</div>
  const steps = file.sections
  const codeFor = (i) => {
    const s = steps[i]
    const folds = steps.filter((t, j) => t.fold && j !== i).map((t) => ({ name: "fold", query: `${t.end - t.start + 1} lines folded · ${t.lead}`, fromLineNumber: t.start, toLineNumber: t.end }))
    const annotations = [...base, ...folds, { name: "focus", query: "", fromLineNumber: s.start, toLineNumber: s.end }]
    return <Pre key={i} code={{ ...hl, annotations }} handlers={handlers} className="pre" style={{ ...hl.style }} />
  }
  return (
    <SelectionProvider className="split" rootMargin={{ top: 140, height: 260 }}>
      <GoToLine sections={steps} line={line} />
      <div className="steps">
        <div className="filehead">
          <div className="path">{file.path}</div>
          <h2>{file.title}</h2>
          <div className="blurb" dangerouslySetInnerHTML={{ __html: file.blurb }} />
        </div>
        {steps.map((s, i) => (
          <Selectable key={i} index={i} selectOn={["click", "scroll"]} className={"step" + (s.fold ? " folded" : "")} id={"step-" + i}>
            <Step fi={fi} si={i} section={s} file={file} cards={cards} checks={checks} onCompare={setCompare} />
          </Selectable>
        ))}
      </div>
      <div className="panel">
        <Selection from={steps.map((_, i) => codeFor(i))} />
      </div>
      {compare && <Modal onClose={() => setCompare(null)}><Card card={compare} checks={checks} compact /></Modal>}
    </SelectionProvider>
  )
}
