// Equivalence cards: our function against the upstream one in a Monaco diff, both implementations' outputs on one
// fixed input, the Hypothesis differential test's outcome, and the pytest proofs, runnable here.
import * as monaco from "monaco-editor"
import { DiffEditor, loader } from "@monaco-editor/react"
import RunButton from "./RunButton.jsx"

// No worker: Monaco computes the diff on the main thread (fine for a few hundred lines) and warns once in the console.
loader.config({ monaco })

function checkFor(node, checks) {
  const id = checks.indexOf(node)
  return id >= 0 ? { id, node } : null
}

export function Card({ card, checks, compact = false }) {
  const lines = Math.max((card.ours_code || "").split("\n").length, (card.upstream_code || "").split("\n").length)
  const height = Math.min(560, 40 + lines * 19)
  const sample = card.sample || {}
  const hyp = card.hypothesis || null
  const tests = [...(card.tests || []), ...(hyp && hyp.test ? [hyp.test] : [])].map((t) => checkFor(t, checks)).filter(Boolean)
  return (
    <section className={"ecard" + (compact ? " compact" : "")} id={"card-" + card.id}>
      <h3>{card.title}</h3>
      {card.changes && <p className="changes">{card.changes}</p>}
      <div className="diffhead">
        <span>upstream: {card.upstream?.url ? <a href={card.upstream.url} target="_blank" rel="noopener">{card.upstream.label || card.upstream.url}</a> : card.upstream?.label || "a single large batch"}</span>
        <span>ours: {card.ours?.path}:{card.ours?.start}–{card.ours?.end}</span>
      </div>
      {card.upstream_code && card.ours_code ? (
        <div className="diff" style={{ height }}>
          <DiffEditor original={card.upstream_code} modified={card.ours_code} language="python" theme="vs"
            options={{ readOnly: true, renderSideBySide: true, minimap: { enabled: false }, fontSize: 12, lineNumbers: "on", scrollBeyondLastLine: false, renderOverviewRuler: false, wordWrap: "off", originalEditable: false, ignoreTrimWhitespace: true, scrollbar: { alwaysConsumeMouseWheel: false } }} />
        </div>
      ) : <p className="muted">no upstream text for this card</p>}
      {sample.ours_png && (
        <div className="sample">
          <p className="desc">{sample.description}</p>
          <div className="imgs">
            <figure><img src={sample.ours_png} alt="ours" /><figcaption>ours</figcaption></figure>
            <figure><img src={sample.theirs_png} alt="upstream" /><figcaption>upstream</figcaption></figure>
            <figure><img src={sample.diff_png} alt="difference" /><figcaption>|difference|</figcaption></figure>
          </div>
          <p className="numbers">max |difference| <b>{fmt(sample.max_abs_diff)}</b> · tolerance {fmt(sample.tolerance)} · <span className={sample.max_abs_diff <= sample.tolerance ? "ok" : "bad"}>{sample.max_abs_diff <= sample.tolerance ? "within tolerance" : "outside tolerance"}</span></p>
        </div>
      )}
      {hyp && hyp.examples != null && (
        <p className="numbers">Hypothesis: <b>{hyp.examples}</b> random inputs, <b>{hyp.failures}</b> failures{hyp.max_abs_diff != null && <>, max |difference| {fmt(hyp.max_abs_diff)}</>}</p>
      )}
      {tests.length > 0 && <div className="tests">{tests.map((t) => <RunButton key={t.node} check={t} />)}</div>}
    </section>
  )
}

function fmt(x) {
  if (x == null) return "–"
  if (x === 0) return "0"
  return Math.abs(x) < 1e-3 || Math.abs(x) >= 1e4 ? x.toExponential(1) : x.toPrecision(3)
}

export default function Equivalence({ cards, checks }) {
  if (!cards.length) return <div className="empty"><p>No equivalence cards yet: run <code>python tour/equivalence.py</code> and reload.</p></div>
  return (
    <div className="eqpage">
      <p className="intro">Where our code re-implements or cuts short someone else's, the two run side by side: the code in a diff, the outputs on one fixed input, a differential test over random inputs, and the pytest proof you can run here.</p>
      {cards.map((c) => <Card key={c.id} card={c} checks={checks} />)}
    </div>
  )
}
