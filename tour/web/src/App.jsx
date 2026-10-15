import { useEffect, useState } from "react"
import { loadTour, readHash } from "./api.js"
import FileView from "./FileView.jsx"
import Equivalence from "./Equivalence.jsx"

export default function App() {
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)
  const [where, setWhere] = useState(readHash())
  useEffect(() => { loadTour().then(setData).catch((e) => setError(String(e))) }, [])
  useEffect(() => {
    const onHash = () => setWhere(readHash())
    const onGoto = (e) => {
      const i = data ? data.files.findIndex((f) => f.path === e.detail.def_path) : -1
      if (i >= 0) window.location.hash = `f${i}-${e.detail.def_line}`
    }
    window.addEventListener("hashchange", onHash)
    window.addEventListener("tour:goto", onGoto)
    return () => { window.removeEventListener("hashchange", onHash); window.removeEventListener("tour:goto", onGoto) }
  }, [data])
  if (error) return <div className="empty">could not load /api/tour: {error}</div>
  if (!data) return <div className="empty">loading…</div>
  const fileIndex = where.view === "file" ? Math.min(where.file, data.files.length - 1) : -1
  return (
    <>
      <header>
        <span className="brand">table-image-icl, line by line</span>
        <nav>
          {data.files.map((f, i) => <a key={f.path} href={`#f${i}`} className={i === fileIndex ? "on" : ""}>{f.path.split("/").pop()}</a>)}
          <a href="#eq" className={where.view === "eq" ? "on eq" : "eq"}>equivalence{data.equivalence.length ? ` (${data.equivalence.length})` : ""}</a>
          <a href="/classic" className="eq">classic</a>
        </nav>
      </header>
      <main>
        {where.view === "eq"
          ? <Equivalence cards={data.equivalence} checks={data.checks} />
          : <FileView key={fileIndex} fi={fileIndex} file={data.files[fileIndex]} traces={data.traces} definitions={data.definitions} cards={data.equivalence} checks={data.checks} line={where.line} />}
      </main>
    </>
  )
}
