export async function loadTour() {
  const r = await fetch("/api/tour")
  return r.json()
}

export async function loadUpstream(fi, si, li) {
  const r = await fetch(`/api/upstream/${fi}/${si}/${li}`)
  return r.json()
}

/** Runs one registered test on the server and calls onChunk with the output as it streams. */
export async function runCheck(id, onChunk) {
  const r = await fetch(`/run/${id}`, { method: "POST" })
  const reader = r.body.getReader()
  const decoder = new TextDecoder()
  for (;;) {
    const { value, done } = await reader.read()
    if (done) break
    onChunk(decoder.decode(value, { stream: true }))
  }
}

/** The location hash as state: #f2 (file 2), #f2-118 (file 2, the section holding line 118), #eq (equivalence). */
export function readHash() {
  const h = window.location.hash.slice(1)
  if (h === "eq") return { view: "eq" }
  const m = /^f(\d+)(?:-(\d+))?$/.exec(h)
  return m ? { view: "file", file: Number(m[1]), line: m[2] ? Number(m[2]) : null } : { view: "file", file: 0, line: null }
}
