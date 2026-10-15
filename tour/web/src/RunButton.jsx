import { useState } from "react"
import { runCheck } from "./api.js"

/** Runs one pytest node on the server and shows its output as it streams. */
export default function RunButton({ check }) {
  const [out, setOut] = useState(null)
  const [state, setState] = useState("idle")
  const name = check.node.includes("::") ? check.node.split("::").pop() : check.node
  const run = async () => {
    setState("running"); setOut("")
    try {
      await runCheck(check.id, (chunk) => setOut((o) => o + chunk))
      setState("done")
    } catch (e) { setOut((o) => (o || "") + "\ncould not run: " + e); setState("done") }
  }
  const ok = out && /\[exit 0\]/.test(out)
  return (
    <div className="check">
      <button className={"run " + state} onClick={run} disabled={state === "running"} title={check.node}>{state === "running" ? "running…" : "run"}</button>
      <code className="name">{name}</code>
      {out !== null && <pre className={"out " + (state === "done" ? (ok ? "ok" : "bad") : "")}>{out}</pre>}
    </div>
  )
}
