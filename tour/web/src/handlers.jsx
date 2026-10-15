// Annotation handlers for Code Hike's Pre: line numbers, focus (the current step's lines; the rest dimmed), shapes
// recorded from a real run at the end of a line, folded regions, and hover definitions on identifiers.
import { InnerLine, InnerPre, getPreRef } from "codehike/code"
import { useLayoutEffect, useRef, useState } from "react"
import { FloatingPortal, flip, offset, shift, useFloating, useHover, useInteractions, safePolygon } from "@floating-ui/react"

function PreWithRef(props) {
  const ref = getPreRef(props)
  useScrollToFocus(ref)
  return <InnerPre merge={props} />
}

/** Scrolls the code panel so the focused lines sit in view (after Code Hike's focus.client.tsx). */
function useScrollToFocus(ref) {
  const firstRender = useRef(true)
  useLayoutEffect(() => {
    const pre = ref.current
    if (!pre) return
    const focused = pre.querySelectorAll("[data-focus=true]")
    if (!focused.length) return
    const box = pre.getBoundingClientRect()
    let top = Infinity, bottom = -Infinity
    focused.forEach((el) => {
      const r = el.getBoundingClientRect()
      top = Math.min(top, r.top - box.top)
      bottom = Math.max(bottom, r.bottom - box.top)
    })
    if (bottom > box.height - 24 || top < 24) {
      pre.scrollTo({ top: pre.scrollTop + top - Math.max(40, (box.height - (bottom - top)) / 3), behavior: firstRender.current ? "instant" : "smooth" })
    }
    firstRender.current = false
  })
}

export const focus = {
  name: "focus",
  PreWithRef,
  Line: (props) => <InnerLine merge={props} className="ln" />,
  AnnotatedLine: ({ annotation, ...props }) => <InnerLine merge={props} data-focus={true} />,
}

export const lineNumbers = {
  name: "line-numbers",
  Line: (props) => (
    <div className="line">
      <span className="num">{props.lineNumber}</span>
      <InnerLine merge={props} className="content" />
    </div>
  ),
}

/** `# !shape` annotations: the query is the shapes text, drawn after the line by CSS. */
export const shape = {
  name: "shape",
  AnnotatedLine: ({ annotation, ...props }) => <InnerLine merge={props} data-shape={annotation.query} />,
}

export const fold = {
  name: "fold",
  Block: ({ annotation, children }) => {
    const [open, setOpen] = useState(false)
    if (open) return <div className="unfolded" onDoubleClick={() => setOpen(false)}>{children}</div>
    return <div className="fold" onClick={() => setOpen(true)} title="click to unfold">{annotation.query}</div>
  },
}

function DefSpan({ data, children }) {
  const [open, setOpen] = useState(false)
  const { refs, floatingStyles, context } = useFloating({ open, onOpenChange: setOpen, placement: "top-start", middleware: [offset(6), flip(), shift({ padding: 8 })] })
  const hover = useHover(context, { delay: { open: 250, close: 80 }, handleClose: safePolygon() })
  const { getReferenceProps, getFloatingProps } = useInteractions([hover])
  const where = data.external ? data.def_path.replace(/^.*site-packages\//, "").replace(/^.*torch\/hub\//, "") : data.def_path
  const go = () => window.dispatchEvent(new CustomEvent("tour:goto", { detail: data }))
  return (
    <>
      <span ref={refs.setReference} {...getReferenceProps()} className={"def" + (data.external ? " ext" : "")} onClick={data.external ? undefined : go}>{children}</span>
      {open && (
        <FloatingPortal>
          <div ref={refs.setFloating} style={floatingStyles} {...getFloatingProps()} className="card">
            <div className="where">{data.kind} · {where}:{data.def_line}{data.external ? "" : " · click to go"}</div>
            <code>{data.doc}</code>
          </div>
        </FloatingPortal>
      )}
    </>
  )
}

export const def = {
  name: "def",
  Inline: ({ annotation, children }) => <DefSpan data={annotation.data}>{children}</DefSpan>,
}

export const handlers = [lineNumbers, focus, shape, fold, def]
