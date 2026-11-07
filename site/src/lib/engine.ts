// Bridge to the in-browser engine (public/worker.js): Pyodide runs the real `resenh` package,
// ONNX Runtime Web runs the network. Nothing leaves the device.
import { useCallback, useEffect, useRef, useState } from "react"

export type Mode = "hybrid" | "dl" | "classical"

export interface Report {
  mode: string
  model: string | null
  width: number
  height: number
  timings: Record<string, number>
  analyzer: { brightness: number; dark_frac: number; highlight_frac: number; noise_sigma: number; cast: number; nonuniformity: number }
  params: { low_light: boolean; retinex_gamma: number; denoise: number; clahe_clip: number; wb: number; highlight_protect: number }
  quality: { noise_ratio: number; epi: number }
  decision: {
    weights: Record<string, number>
    guards: Record<string, Record<string, number>>
    penalties: Record<string, number>
    notes: string[]
  } | null
}

export interface Result {
  output: string
  input: string
  panels: { url: string; label: string }[]
  report: Report
  ms: number
  netMs: number
  backend: string
}

export type Phase = "loading" | "ready" | "running" | "error"

const WHEEL = "py/resenh-0.1.0-py3-none-any.whl"
const url = (b: Uint8Array) => URL.createObjectURL(new Blob([b as BlobPart], { type: "image/png" }))

export function useEngine() {
  const worker = useRef<Worker | null>(null)
  const [phase, setPhase] = useState<Phase>("loading")
  const [progress, setProgress] = useState({ pct: 2, label: "Starting the engine" })
  const [backend, setBackend] = useState<string>("")
  const [error, setError] = useState<string>("")
  const [result, setResult] = useState<Result | null>(null)
  const pending = useRef<null | (() => void)>(null)

  useEffect(() => {
    const w = new Worker("/worker.js?v=2")
    worker.current = w
    w.onmessage = (e: MessageEvent) => {
      const m = e.data
      if (m.type === "progress") setProgress({ pct: m.pct, label: m.label })
      else if (m.type === "ready") {
        setBackend(m.backend === "webgpu" ? "WebGPU" : "WebAssembly")
        setPhase("ready")
        pending.current?.()
        pending.current = null
      } else if (m.type === "result") {
        const report: Report = JSON.parse(m.report)
        setResult((old) => {
          old?.panels.forEach((p) => URL.revokeObjectURL(p.url))
          return {
            output: url(m.image), input: url(m.input), report, ms: m.ms, netMs: m.netMs, backend: m.backend,
            panels: m.panels.map(([b, label]: [Uint8Array, string]) => ({ url: url(b), label })),
          }
        })
        setPhase("ready")
      } else if (m.type === "error") {
        setError(m.message)
        setPhase((p) => (p === "loading" ? "error" : "ready"))
      }
    }
    w.onerror = (e) => { setError(e.message || "The engine could not start in this browser."); setPhase("error") }
    w.postMessage({ type: "init", cfg: { base: location.origin + "/", wheel: WHEEL } })
    return () => w.terminate()
  }, [])

  const run = useCallback((bytes: Uint8Array, mode: Mode, preset: string, side: number) => {
    const go = () => {
      setError("")
      setPhase("running")
      setProgress({ pct: 5, label: "Enhancing" })
      worker.current?.postMessage({ type: "run", bytes: bytes.slice(), mode, preset, side })
    }
    if (phase === "loading") pending.current = go
    else go()
  }, [phase])

  return { phase, progress, backend, error, result, run }
}
