import { useEffect, useRef, useState } from "react"
import { BrainCircuit, Check, Copy, Download, ImagePlus, Layers, ShieldCheck, SlidersHorizontal, Sparkles, X } from "lucide-react"
import { Compare } from "@/components/Compare"
import { Segmented } from "@/components/Segmented"
import { Shape } from "@/components/Shape"
import { Button } from "@/components/ui/button"
import { Dialog, DialogContent, DialogTitle } from "@/components/ui/dialog"
import { Progress } from "@/components/ui/progress"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { useEngine, type Mode, type Report } from "@/lib/engine"
import { shapeClip, type ShapeName } from "@/lib/shapes"
import { cn } from "@/lib/utils"

const REPO = "https://github.com/SharadhNaidu/resen"
const SAMPLES: { id: string; shape: ShapeName; place: string }[] = [
  { id: "lime_1", shape: "flower", place: "Courtyard at dusk" },
  { id: "vv_5", shape: "cookie", place: "Backlit valley" },
  { id: "dicm_3", shape: "clover", place: "Indoor, no flash" },
  { id: "mef_2", shape: "sunny", place: "Mixed exposure" },
  { id: "npe_0", shape: "soft-burst", place: "Uneven lighting" },
  { id: "dicm_20", shape: "flower", place: "Night portrait" },
]
const MODES: { value: Mode; label: string; icon: typeof Sparkles; blurb: string }[] = [
  { value: "hybrid", label: "Hybrid", icon: Sparkles, blurb: "Classical and network, checked and fused" },
  { value: "dl", label: "Deep learning", icon: BrainCircuit, blurb: "Retinexformer alone, no safety checks" },
  { value: "classical", label: "Classical", icon: SlidersHorizontal, blurb: "Retinex, CLAHE and denoising, no network" },
]

export default function App() {
  const engine = useEngine()
  const [mode, setMode] = useState<Mode>("hybrid")
  const [side, setSide] = useState<"640" | "960">("640")
  const [preset, setPreset] = useState<"robust" | "benchmark">("robust")
  const [photo, setPhoto] = useState<{ bytes: Uint8Array; name: string; sample?: string } | null>(null)
  const fileRef = useRef<HTMLInputElement>(null)
  const studioRef = useRef<HTMLDivElement>(null)
  const [dragOver, setDragOver] = useState(false)

  const start = (bytes: Uint8Array, name: string, sample?: string, m: Mode = mode) => {
    setPhoto({ bytes, name, sample })
    engine.run(bytes, m, preset, Number(side))
  }
  const fromFile = async (f: File | Blob | null | undefined, name = "photo") => {
    if (!f) return
    start(new Uint8Array(await f.arrayBuffer()), (f as File).name || name)
  }
  const fromSample = async (id: string) => {
    const b = await (await fetch(`/examples/${id}.jpg`)).arrayBuffer()
    start(new Uint8Array(b), id, id)
  }
  const switchMode = (m: Mode) => {
    setMode(m)
    if (photo) engine.run(photo.bytes, m, preset, Number(side))
  }

  useEffect(() => {
    const onPaste = (e: ClipboardEvent) => {
      const item = [...(e.clipboardData?.items ?? [])].find((i) => i.type.startsWith("image/"))
      if (item) fromFile(item.getAsFile(), "pasted")
    }
    window.addEventListener("paste", onPaste)
    return () => window.removeEventListener("paste", onPaste)
  })

  const r = engine.result
  const busy = engine.phase === "running" || (engine.phase === "loading" && !!photo)

  return (
    <div className="relative min-h-dvh overflow-x-clip">
      <Atmosphere />
      <Nav />

      <header className="relative mx-auto max-w-5xl px-4 pt-14 pb-12 text-center sm:pt-20">
        <Ornament />
        <div className="mx-auto mt-6 inline-flex flex-col items-center">
          <span className="h-px w-56 bg-gradient-to-r from-transparent via-periwinkle-deep/40 to-transparent" />
          <span className="py-2 text-[15px] text-[#2f3f9e]">Low-light photo restoration, in your browser</span>
          <span className="h-px w-56 bg-gradient-to-r from-transparent via-periwinkle-deep/40 to-transparent" />
        </div>
        <h1 className="mx-auto mt-6 max-w-3xl text-[clamp(2.4rem,6.4vw,4.4rem)] leading-[1.02] font-[440] text-[#1e2033] animate-[rise_.9s_cubic-bezier(.2,0,0,1)_both]">
          Photos from the dark, brought back to life
        </h1>
        <p className="mx-auto mt-5 max-w-xl text-[17px] leading-relaxed text-[#4a5072]">
          resenh recovers detail, natural colour and clean shadows from underexposed photos, without blowing out lamps or
          amplifying noise. Your photo never leaves this device.
        </p>
        <div className="mt-8 flex flex-wrap items-center justify-center gap-3">
          <button
            onClick={() => { studioRef.current?.scrollIntoView({ block: "start" }); fileRef.current?.click() }}
            className="pill-primary state-layer inline-flex h-12 items-center gap-2 rounded-full px-6 text-[15px] font-medium outline-none focus-visible:ring-4 focus-visible:ring-periwinkle/60"
          >
            <ImagePlus className="size-4.5" /> Enhance a photo
          </button>
          <CopyPill text="pip install resenh" />
        </div>
      </header>

      {/* Studio */}
      <section ref={studioRef} id="studio" className="relative mx-auto max-w-6xl scroll-mt-20 px-3 sm:px-4">
        <div className="bezel rounded-[28px] p-1.5">
          <div role="tablist" aria-label="Method" className="grid grid-cols-3 gap-1 rounded-t-[22px] p-1">
            {MODES.map(({ value, label, icon: Icon }) => (
              <button
                key={value}
                role="tab"
                aria-selected={mode === value}
                onClick={() => switchMode(value)}
                className={cn(
                  "state-layer flex h-11 items-center justify-center gap-2 rounded-[16px] px-2 text-[13.5px] sm:text-[14px] font-medium transition-all duration-300 outline-none",
                  "focus-visible:ring-3 focus-visible:ring-periwinkle/60",
                  mode === value ? "bg-white text-[#2f3f9e] shadow-[0_1px_3px_rgba(30,32,51,.12)]" : "text-[#5d6380]",
                )}
              >
                <Icon className="hidden size-4 sm:block" />
                <span className="whitespace-nowrap">{label}</span>
              </button>
            ))}
          </div>

          <div className="relative overflow-hidden rounded-[22px] bg-white">
            <div className="pointer-events-none absolute inset-x-0 top-0 h-64 bg-[radial-gradient(60%_100%_at_50%_0%,#e6ecff_0%,transparent_70%)]" />
            <div className="relative grid grid-cols-[minmax(0,1fr)] gap-5 p-3 sm:p-5 lg:grid-cols-[minmax(0,1fr)_340px]">
              {/* Stage */}
              <div className="relative self-start overflow-hidden rounded-[18px] bg-[#0d1020]">
                <Compare
                  className="max-h-[78vh]"
                  before={r ? r.input : "/hero/lime_1_in.webp"}
                  after={r ? r.output : "/hero/lime_1_out.webp"}
                  afterLabel={r ? `resenh · ${labelOf(r.report.mode)}` : "resenh · Hybrid"}
                />
                {busy && <RunningVeil label={engine.progress.label} pct={engine.progress.pct} />}
              </div>

              {/* Dock */}
              <div className="flex flex-col gap-5">
                <label
                  onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
                  onDragLeave={() => setDragOver(false)}
                  onDrop={(e) => { e.preventDefault(); setDragOver(false); fromFile(e.dataTransfer.files[0]) }}
                  className={cn(
                    "state-layer group relative flex cursor-pointer flex-col items-center gap-2 overflow-hidden rounded-[22px] border border-dashed px-4 py-6 text-center transition-colors",
                    "focus-within:ring-3 focus-within:ring-periwinkle/60",
                    dragOver ? "border-periwinkle-deep bg-[#eef2ff]" : "border-[#1e2033]/18 bg-[#f7f8fd]",
                  )}
                >
                  <input ref={fileRef} type="file" accept="image/*" className="sr-only" onChange={(e) => fromFile(e.target.files?.[0])} />
                  <span className="relative grid size-14 place-items-center">
                    <Shape name="cookie" palette="mist" className="absolute inset-0 size-14 transition-transform duration-500 group-hover:rotate-45" />
                    <ImagePlus className="relative size-5 text-[#1e2033]" />
                  </span>
                  <span className="text-[15px] font-medium text-[#1e2033]">Choose a dark photo</span>
                  <span className="text-[13px] text-[#5d6380]">or drop it here, or paste it. JPEG, PNG or WebP.</span>
                </label>

                <div>
                  <p className="mb-2.5 text-[13px] font-medium text-[#3d4260]">Or try one of these</p>
                  <div className="grid grid-cols-6 gap-1.5 sm:gap-2 lg:grid-cols-3">
                    {SAMPLES.map((s) => (
                      <button
                        key={s.id}
                        title={s.place}
                        aria-label={`Sample: ${s.place}`}
                        onClick={() => fromSample(s.id)}
                        className="group relative aspect-square outline-none focus-visible:[&>span]:ring-3"
                      >
                        <span
                          className={cn(
                            "absolute inset-0 transition-transform duration-500 ease-[cubic-bezier(.2,0,0,1)] group-hover:scale-[1.06] group-hover:rotate-[10deg]",
                            photo?.sample === s.id && "scale-[1.04]",
                          )}
                          style={{ clipPath: shapeClip(s.shape) }}
                        >
                          <img src={`/hero/${s.id}_out.webp`} alt="" className="size-full object-cover" loading="lazy" />
                        </span>
                        {photo?.sample === s.id && (
                          <span className="absolute -right-0.5 -bottom-0.5 grid size-5 place-items-center rounded-full bg-[#1e2033] text-white ring-2 ring-white">
                            <Check className="size-3" />
                          </span>
                        )}
                      </button>
                    ))}
                  </div>
                </div>

                <div className="space-y-3">
                  <Field label="Detail">
                    <Segmented label="Working resolution" value={side} onChange={(v) => setSide(v)}
                      options={[{ value: "640", label: "Standard", hint: "640 px, fast on every device" },
                                { value: "960", label: "High", hint: "960 px, for laptops (about 2 GB of memory)" }]} />
                  </Field>
                  <Field label="Tuned for">
                    <Segmented label="Preset" value={preset} onChange={(v) => setPreset(v)}
                      options={[{ value: "robust", label: "Real photos", hint: "Removes sensor noise before brightening" },
                                { value: "benchmark", label: "Benchmarks", hint: "Exactly as evaluated on LOL" }]} />
                  </Field>
                </div>

                <div className="mt-auto space-y-3">
                  <button
                    disabled={!photo || busy}
                    onClick={() => photo && engine.run(photo.bytes, mode, preset, Number(side))}
                    className="pill-primary state-layer inline-flex h-12 w-full items-center justify-center gap-2 rounded-full text-[15px] font-medium outline-none focus-visible:ring-4 focus-visible:ring-periwinkle/60 disabled:opacity-45"
                  >
                    <Sparkles className="size-4" /> {photo ? "Enhance again" : "Pick a photo to start"}
                  </button>
                  <EngineStatus engine={engine} />
                </div>
              </div>
            </div>

            {r && <Insights r={r} />}
          </div>
        </div>
        <p className="mt-4 px-2 text-center text-[13px] text-balance text-[#5d6380]">
          <ShieldCheck className="mr-1.5 inline size-4 -translate-y-px align-middle text-periwinkle-deep" />
          Runs on your device with WebAssembly or WebGPU. Nothing is uploaded.
        </p>
      </section>

      <HowItWorks />
      <Results />
      <Python />
      <Footer />
    </div>
  )
}

const labelOf = (m: string) => MODES.find((x) => x.value === m)?.label ?? m

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="grid grid-cols-[76px_1fr] items-center gap-3">
      <span className="text-[13px] font-medium text-[#3d4260]">{label}</span>
      {children}
    </div>
  )
}

function RunningVeil({ label, pct }: { label: string; pct: number }) {
  return (
    <div className="absolute inset-0 grid place-items-center bg-[#0d1020]/55 backdrop-blur-[3px]">
      <div className="flex flex-col items-center gap-4 text-white">
        <Shape name="flower" palette="dusk" spin className="size-16" />
        <div className="text-[15px] font-medium">{label}…</div>
        <div className="h-1 w-48 overflow-hidden rounded-full bg-white/20">
          <div className="h-full rounded-full bg-gradient-to-r from-periwinkle to-sunrise-soft transition-[width] duration-500" style={{ width: `${pct}%` }} />
        </div>
      </div>
    </div>
  )
}

function EngineStatus({ engine }: { engine: ReturnType<typeof useEngine> }) {
  const { phase, progress, backend, error, result } = engine
  if (error)
    return <p role="alert" className="rounded-2xl bg-[#fdecea] px-4 py-3 text-[13px] text-[#8c1d13]">{error}</p>
  if (phase === "loading")
    return (
      <div className="space-y-2" aria-live="polite">
        <div className="flex justify-between text-[13px] text-[#5d6380]"><span>{progress.label}…</span><span className="tabular">{progress.pct}%</span></div>
        <Progress value={progress.pct} className="h-1.5" />
        <p className="text-[12px] text-[#7a8099]">First visit downloads about 40 MB, then it's cached.</p>
      </div>
    )
  return (
    <div className="flex items-center justify-between text-[13px] text-[#5d6380]" aria-live="polite">
      <span className="inline-flex items-center gap-2">
        <span className="size-2 rounded-full bg-emerald-500 shadow-[0_0_0_3px_rgba(16,185,129,.18)]" />
        Engine ready on {backend}
      </span>
      {result && <span className="tabular">{(result.ms / 1000).toFixed(2)} s</span>}
    </div>
  )
}

function Insights({ r }: { r: NonNullable<ReturnType<typeof useEngine>["result"]> }) {
  const [zoom, setZoom] = useState<null | { url: string; label: string }>(null)
  const rep = r.report
  return (
    <div className="relative border-t border-[#1e2033]/8 bg-[#fbfcff] p-3 sm:p-5">
      <Tabs defaultValue="analysis">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <TabsList className="h-11 w-full rounded-full bg-[#eef1fb] p-1 sm:w-auto">
            <TabsTrigger value="analysis" className="flex-1 rounded-full px-2 text-[13px] sm:px-4 sm:text-sm">What it measured</TabsTrigger>
            <TabsTrigger value="stages" className="flex-1 rounded-full px-2 text-[13px] sm:px-4 sm:text-sm">Every stage</TabsTrigger>
            <TabsTrigger value="controller" className="flex-1 rounded-full px-2 text-[13px] sm:px-4 sm:text-sm">Quality check</TabsTrigger>
          </TabsList>
          <a
            href={r.output}
            download="resenh.png"
            className="state-layer inline-flex h-10 w-full items-center justify-center gap-2 rounded-full border sm:w-auto border-[#1e2033]/15 bg-white px-4 text-[14px] font-medium text-[#1e2033] outline-none focus-visible:ring-3 focus-visible:ring-periwinkle/60"
          >
            <Download className="size-4" /> Download PNG
          </a>
        </div>

        <TabsContent value="analysis" className="mt-5">
          <div className="grid grid-cols-2 gap-2 sm:gap-3 lg:grid-cols-4">
            <Metric label="Processing time" value={`${(r.ms / 1000).toFixed(2)} s`} note={r.netMs ? `network ${(r.netMs / 1000).toFixed(2)} s on ${r.backend === "webgpu" ? "WebGPU" : "WebAssembly"}` : "no network needed"} />
            <Metric label="Edges preserved" value={rep.quality.epi.toFixed(3)} note="1.000 keeps every edge of the input" />
            <Metric label="Noise vs. simple brightening" value={`×${rep.quality.noise_ratio.toFixed(2)}`} note="below 1 means cleaner than raising exposure" />
            <Metric label="Output size" value={`${rep.width} × ${rep.height}`} note="the Python package runs at full resolution" />
          </div>
          <div className="mt-3 grid gap-3 lg:grid-cols-2">
            <KV title="The photo" rows={[
              ["Mean brightness", rep.analyzer.brightness.toFixed(3)],
              ["Pixels in deep shadow", `${(rep.analyzer.dark_frac * 100).toFixed(0)}%`],
              ["Near-white highlights", `${(rep.analyzer.highlight_frac * 100).toFixed(1)}%`],
              ["Sensor noise σ", rep.analyzer.noise_sigma.toFixed(4)],
              ["Colour cast", rep.analyzer.cast.toFixed(2)],
              ["Uneven lighting", rep.analyzer.nonuniformity.toFixed(2)],
            ]} />
            <KV title="Settings it chose" rows={[
              ["Treated as low-light", rep.params.low_light ? "Yes" : "No, brightening capped"],
              ["Retinex gamma", rep.params.retinex_gamma.toFixed(2)],
              ["Denoising strength", rep.params.denoise.toFixed(2)],
              ["Local contrast (CLAHE)", rep.params.clahe_clip.toFixed(1)],
              ["White balance", rep.params.wb.toFixed(2)],
              ["Highlight protection", rep.params.highlight_protect.toFixed(2)],
            ]} />
          </div>
        </TabsContent>

        <TabsContent value="stages" className="mt-5">
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4">
            {r.panels.map((p) => (
              <button key={p.label} onClick={() => setZoom(p)} className="state-layer group overflow-hidden rounded-[18px] border border-[#1e2033]/8 bg-white text-left outline-none focus-visible:ring-3 focus-visible:ring-periwinkle/60">
                <img src={p.url} alt={p.label} className="aspect-[4/3] w-full object-cover transition-transform duration-500 group-hover:scale-[1.03]" />
                <span className="block px-3 py-2.5 text-[12.5px] leading-snug text-[#3d4260]">{p.label.replace(/^\d+\.\s*/, "")}</span>
              </button>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="controller" className="mt-5">
          <Controller rep={rep} />
        </TabsContent>
      </Tabs>

      <Dialog open={!!zoom} onOpenChange={(o) => !o && setZoom(null)}>
        <DialogContent className="max-w-[min(96vw,1100px)] rounded-[24px] p-3 sm:max-w-[min(96vw,1100px)]" showCloseButton={false}>
          <DialogTitle className="flex items-center justify-between px-2 pt-1 text-[15px] font-medium">
            {zoom?.label.replace(/^\d+\.\s*/, "")}
            <button onClick={() => setZoom(null)} aria-label="Close" className="state-layer grid size-9 place-items-center rounded-full"><X className="size-4" /></button>
          </DialogTitle>
          {zoom && <img src={zoom.url} alt={zoom.label} className="max-h-[78vh] w-full rounded-[16px] object-contain" />}
        </DialogContent>
      </Dialog>
    </div>
  )
}

function Metric({ label, value, note }: { label: string; value: string; note: string }) {
  return (
    <div className="min-w-0 rounded-[20px] bg-white p-3.5 shadow-[0_0_0_1px_rgba(30,32,51,.07)] sm:p-4">
      <div className="text-[12.5px] leading-snug text-[#5d6380] sm:text-[13px]">{label}</div>
      <div className="tabular mt-1 text-[21px] sm:text-[26px] font-[450] tracking-tight text-[#1e2033]">{value}</div>
      <div className="mt-1 text-[12px] leading-snug text-[#7a8099]">{note}</div>
    </div>
  )
}

function KV({ title, rows }: { title: string; rows: [string, string][] }) {
  return (
    <div className="rounded-[20px] bg-white p-4 shadow-[0_0_0_1px_rgba(30,32,51,.07)]">
      <h3 className="mb-2 text-[15px] font-medium text-[#1e2033]">{title}</h3>
      <dl className="divide-y divide-[#1e2033]/6">
        {rows.map(([k, v]) => (
          <div key={k} className="flex justify-between py-2 text-[14px]">
            <dt className="text-[#5d6380]">{k}</dt>
            <dd className="tabular font-medium text-[#1e2033]">{v}</dd>
          </div>
        ))}
      </dl>
    </div>
  )
}

function Controller({ rep }: { rep: Report }) {
  const d = rep.decision
  if (!d)
    return (
      <p className="rounded-[20px] bg-white p-5 text-[14px] text-[#4a5072] shadow-[0_0_0_1px_rgba(30,32,51,.07)]">
        The quality check runs in <b>Hybrid</b> mode, where it compares the classical and network results before
        fusing them. Switch to Hybrid above to see it.
      </p>
    )
  const names: Record<string, string> = {
    clipped: "Newly blown highlights", chroma_shift: "Colour shift", edge_keep: "Edges kept",
    noise_amp: "Noise amplification", brightness: "Brightness",
  }
  const net = d.weights.network ?? 0
  return (
    <div className="grid grid-cols-[minmax(0,1fr)] gap-3 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.2fr)]">
      <div className="rounded-[20px] bg-white p-5 shadow-[0_0_0_1px_rgba(30,32,51,.07)]">
        <h3 className="text-[15px] font-medium">How the result was blended</h3>
        <div className="mt-4 flex h-11 overflow-hidden rounded-full bg-[#eef1fb] text-[13px] font-medium">
          <div className="flex items-center justify-center bg-gradient-to-r from-[#6a88e2] to-[#8ea6f0] text-white transition-[width] duration-700" style={{ width: `${net * 100}%` }}>
            {net > 0.15 && `Network ${(net * 100).toFixed(0)}%`}
          </div>
          <div className="flex flex-1 items-center justify-center bg-gradient-to-r from-[#fdc58a] to-[#f7a65a] text-[#4a2a07]">
            {1 - net > 0.15 && `Classical ${((1 - net) * 100).toFixed(0)}%`}
          </div>
        </div>
        <p className="mt-4 text-[13.5px] leading-relaxed text-[#4a5072]">
          Each candidate is scored for problems a viewer would notice. The one with fewer problems gets more weight.
        </p>
        {d.notes.map((n) => (
          <p key={n} className="mt-2 rounded-xl bg-[#fff4e6] px-3 py-2 text-[13px] text-[#7a3e00]">{n}</p>
        ))}
      </div>
      <div className="rounded-[20px] bg-white p-5 shadow-[0_0_0_1px_rgba(30,32,51,.07)]">
        <table className="w-full text-[14px]">
          <thead>
            <tr className="text-[12.5px] text-[#7a8099]">
              <th className="pb-2 text-left font-normal">Check</th>
              <th className="pb-2 text-right font-normal">Network</th>
              <th className="pb-2 text-right font-normal">Classical</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-[#1e2033]/6">
            {Object.keys(names).map((k) => (
              <tr key={k}>
                <td className="py-2 text-[#5d6380]">{names[k]}</td>
                <td className="tabular py-2 text-right font-medium">{d.guards.network?.[k]?.toFixed(3)}</td>
                <td className="tabular py-2 text-right font-medium">{d.guards.classical?.[k]?.toFixed(3)}</td>
              </tr>
            ))}
            <tr>
              <td className="py-2 font-medium">Total penalty</td>
              <td className="tabular py-2 text-right font-medium">{d.penalties.network?.toFixed(2)}</td>
              <td className="tabular py-2 text-right font-medium">{d.penalties.classical?.toFixed(2)}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  )
}

/* ---------------------------------------------------------------- page sections */

function Atmosphere() {
  return (
    <div aria-hidden className="pointer-events-none absolute inset-x-0 top-0 -z-0 h-[980px]">
      <div className="absolute inset-0 bg-[radial-gradient(42%_34%_at_50%_0%,#f7862a_0%,rgba(253,197,138,.75)_38%,transparent_75%)] opacity-90" />
      <div className="absolute inset-0 bg-[radial-gradient(38%_45%_at_8%_22%,#a5bbfc_0%,transparent_70%),radial-gradient(38%_45%_at_92%_22%,#a5bbfc_0%,transparent_70%)] opacity-90" />
      <div className="absolute inset-0 bg-[radial-gradient(70%_55%_at_50%_10%,rgba(213,226,255,.0)_0%,#fafbff_78%)]" />
      <div className="absolute inset-x-0 bottom-0 h-64 bg-gradient-to-b from-transparent to-[#fafbff]" />
    </div>
  )
}

function Nav() {
  return (
    <nav className="sticky top-0 z-30 border-b border-transparent bg-[#fafbff]/80 backdrop-blur-xl">
      <div className="mx-auto flex h-16 max-w-6xl items-center justify-between px-4">
        <a href="/" className="flex items-center gap-2 text-[21px] font-[560] tracking-[-0.03em] text-[#1e2033]">
          <Shape name="flower" palette="dusk" className="size-7" />
          resenh
        </a>
        <div className="hidden items-center gap-7 text-[15px] text-[#2b2f45] md:flex">
          <a href="#studio" className="hover:text-black">Try it</a>
          <a href="#how" className="hover:text-black">How it works</a>
          <a href="#results" className="hover:text-black">Results</a>
          <a href="#python" className="hover:text-black">Python</a>
        </div>
        <div className="flex items-center gap-2">
          <a href={REPO} className="pill-primary state-layer inline-flex h-10 items-center gap-2 rounded-full px-4 text-[14px] font-medium">
            <GithubMark className="size-4" /> GitHub
          </a>
        </div>
      </div>
    </nav>
  )
}

function Ornament() {
  // horizon with a rising sun and scrolling light rays: night turning into day
  return (
    <svg viewBox="0 0 240 64" className="mx-auto h-14 w-auto drop-shadow-[0_2px_10px_rgba(255,255,255,.6)]" aria-hidden>
      <g fill="none" stroke="#fff" strokeLinecap="round">
        <path d="M18 52h204" strokeWidth="2.2" />
        <path d="M84 52a36 36 0 0 1 72 0" strokeWidth="2.6" />
        <path d="M100 52a20 20 0 0 1 40 0" strokeWidth="2.2" opacity=".85" />
        {[-62, -38, -14, 14, 38, 62].map((a) => (
          <path key={a} d={`M120 52 m${Math.sin((a * Math.PI) / 180) * 44} ${-Math.cos((a * Math.PI) / 180) * 44} l${Math.sin((a * Math.PI) / 180) * 9} ${-Math.cos((a * Math.PI) / 180) * 9}`} strokeWidth="2.2" />
        ))}
        <path d="M30 44c10-8 22-8 30 0M180 44c8-8 20-8 30 0" strokeWidth="2" opacity=".9" />
      </g>
    </svg>
  )
}

function CopyPill({ text }: { text: string }) {
  const [done, setDone] = useState(false)
  return (
    <button
      onClick={() => { navigator.clipboard?.writeText(text); setDone(true); setTimeout(() => setDone(false), 1600) }}
      className="state-layer inline-flex h-12 items-center gap-2.5 rounded-full border border-[#1e2033]/15 bg-white/80 px-5 text-[15px] text-[#1e2033] backdrop-blur outline-none focus-visible:ring-4 focus-visible:ring-periwinkle/60"
    >
      <code className="font-mono text-[14px]">{text}</code>
      {done ? <Check className="size-4 text-emerald-600" /> : <Copy className="size-4 text-[#5d6380]" />}
    </button>
  )
}

const STEPS: { title: string; body: string; shape: ShapeName; palette: "dusk" | "mist" | "night" }[] = [
  { title: "Measure the photo", body: "Brightness, shadow share, sensor noise, colour cast, uneven light and highlights are measured, and every later setting is chosen from them.", shape: "cookie", palette: "mist" },
  { title: "Classical restoration", body: "An illumination map from max-RGB and a guided filter drives Retinex brightening, then noise-aware denoising, CLAHE and white balance.", shape: "clover", palette: "dusk" },
  { title: "Learned restoration", body: "Retinexformer (ICCV 2023), fine-tuned with the classical illumination and outputs as extra inputs, restores colour and texture.", shape: "flower", palette: "mist" },
  { title: "Check, then fuse", body: "Both results are scored for blown highlights, colour shift, lost edges and amplified noise. The safer one gets the weight.", shape: "sunny", palette: "dusk" },
]

function HowItWorks() {
  return (
    <section id="how" className="mx-auto max-w-6xl scroll-mt-20 px-4 pt-28">
      <div className="grid gap-6 md:grid-cols-[1fr_auto] md:items-end">
        <div>
          <h2 className="text-[clamp(2rem,4vw,2.9rem)] leading-[1.08] font-[440] text-[#1e2033]">Two ways to see in the dark, and a judge between them</h2>
          <p className="mt-4 max-w-2xl text-[17px] leading-relaxed text-[#4a5072]">
            Classical methods are predictable but flat. Networks are vivid but sometimes invent colour or blow out lights.
            resenh runs both and keeps what each does well.
          </p>
        </div>
        <a href={`${REPO}/blob/main/docs/design.md`} className="state-layer inline-flex h-11 items-center rounded-full border border-[#1e2033]/15 bg-white px-5 text-[15px] font-medium text-[#1e2033]">
          Read the design
        </a>
      </div>
      <ol className="mt-10 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        {STEPS.map((s, i) => (
          <li key={s.title} className="group relative overflow-hidden rounded-[24px] bg-white p-6 shadow-[0_0_0_1px_rgba(30,32,51,.07)] transition-shadow duration-300 hover:shadow-[0_0_0_1px_rgba(106,136,226,.35),0_18px_40px_-20px_rgba(54,72,160,.35)]">
            <Shape name={s.shape} palette={s.palette} className="absolute -top-8 -right-8 size-24 opacity-90 sm:-top-10 sm:-right-10 sm:size-32 transition-transform duration-700 ease-[cubic-bezier(.2,0,0,1)] group-hover:rotate-45" />
            <span className="tabular relative text-[13px] text-[#7a8099]">Step {i + 1}</span>
            <h3 className="relative mt-9 text-[19px] sm:mt-16 font-[520] text-[#1e2033]">{s.title}</h3>
            <p className="relative mt-2 text-[14.5px] leading-relaxed text-[#4a5072]">{s.body}</p>
          </li>
        ))}
      </ol>
    </section>
  )
}

const TABLE = [
  { name: "Classical branch alone", psnr: "16.83", ssim: "0.758", lpips: "0.306", ours: false },
  { name: "Retinexformer, official weights", psnr: "25.67", ssim: "0.930", lpips: "0.059", ours: false },
  { name: "resenh network (fine-tuned with priors)", psnr: "25.69", ssim: "0.936", lpips: "0.050", ours: true },
]

function Results() {
  return (
    <section id="results" className="mx-auto max-w-6xl scroll-mt-20 px-4 pt-28">
      <div className="relative overflow-hidden rounded-[32px] bg-gradient-to-b from-[#232741] to-[#151829] p-5 text-white sm:p-10">
        <Shape name="flower" palette="dusk" spin className="pointer-events-none absolute -top-28 -right-28 size-60 opacity-50 blur-[1px] sm:-top-24 sm:-right-20 sm:size-80 sm:opacity-80" />
        <div className="pointer-events-none absolute -bottom-40 -left-24 size-96 rounded-full bg-[radial-gradient(circle,#6a88e2_0%,transparent_65%)] opacity-50" />
        <div className="relative max-w-2xl">
          <h2 className="text-[clamp(2rem,4vw,2.9rem)] leading-[1.08] font-[440]">Measured the strict way</h2>
          <p className="mt-4 text-[16.5px] leading-relaxed text-white/75">
            Plain PSNR on the saved image: no ground-truth brightness correction, no per-image tuning, no checkpoint picked
            on the test set. Our harness reproduces the published Retinexformer numbers exactly before comparing anything.
          </p>
        </div>
        <div className="relative mt-8 overflow-x-auto rounded-[18px] bg-white/[.06] sm:rounded-[22px] ring-1 ring-white/10 backdrop-blur">
          <table className="w-full text-left text-[14px] sm:text-[15px]">
            <thead>
              <tr className="text-[13px] text-white/60">
                <th className="py-4 pr-2 pl-4 font-normal sm:px-5">LOL-v2 Synthetic, 100 test pairs</th>
                <th className="px-2 py-4 text-right font-normal whitespace-nowrap sm:px-5"><span className="sm:hidden">PSNR ↑</span><span className="hidden sm:inline">PSNR (dB), higher is better</span></th>
                <th className="px-2 py-4 text-right font-normal whitespace-nowrap sm:px-5"><span className="sm:hidden">SSIM ↑</span><span className="hidden sm:inline">SSIM, higher is better</span></th>
                <th className="py-4 pr-4 pl-2 text-right font-normal whitespace-nowrap sm:px-5"><span className="sm:hidden">LPIPS ↓</span><span className="hidden sm:inline">LPIPS, lower is better</span></th>
              </tr>
            </thead>
            <tbody>
              {TABLE.map((t) => (
                <tr key={t.name} className={cn("border-t border-white/10", t.ours && "bg-white/[.07]")}>
                  <td className="py-4 pr-2 pl-4 sm:px-5">
                    <span className="inline-flex items-center gap-2">
                      {t.ours && <Shape name="cookie" palette="dusk" className="size-4" />}
                      {t.name}
                    </span>
                  </td>
                  <td className="tabular px-2 py-4 text-right font-medium sm:px-5">{t.psnr}</td>
                  <td className="tabular px-2 py-4 text-right font-medium sm:px-5">{t.ssim}</td>
                  <td className="tabular py-4 pr-4 pl-2 text-right font-medium sm:px-5">{t.lpips}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="relative mt-6 flex flex-wrap gap-3">
          <a href={`${REPO}/blob/main/eval_P1_T46.md`} className="inline-flex h-11 items-center rounded-full bg-white px-5 text-[15px] font-medium text-[#1e2033]">Full results on every benchmark</a>
          <a href={`${REPO}/blob/main/docs/failures.md`} className="inline-flex h-11 items-center rounded-full border border-white/25 px-5 text-[15px] font-medium text-white">Where it still fails</a>
        </div>
      </div>
    </section>
  )
}

const CODE = {
  python: `import resenh

# a path, URL, PIL image or NumPy array
res = resenh.enhance("night.jpg")
res.save("night_restored.png")

res            # notebook: before / after
res.summary()  # what it measured and chose
res.stages     # every stage, labelled`,
  cli: `pip install "resenh[dl]"

resenh enhance night.jpg -o restored.png
resenh enhance photos/ -o restored/
resenh --env   # versions, device, weights`,
}

function Python() {
  const [copied, setCopied] = useState("")
  return (
    <section id="python" className="mx-auto max-w-6xl scroll-mt-20 px-4 pt-28">
      <div className="text-center">
        <h2 className="text-[clamp(2rem,4vw,2.9rem)] leading-[1.08] font-[440] text-[#1e2033]">The same engine, as a Python package</h2>
        <p className="mx-auto mt-4 max-w-xl text-[17px] text-[#4a5072]">Full resolution, GPU when you have one, and every stage inspectable.</p>
      </div>
      <div className="mt-10 grid grid-cols-[minmax(0,1fr)] items-start gap-3 lg:grid-cols-[minmax(0,1.25fr)_minmax(0,1fr)]">
        <div className="min-w-0 rounded-[28px] bg-gradient-to-b from-white to-[#eef2ff] p-3 shadow-[0_0_0_1px_rgba(30,32,51,.07)] sm:p-6">
          <Tabs defaultValue="python">
            <div className="flex items-center justify-between">
              <TabsList className="h-10 rounded-full bg-[#eef1fb] p-1">
                <TabsTrigger value="python" className="rounded-full px-4">Python</TabsTrigger>
                <TabsTrigger value="cli" className="rounded-full px-4">Command line</TabsTrigger>
              </TabsList>
            </div>
            {(["python", "cli"] as const).map((k) => (
              <TabsContent key={k} value={k} className="relative mt-4">
                <pre className="overflow-x-auto rounded-[20px] bg-white p-4 pt-12 text-[12px] sm:p-5 sm:pt-5 sm:text-[13.5px] leading-[1.75] text-[#26293d] shadow-[0_0_0_1px_rgba(30,32,51,.06)]"><code>{CODE[k]}</code></pre>
                <Button
                  variant="ghost"
                  size="sm"
                  className="absolute top-3 right-3 rounded-full"
                  onClick={() => { navigator.clipboard?.writeText(CODE[k]); setCopied(k); setTimeout(() => setCopied(""), 1500) }}
                >
                  {copied === k ? <Check className="size-3.5" /> : <Copy className="size-3.5" />} {copied === k ? "Copied" : "Copy"}
                </Button>
              </TabsContent>
            ))}
          </Tabs>
        </div>
        <div className="grid grid-cols-[minmax(0,1fr)] gap-3 sm:grid-cols-2 lg:grid-cols-1">
          <Card title="Notebook ready" body="Open the quickstart in Colab or Kaggle and run it top to bottom, GPU or not." href={`https://colab.research.google.com/github/SharadhNaidu/resen/blob/main/notebooks/quickstart.ipynb`} cta="Open in Colab" shape="clover" />
          <Card title="Works without PyTorch" body="pip install resenh alone gives the classical pipeline. Add [dl] for the network." href="https://pypi.org/project/resenh/" cta="View on PyPI" shape="cookie" />
          <Card title="Every stage, inspectable" body="Illumination, reflectance, gain, denoising and the controller's decision come back with the image." href={`${REPO}#use`} cta="Read the API" shape="flower" icon />
        </div>
      </div>
    </section>
  )
}

function Card({ title, body, href, cta, shape, icon }: { title: string; body: string; href: string; cta: string; shape: ShapeName; icon?: boolean }) {
  return (
    <a href={href} className="group relative overflow-hidden rounded-[24px] bg-white p-5 shadow-[0_0_0_1px_rgba(30,32,51,.07)] transition-shadow hover:shadow-[0_0_0_1px_rgba(106,136,226,.35),0_14px_34px_-18px_rgba(54,72,160,.35)]">
      <Shape name={shape} palette="mist" className="absolute -top-8 -right-8 size-24 transition-transform duration-700 group-hover:rotate-45" />
      {icon && <Layers className="relative size-5 text-[#2f3f9e]" />}
      <h3 className="relative mt-6 text-[17px] font-[520] text-[#1e2033]">{title}</h3>
      <p className="relative mt-1.5 text-[14px] leading-relaxed text-[#4a5072]">{body}</p>
      <span className="relative mt-3 inline-block text-[14px] font-medium text-[#2f3f9e] underline-offset-4 group-hover:underline">{cta}</span>
    </a>
  )
}

function Footer() {
  return (
    <footer className="mx-auto mt-28 max-w-6xl px-4 pb-14">
      <div className="flex flex-col gap-4 border-t border-[#1e2033]/10 pt-8 text-[13.5px] text-[#5d6380] sm:flex-row sm:items-center sm:justify-between">
        <span className="flex items-center gap-2 text-[#1e2033]"><Shape name="flower" palette="dusk" className="size-5" /> resenh, by Sharadh Naidu. MIT licence.</span>
        <span className="max-w-xl">
          Built on Retinexformer (Cai et al., ICCV 2023), the GT-mean loss (ICCV 2025), LIME and the guided filter. Enhanced
          photos can contain invented detail, so do not use them as evidence.
        </span>
      </div>
    </footer>
  )
}

function GithubMark({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" fill="currentColor" aria-hidden className={className}>
      <path d="M12 .5a11.5 11.5 0 0 0-3.64 22.41c.58.1.79-.25.79-.56v-2c-3.2.7-3.88-1.37-3.88-1.37-.53-1.33-1.28-1.69-1.28-1.69-1.05-.71.08-.7.08-.7 1.16.08 1.77 1.19 1.77 1.19 1.03 1.77 2.7 1.26 3.36.96.1-.75.4-1.26.73-1.55-2.56-.29-5.25-1.28-5.25-5.69 0-1.26.45-2.29 1.19-3.1-.12-.29-.52-1.46.11-3.05 0 0 .97-.31 3.17 1.18a11 11 0 0 1 5.77 0c2.2-1.49 3.17-1.18 3.17-1.18.63 1.59.23 2.76.11 3.05.74.81 1.19 1.84 1.19 3.1 0 4.42-2.7 5.39-5.27 5.68.41.36.78 1.06.78 2.14v3.17c0 .31.21.67.8.56A11.5 11.5 0 0 0 12 .5Z" />
    </svg>
  )
}
