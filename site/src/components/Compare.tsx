import { useCallback, useRef, useState } from "react"
import { cn } from "@/lib/utils"

interface Props {
  before: string
  after: string
  beforeLabel?: string
  afterLabel?: string
  className?: string
}

/** Drag, tap or use the arrow keys to compare the original with the result. */
export function Compare({ before, after, beforeLabel = "Original", afterLabel = "resenh", className }: Props) {
  const ref = useRef<HTMLDivElement>(null)
  const [pos, setPos] = useState(50)
  const [drag, setDrag] = useState(false)
  const [ratio, setRatio] = useState<number | null>(null)

  const move = useCallback((clientX: number) => {
    const r = ref.current?.getBoundingClientRect()
    if (!r) return
    setPos(Math.min(100, Math.max(0, ((clientX - r.left) / r.width) * 100)))
  }, [])

  return (
    <div
      ref={ref}
      role="slider"
      tabIndex={0}
      aria-label="Compare original and enhanced photo"
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={Math.round(pos)}
      onPointerDown={(e) => { setDrag(true); e.currentTarget.setPointerCapture(e.pointerId); move(e.clientX) }}
      onPointerMove={(e) => drag && move(e.clientX)}
      onPointerUp={() => setDrag(false)}
      onKeyDown={(e) => {
        if (e.key === "ArrowLeft") setPos((p) => Math.max(0, p - 4))
        if (e.key === "ArrowRight") setPos((p) => Math.min(100, p + 4))
      }}
      style={ratio ? { aspectRatio: String(ratio) } : undefined}
      className={cn(
        "group relative w-full cursor-ew-resize touch-none select-none overflow-hidden bg-[#0d1020] outline-none",
        "focus-visible:ring-4 focus-visible:ring-periwinkle/60",
        className,
      )}
    >
      <img
        src={after}
        alt="Enhanced result"
        draggable={false}
        onLoad={(e) => setRatio(e.currentTarget.naturalWidth / e.currentTarget.naturalHeight)}
        className="block h-full w-full object-contain"
      />
      <img
        src={before}
        alt="Original photo"
        draggable={false}
        className="absolute inset-0 block h-full w-full object-contain"
        style={{ clipPath: `inset(0 ${100 - pos}% 0 0)` }}
      />
      <div className="pointer-events-none absolute inset-y-0" style={{ left: `${pos}%` }}>
        <div className="absolute inset-y-0 -left-px w-0.5 bg-white/90 shadow-[0_0_12px_rgba(0,0,0,.45)]" />
        <div
          className={cn(
            "absolute top-1/2 -left-5 grid size-10 -translate-y-1/2 place-items-center rounded-full bg-white text-[#1e2033]",
            "shadow-[0_2px_10px_rgba(10,14,40,.35)] transition-transform duration-200 ease-[cubic-bezier(.2,0,0,1)]",
            drag ? "scale-90" : "group-hover:scale-110",
          )}
        >
          <svg viewBox="0 0 24 24" className="size-5" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="m9 7-5 5 5 5M15 7l5 5-5 5" />
          </svg>
        </div>
      </div>
      <span className="pointer-events-none absolute top-3 left-3 rounded-full bg-black/45 px-3 py-1 text-xs font-medium text-white backdrop-blur-md">
        {beforeLabel}
      </span>
      <span className="pointer-events-none absolute top-3 right-3 rounded-full bg-white/85 px-3 py-1 text-xs font-medium text-[#1e2033] backdrop-blur-md">
        {afterLabel}
      </span>
    </div>
  )
}
