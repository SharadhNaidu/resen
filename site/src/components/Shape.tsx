import { useId } from "react"
import { shapePath, type ShapeName } from "@/lib/shapes"
import { cn } from "@/lib/utils"

const PALETTES = {
  dusk: ["#fdc58a", "#f7862a", "#a5bbfc"],
  mist: ["#e6ecff", "#a5bbfc", "#6a88e2"],
  night: ["#6a88e2", "#3a3f5c", "#1e2033"],
} as const

/** A Material 3 Expressive shape filled with a soft radial gradient (Sarvam-style bloom). */
export function Shape({ name, palette = "mist", className, spin = false }: {
  name: ShapeName
  palette?: keyof typeof PALETTES
  className?: string
  spin?: boolean
}) {
  const id = useId().replace(/:/g, "")
  const [a, b, c] = PALETTES[palette]
  return (
    <svg viewBox="0 0 100 100" aria-hidden className={cn(spin && "animate-[breathe_14s_ease-in-out_infinite]", className)}>
      <defs>
        <radialGradient id={`g${id}`} cx="35%" cy="30%" r="80%">
          <stop offset="0%" stopColor={a} />
          <stop offset="55%" stopColor={b} />
          <stop offset="100%" stopColor={c} />
        </radialGradient>
      </defs>
      <path d={shapePath(name)} fill={`url(#g${id})`} />
    </svg>
  )
}
