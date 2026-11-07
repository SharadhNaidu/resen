// Material 3 Expressive shape library (flower, cookie, clover, sunny, pill-star), generated as
// smooth polar curves so they can be used as SVG fills and CSS clip-paths at any size.

export type ShapeName = "flower" | "cookie" | "clover" | "sunny" | "soft-burst" | "circle"

const SPEC: Record<ShapeName, { lobes: number; depth: number; sharp: number; rot: number }> = {
  flower: { lobes: 8, depth: 0.16, sharp: 0.55, rot: 0 },
  cookie: { lobes: 9, depth: 0.07, sharp: 1, rot: 0 },
  clover: { lobes: 4, depth: 0.22, sharp: 0.7, rot: Math.PI / 4 },
  sunny: { lobes: 12, depth: 0.06, sharp: 0.35, rot: 0 },
  "soft-burst": { lobes: 10, depth: 0.1, sharp: 0.5, rot: 0 },
  circle: { lobes: 1, depth: 0, sharp: 1, rot: 0 },
}

/** Path in a 0..100 box. `sharp` < 1 makes lobes rounder and valleys tighter. */
export function shapePath(name: ShapeName, steps = 240): string {
  const { lobes, depth, sharp, rot } = SPEC[name]
  const pts: string[] = []
  for (let i = 0; i < steps; i++) {
    const t = (i / steps) * Math.PI * 2
    const c = Math.cos(lobes * (t + rot))
    const wave = Math.sign(c) * Math.pow(Math.abs(c), sharp)
    const r = 50 * (1 - depth) + 50 * depth * wave
    pts.push(`${(50 + r * Math.cos(t)).toFixed(2)},${(50 + r * Math.sin(t)).toFixed(2)}`)
  }
  return `M${pts.join("L")}Z`
}

/** CSS clip-path polygon (percentages) for the same shape. */
export function shapeClip(name: ShapeName, steps = 120): string {
  const d = shapePath(name, steps).slice(1, -1).split("L")
  return `polygon(${d.map((p) => p.split(",").map((v) => `${v}%`).join(" ")).join(",")})`
}
