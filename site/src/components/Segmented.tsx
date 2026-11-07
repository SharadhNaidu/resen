import { Check } from "lucide-react"
import { cn } from "@/lib/utils"

interface Option<T extends string> { value: T; label: string; hint?: string }

/** Material 3 segmented button: outlined pill group, the selected segment is tonal with a check. */
export function Segmented<T extends string>({ value, onChange, options, label }: {
  value: T
  onChange: (v: T) => void
  options: Option<T>[]
  label: string
}) {
  return (
    <div role="radiogroup" aria-label={label} className="flex w-full overflow-hidden rounded-full border border-[#1e2033]/15 bg-white">
      {options.map((o, i) => {
        const on = o.value === value
        return (
          <button
            key={o.value}
            type="button"
            role="radio"
            aria-checked={on}
            title={o.hint}
            onClick={() => onChange(o.value)}
            className={cn(
              "state-layer flex h-10 flex-1 items-center justify-center gap-1.5 px-2 whitespace-nowrap text-[13px] font-medium transition-colors",
              "outline-none focus-visible:z-10 focus-visible:ring-3 focus-visible:ring-periwinkle/60",
              i > 0 && "border-l border-[#1e2033]/15",
              on ? "bg-[#dfe6fd] text-[#1d2a6b]" : "text-[#3d4260]",
            )}
          >
            <Check className={cn("size-3.5 transition-all duration-200", on ? "w-3.5 opacity-100" : "w-0 opacity-0")} />
            {o.label}
          </button>
        )
      })}
    </div>
  )
}
