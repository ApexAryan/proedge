import { useEffect, useRef, useState } from 'react'
import type { SortKey } from '../lib/format'

const OPTIONS: { id: SortKey; label: string }[] = [
  { id: 'edge', label: 'Edge' },
  { id: 'confidence', label: 'Confidence' },
  { id: 'move', label: 'Line move' },
]

export function SortMenu({ value, onChange }: { value: SortKey; onChange: (s: SortKey) => void }) {
  const [open, setOpen] = useState(false)
  const root = useRef<HTMLDivElement>(null)
  const current = OPTIONS.find((o) => o.id === value)?.label ?? 'Edge'

  useEffect(() => {
    if (!open) return
    const onDoc = (e: MouseEvent) => {
      if (!root.current?.contains(e.target as Node)) setOpen(false)
    }
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setOpen(false)
    }
    document.addEventListener('mousedown', onDoc)
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('mousedown', onDoc)
      document.removeEventListener('keydown', onKey)
    }
  }, [open])

  return (
    <div className={`sort-menu ${open ? 'open' : ''}`} ref={root}>
      <button
        type="button"
        className="sort-trigger"
        aria-haspopup="listbox"
        aria-expanded={open}
        onClick={() => setOpen((v) => !v)}
      >
        <span className="sort-kicker">Sort</span>
        <span className="sort-value">{current}</span>
        <span className="sort-caret" aria-hidden>
          ▾
        </span>
      </button>
      {open && (
        <div className="sort-list" role="listbox">
          {OPTIONS.map((o) => (
            <button
              key={o.id}
              type="button"
              role="option"
              aria-selected={o.id === value}
              className={o.id === value ? 'on' : ''}
              onClick={() => {
                onChange(o.id)
                setOpen(false)
              }}
            >
              {o.label}
            </button>
          ))}
        </div>
      )}
    </div>
  )
}
