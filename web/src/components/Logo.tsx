type Props = {
  className?: string
  showWordmark?: boolean
}

export function Logo({ className = '', showWordmark = true }: Props) {
  return (
    <div className={`brand-lockup ${className}`.trim()} aria-label="ProEdge">
      <svg className="brand-mark" viewBox="0 0 32 32" fill="none" aria-hidden>
        <rect x="0.5" y="0.5" width="31" height="31" rx="5" fill="var(--bg-2)" stroke="var(--hair)" />
        <line
          x1="7"
          y1="22"
          x2="25"
          y2="22"
          stroke="var(--ink-faint)"
          strokeWidth="1.5"
          strokeLinecap="round"
        />
        <path
          d="M7 22 L13.5 14.5 L18.5 18 L25 10"
          stroke="var(--gold)"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
        <circle cx="25" cy="10" r="2" fill="var(--gold)" />
      </svg>
      {showWordmark && (
        <span className="brand-word">
          Pro<span className="brand-accent">Edge</span>
        </span>
      )}
    </div>
  )
}
