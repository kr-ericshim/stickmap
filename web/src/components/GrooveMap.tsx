import { useEffect, useRef, useState } from 'react'
import { FAMILY, type Family, type Inst, type Score } from '../api'
import type { Timeline } from '../timeline'

// Top to bottom, the way the kit sits on a staff.
const ROWS: { inst: Family; label: string; color: string }[] = [
  { inst: 'cymbal', label: '심벌', color: '#4f86a0' },
  { inst: 'hihat', label: '하이햇', color: '#c9982f' },
  { inst: 'tom', label: '탐', color: '#4e7c6b' },
  { inst: 'snare', label: '스네어', color: '#c2452d' },
  { inst: 'kick', label: '킥', color: '#14363f' },
]
export const FAMILY_COLOR = Object.fromEntries(ROWS.map((row) => [row.inst, row.color])) as Record<Family, string>
// Within a row, what was played has its own lane, top to bottom as on the staff.
// A plain hi-hat or cymbal, whose articulation is not known, fills the row.
const LANES: Partial<Record<Inst, number>> = {
  tom_high: 0, tom: 1, tom_low: 2,
  ride_bell: 1, ride: 2,
  hihat_open: 0, hihat_pedal: 2,
}
const ROW_HEIGHT = 11
const RULER = 16
const HEIGHT = RULER + ROWS.length * ROW_HEIGHT
const GUTTER = 52 // room for the row labels

type Props = {
  score: Score
  timeline: Timeline
  clock: () => number
  onSeek: (seconds: number) => void
}

/** The whole song at a glance: every hit as a mark, one row per drum. */
export function GrooveMap({ score, timeline, clock, onSeek }: Props) {
  const trackRef = useRef<HTMLDivElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const playheadRef = useRef<HTMLDivElement>(null)
  const [hover, setHover] = useState<{ x: number; bar: number } | null>(null)

  useEffect(() => {
    const track = trackRef.current!
    const canvas = canvasRef.current!
    const draw = () => {
      const width = track.clientWidth
      const ratio = window.devicePixelRatio || 1
      canvas.width = width * ratio
      canvas.height = HEIGHT * ratio
      const context = canvas.getContext('2d')!
      context.scale(ratio, ratio)
      context.clearRect(0, 0, width, HEIGHT)
      const end = timeline.end || 1
      const x = (seconds: number) => (seconds / end) * width

      context.font = "10px 'IBM Plex Mono', monospace"
      context.textBaseline = 'top'
      const labelEvery = Math.max(4, Math.ceil(score.bars.length / (width / 46) / 4) * 4)
      score.bars.forEach((bar, index) => {
        const left = x(bar.beat_times[0])
        const major = index % labelEvery === 0
        context.fillStyle = major ? 'rgba(21,33,31,0.22)' : 'rgba(21,33,31,0.06)'
        context.fillRect(left, major ? 0 : RULER, 1, major ? HEIGHT : HEIGHT - RULER)
        if (major) {
          context.fillStyle = '#5b6b68'
          context.fillText(String(index + 1), left + 3, 2)
        }
      })

      context.globalAlpha = 0.7
      ROWS.forEach((row, rowIndex) => {
        context.fillStyle = row.color
        const top = RULER + rowIndex * ROW_HEIGHT + 2
        for (const bar of score.bars) {
          for (const note of bar.notes) {
            const left = x(timeline.tickTime(bar, note.tick))
            if (FAMILY[note.inst] !== row.inst) continue
            const lane = LANES[note.inst]
            const height = ROW_HEIGHT - 4
            if (lane !== undefined) {
              context.fillRect(left, top + (lane * height) / 3, 1.5, height / 3)
            } else {
              // A ghost note or a side stick is a shorter mark.
              const inset = note.ghost || note.inst === 'sidestick' ? height / 4 : 0
              context.fillRect(left, top + inset, 1.5, height - 2 * inset)
            }
          }
        }
      })
      context.globalAlpha = 1
    }
    draw()
    const observer = new ResizeObserver(draw)
    observer.observe(track)
    // The ruler uses a web font; redraw once it is ready.
    document.fonts?.ready.then(draw)
    return () => observer.disconnect()
  }, [score, timeline])

  useEffect(() => {
    let frame = 0
    const move = () => {
      frame = requestAnimationFrame(move)
      const track = trackRef.current
      const playhead = playheadRef.current
      if (!track || !playhead) return
      const end = timeline.end || 1
      playhead.style.transform = `translateX(${(Math.min(clock(), end) / end) * track.clientWidth}px)`
    }
    frame = requestAnimationFrame(move)
    return () => cancelAnimationFrame(frame)
  }, [timeline, clock])

  const barUnder = (event: React.MouseEvent<HTMLDivElement>) => {
    const rect = event.currentTarget.getBoundingClientRect()
    const x = Math.max(0, Math.min(rect.width, event.clientX - rect.left))
    return { x, bar: timeline.barAt((x / rect.width) * timeline.end) }
  }

  return (
    <div className="flex select-none" style={{ height: HEIGHT }}>
      <div className="shrink-0 font-mono text-[10px] text-mute" style={{ width: GUTTER, paddingTop: RULER }}>
        {ROWS.map((row) => (
          <div key={row.inst} className="flex items-center gap-1.5" style={{ height: ROW_HEIGHT }}>
            <span className="size-1.5 rounded-full" style={{ background: row.color }} />
            <span className="font-sans">{row.label}</span>
          </div>
        ))}
      </div>
      <div
        ref={trackRef}
        className="relative min-w-0 flex-1 cursor-pointer"
        onMouseMove={(event) => setHover(barUnder(event))}
        onMouseLeave={() => setHover(null)}
        onClick={(event) => onSeek(timeline.barStart(barUnder(event).bar))}
      >
        <canvas ref={canvasRef} className="block h-full w-full" />
        <div
          ref={playheadRef}
          className="pointer-events-none absolute top-0 left-0 h-full w-0.5 bg-brass-deep"
        />
        {hover && (
          <div
            className="pointer-events-none absolute -top-5 -translate-x-1/2 rounded bg-ink px-1.5 py-0.5 font-mono text-[10px] text-paper"
            style={{ left: hover.x }}
          >
            {hover.bar + 1}마디
          </div>
        )}
      </div>
    </div>
  )
}
