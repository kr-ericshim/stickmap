import { OpenSheetMusicDisplay, type VexFlowGraphicalNote } from 'opensheetmusicdisplay'
import { useCallback, useEffect, useRef, useState } from 'react'
import type { Timeline } from '../timeline'

type Box = { x: number; y: number; width: number; height: number }

type Engraving = {
  boxes: Box[] // one per bar, in pixels
  // Per bar: where each note or rest sits, as whole notes from the bar's start and pixels across.
  anchors: { whole: number; x: number }[][]
}

type Props = {
  xml: string
  timeline: Timeline
  clock: () => number
  playing: boolean
  onSeek: (seconds: number) => void
}

const UNIT = 10 // OSMD layout units to pixels at zoom 1
// OSMD does not draw parenthesized note heads. Ghost notes get a colour in the
// XML it is given, which it keeps on the note, and the parentheses are drawn here.
const GHOST_HEAD = '<notehead parentheses="yes"'
const GHOST_TAG = '#000000'
const SVG = 'http://www.w3.org/2000/svg'

function parenthesize(head: SVGGraphicsElement) {
  const box = head.getBBox()
  const middle = box.y + box.height / 2
  const reach = box.height * 0.75
  const arc = (x: number, bulge: number) =>
    `M${x} ${middle - reach}Q${x + bulge} ${middle} ${x} ${middle + reach}`
  const path = document.createElementNS(SVG, 'path')
  path.setAttribute('d', arc(box.x - 2, -4) + arc(box.x + box.width + 2, 4))
  path.setAttribute('fill', 'none')
  path.setAttribute('stroke', 'currentColor')
  path.setAttribute('stroke-width', '1.2')
  path.setAttribute('stroke-linecap', 'round')
  head.parentNode?.appendChild(path)
}
const STAFF_PAD = 2.5 // units above and below the staff that count as the bar

const ZOOM_MIN = 0.5
const ZOOM_MAX = 2
const ZOOM_STEP = 0.1
const ZOOM_KEY = 'groovemap.scoreZoom'

function clampZoom(zoom: number) {
  return Math.min(ZOOM_MAX, Math.max(ZOOM_MIN, Math.round(zoom * 100) / 100))
}

function savedZoom() {
  try {
    const value = Number(window.localStorage.getItem(ZOOM_KEY))
    return Number.isFinite(value) && value > 0 ? clampZoom(value) : 1
  } catch {
    return 1
  }
}

/** Engraved score with a playhead that follows the recording. */
export function ScoreView({ xml, timeline, clock, playing, onSeek }: Props) {
  const hostRef = useRef<HTMLDivElement>(null)
  const playheadRef = useRef<HTMLDivElement>(null)
  const engraving = useRef<Engraving | null>(null)
  const [failure, setFailure] = useState<string | null>(null)
  const [zoom, setZoom] = useState(savedZoom)
  const zoomRef = useRef(zoom)
  // The live engraver, so a zoom change can re-render without reloading the score.
  const rezoom = useRef<((zoom: number) => void) | null>(null)

  useEffect(() => {
    const host = hostRef.current!
    let cancelled = false
    let width = host.clientWidth
    let timer = 0
    setFailure(null)

    const osmd = new OpenSheetMusicDisplay(host, {
      backend: 'svg',
      autoResize: false,
      drawTitle: false,
      drawSubtitle: false,
      drawComposer: false,
      drawPartNames: false,
      drawMeasureNumbersOnlyAtSystemStart: true,
      // Every bar is drawn, so each has a place for the playhead and a spot to click.
      autoGenerateMultipleRestMeasuresFromRestMeasures: false,
    })
    // Flat beams, as in the PDF: a beam's height should not jump with the drums under it.
    osmd.EngravingRules.FlatBeams = true
    osmd.EngravingRules.TupletsBracketedUseXMLValue = true
    osmd.EngravingRules.TupletNumberLimitConsecutiveRepetitions = false
    osmd.zoom = zoomRef.current

    const engrave = () => {
      osmd.render()
      const scale = UNIT * osmd.zoom
      const boxes: Box[] = []
      const anchors: Engraving['anchors'] = []
      for (const staves of osmd.GraphicSheet.MeasureList) {
        const shape = staves[0]?.PositionAndShape
        if (!shape) {
          // A bar OSMD did not draw: the playhead waits at the end of the one before.
          const last = boxes[boxes.length - 1] ?? { x: 0, y: 0, width: 0, height: 0 }
          boxes.push({ ...last, x: last.x + last.width, width: 0 })
          anchors.push([])
          continue
        }
        const box = {
          x: shape.AbsolutePosition.x * scale,
          y: (shape.AbsolutePosition.y - STAFF_PAD) * scale,
          width: shape.Size.width * scale,
          height: (4 + 2 * STAFF_PAD) * scale,
        }
        boxes.push(box)
        const staffEntries = staves.flatMap((measure) => measure.staffEntries)
        const entries = staffEntries
          .map((entry) => ({
            whole: entry.relInMeasureTimestamp.RealValue,
            x: entry.PositionAndShape.AbsolutePosition.x * scale,
          }))
          .sort((a, b) => a.whole - b.whole)
        // A whole-bar rest sits in the middle of its bar; the playhead crosses the bar from its left edge.
        const resting = staffEntries.every((entry) =>
          entry.graphicalVoiceEntries.every((voice) => voice.notes.every((note) => note.sourceNote.isRest())),
        )
        anchors.push(resting ? [] : entries)
        for (const entry of staves.flatMap((measure) => measure.staffEntries)) {
          for (const note of entry.graphicalVoiceEntries.flatMap((voice) => voice.notes)) {
            if (note.sourceNote.NoteheadColorXml !== GHOST_TAG) continue
            const drawn = note as VexFlowGraphicalNote
            const heads = drawn.getSVGGElement()?.querySelectorAll<SVGGraphicsElement>('.vf-notehead')
            const head = heads?.[drawn.vfnoteIndex]
            if (head) parenthesize(head)
          }
        }
      }
      engraving.current = { boxes, anchors }
    }

    osmd
      .load(xml.replaceAll(GHOST_HEAD, `${GHOST_HEAD} color="${GHOST_TAG}"`))
      .then(() => {
        if (cancelled) return
        engrave()
        rezoom.current = (next) => {
          osmd.zoom = next
          engrave()
        }
      })
      .catch((error: unknown) => {
        if (!cancelled) setFailure(error instanceof Error ? error.message : String(error))
      })

    // Rendering changes the height, so only a width change means a real resize.
    const observer = new ResizeObserver(() => {
      if (host.clientWidth === width || !engraving.current) return
      width = host.clientWidth
      window.clearTimeout(timer)
      timer = window.setTimeout(engrave, 150)
    })
    observer.observe(host)

    return () => {
      cancelled = true
      rezoom.current = null
      observer.disconnect()
      window.clearTimeout(timer)
      engraving.current = null
      host.replaceChildren()
    }
  }, [xml])

  // Bars move when the zoom changes; the playhead follows from the clock, so the place in the song is kept.
  useEffect(() => {
    if (zoomRef.current === zoom) return
    zoomRef.current = zoom
    try {
      window.localStorage.setItem(ZOOM_KEY, String(zoom))
    } catch {
      // Storage may be unavailable; the zoom still applies for this visit.
    }
    rezoom.current?.(zoom)
  }, [zoom])

  const zoomBy = useCallback((steps: number) => {
    setZoom((current) => (steps === 0 ? 1 : clampZoom(current + steps * ZOOM_STEP)))
  }, [])

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.ctrlKey || event.metaKey || event.altKey) return
      const target = event.target as HTMLElement | null
      if (target?.closest?.('input, textarea, select, [contenteditable="true"]')) return
      const steps = event.key === '+' || event.key === '=' ? 1 : event.key === '-' ? -1 : event.key === '0' ? 0 : null
      if (steps === null) return
      event.preventDefault()
      zoomBy(steps)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [zoomBy])

  useEffect(() => {
    let frame = 0
    let shownLine = NaN
    let shownZoom = zoomRef.current

    const follow = () => {
      frame = requestAnimationFrame(follow)
      const state = engraving.current
      const playhead = playheadRef.current
      if (!state || !playhead) return
      const time = clock()
      const bar = timeline.barAt(time)
      const box = state.boxes[bar]
      if (!box) return

      // Glide from each note to the next, and from the last one to the bar line.
      const whole = timeline.wholeAt(time) - timeline.wholes[bar]
      const length = timeline.bars[bar].beats / (timeline.bars[bar].denominator ?? 4)
      const anchors = state.anchors[bar]
      let from = { whole: 0, x: anchors[0]?.x ?? box.x + Math.min(12, box.width / 4) }
      let to = { whole: length, x: box.x + box.width }
      for (const anchor of anchors) {
        if (anchor.whole <= whole) from = anchor
        else {
          to = anchor
          break
        }
      }
      const span = to.whole - from.whole
      const x = from.x + (to.x - from.x) * (span > 0 ? Math.min(1, (whole - from.whole) / span) : 0)
      playhead.style.transform = `translate(${x}px, ${box.y}px)`
      playhead.style.height = `${box.height}px`
      playhead.style.opacity = '1'
      // After a zoom the line moved even if its y did not: bring the playhead back into view.
      if (box.y !== shownLine || shownZoom !== zoomRef.current) {
        shownZoom = zoomRef.current
        playhead.scrollIntoView({ block: 'center', behavior: playing && !Number.isNaN(shownLine) ? 'smooth' : 'auto' })
        shownLine = box.y
      }
    }

    frame = requestAnimationFrame(follow)
    return () => cancelAnimationFrame(frame)
  }, [timeline, clock, playing])

  const seekToClick = (event: React.MouseEvent<HTMLDivElement>) => {
    const state = engraving.current
    if (!state) return
    const origin = event.currentTarget.getBoundingClientRect()
    const x = event.clientX - origin.left
    const y = event.clientY - origin.top
    const bar = state.boxes.findIndex(
      (box) => x >= box.x && x < box.x + box.width && y >= box.y && y < box.y + box.height,
    )
    if (bar >= 0) onSeek(timeline.barStart(bar))
  }

  return (
    <div>
      <div className="mb-2 flex items-center justify-end gap-1 text-xs text-ink" role="group" aria-label="악보 확대">
        <button
          type="button"
          onClick={() => zoomBy(-1)}
          disabled={zoom <= ZOOM_MIN}
          title="축소 (-)"
          aria-label="축소"
          className={ZOOM_BUTTON}
        >
          −
        </button>
        <button
          type="button"
          onClick={() => zoomBy(0)}
          title="원래 크기 (0)"
          aria-label={`확대 ${Math.round(zoom * 100)}%, 누르면 100%로`}
          className="h-7 min-w-14 rounded-full px-2 font-mono tabular-nums hover:bg-ink/10"
        >
          {Math.round(zoom * 100)}%
        </button>
        <button
          type="button"
          onClick={() => zoomBy(1)}
          disabled={zoom >= ZOOM_MAX}
          title="확대 (+)"
          aria-label="확대"
          className={ZOOM_BUTTON}
        >
          +
        </button>
      </div>
      <div className="relative cursor-pointer" onClick={seekToClick}>
        <div ref={hostRef} className="relative" />
        <div
          ref={playheadRef}
          className="pointer-events-none absolute top-0 left-0 -ml-px w-0.5 rounded-full bg-brass opacity-0"
        />
        {failure && (
          <p className="p-6 text-alarm">악보를 그리지 못했습니다: {failure}</p>
        )}
      </div>
    </div>
  )
}

const ZOOM_BUTTON =
  'flex h-7 w-7 items-center justify-center rounded-full bg-ink/6 text-sm font-medium hover:bg-ink/10 disabled:opacity-40 disabled:hover:bg-ink/6'
