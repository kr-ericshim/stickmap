import type { Bar, Note, Score } from './api'

/** Maps between seconds in the recording and positions in the score. */
export class Timeline {
  readonly bars: Bar[]
  readonly ticksPerBeat: number
  /** Start of each bar in whole notes, the unit OSMD's cursor uses. */
  readonly wholes: number[]
  private readonly starts: number[]
  private readonly onsetAnchors: { time: number; tick: number }[][]

  constructor(score: Score) {
    this.bars = score.bars
    this.ticksPerBeat = score.divisions ?? 12
    this.wholes = []
    let whole = 0
    for (const bar of score.bars) {
      this.wholes.push(whole)
      whole += bar.beats / (bar.denominator ?? 4)
    }
    this.starts = this.bars.map((bar) => Math.min(bar.beat_times[0], ...bar.notes.flatMap((note) => note.raw_time === undefined ? [] : [note.raw_time])))
    this.onsetAnchors = this.bars.map((bar, index) => {
      const groups = new Map<number, number[]>()
      for (const note of bar.notes) {
        if (note.raw_time === undefined) continue
        const group = groups.get(note.tick) ?? []
        group.push(note.raw_time)
        groups.set(note.tick, group)
      }
      const anchors = [...groups].sort(([a], [b]) => a - b).map(([tick, times]) => {
        times.sort((a, b) => a - b)
        const middle = times.length >> 1
        return { tick, time: times.length % 2 ? times[middle] : (times[middle - 1] + times[middle]) / 2 }
      })
      if (!anchors.length) return anchors
      if (anchors[0].tick > 0) anchors.unshift({ time: this.barStart(index), tick: 0 })
      anchors.push({ time: this.barEnd(index), tick: bar.beats * this.ticksPerBeat * 4 / (bar.denominator ?? 4) })
      // Cache the monotone mapping once, instead of rebuilding it every frame.
      for (let i = 1; i < anchors.length; i++) anchors[i].time = Math.max(anchors[i].time, anchors[i - 1].time + 1e-6)
      return anchors
    })
  }

  get end(): number {
    const last = this.bars[this.bars.length - 1]
    return last ? last.beat_times[last.beat_times.length - 1] : 0
  }

  barStart(index: number): number {
    const clamped = Math.max(0, Math.min(this.bars.length - 1, index))
    return this.starts[clamped] ?? 0
  }

  /** The selected end bar is included in an A–B loop. */
  barEnd(index: number): number {
    const clamped = Math.max(0, Math.min(this.bars.length - 1, index))
    return this.starts[clamped + 1] ?? this.bars[clamped]?.beat_times.at(-1) ?? 0
  }

  /** Index of the bar sounding at `time`, clamped to the score. */
  barAt(time: number): number {
    let low = 0
    let high = this.bars.length - 1
    while (low < high) {
      const mid = (low + high + 1) >> 1
      if (this.starts[mid] <= time) low = mid
      else high = mid - 1
    }
    return low
  }

  /** Score position at `time` in whole notes from the start. */
  wholeAt(time: number): number {
    if (!this.bars.length) return 0
    const index = this.barAt(time)
    const bar = this.bars[index]
    const raw = this.onsetAnchors[index]
    if (raw.length) {
      let from = raw[0]
      let to = raw[raw.length - 1]
      for (const anchor of raw) {
        if (anchor.time <= time) from = anchor
        else { to = anchor; break }
      }
      const fraction = Math.max(0, Math.min(1, (time - from.time) / Math.max(1e-6, to.time - from.time)))
      const tick = from.tick + fraction * (to.tick - from.tick)
      return this.wholes[index] + tick / (4 * this.ticksPerBeat)
    }
    const times = bar.beat_times
    let beat = 0
    while (beat < times.length - 2 && times[beat + 1] <= time) beat++
    const span = times[beat + 1] - times[beat]
    const fraction = span > 0 ? Math.max(0, Math.min(1, (time - times[beat]) / span)) : 0
    return this.wholes[index] + (beat + fraction) / (bar.denominator ?? 4)
  }

  /** Seconds at which a note at `tick` of `bar` sounds. */
  tickTime(bar: Bar, tick: number): number {
    const unitTicks = this.ticksPerBeat * 4 / (bar.denominator ?? 4)
    const beat = Math.min(bar.beats - 1, Math.floor(tick / unitTicks))
    const start = bar.beat_times[beat]
    const end = bar.beat_times[beat + 1]
    return start + ((end - start) * (tick - beat * unitTicks)) / unitTicks
  }

  noteTime(bar: Bar, note: Note): number {
    return note.raw_time ?? this.tickTime(bar, note.tick)
  }
}

export function formatClock(seconds: number): string {
  const whole = Math.max(0, Math.floor(seconds))
  return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, '0')}`
}

export function formatBytes(bytes: number): string {
  if (bytes >= 1e9) return `${(bytes / 1e9).toFixed(1)}GB`
  return `${Math.max(1, Math.round(bytes / 1e6))}MB`
}
