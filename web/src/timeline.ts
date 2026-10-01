import type { Bar, Score } from './api'

/** Maps between seconds in the recording and positions in the score. */
export class Timeline {
  readonly bars: Bar[]
  readonly ticksPerBeat: number
  /** Start of each bar in whole notes, the unit OSMD's cursor uses. */
  readonly wholes: number[]

  constructor(score: Score) {
    this.bars = score.bars
    this.ticksPerBeat = score.divisions ?? 12
    this.wholes = []
    let whole = 0
    for (const bar of score.bars) {
      this.wholes.push(whole)
      whole += bar.beats / (bar.denominator ?? 4)
    }
  }

  get end(): number {
    const last = this.bars[this.bars.length - 1]
    return last ? last.beat_times[last.beat_times.length - 1] : 0
  }

  barStart(index: number): number {
    const clamped = Math.max(0, Math.min(this.bars.length - 1, index))
    return this.bars[clamped]?.beat_times[0] ?? 0
  }

  /** Index of the bar sounding at `time`, clamped to the score. */
  barAt(time: number): number {
    let low = 0
    let high = this.bars.length - 1
    while (low < high) {
      const mid = (low + high + 1) >> 1
      if (this.bars[mid].beat_times[0] <= time) low = mid
      else high = mid - 1
    }
    return low
  }

  /** Score position at `time` in whole notes from the start. */
  wholeAt(time: number): number {
    if (!this.bars.length) return 0
    const index = this.barAt(time)
    const times = this.bars[index].beat_times
    let beat = 0
    while (beat < times.length - 2 && times[beat + 1] <= time) beat++
    const span = times[beat + 1] - times[beat]
    const fraction = span > 0 ? Math.max(0, Math.min(1, (time - times[beat]) / span)) : 0
    return this.wholes[index] + (beat + fraction) / (this.bars[index].denominator ?? 4)
  }

  /** Seconds at which a note at `tick` of `bar` sounds. */
  tickTime(bar: Bar, tick: number): number {
    const unitTicks = this.ticksPerBeat * 4 / (bar.denominator ?? 4)
    const beat = Math.min(bar.beats - 1, Math.floor(tick / unitTicks))
    const start = bar.beat_times[beat]
    const end = bar.beat_times[beat + 1]
    return start + ((end - start) * (tick - beat * unitTicks)) / unitTicks
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
