import type { Bar, BeatGrid } from './api'

export type Loop = { start: number; end: number }

function firstAtOrAfter(times: number[], time: number): number {
  let low = 0
  let high = times.length
  while (low < high) {
    const mid = (low + high) >> 1
    if (times[mid] < time) low = mid + 1
    else high = mid
  }
  return low
}

/** The recording remains the clock; WebAudio only mixes in the practice clicks. */
export class PracticeClick {
  private readonly context = new AudioContext()
  private readonly output = this.context.createGain()
  private readonly sounds = new Set<OscillatorNode>()

  constructor(volume: number) {
    this.output.gain.value = volume
    this.output.connect(this.context.destination)
  }

  get time(): number {
    return this.context.currentTime
  }

  resume(): Promise<void> {
    return this.context.resume()
  }

  setVolume(volume: number): void {
    this.output.gain.setTargetAtTime(volume, this.time, 0.01)
  }

  silence(): void {
    for (const sound of this.sounds) sound.stop()
    this.sounds.clear()
  }

  close(): void {
    this.silence()
    void this.context.close()
  }

  private click(time: number, accent: boolean): void {
    const sound = this.context.createOscillator()
    const envelope = this.context.createGain()
    sound.frequency.value = accent ? 1800 : 1100
    envelope.gain.setValueAtTime(0, time)
    envelope.gain.linearRampToValueAtTime(accent ? 0.6 : 0.35, time + 0.002)
    envelope.gain.exponentialRampToValueAtTime(0.001, time + 0.03)
    sound.connect(envelope)
    envelope.connect(this.output)
    sound.onended = () => {
      this.sounds.delete(sound)
      sound.disconnect()
      envelope.disconnect()
    }
    this.sounds.add(sound)
    sound.start(time)
    sound.stop(time + 0.035)
  }

  /** Schedule only 100 ms ahead, and throw away that window on every transport change. */
  follow(media: HTMLMediaElement, grid: BeatGrid, loop: Loop | null): () => void {
    let next = 0
    let running = !media.paused && !media.seeking && media.readyState >= 3
    const reset = (tolerance = 0.02) => {
      this.silence()
      next = firstAtOrAfter(grid.beats, Math.max(loop?.start ?? 0, media.currentTime - tolerance * media.playbackRate))
    }
    const start = () => {
      reset()
      running = true
      schedule()
    }
    const stop = () => {
      running = false
      reset()
    }
    const seeked = () => {
      reset()
      running = !media.paused && media.readyState >= 3
      schedule()
    }
    const changedRate = () => {
      reset(0) // a beat already heard must not be repeated when the speed changes
      schedule()
    }
    const schedule = () => {
      if (!running || media.paused || media.seeking || media.readyState < 3 || this.context.state !== 'running') return
      const seconds = media.currentTime
      const rate = media.playbackRate
      while (next < grid.beats.length) {
        const beat = grid.beats[next]
        if (loop && beat >= loop.end) break
        const delay = (beat - seconds) / rate
        if (delay > 0.1) break
        next++
        if (delay < -0.02 || (loop && beat < loop.start)) continue
        const downbeat = firstAtOrAfter(grid.downbeats, beat - 0.03)
        const accent = downbeat < grid.downbeats.length && Math.abs(grid.downbeats[downbeat] - beat) <= 0.03
        this.click(this.time + Math.max(0.005, delay), accent)
      }
    }
    reset()
    media.addEventListener('playing', start)
    media.addEventListener('pause', stop)
    media.addEventListener('ended', stop)
    media.addEventListener('waiting', stop)
    media.addEventListener('emptied', stop)
    media.addEventListener('seeking', stop)
    media.addEventListener('seeked', seeked)
    media.addEventListener('ratechange', changedRate)
    const timer = window.setInterval(schedule, 25)
    schedule()
    return () => {
      window.clearInterval(timer)
      media.removeEventListener('playing', start)
      media.removeEventListener('pause', stop)
      media.removeEventListener('ended', stop)
      media.removeEventListener('waiting', stop)
      media.removeEventListener('emptied', stop)
      media.removeEventListener('seeking', stop)
      media.removeEventListener('seeked', seeked)
      media.removeEventListener('ratechange', changedRate)
      this.silence()
    }
  }

  /** One score bar, in denominator-note beats; use its local timing at the chosen rate. */
  countIn(bar: Bar, rate: number): { start: number; end: number; beats: number[] } {
    this.silence()
    const start = this.time + 0.05
    const origin = bar.beat_times[0]
    const beats = bar.beat_times.slice(0, -1).map((time) => start + (time - origin) / rate)
    beats.forEach((time, index) => this.click(time, index === 0))
    return { start, end: start + (bar.beat_times.at(-1)! - origin) / rate, beats }
  }
}
