import { useCallback, useEffect, useRef, useState, type RefObject } from 'react'
import { api, type BeatGrid, type Song } from './api'
import { PracticeClick, type Loop } from './practice'
import type { Timeline } from './timeline'

export function usePracticeClick(
  mediaRef: RefObject<HTMLVideoElement | null>, song: Song, timeline: Timeline,
  rate: number, loop: Loop | null, onError: (message: string) => void,
) {
  const engine = useRef<PracticeClick | null>(null)
  const countdown = useRef(0)
  const generation = useRef(0)
  const waiting = useRef<(() => void) | null>(null)
  const countActive = useRef(false)
  const [grid, setGrid] = useState<BeatGrid | null>(null)
  const [beatError, setBeatError] = useState<string | null>(null)
  const [enabled, setEnabled] = useState(false)
  const [countIn, setCountIn] = useState(false)
  const [counting, setCounting] = useState<number | null>(null) // 0: loading; 1..N: count-in beat
  const [volume, setVolume] = useState(0.35)

  const cancelCountIn = useCallback(() => {
    generation.current++
    waiting.current?.()
    waiting.current = null
    window.clearInterval(countdown.current)
    countdown.current = 0
    if (countActive.current) engine.current?.silence()
    countActive.current = false
    setCounting(null)
  }, [])

  const silenceClicks = useCallback(() => engine.current?.silence(), [])

  const ensureEngine = useCallback(() => {
    engine.current ??= new PracticeClick(volume)
    return engine.current
  }, [volume])

  useEffect(() => {
    let cancelled = false
    setGrid(null)
    setBeatError(null)
    api.beats(song.slug).then((loaded) => {
      if (!cancelled) {
        setGrid(loaded)
        if (!loaded.beats.length) {
          setEnabled(false)
          setBeatError('클릭에 쓸 박자가 없습니다')
        }
      }
    }).catch(() => {
      if (!cancelled) {
        setEnabled(false)
        setBeatError('박자 캐시를 불러오지 못했습니다')
      }
    })
    return () => { cancelled = true }
  }, [song.slug, song.updated_at, song.has_drums])

  useEffect(() => {
    engine.current?.setVolume(volume)
  }, [volume])

  // Transport edits cancel a count-in rather than leaving an old delayed play behind.
  useEffect(() => {
    cancelCountIn()
  }, [countIn, rate, loop, timeline, cancelCountIn])

  useEffect(() => {
    if (!enabled || !grid?.beats.length || !engine.current || !mediaRef.current) return
    return engine.current.follow(mediaRef.current, grid, loop)
  }, [enabled, grid, loop, rate, mediaRef])

  useEffect(() => () => {
    cancelCountIn()
    engine.current?.close()
    engine.current = null
  }, [cancelCountIn])

  const toggleClick = () => {
    if (enabled) {
      setEnabled(false)
      return
    }
    try {
      const click = ensureEngine()
      setEnabled(true)
      void click.resume().catch((error: Error) => {
        setEnabled(false)
        onError(`클릭을 켜지 못했습니다: ${error.message}`)
      })
    } catch (error) {
      onError(`클릭을 켜지 못했습니다: ${error instanceof Error ? error.message : String(error)}`)
    }
  }

  const startPlayback = useCallback(async (play: () => void) => {
    cancelCountIn()
    const media = mediaRef.current
    if (media?.ended) media.currentTime = 0
    const bar = timeline.bars[timeline.barAt(media?.currentTime ?? 0)]
    if (!countIn || !media || !bar) {
      if (enabled && engine.current) {
        void engine.current.resume().catch((error: Error) => onError(`클릭을 켜지 못했습니다: ${error.message}`))
      }
      play()
      return
    }
    const request = generation.current
    countActive.current = true
    setCounting(0)
    try {
      const click = ensureEngine()
      await click.resume()
      if (request !== generation.current) return
      // Finish loading/seeking before counting, so network buffering does not eat the count-in.
      if (media.readyState < 3 || media.seeking) {
        const ready = await new Promise<boolean>((resolve) => {
          const finish = (ok: boolean) => {
            media.removeEventListener('canplay', canPlay)
            media.removeEventListener('seeked', seeked)
            media.removeEventListener('error', failed)
            waiting.current = null
            resolve(ok)
          }
          const canPlay = () => { if (!media.seeking) finish(true) }
          const seeked = () => { if (media.readyState >= 3) finish(true) }
          const failed = () => finish(false)
          waiting.current = failed
          media.addEventListener('canplay', canPlay)
          media.addEventListener('seeked', seeked)
          media.addEventListener('error', failed)
        })
        if (request !== generation.current) return
        if (!ready) {
          cancelCountIn()
          return
        }
      }
      const plan = click.countIn(bar, rate)
      setCounting(1)
      countdown.current = window.setInterval(() => {
        if (click.time >= plan.end) {
          window.clearInterval(countdown.current)
          countdown.current = 0
          countActive.current = false
          setCounting(null)
          play()
        } else {
          const beat = plan.beats.filter((time) => time <= click.time).length
          setCounting(Math.max(1, beat))
        }
      }, 15)
    } catch (error) {
      if (request !== generation.current) return
      cancelCountIn()
      onError(`카운트인을 시작하지 못했습니다: ${error instanceof Error ? error.message : String(error)}`)
    }
  }, [cancelCountIn, countIn, enabled, ensureEngine, mediaRef, onError, rate, timeline])

  return { enabled, toggleClick, countIn, setCountIn, counting, cancelCountIn, silenceClicks, startPlayback,
    volume, setVolume, beatError, hasBeats: !!grid?.beats.length }
}
