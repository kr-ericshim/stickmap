import { useEffect, useRef, useState } from 'react'
import { api } from '../api'

const SPEEDS = [0.25, 0.5, 1]

type Props = {
  slug: string
  hasVideo: boolean
  hasDrums: boolean
  time: number // the stroke
  start: number // the loop
  end: number
}

// One stroke, looped slowly: the picture shows which drum was hit, the sound what it sounds like.
// The video plays muted and a separate audio element carries the chosen stem, kept on the same clock.
export function ClipMonitor({ slug, hasVideo, hasDrums, time, start, end }: Props) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const audioRef = useRef<HTMLAudioElement>(null)
  const [speed, setSpeed] = useState(0.25)
  const [stem, setStem] = useState<'drums' | 'mix'>(hasDrums ? 'drums' : 'mix')
  const [progress, setProgress] = useState(0)
  const [playing, setPlaying] = useState(true)

  useEffect(() => {
    let frame = 0
    const media = [videoRef.current, audioRef.current].filter(Boolean) as HTMLMediaElement[]
    const restart = () => {
      for (const m of media) m.currentTime = start
    }
    restart()
    for (const m of media) {
      m.playbackRate = speed
      if (playing) void m.play().catch(() => setPlaying(false))
      else m.pause()
    }
    const tick = () => {
      const lead = audioRef.current ?? videoRef.current
      if (lead) {
        if (lead.currentTime >= end) restart()
        const video = videoRef.current
        // Resync the picture to the sound when they drift apart by more than a frame or two.
        if (video && audioRef.current && Math.abs(video.currentTime - audioRef.current.currentTime) > 0.06) {
          video.currentTime = audioRef.current.currentTime
        }
        setProgress((lead.currentTime - start) / (end - start))
      }
      frame = requestAnimationFrame(tick)
    }
    frame = requestAnimationFrame(tick)
    return () => {
      cancelAnimationFrame(frame)
      for (const m of media) m.pause()
    }
  }, [start, end, speed, stem, playing])

  const strike = (time - start) / (end - start)

  return (
    <div className="grid gap-2">
      <div className="relative aspect-video overflow-hidden rounded-md bg-ink">
        {hasVideo ? (
          <video
            ref={videoRef}
            src={api.videoUrl(slug)}
            muted
            playsInline
            preload="auto"
            onClick={() => setPlaying(!playing)}
            className="size-full cursor-pointer object-contain"
          />
        ) : (
          <p className="flex size-full items-center justify-center text-sm text-line">영상이 없는 곡: 소리만 들려줍니다</p>
        )}
        <audio ref={audioRef} src={api.audioUrl(slug, stem)} preload="auto" />
        <div className="absolute inset-x-3 bottom-3 h-[3px] rounded-full bg-lacquer-line">
          <div className="h-full rounded-full bg-line" style={{ width: `${Math.min(100, progress * 100)}%` }} />
          <div className="absolute -top-1 h-[11px] w-0.5 bg-brass" style={{ left: `${strike * 100}%` }} />
        </div>
      </div>
      <div className="flex flex-wrap items-center gap-1.5 text-xs">
        {SPEEDS.map((value) => (
          <button
            key={value}
            aria-pressed={speed === value}
            onClick={() => setSpeed(value)}
            className={`rounded-full px-3 py-1 font-medium ${speed === value ? 'bg-ink text-paper' : 'bg-ink/6 text-mute hover:text-ink'}`}
          >
            {value}×
          </button>
        ))}
        {hasDrums && (
          <>
            <span className="w-2" />
            {(['drums', 'mix'] as const).map((value) => (
              <button
                key={value}
                aria-pressed={stem === value}
                onClick={() => setStem(value)}
                className={`rounded-full px-3 py-1 font-medium ${stem === value ? 'bg-ink text-paper' : 'bg-ink/6 text-mute hover:text-ink'}`}
              >
                {value === 'drums' ? '드럼만' : '원곡'}
              </button>
            ))}
          </>
        )}
        <button
          onClick={() => setPlaying(!playing)}
          className="ml-auto rounded-full bg-ink/6 px-3 py-1 font-medium text-mute hover:text-ink"
        >
          {playing ? '멈춤' : '반복 재생'}
        </button>
      </div>
    </div>
  )
}
