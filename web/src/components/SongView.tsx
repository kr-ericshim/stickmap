import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { api, type Options, type Song, type TimeSignature, type TimeSignatureSuggestion } from '../api'
import { formatBytes, formatClock, Timeline } from '../timeline'
import { GrooveMap } from './GrooveMap'
import { ScoreView } from './ScoreView'
import { SoundPanel } from './SoundPanel'

const RATES = [0.5, 0.75, 1]

type Props = {
  song: Song
  onChange: (song: Song) => void
  onDeleted: () => void
  onRetranscribe: () => void
  onError: (message: string) => void
}

export function SongView({ song, onChange, onDeleted, onRetranscribe, onError }: Props) {
  const mediaRef = useRef<HTMLVideoElement>(null)
  const [xml, setXml] = useState<string | null>(null)
  const [playing, setPlaying] = useState(false)
  const [rate, setRate] = useState(1)
  const [stem, setStem] = useState<'mix' | 'drums'>('mix')
  const [busy, setBusy] = useState(false)
  const [position, setPosition] = useState({ bar: 0, seconds: 0 })
  const [showSounds, setShowSounds] = useState(false)

  const timeline = useMemo(() => new Timeline(song.score), [song.score])
  const clock = useCallback(() => mediaRef.current?.currentTime ?? 0, [])
  const showVideo = song.has_video && stem === 'mix'
  const source =
    stem === 'drums'
      ? api.audioUrl(song.slug, 'drums')
      : song.has_video
        ? api.videoUrl(song.slug)
        : api.audioUrl(song.slug, 'mix')

  useEffect(() => {
    let cancelled = false
    api
      .musicxml(song.slug)
      .then((text) => !cancelled && setXml(text))
      .catch((error: Error) => onError(error.message))
    return () => {
      cancelled = true
    }
  }, [song.slug, song.updated_at, onError])

  // Swap the source without losing the place in the song.
  useEffect(() => {
    const media = mediaRef.current!
    const resumeAt = media.currentTime
    const wasPlaying = !media.paused
    media.src = source
    media.currentTime = resumeAt
    if (wasPlaying) void media.play()
  }, [source])

  useEffect(() => {
    mediaRef.current!.playbackRate = rate
  }, [rate, source])

  const seek = useCallback((seconds: number) => {
    if (mediaRef.current) mediaRef.current.currentTime = Math.max(0, seconds)
  }, [])

  const toggle = useCallback(() => {
    const media = mediaRef.current
    if (!media) return
    if (media.paused) void media.play()
    else media.pause()
  }, [])

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement
      if (target.closest('input, textarea, select')) return
      if (event.code === 'Space') {
        if (target.closest('button')) return // the focused button handles it
        event.preventDefault()
        toggle()
      } else if (event.code === 'ArrowLeft' || event.code === 'ArrowRight') {
        event.preventDefault()
        const bar = timeline.barAt(clock())
        // Left goes to the start of this bar first, then to the one before.
        const intoBar = clock() - timeline.barStart(bar) > 0.35
        seek(timeline.barStart(event.code === 'ArrowRight' ? bar + 1 : intoBar ? bar : bar - 1))
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [toggle, seek, clock, timeline])

  useEffect(() => {
    const timer = window.setInterval(() => {
      const seconds = clock()
      const bar = timeline.barAt(seconds)
      setPosition((previous) =>
        previous.bar === bar && Math.floor(previous.seconds) === Math.floor(seconds)
          ? previous
          : { bar, seconds },
      )
    }, 120)
    return () => window.clearInterval(timer)
  }, [clock, timeline])

  const act = async (work: () => Promise<Song | void>) => {
    setBusy(true)
    try {
      const updated = await work()
      if (updated) onChange(updated)
    } catch (error) {
      onError(error instanceof Error ? error.message : String(error))
    } finally {
      setBusy(false)
    }
  }

  const setOption = (patch: Partial<Options>) => act(() => api.requantize(song.slug, patch))

  const clearCache = () => {
    if (!window.confirm(`분리한 드럼 음원과 분석 결과 ${formatBytes(song.cache_bytes)}를 지웁니다. 악보와 원곡은 남습니다.`)) return
    setStem('mix')
    void act(() => api.clearCache(song.slug))
  }

  const deleteSong = () => {
    if (!window.confirm(`“${song.title}”을(를) 음원, 영상, 악보까지 모두 삭제합니다. 되돌릴 수 없습니다.`)) return
    void act(async () => {
      await api.deleteSong(song.slug)
      onDeleted()
    })
  }

  const meters = [...new Set(song.score.bars.map((bar) => `${bar.beats}/${bar.denominator ?? 4}`))]
  const signature = song.options.time_signature
  const signatures = SIGNATURES.includes(signature as TimeSignature) || !signature ? SIGNATURES : [...SIGNATURES, signature]
  const suggestion = signature ? undefined : song.score.time_signature_suggestion
  // A new signature redefines what a beat is, so the other bar corrections start over.
  const setSignature = (time_signature: TimeSignature | null) =>
    setOption({ time_signature, beat_shift: 0, tempo_scale: 1 })

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <header className="flex items-start gap-6 px-8 pt-5 pb-4">
        <div className="min-w-0 flex-1">
          <h1 className="truncate font-display text-[32px] leading-tight font-bold tracking-tight [font-stretch:85%]">
            {song.title}
          </h1>
          <p className="mt-1 font-mono text-xs text-mute">
            {song.uploader && <span className="font-sans">{song.uploader} · </span>}
            {Math.round(song.score.tempo)} BPM · {meters.join(' ')} ·{' '}
            {song.score.bars.length}마디
          </p>
          {suggestion && (
            <p className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1 text-sm">
              <span>
                <span className="font-medium">{suggestion.time_signature}</span> 박자일 수 있습니다
                <span className="text-mute"> · {suggestionReasons(suggestion)}</span>
              </span>
              <button
                disabled={busy}
                onClick={() => setSignature(suggestion.time_signature)}
                className="rounded-full bg-ink/6 px-3 py-1 text-xs font-medium hover:bg-ink/10 disabled:opacity-50"
              >
                {suggestion.time_signature}로 바꾸기
              </button>
            </p>
          )}

          <div className="mt-4 flex flex-wrap items-center gap-x-5 gap-y-3">
            <button
              onClick={toggle}
              className="flex h-11 w-24 items-center justify-center gap-2 rounded-full bg-lacquer font-medium text-paper hover:bg-lacquer-soft"
            >
              {playing ? '일시정지' : '재생'}
            </button>
            <p className="font-mono text-sm tabular-nums">
              <span className="text-lg font-medium">{position.bar + 1}</span>
              <span className="text-mute"> / {song.score.bars.length}마디 · {formatClock(position.seconds)}</span>
            </p>
            <Segmented
              label="속도"
              value={rate}
              options={RATES.map((value) => ({ value, label: `${value}×` }))}
              onChange={setRate}
            />
            {song.has_drums && (
              <button
                aria-pressed={showSounds}
                onClick={() => {
                  // The panel plays its own clips; the song stops so the two do not play over each other.
                  mediaRef.current?.pause()
                  setShowSounds(!showSounds)
                }}
                className={`rounded-full px-4 py-2 text-xs font-medium ${
                  showSounds ? 'bg-ink text-paper' : 'bg-ink/6 text-ink hover:bg-ink/10'
                }`}
              >
                소리 확인{Object.keys(song.options.sound_names ?? {}).length ? ` · ${Object.keys(song.options.sound_names).length}` : ''}
              </button>
            )}
            {song.has_drums && (
              <Segmented
                label="듣기"
                value={stem}
                options={[
                  { value: 'mix', label: '원곡' },
                  { value: 'drums', label: '드럼만' },
                ]}
                onChange={setStem}
              />
            )}
          </div>

          <div className="mt-3 flex flex-wrap items-center gap-x-5 gap-y-3">
            <Segmented
              label="음표"
              disabled={busy}
              value={song.options.grid}
              options={[
                { value: 16, label: '16분' },
                { value: 8, label: '8분만' },
              ]}
              onChange={(grid) => setOption({ grid })}
            />
            <Segmented
              label="킥"
              disabled={busy}
              value={song.options.voices}
              options={[
                { value: 1, label: '한 성부' },
                { value: 2, label: '아래로 분리' },
              ]}
              onChange={(voices) => setOption({ voices })}
            />
            <Segmented
              label="셋잇단"
              disabled={busy}
              value={song.options.triplets}
              options={[
                { value: 'auto', label: '자동' },
                { value: 'off', label: '끄기' },
              ]}
              onChange={(triplets) => setOption({ triplets })}
            />
          </div>

          <div className="mt-3 flex flex-wrap items-center gap-x-5 gap-y-3">
            <label className="flex items-center gap-2 text-xs">
              <span className="text-mute">박자</span>
              <select
                disabled={busy}
                value={signature ?? ''}
                onChange={(event) => setSignature((event.target.value || null) as TimeSignature | null)}
                className="rounded-full bg-ink/6 px-3 py-1 font-mono font-medium disabled:opacity-50"
              >
                <option value="">자동</option>
                {signatures.map((value) => (
                  <option key={value} value={value}>
                    {value}
                  </option>
                ))}
              </select>
            </label>
            <Segmented
              label="템포"
              disabled={busy}
              value={song.options.tempo_scale}
              options={[
                { value: 0.5, label: '절반' },
                { value: 1, label: '기본' },
                { value: 2, label: '두 배' },
              ]}
              onChange={(tempo_scale) => setOption({ tempo_scale, beat_shift: 0 })}
            />
            <div className="flex items-center gap-2 text-xs">
              <span className="text-mute">마디 시작</span>
              <div className="flex items-center rounded-full bg-ink/6 p-0.5">
                <button
                  disabled={busy}
                  aria-label="마디 시작을 앞으로"
                  onClick={() => setOption({ beat_shift: song.options.beat_shift - 0.25 })}
                  className="rounded-full px-2.5 py-1 font-medium text-mute hover:text-ink disabled:opacity-50"
                >
                  ◀
                </button>
                <span className="w-16 text-center font-mono tabular-nums">
                  {song.options.beat_shift > 0 ? '+' : ''}
                  {song.options.beat_shift}박
                </span>
                <button
                  disabled={busy}
                  aria-label="마디 시작을 뒤로"
                  onClick={() => setOption({ beat_shift: song.options.beat_shift + 0.25 })}
                  className="rounded-full px-2.5 py-1 font-medium text-mute hover:text-ink disabled:opacity-50"
                >
                  ▶
                </button>
              </div>
            </div>
            <Segmented
              label="감도"
              disabled={busy}
              value={song.options.sensitivity}
              options={[
                { value: 1, label: '보통' },
                { value: 1.7, label: '높게' },
                { value: 3, label: '아주 높게' },
              ]}
              onChange={(sensitivity) => setOption({ sensitivity })}
            />
            <Segmented
              label="변박"
              disabled={busy || signature !== null}
              value={song.options.odd_bars}
              options={[
                { value: 'none', label: '없음' },
                { value: 'auto', label: '자동' },
                { value: 'detected', label: '검출대로' },
              ]}
              onChange={(odd_bars) => setOption({ odd_bars })}
            />
            <div className="flex items-center gap-3 text-xs">
              <span className="text-mute">내려받기</span>
              {(['pdf', 'musicxml', 'midi'] as const).map((kind) => (
                <a
                  key={kind}
                  href={api.downloadUrl(song.slug, kind)}
                  className="font-mono text-brass-deep underline decoration-brass/40 underline-offset-4 hover:decoration-brass-deep"
                >
                  {kind === 'musicxml' ? 'MusicXML' : kind.toUpperCase()}
                </a>
              ))}
            </div>
            <div className="ml-auto flex items-center gap-3 text-xs text-mute">
              <button disabled={busy} onClick={onRetranscribe} className="hover:text-ink disabled:opacity-50">
                다시 채보
              </button>
              {song.cache_bytes > 0 && (
                <button disabled={busy} onClick={clearCache} className="hover:text-ink disabled:opacity-50">
                  캐시 삭제 {formatBytes(song.cache_bytes)}
                </button>
              )}
              <button disabled={busy} onClick={deleteSong} className="hover:text-alarm disabled:opacity-50">
                곡 삭제
              </button>
            </div>
          </div>
        </div>

        <video
          ref={mediaRef}
          playsInline
          onPlay={() => setPlaying(true)}
          onPause={() => setPlaying(false)}
          onClick={toggle}
          className={
            showVideo
              ? 'aspect-video w-72 shrink-0 cursor-pointer rounded-md bg-ink object-contain shadow-sm'
              : 'hidden'
          }
        />
      </header>

      <div className="border-y border-line bg-paper/60 px-8 py-2">
        <GrooveMap score={song.score} timeline={timeline} clock={clock} onSeek={seek} />
      </div>

      <div className="flex min-h-0 flex-1">
        <div className="min-h-0 min-w-0 flex-1 overflow-y-auto px-8 py-6">
          <div className="mx-auto max-w-[1100px] rounded-sm bg-paper px-6 py-8 shadow-[0_1px_0_var(--color-line),0_12px_32px_-16px_rgba(20,54,63,0.35)]">
            {xml ? (
              <ScoreView xml={xml} timeline={timeline} clock={clock} playing={playing} onSeek={seek} />
            ) : (
              <p className="py-16 text-center text-mute">악보를 불러오는 중…</p>
            )}
          </div>
        </div>
        {showSounds && (
          <aside className="min-h-0 w-[440px] shrink-0 overflow-y-auto border-l border-line bg-room">
            <SoundPanel song={song} onChange={onChange} onError={onError} />
          </aside>
        )}
      </div>
    </div>
  )
}

type SegmentedProps<T extends string | number> = {
  label: string
  value: T
  options: { value: T; label: string }[]
  onChange: (value: T) => void
  disabled?: boolean
}

function Segmented<T extends string | number>({ label, value, options, onChange, disabled }: SegmentedProps<T>) {
  return (
    <div className="flex items-center gap-2 text-xs">
      <span className="text-mute">{label}</span>
      <div className="flex rounded-full bg-ink/6 p-0.5">
        {options.map((option) => (
          <button
            key={String(option.value)}
            disabled={disabled}
            aria-pressed={option.value === value}
            onClick={() => option.value !== value && onChange(option.value)}
            className={`rounded-full px-3 py-1 font-medium disabled:opacity-50 ${
              option.value === value ? 'bg-paper text-ink shadow-sm' : 'text-mute hover:text-ink'
            }`}
          >
            {option.label}
          </button>
        ))}
      </div>
    </div>
  )
}

// Bar lengths offered for a correction; any other N/4, N/8 or N/16 still shows when saved.
const SIGNATURES: TimeSignature[] = [
  '2/4', '3/4', '4/4', '5/4', '6/4', '7/4',
  '3/8', '5/8', '6/8', '7/8', '9/8', '11/8', '12/8', '13/8', '15/8',
  '5/16', '7/16', '9/16', '11/16', '13/16', '15/16',
]

function suggestionReasons(suggestion: TimeSignatureSuggestion): string {
  const reasons = suggestion.reasons.map((reason) =>
    reason === 'title_hint'
      ? '제목에 적힌 박자'
      : reason === 'bar_period_mismatch'
        ? `패턴이 ${suggestion.suggested_bar_seconds.toFixed(2)}초마다 반복`
        : '마디 첫 박이 들쭉날쭉',
  )
  return reasons.join(' · ')
}
