import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { api, type MeterChange, type Options, type Song, type TimeSignature, type TimeSignatureSuggestion } from '../api'
import { friendlyError, responseError } from '../errors'
import { formatBytes, formatClock, Timeline } from '../timeline'
import { usePracticeClick } from '../usePracticeClick'
import { GrooveMap } from './GrooveMap'
import { ScoreView } from './ScoreView'
import { SoundPanel } from './SoundPanel'

const RATES = [0.5, 0.75, 1, 1.25]
const SPEED_KEYS: Record<string, number> = { '1': 0.5, '2': 0.75, '3': 1 }

type DownloadKind = 'pdf' | 'musicxml' | 'midi'
const EXTENSIONS: Record<DownloadKind, string> = { pdf: 'pdf', musicxml: 'musicxml', midi: 'mid' }

/** The file name a Content-Disposition header suggests, if any. */
function fileName(header: string | null): string | null {
  if (!header) return null
  const encoded = /filename\*=(?:UTF-8'')?([^;]+)/i.exec(header)
  if (encoded) {
    try {
      return decodeURIComponent(encoded[1].trim().replace(/^"|"$/g, ''))
    } catch {
      // fall through to the plain name
    }
  }
  const plain = /filename="?([^";]+)"?/i.exec(header)
  return plain ? plain[1].trim() : null
}

function Spinner() {
  return <span aria-hidden="true" className="size-3.5 animate-spin rounded-full border-2 border-current border-t-transparent" />
}

type Props = {
  song: Song
  onChange: (song: Song) => void
  onDeleted: () => void
  onRetranscribe: () => void
  onError: (message: string) => void
}

export function SongView({ song, onChange, onDeleted, onRetranscribe, onError }: Props) {
  const mediaRef = useRef<HTMLVideoElement>(null)
  const pendingSource = useRef<{ seconds: number; resume: boolean; countIn?: boolean } | null>(null)
  const [xml, setXml] = useState<string | null>(null)
  const [playing, setPlaying] = useState(false)
  const [rate, setRate] = useState(1)
  const [stem, setStem] = useState<'mix' | 'drums' | 'nodrums'>('mix')
  const [volumes, setVolumes] = useState({ mix: 1, drums: 1, nodrums: 1 })
  const [busy, setBusy] = useState(false)
  const [position, setPosition] = useState({ bar: 0, seconds: 0 })
  const [showSounds, setShowSounds] = useState(false)
  const [loopBars, setLoopBars] = useState<{ start: number | null; end: number | null }>({ start: null, end: null })
  const [picking, setPicking] = useState<'start' | 'end' | null>(null)
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [videoMode, setVideoMode] = useState<'small' | 'large' | 'off'>('small')
  const [downloading, setDownloading] = useState<DownloadKind | null>(null)

  const timeline = useMemo(() => new Timeline(song.score), [song.score])
  const loop = useMemo(() => {
    if (loopBars.start === null || loopBars.end === null) return null
    const start = timeline.barStart(loopBars.start)
    const end = timeline.barEnd(loopBars.end)
    return end > start ? { start, end } : null
  }, [loopBars, timeline])
  const practice = usePracticeClick(mediaRef, song, timeline, rate, loop, onError)
  const { cancelCountIn, silenceClicks, startPlayback, counting } = practice
  const playbackRequest = useRef(startPlayback)
  useEffect(() => { playbackRequest.current = startPlayback }, [startPlayback])
  const clock = useCallback(() => pendingSource.current?.seconds ?? mediaRef.current?.currentTime ?? 0, [])
  const showVideo = song.has_video && stem === 'mix'
  const source =
    stem !== 'mix'
      ? api.audioUrl(song.slug, stem)
      : song.has_video
        ? api.videoUrl(song.slug)
        : api.audioUrl(song.slug, 'mix')

  const playMedia = useCallback(() => {
    const media = mediaRef.current
    if (!media) return
    void media.play().catch((error: Error) => {
      // Changing a source or pausing while play() is pending aborts that request.
      if (error.name !== 'AbortError') {
        setPlaying(false)
        onError(`재생하지 못했습니다: ${error.message}`)
      }
    })
  }, [onError])

  useEffect(() => {
    let cancelled = false
    api
      .musicxml(song.slug)
      .then((text) => !cancelled && setXml(text))
      .catch((error: unknown) => !cancelled && onError(friendlyError(error)))
    return () => {
      cancelled = true
    }
  }, [song.slug, song.updated_at, onError])

  // Swap the source without losing the place in the song.
  useEffect(() => {
    const media = mediaRef.current!
    cancelCountIn()
    silenceClicks()
    pendingSource.current ??= { seconds: media.currentTime, resume: !media.paused }
    media.pause()
    const restore = () => {
      const pending = pendingSource.current
      if (!pending) return
      media.currentTime = Math.min(pending.seconds, Number.isFinite(media.duration) ? media.duration : pending.seconds)
      media.playbackRate = media.defaultPlaybackRate
      media.preservesPitch = true
      pendingSource.current = null
      if (pending.resume) {
        if (pending.countIn) {
          setPlaying(false)
          void playbackRequest.current(playMedia)
        }
        else playMedia()
      }
    }
    media.addEventListener('loadedmetadata', restore, { once: true })
    media.src = source
    return () => media.removeEventListener('loadedmetadata', restore)
  }, [source, playMedia, cancelCountIn, silenceClicks])

  useEffect(() => {
    const media = mediaRef.current!
    media.defaultPlaybackRate = rate
    media.playbackRate = rate
    media.preservesPitch = true
  }, [rate, source])

  useEffect(() => {
    mediaRef.current!.volume = volumes[stem]
  }, [volumes, stem, source])

  const seek = useCallback((seconds: number) => {
    cancelCountIn()
    silenceClicks()
    if (loop && (seconds < loop.start || seconds >= loop.end)) seconds = loop.start
    if (pendingSource.current) pendingSource.current.seconds = Math.max(0, seconds)
    if (mediaRef.current) mediaRef.current.currentTime = Math.max(0, seconds)
  }, [loop, cancelCountIn, silenceClicks])

  const chooseBar = (edge: 'start' | 'end', index: number) => {
    index = Math.max(0, Math.min(timeline.bars.length - 1, index))
    setLoopBars((previous) => edge === 'start'
      ? { start: index, end: previous.end !== null ? Math.max(index, previous.end) : null }
      : { start: previous.start !== null ? Math.min(index, previous.start) : null, end: index })
    setPicking(null)
  }

  const selectOnScore = (seconds: number) => {
    if (picking) chooseBar(picking, timeline.barAt(seconds))
    else seek(seconds)
  }

  // A new score for the same song (re-transcription, settings) keeps the loop while it still fits;
  // another song remounts this view (App keys it by slug), so it starts with no loop.
  useEffect(() => {
    const count = timeline.bars.length
    setLoopBars((previous) => {
      if (previous.start === null && previous.end === null) return previous
      if ((previous.start ?? 0) >= count) return { start: null, end: null }
      if (previous.end !== null && previous.end >= count) return { ...previous, end: count - 1 }
      return previous
    })
  }, [timeline])

  // L: first press marks A at the playhead's bar, the second marks B, the third clears.
  const markLoop = useCallback(() => {
    const bar = timeline.barAt(clock())
    setPicking(null)
    setLoopBars((previous) => {
      if (previous.start === null) return { start: bar, end: null }
      if (previous.end === null) return { start: Math.min(previous.start, bar), end: Math.max(previous.start, bar) }
      return { start: null, end: null }
    })
  }, [timeline, clock])

  useEffect(() => {
    if (!loop) return
    seek(clock())
    const timer = window.setInterval(() => {
      const media = mediaRef.current
      if (media && !media.paused && !media.seeking && media.currentTime >= loop.end) seek(loop.start)
    }, 25)
    return () => window.clearInterval(timer)
  }, [loop, seek, clock])

  const toggle = useCallback(() => {
    const media = mediaRef.current
    if (!media) return
    if (counting !== null) {
      cancelCountIn()
      setPlaying(false)
    }
    else if (pendingSource.current) {
      pendingSource.current.resume = !pendingSource.current.resume
      pendingSource.current.countIn = true
      setPlaying(pendingSource.current.resume)
    } else if (media.paused) void startPlayback(playMedia)
    else media.pause()
  }, [playMedia, counting, cancelCountIn, startPlayback])

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement
      if (target.closest('input, textarea, select, [contenteditable="true"]')) return
      if (event.metaKey || event.ctrlKey || event.altKey) return
      if (event.code === 'KeyL') {
        if (!timeline.bars.length) return
        event.preventDefault()
        markLoop()
      } else if (SPEED_KEYS[event.key] !== undefined) {
        event.preventDefault()
        setRate(SPEED_KEYS[event.key])
      } else if (event.code === 'Space') {
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
  }, [toggle, seek, clock, timeline, markLoop])

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
      onError(friendlyError(error))
    } finally {
      setBusy(false)
    }
  }

  // Fetch first, so a server error (e.g. no PDF renderer) shows in the banner instead of a JSON tab.
  const download = async (kind: DownloadKind) => {
    if (downloading) return
    setDownloading(kind)
    try {
      const response = await fetch(api.downloadUrl(song.slug, kind))
      if (!response.ok) {
        onError(await responseError(response))
        return
      }
      const blob = await response.blob()
      const url = URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = fileName(response.headers.get('content-disposition')) ?? `${song.slug}.${EXTENSIONS[kind]}`
      document.body.append(link)
      link.click()
      link.remove()
      window.setTimeout(() => URL.revokeObjectURL(url), 1000)
    } catch (error) {
      onError(friendlyError(error))
    } finally {
      setDownloading(null)
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
    setOption({ time_signature, beat_shift: 0, tempo_scale: 1, meter_changes: [] })
  const meterChanges = song.options.meter_changes ?? []
  const denominator = signature?.split('+')[0].split('/')[1]
  const changeChoices = SIGNATURES.filter((value) => !value.includes('+') && value.endsWith(`/${denominator}`))
  // Whole-beat bar starts for the dropdown; the arrows keep quarter-beat steps.
  const barBeats = signature
    ? Math.max(...signature.split('+').map((value) => Number(value.split('/')[0])))
    : Math.max(...song.score.bars.map((bar) => bar.beats), 1)
  const reach = Math.min(11, barBeats - 1)
  const shiftChoices = [...new Set([...Array.from({ length: 2 * reach + 1 }, (_, i) => i - reach), song.options.beat_shift])]
    .sort((a, b) => a - b)
  const barAt = (time: number) => {
    const starts = song.score.bars.map((bar) => bar.beat_times[0])
    return starts.reduce((best, start, i) => (Math.abs(start - time) < Math.abs(starts[best] - time) ? i : best), 0)
  }
  const setMeterChanges = (changes: MeterChange[]) =>
    setOption({ meter_changes: [...changes].sort((a, b) => a.time - b.time) })
  // The bar under the playhead becomes the first bar in the new signature.
  const changeMeterHere = (time_signature: TimeSignature) => {
    const time = Math.round(song.score.bars[position.bar].beat_times[0] * 1000) / 1000
    setMeterChanges([...meterChanges.filter((change) => Math.abs(change.time - time) > 0.05), { time, time_signature }])
  }

  const options = song.options
  const rideRuns = options.ride_runs ?? true // songs saved before the option read as on
  // Reading corrections away from the defaults the server starts every song with (library.DEFAULT_OPTIONS).
  const changedSettings = [
    options.grid !== 16,
    options.voices !== 1,
    options.triplets !== 'auto',
    options.sensitivity !== 1,
    signature !== null,
    meterChanges.length > 0,
    options.tempo_scale !== 1,
    options.beat_shift !== 0,
    options.odd_bars !== 'none',
    rideRuns === false,
  ].filter(Boolean).length
  const soundNames = Object.keys(options.sound_names ?? {}).length
  const videoShown = showVideo && videoMode !== 'off'
  const loopSet = loopBars.start !== null || loopBars.end !== null

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <header className="flex items-start gap-5 px-6 pt-3 pb-3">
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-4">
            <div className="flex min-w-0 flex-1 items-baseline gap-3">
              <h1 className="truncate font-display text-[22px] leading-tight font-bold tracking-[-0.02em] [font-stretch:85%]">
                {song.title}
              </h1>
              <p className="hidden min-w-0 shrink-[100] truncate font-mono text-xs text-mute tabular-nums lg:block">
                {song.uploader && <span className="font-sans">{song.uploader} · </span>}
                {Math.round(song.score.tempo)} BPM · {meters.join(' ')} · {song.score.bars.length}마디
              </p>
            </div>
            <div className="flex shrink-0 items-center gap-1.5">
              {showVideo && videoMode === 'off' && (
                <button onClick={() => setVideoMode('small')} className={CHIP}>
                  <VideoIcon /> 영상 보기
                </button>
              )}
              {song.has_drums && (
                <button
                  aria-pressed={showSounds}
                  onClick={() => {
                    // The panel plays its own clips; the song stops so the two do not play over each other.
                    mediaRef.current?.pause()
                    cancelCountIn()
                    setShowSounds(!showSounds)
                  }}
                  className={showSounds ? CHIP_ON : CHIP}
                >
                  소리 확인{soundNames ? ` · ${soundNames}` : ''}
                </button>
              )}
              <button
                aria-expanded={settingsOpen}
                aria-controls="reading-settings"
                onClick={() => setSettingsOpen(!settingsOpen)}
                className={settingsOpen ? CHIP_ON : CHIP}
              >
                <SlidersIcon />
                채보 설정
                {busy ? (
                  <span className="font-normal opacity-70">· 적는 중…</span>
                ) : changedSettings > 0 && (
                  <span
                    aria-label={`${changedSettings}개 바꿈`}
                    className={`grid h-4 min-w-4 place-items-center rounded-full px-1 font-mono text-[10px] ${settingsOpen ? 'bg-brass text-ink' : 'bg-brass/25 text-brass-deep'}`}
                  >
                    {changedSettings}
                  </span>
                )}
              </button>
              <Popover
                label="더 보기"
                trigger={<MoreIcon />}
                className={`${CHIP} w-8 justify-center px-0`}
              >
                {(close) => (
                  <div role="menu" className="w-56 py-1.5 text-sm">
                    <p className="px-3 pt-1 pb-1.5 text-xs text-mute">내려받기</p>
                    {(['pdf', 'musicxml', 'midi'] as const).map((kind) => (
                      <button
                        key={kind}
                        role="menuitem"
                        disabled={downloading !== null}
                        aria-busy={downloading === kind}
                        onClick={() => void download(kind).then(close)}
                        className={MENU_ITEM}
                      >
                        {downloading === kind ? <Spinner /> : <DownloadIcon />}
                        {kind === 'musicxml' ? 'MusicXML' : kind.toUpperCase()}
                        {downloading === kind && <span className="ml-auto text-xs text-mute">만드는 중…</span>}
                      </button>
                    ))}
                    <div className="my-1.5 border-t border-line" />
                    <button role="menuitem" disabled={busy} onClick={() => { close(); onRetranscribe() }} className={MENU_ITEM}>
                      다시 채보
                    </button>
                    {song.cache_bytes > 0 && (
                      <button role="menuitem" disabled={busy} onClick={() => { close(); clearCache() }} className={MENU_ITEM}>
                        캐시 삭제
                        <span className="ml-auto font-mono text-xs text-mute">{formatBytes(song.cache_bytes)}</span>
                      </button>
                    )}
                    <button role="menuitem" disabled={busy} onClick={() => { close(); deleteSong() }} className={`${MENU_ITEM} text-alarm`}>
                      곡 삭제…
                    </button>
                  </div>
                )}
              </Popover>
            </div>
          </div>

          {suggestion && (
            <p className="mt-1.5 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs">
              <span>
                <span className="font-medium">{suggestion.time_signature}</span> 박자일 수 있습니다
                <span className="text-mute"> · {suggestionReasons(suggestion)}</span>
              </span>
              <button
                disabled={busy}
                onClick={() => setSignature(suggestion.time_signature)}
                className="rounded-full bg-brass/20 px-2.5 py-0.5 font-medium text-brass-deep hover:bg-brass/30 disabled:opacity-50"
              >
                {suggestion.time_signature}로 바꾸기
              </button>
            </p>
          )}

          <div className="mt-2.5 flex flex-wrap items-center gap-x-4 gap-y-2 text-xs">
            <div className="flex items-center gap-3">
              <button
                onClick={toggle}
                className="flex h-9 w-[5.5rem] items-center justify-center gap-1.5 rounded-full bg-lacquer text-sm font-medium text-paper hover:bg-lacquer-soft"
              >
                {counting !== null ? '취소' : playing ? <><PauseIcon /> 일시정지</> : <><PlayIcon /> 재생</>}
              </button>
              <p className="min-w-[8.5rem] font-mono text-sm tabular-nums">
                <span className="text-base font-medium">{position.bar + 1}</span>
                <span className="text-mute"> / {song.score.bars.length}마디 · {formatClock(position.seconds)}</span>
              </p>
              {counting !== null && (
                <span role="status" className="rounded-full bg-brass/20 px-2.5 py-0.5 font-medium text-brass-deep">
                  {counting === 0 ? '카운트인 준비 중…' : `카운트인 · ${counting}박`}
                </span>
              )}
            </div>

            <Divider />
            <Segmented
              label="속도"
              value={rate}
              options={RATES.map((value) => ({ value, label: `${value}×` }))}
              onChange={setRate}
            />

            <Divider />
            <div className="flex items-center gap-2">
              <Segmented
                label="듣기"
                value={stem}
                options={[
                  { value: 'mix', label: '원곡' },
                  { value: 'drums', label: '드럼만', disabled: !song.has_drums },
                  { value: 'nodrums', label: '드럼 뺀 MR', disabled: !song.has_mr },
                ]}
                onChange={setStem}
              />
              <input
                type="range" min={0} max={1} step={0.01}
                aria-label={`${stem === 'nodrums' ? 'MR' : stem === 'drums' ? '드럼' : '원곡'} 음량`}
                title={`음량 ${Math.round(volumes[stem] * 100)}%`}
                value={volumes[stem]}
                onChange={(event) => {
                  const volume = event.target.valueAsNumber
                  setVolumes((previous) => ({ ...previous, [stem]: volume }))
                }}
                className="w-20 accent-brass"
              />
            </div>

            <Divider />
            <div className="flex items-center rounded-full bg-ink/6 p-0.5">
              <button
                aria-pressed={practice.enabled}
                disabled={!practice.hasBeats || counting !== null}
                title={practice.beatError ?? '첫 박을 강하게 · 현재 속도로 재생'}
                onClick={practice.toggleClick}
                className={`rounded-full px-3 py-1 font-medium disabled:opacity-50 ${
                  practice.enabled ? 'bg-ink text-paper' : 'text-ink hover:bg-ink/6'
                }`}
              >
                메트로놈
              </button>
              <Popover
                label="메트로놈 설정"
                align="start"
                trigger={<>{practice.countIn && <span>카운트인</span>}<ChevronIcon /></>}
                className="flex items-center gap-1 rounded-full px-2 py-1 font-medium text-mute hover:text-ink"
              >
                {() => (
                  <div className="w-64 space-y-3 p-3.5 text-xs">
                    <label className="flex items-center gap-2">
                      <span className="w-14 text-mute">클릭 음량</span>
                      <input
                        type="range" min={0} max={1} step={0.01}
                        value={practice.volume}
                        onChange={(event) => practice.setVolume(event.target.valueAsNumber)}
                        className="min-w-0 flex-1 accent-brass"
                      />
                      <span className="w-8 text-right font-mono text-mute tabular-nums">{Math.round(practice.volume * 100)}%</span>
                    </label>
                    <label className="flex items-center gap-2">
                      <input
                        type="checkbox"
                        checked={practice.countIn}
                        disabled={!timeline.bars.length}
                        onChange={(event) => practice.setCountIn(event.target.checked)}
                        className="accent-brass"
                      />
                      재생 전에 한 마디 카운트인
                    </label>
                    <p className={practice.beatError ? 'text-alarm' : 'text-mute'}>
                      {practice.beatError ?? '첫 박을 강하게 치고, 지금 속도에 맞춰 울립니다.'}
                    </p>
                  </div>
                )}
              </Popover>
            </div>

            <Divider />
            <div className="flex items-center gap-1.5" title="끝 마디 포함">
              <span className={loop ? 'font-medium text-brass-deep' : 'text-mute'}>반복</span>
              {(['start', 'end'] as const).map((edge) => (
                <div key={edge} className="flex items-center gap-1.5">
                  {edge === 'end' && <span className="text-mute">–</span>}
                  <div className={`flex items-center rounded-full p-0.5 ${picking === edge ? 'bg-brass/25' : 'bg-ink/6'}`}>
                    <button
                      disabled={!timeline.bars.length}
                      aria-pressed={picking === edge}
                      aria-label={edge === 'start' ? '악보에서 반복 시작 마디 고르기' : '악보에서 반복 끝 마디 고르기'}
                      title={edge === 'start' ? 'A: 악보나 타임라인에서 시작 마디 고르기' : 'B: 악보나 타임라인에서 끝 마디 고르기'}
                      onClick={() => setPicking(picking === edge ? null : edge)}
                      className={`grid size-6 place-items-center rounded-full font-mono font-medium disabled:opacity-50 ${
                        picking === edge ? 'bg-ink text-paper' : 'text-ink hover:bg-paper'
                      }`}
                    >
                      {edge === 'start' ? 'A' : 'B'}
                    </button>
                    <input
                      type="number"
                      aria-label={edge === 'start' ? '반복 시작 마디' : '반복 끝 마디'}
                      min={1}
                      max={timeline.bars.length}
                      step={1}
                      value={loopBars[edge] === null ? '' : loopBars[edge] + 1}
                      placeholder="—"
                      disabled={!timeline.bars.length}
                      onChange={(event) => {
                        const value = event.target.valueAsNumber
                        if (Number.isInteger(value)) chooseBar(edge, value - 1)
                        else setLoopBars((previous) => ({ ...previous, [edge]: null }))
                      }}
                      className="w-11 bg-transparent pr-1 text-center font-mono text-ink tabular-nums placeholder:text-mute"
                    />
                  </div>
                </div>
              ))}
              {(loopSet || picking) && (
                <button
                  aria-label="반복 해제"
                  title="반복 해제"
                  onClick={() => {
                    setLoopBars({ start: null, end: null })
                    setPicking(null)
                  }}
                  className="grid size-6 place-items-center rounded-full text-mute hover:bg-ink/6 hover:text-ink"
                >
                  <CloseIcon />
                </button>
              )}
              <span role="status" className="font-medium text-brass-deep">
                {picking && `악보에서 ${picking === 'start' ? '시작' : '끝'} 마디를 클릭하세요`}
              </span>
            </div>
          </div>
          <p className="mt-1.5 text-[11px] text-mute" aria-hidden="true">
            Space 재생 · ←/→ 마디 이동 · L 구간 반복 · 1/2/3 속도
          </p>
        </div>

        <div
          className={
            videoShown
              ? `group relative shrink-0 overflow-hidden rounded-md bg-ink shadow-sm ${videoMode === 'large' ? 'w-80' : 'w-40'}`
              : 'hidden'
          }
        >
          <video
            ref={mediaRef}
            playsInline
            onPlay={() => setPlaying(true)}
            onPause={() => setPlaying(false)}
            onError={() => {
              pendingSource.current = null
              cancelCountIn()
              setPlaying(false)
              onError('음원을 불러오지 못했습니다. 소스를 바꾸거나 다시 시도하세요.')
            }}
            onEnded={() => {
              if (loop) {
                seek(loop.start)
                playMedia()
              } else setPlaying(false)
            }}
            onClick={toggle}
            className="block aspect-video w-full cursor-pointer object-contain"
          />
          <div className="absolute top-1 right-1 flex gap-1 opacity-0 transition-opacity group-focus-within:opacity-100 group-hover:opacity-100">
            <button
              aria-label={videoMode === 'large' ? '영상 작게' : '영상 크게'}
              title={videoMode === 'large' ? '영상 작게' : '영상 크게'}
              onClick={() => setVideoMode(videoMode === 'large' ? 'small' : 'large')}
              className={VIDEO_BUTTON}
            >
              {videoMode === 'large' ? <ShrinkIcon /> : <ExpandIcon />}
            </button>
            <button aria-label="영상 숨기기" title="영상 숨기기" onClick={() => setVideoMode('off')} className={VIDEO_BUTTON}>
              <CloseIcon />
            </button>
          </div>
        </div>
      </header>

      {settingsOpen && (
        <section id="reading-settings" aria-label="채보 설정" className="border-t border-line bg-paper/55 px-6 py-3 text-xs">
          <div className="flex flex-wrap items-start gap-x-10 gap-y-3">
            <div className="space-y-2">
              <Segmented
                label="기본 해상도"
                labelClassName="w-20"
                disabled={busy}
                value={options.grid}
                options={[
                  { value: 16, label: '16분' },
                  { value: 8, label: '8분만' },
                ]}
                onChange={(grid) => setOption({ grid })}
              />
              <Segmented
                label="셋잇단·6연음"
                labelClassName="w-20"
                disabled={busy}
                value={options.triplets}
                options={[
                  { value: 'auto', label: '자동' },
                  { value: 'off', label: '끄기' },
                ]}
                onChange={(triplets) => setOption({ triplets })}
              />
              <Segmented
                label="감도"
                labelClassName="w-20"
                disabled={busy}
                value={options.sensitivity}
                options={[
                  { value: 1, label: '보통' },
                  { value: 1.7, label: '높게' },
                  { value: 3, label: '아주 높게' },
                ]}
                onChange={(sensitivity) => setOption({ sensitivity })}
              />
            </div>

            <div className="space-y-2">
              <label className="flex items-center gap-2">
                <span className="w-16 text-mute">박자</span>
                <select
                  disabled={busy}
                  value={signature ?? ''}
                  onChange={(event) => {
                    if (event.target.value !== CUSTOM) return setSignature((event.target.value || null) as TimeSignature | null)
                    const typed = window.prompt('박자를 입력하세요 (예: 17/8, 7/8+6/8)', signature ?? '')
                    const value = typed?.replace(/\s+/g, '').replaceAll('·', '+')
                    if (!value) return
                    if (isSignature(value)) setSignature(value)
                    else window.alert('N/4, N/8, N/16 꼴로 적어 주세요 (N은 1~32). 섞인 박자는 분모를 같게 하고 +로 이어 최대 4개까지.')
                  }}
                  className="rounded-full bg-ink/6 px-3 py-1 font-mono font-medium disabled:opacity-50"
                >
                  <option value="">자동</option>
                  {signatures.filter((value) => !value.includes('+')).map((value) => (
                    <option key={value} value={value}>
                      {value}
                    </option>
                  ))}
                  <optgroup label="섞인 박자">
                    {signatures.filter((value) => value.includes('+')).map((value) => (
                      <option key={value} value={value}>
                        {value.replaceAll('+', ' · ')}
                      </option>
                    ))}
                  </optgroup>
                  <option value={CUSTOM}>직접 입력…</option>
                </select>
              </label>
              <Segmented
                label="템포"
                labelClassName="w-16"
                disabled={busy}
                value={options.tempo_scale}
                options={[
                  { value: 0.5, label: '절반' },
                  { value: 1, label: '기본' },
                  { value: 2, label: '두 배' },
                ]}
                onChange={(tempo_scale) => setOption({ tempo_scale, beat_shift: 0 })}
              />
              <div className="flex items-center gap-2">
                <span className="w-16 text-mute">마디 시작</span>
                <div className="flex items-center rounded-full bg-ink/6 p-0.5">
                  <button
                    disabled={busy}
                    aria-label="마디 시작을 앞으로"
                    onClick={() => setOption({ beat_shift: options.beat_shift - 0.25 })}
                    className="rounded-full px-2.5 py-1 font-medium text-mute hover:text-ink disabled:opacity-50"
                  >
                    <ChevronIcon className="rotate-90" />
                  </button>
                  <select
                    disabled={busy}
                    aria-label="마디 시작"
                    value={options.beat_shift}
                    onChange={(event) => setOption({ beat_shift: Number(event.target.value) })}
                    className="w-16 cursor-pointer appearance-none bg-transparent text-center font-mono tabular-nums disabled:opacity-50"
                  >
                    {shiftChoices.map((value) => (
                      <option key={value} value={value}>
                        {value > 0 ? '+' : ''}
                        {value}박
                      </option>
                    ))}
                  </select>
                  <button
                    disabled={busy}
                    aria-label="마디 시작을 뒤로"
                    onClick={() => setOption({ beat_shift: options.beat_shift + 0.25 })}
                    className="rounded-full px-2.5 py-1 font-medium text-mute hover:text-ink disabled:opacity-50"
                  >
                    <ChevronIcon className="-rotate-90" />
                  </button>
                </div>
              </div>
              {signature !== null && (
                <div className="flex flex-wrap items-center gap-2">
                  <span className="w-16 text-mute">박자 바꿈</span>
                  {meterChanges.map((change) => (
                    <span key={change.time} className="flex items-center gap-1 rounded-full bg-brass/15 py-0.5 pr-1 pl-2.5 font-mono">
                      {barAt(change.time) + 1}마디~ {change.time_signature}
                      <button
                        disabled={busy}
                        aria-label={`${barAt(change.time) + 1}마디 박자 바꿈 지우기`}
                        onClick={() => setMeterChanges(meterChanges.filter((other) => other !== change))}
                        className="rounded-full px-1.5 text-mute hover:text-ink disabled:opacity-50"
                      >
                        ×
                      </button>
                    </span>
                  ))}
                  <select
                    disabled={busy}
                    value=""
                    onChange={(event) => event.target.value && changeMeterHere(event.target.value as TimeSignature)}
                    title="재생 위치의 마디부터 박자를 바꿉니다. 지정하면 그 지점들만 따르고 자동 판단은 끕니다"
                    className="rounded-full bg-ink/6 px-3 py-1 font-medium disabled:opacity-50"
                  >
                    <option value="">{position.bar + 1}마디부터…</option>
                    {changeChoices.map((value) => (
                      <option key={value} value={value}>
                        {value}
                      </option>
                    ))}
                  </select>
                </div>
              )}
              <Segmented
                label="변박"
                labelClassName="w-16"
                disabled={busy || signature !== null}
                value={options.odd_bars}
                options={[
                  { value: 'none', label: '없음' },
                  { value: 'auto', label: '자동' },
                  { value: 'detected', label: '검출대로' },
                ]}
                onChange={(odd_bars) => setOption({ odd_bars })}
              />
            </div>

            <div className="space-y-2">
              <Segmented
                label="킥"
                labelClassName="w-8"
                disabled={busy}
                value={options.voices}
                options={[
                  { value: 1, label: '한 성부' },
                  { value: 2, label: '아래로 분리' },
                ]}
                onChange={(voices) => setOption({ voices })}
              />
              <label className="flex items-start gap-2">
                <input
                  type="checkbox"
                  checked={rideRuns}
                  disabled={busy}
                  onChange={(event) => setOption({ ride_runs: event.target.checked })}
                  className="mt-0.5 accent-brass"
                />
                <span>
                  연속 심벌을 라이드로
                  <span className="block text-mute">끄면 모든 심벌을 크래시로 표기</span>
                </span>
              </label>
              <p className="max-w-[38ch] leading-relaxed text-mute">
                16분을 기본으로 읽기 쉽게 적고, 빠른 타격이 분명할 때 32분을 사용합니다. 자동은 8분 셋잇단과 6연음도 구간별로 판단합니다.
              </p>
            </div>

            <div className="ml-auto flex items-center gap-3 self-start">
              <span role="status" className="text-mute">{busy ? '악보를 다시 적는 중…' : ''}</span>
              <button
                aria-label="채보 설정 닫기"
                title="닫기"
                onClick={() => setSettingsOpen(false)}
                className="grid size-7 place-items-center rounded-full text-mute hover:bg-ink/6 hover:text-ink"
              >
                <CloseIcon />
              </button>
            </div>
          </div>
        </section>
      )}

      <div className="border-y border-line bg-paper/60 px-6 py-1.5">
        <GrooveMap score={song.score} timeline={timeline} clock={clock} onSeek={selectOnScore} loop={loop} />
      </div>

      <div className="flex min-h-0 flex-1">
        <div className="min-h-0 min-w-0 flex-1 overflow-y-auto px-6 py-4">
          <div className="mx-auto max-w-[1100px] rounded-sm bg-paper px-6 py-6 shadow-[0_1px_0_var(--color-line),0_12px_32px_-16px_rgba(20,54,63,0.35)]">
            {xml ? (
              <ScoreView xml={xml} timeline={timeline} clock={clock} playing={playing} onSeek={selectOnScore} />
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

const CHIP = 'flex h-8 items-center gap-1.5 rounded-full bg-ink/6 px-3 text-xs font-medium text-ink hover:bg-ink/10'
const CHIP_ON = 'flex h-8 items-center gap-1.5 rounded-full bg-ink px-3 text-xs font-medium text-paper'
const MENU_ITEM = 'flex w-full items-center gap-2 px-3 py-1.5 text-left hover:bg-ink/6 disabled:opacity-50'
const VIDEO_BUTTON = 'grid size-6 place-items-center rounded-full bg-ink/70 text-paper hover:bg-ink'

function Divider() {
  return <span aria-hidden className="hidden h-5 w-px bg-line xl:block" />
}

type PopoverProps = {
  label: string
  trigger: React.ReactNode
  className: string
  align?: 'start' | 'end'
  children: (close: () => void) => React.ReactNode
}

/** A button with a small panel under it; outside clicks and Escape close it. */
function Popover({ label, trigger, className, align = 'end', children }: PopoverProps) {
  const [open, setOpen] = useState(false)
  const root = useRef<HTMLDivElement>(null)
  const button = useRef<HTMLButtonElement>(null)
  useEffect(() => {
    if (!open) return
    const onDown = (event: PointerEvent) => {
      if (!root.current?.contains(event.target as Node)) setOpen(false)
    }
    const onKey = (event: KeyboardEvent) => {
      if (event.key !== 'Escape') return
      setOpen(false)
      button.current?.focus()
    }
    document.addEventListener('pointerdown', onDown)
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('pointerdown', onDown)
      document.removeEventListener('keydown', onKey)
    }
  }, [open])
  return (
    <div ref={root} className="relative">
      <button
        ref={button}
        aria-label={label}
        title={label}
        aria-haspopup="true"
        aria-expanded={open}
        onClick={() => setOpen(!open)}
        className={className}
      >
        {trigger}
      </button>
      {open && (
        <div className={`absolute top-full z-30 mt-2 rounded-lg ${align === 'end' ? 'right-0' : '-left-24'} border border-line bg-paper text-ink shadow-[0_10px_28px_-12px_rgba(20,54,63,0.45)]`}>
          {children(() => setOpen(false))}
        </div>
      )}
    </div>
  )
}

function Icon({ children, className = '' }: { children: React.ReactNode; className?: string }) {
  return (
    <svg aria-hidden viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth={1.6} strokeLinecap="round" strokeLinejoin="round" className={`size-3.5 shrink-0 ${className}`}>
      {children}
    </svg>
  )
}
const PlayIcon = () => <Icon><path d="M5 3.5v9l7-4.5z" fill="currentColor" /></Icon>
const PauseIcon = () => <Icon><path d="M5.5 3.5v9M10.5 3.5v9" strokeWidth={2.2} /></Icon>
const MoreIcon = () => <Icon><circle cx="3.5" cy="8" r=".6" fill="currentColor" /><circle cx="8" cy="8" r=".6" fill="currentColor" /><circle cx="12.5" cy="8" r=".6" fill="currentColor" /></Icon>
const SlidersIcon = () => <Icon><path d="M2.5 5h6M11.5 5h2M2.5 11h2M7.5 11h6" /><circle cx="10" cy="5" r="1.5" /><circle cx="6" cy="11" r="1.5" /></Icon>
const ChevronIcon = ({ className }: { className?: string }) => <Icon className={className}><path d="M4.5 6.5 8 10l3.5-3.5" /></Icon>
const CloseIcon = () => <Icon><path d="M4.5 4.5l7 7M11.5 4.5l-7 7" /></Icon>
const DownloadIcon = () => <Icon className="text-mute"><path d="M8 2.5v8M4.5 7 8 10.5 11.5 7M3 13.5h10" /></Icon>
const VideoIcon = () => <Icon><rect x="2" y="4" width="8.5" height="8" rx="1.5" /><path d="m10.5 7 3.5-2v6l-3.5-2" /></Icon>
const ExpandIcon = () => <Icon><path d="M9.5 2.5h4v4M6.5 13.5h-4v-4M13.5 2.5 9 7M2.5 13.5 7 9" /></Icon>
const ShrinkIcon = () => <Icon><path d="M13.5 6.5h-4v-4M2.5 9.5h4v4M9.5 6.5 14 2M6.5 9.5 2 14" /></Icon>

type SegmentedProps<T extends string | number> = {
  label: string
  labelClassName?: string
  value: T
  options: { value: T; label: string; disabled?: boolean }[]
  onChange: (value: T) => void
  disabled?: boolean
}

function Segmented<T extends string | number>({ label, labelClassName = '', value, options, onChange, disabled }: SegmentedProps<T>) {
  return (
    <div role="group" aria-label={label} className="flex items-center gap-2 text-xs">
      <span className={`shrink-0 text-mute ${labelClassName}`}>{label}</span>
      <div className="flex rounded-full bg-ink/6 p-0.5">
        {options.map((option) => (
          <button
            key={String(option.value)}
            disabled={disabled || option.disabled}
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

// Bar lengths offered for a correction; any other N/4, N/8 or N/16 can be typed in and still shows when saved.
const SIGNATURES: TimeSignature[] = [
  '2/4', '3/4', '4/4', '5/4', '6/4', '7/4',
  '3/8', '5/8', '6/8', '7/8', '9/8', '11/8', '12/8', '13/8', '15/8', '17/8', '19/8', '21/8',
  '5/16', '7/16', '9/16', '11/16', '13/16', '15/16',
  // Sections that move between bar lengths over one steady pulse.
  '7/8+6/8', '5/8+6/8', '7/8+8/8', '9/8+6/8', '3/4+4/4', '5/4+4/4', '7/4+4/4',
]

const CUSTOM = 'custom'

// Mirrors metre.parse_signatures on the server: up to four N/4, N/8 or N/16 (N 1–32) sharing one denominator.
function isSignature(value: string): value is TimeSignature {
  const parts = value.split('+')
  if (parts.length > 4 || !parts.every((part) => /^([1-9]|[12][0-9]|3[0-2])\/(4|8|16)$/.test(part))) return false
  return new Set(parts.map((part) => part.split('/')[1])).size === 1
}

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
