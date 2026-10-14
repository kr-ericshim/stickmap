import { useCallback, useEffect, useRef, useState } from 'react'
import { api, type Job, type Song, type SongSummary } from './api'
import { SearchBar } from './components/SearchBar'
import { Sidebar } from './components/Sidebar'
import { SongView } from './components/SongView'
import { friendlyError, isConnectionError, OFFLINE_MESSAGE } from './errors'

const slugFromHash = () => decodeURIComponent(window.location.hash.slice(1)) || null

export default function App() {
  const [songs, setSongs] = useState<SongSummary[]>([])
  const [jobs, setJobs] = useState<Job[]>([])
  const [selected, setSelected] = useState<string | null>(slugFromHash)
  const [song, setSong] = useState<Song | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [stopped, setStopped] = useState(false)
  const started = useRef(new Set<string>()) // jobs begun in this tab
  const finished = useRef(new Set<string>())

  const report = useCallback((cause: unknown) => setError(friendlyError(cause)), [])
  const songRef = useRef<Song | null>(null)
  const errorRef = useRef<string | null>(null)
  useEffect(() => {
    songRef.current = song
    errorRef.current = error
  }, [song, error])

  const select = useCallback((slug: string | null) => {
    window.location.hash = slug ? encodeURIComponent(slug) : ''
    setSelected(slug)
  }, [])

  const refreshSongs = useCallback(() => {
    api.songs().then(setSongs).catch(report)
  }, [report])

  useEffect(() => {
    refreshSongs()
    const onHash = () => setSelected(slugFromHash())
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [refreshSongs])

  useEffect(() => {
    let cancelled = false
    const poll = async () => {
      if (stopped) return
      try {
        const latest = await api.jobs()
        if (cancelled) return
        setJobs(latest)
        // The engine answers again: drop the connection banner and catch up on the library.
        if (errorRef.current === OFFLINE_MESSAGE) {
          errorRef.current = null
          setError(null)
          refreshSongs()
        }
        for (const job of latest) {
          if (job.status !== 'done' || finished.current.has(job.id)) continue
          finished.current.add(job.id)
          refreshSongs()
          if (started.current.has(job.id) && job.slug) select(job.slug)
        }
      } catch (cause) {
        // The server may be restarting; keep polling and say so until it answers.
        if (!cancelled && isConnectionError(cause)) setError(OFFLINE_MESSAGE)
      }
    }
    void poll()
    const timer = window.setInterval(poll, 1200)
    return () => {
      cancelled = true
      window.clearInterval(timer)
    }
  }, [refreshSongs, select, stopped])

  const shutdown = async () => {
    const active = jobs.filter((job) => job.status === 'queued' || job.status === 'running').length
    const question = active
      ? `채보 작업 ${active}개가 아직 진행 중입니다. 지금 끄면 작업이 중단되고, 그 곡은 나중에 '채보 시작'으로 다시 돌려야 합니다. 그래도 끌까요?`
      : 'groovemap 엔진(서버)을 끌까요? 다시 쓰려면 바탕화면의 \'groovemap 실행\'을 여세요.'
    if (!window.confirm(question)) return
    try {
      await api.shutdown(active > 0)
      setStopped(true)
    } catch (cause) {
      report(cause)
    }
  }

  // Reload when this song's entry in the library changes: a finished job may have replaced the score.
  // Unrelated jobs finishing leave `updated_at` alone, so the open song (and its loop) stays as is.
  const version = songs.find((item) => item.slug === selected)?.updated_at
  useEffect(() => {
    if (!selected) {
      setSong(null)
      return
    }
    const loaded = songRef.current
    if (loaded && loaded.slug === selected && version !== undefined && loaded.updated_at === version) return
    let cancelled = false
    api
      .song(selected)
      .then((fresh) => {
        if (cancelled) return
        setSong((previous) =>
          previous && previous.slug === fresh.slug && previous.updated_at === fresh.updated_at ? previous : fresh,
        )
      })
      .catch((cause: unknown) => {
        if (cancelled) return
        setSong(null)
        report(cause)
      })
    return () => {
      cancelled = true
    }
  }, [selected, version, report])

  const current = song && song.slug === selected ? song : null
  const summary = songs.find((item) => item.slug === selected)

  if (stopped) return <Stopped />

  return (
    <div className="flex h-full">
      <Sidebar
        songs={songs}
        jobs={jobs}
        selected={selected}
        onSelect={select}
        onShutdown={shutdown}
        onRetranscribe={(scope) => {
          api
            .retranscribe(scope)
            .then((queued) => setJobs((previous) => [...queued.reverse(), ...previous]))
            .catch(report)
        }}
      />
      <main className="flex min-w-0 flex-1 flex-col">
        <SearchBar
          onJob={(job) => {
            started.current.add(job.id)
            setJobs((previous) => [job, ...previous])
          }}
          onError={report}
        />
        {error && (
          <div className="flex items-center gap-4 bg-alarm px-8 py-2 text-sm text-paper">
            <span className="min-w-0 flex-1">{error}</span>
            <button onClick={() => setError(null)} className="shrink-0 underline underline-offset-4">
              닫기
            </button>
          </div>
        )}
        {current?.transcribed ? (
          <SongView
            key={current.slug}
            song={current}
            onChange={(updated) => {
              setSong(updated)
              refreshSongs()
            }}
            onDeleted={() => {
              select(null)
              refreshSongs()
            }}
            onRetranscribe={() => {
              api
                .transcribe(current.slug)
                .then((job) => {
                  started.current.add(job.id)
                  setJobs((previous) => [job, ...previous])
                })
                .catch(report)
            }}
            onError={report}
          />
        ) : (
          <Empty
            hasSongs={songs.length > 0}
            waiting={selected && summary && !summary.transcribed ? summary : null}
            running={jobs.some((job) => job.slug === selected && (job.status === 'queued' || job.status === 'running'))}
            onTranscribe={(slug) => {
              api
                .transcribe(slug)
                .then((job) => {
                  started.current.add(job.id)
                  setJobs((previous) => [job, ...previous])
                })
                .catch(report)
            }}
          />
        )}
      </main>
    </div>
  )
}

function Stopped() {
  return (
    <div className="flex h-full items-center bg-lacquer px-16 text-paper">
      <div className="max-w-xl">
        <p className="font-display text-5xl leading-[1.05] font-bold tracking-tight [font-stretch:80%]">엔진을 껐습니다.</p>
        <p className="mt-5 text-base text-paper/60">
          이 탭은 닫아도 됩니다. 다시 쓰려면 바탕화면의 ‘groovemap 실행’을 여세요.
        </p>
      </div>
    </div>
  )
}

type EmptyProps = {
  hasSongs: boolean
  waiting: SongSummary | null
  running: boolean
  onTranscribe: (slug: string) => void
}

function Empty({ hasSongs, waiting, running, onTranscribe }: EmptyProps) {
  return (
    <div className="flex flex-1 items-center px-16">
      <div className="max-w-xl">
        <p className="font-display text-5xl leading-[1.05] font-bold tracking-tight [font-stretch:80%]">
          {waiting ? (running ? '채보하는 중입니다.' : '이 곡은 아직 채보 전입니다.') : '어떤 곡을 받아 적을까요?'}
        </p>
        {waiting && !running && (
          <button
            onClick={() => onTranscribe(waiting.slug)}
            className="mt-6 h-11 rounded-full bg-lacquer px-6 font-medium text-paper hover:bg-lacquer-soft"
          >
            채보 시작
          </button>
        )}
        <p className="mt-5 text-base text-mute">
          {waiting
            ? running
              ? '왼쪽에서 단계별 진행 상황을 볼 수 있습니다. 끝나면 악보가 바로 열립니다.'
              : '음원은 이미 받아 두었습니다. 분석이 끝나면 여기에 악보가 나옵니다. 걸리는 시간은 왼쪽 작업 목록에 표시됩니다.'
            : hasSongs
              ? '왼쪽에서 곡을 고르거나, 위에서 새 곡을 검색하세요. 걸리는 시간은 왼쪽 작업 목록에 표시됩니다.'
              : '위 검색창에 곡 이름이나 YouTube 주소를 넣으세요. 드럼을 분리해 악보 초안을 만들고, 영상에 맞춰 커서가 따라갑니다.'}
        </p>
      </div>
    </div>
  )
}
