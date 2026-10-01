import { useCallback, useEffect, useRef, useState } from 'react'
import { api, type Job, type Song, type SongSummary } from './api'
import { SearchBar } from './components/SearchBar'
import { Sidebar } from './components/Sidebar'
import { SongView } from './components/SongView'

const slugFromHash = () => decodeURIComponent(window.location.hash.slice(1)) || null

export default function App() {
  const [songs, setSongs] = useState<SongSummary[]>([])
  const [jobs, setJobs] = useState<Job[]>([])
  const [selected, setSelected] = useState<string | null>(slugFromHash)
  const [song, setSong] = useState<Song | null>(null)
  const [error, setError] = useState<string | null>(null)
  const started = useRef(new Set<string>()) // jobs begun in this tab
  const finished = useRef(new Set<string>())

  const select = useCallback((slug: string | null) => {
    window.location.hash = slug ? encodeURIComponent(slug) : ''
    setSelected(slug)
  }, [])

  const refreshSongs = useCallback(() => {
    api.songs().then(setSongs).catch((cause: Error) => setError(cause.message))
  }, [])

  useEffect(() => {
    refreshSongs()
    const onHash = () => setSelected(slugFromHash())
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [refreshSongs])

  useEffect(() => {
    let cancelled = false
    const poll = async () => {
      try {
        const latest = await api.jobs()
        if (cancelled) return
        setJobs(latest)
        for (const job of latest) {
          if (job.status !== 'done' || finished.current.has(job.id)) continue
          finished.current.add(job.id)
          refreshSongs()
          if (started.current.has(job.id) && job.slug) select(job.slug)
        }
      } catch {
        // The server may be restarting; the next poll will tell.
      }
    }
    void poll()
    const timer = window.setInterval(poll, 1200)
    return () => {
      cancelled = true
      window.clearInterval(timer)
    }
  }, [refreshSongs, select])

  useEffect(() => {
    if (!selected) {
      setSong(null)
      return
    }
    let cancelled = false
    api
      .song(selected)
      .then((loaded) => !cancelled && setSong(loaded))
      .catch((cause: Error) => {
        if (cancelled) return
        setSong(null)
        setError(cause.message)
      })
    return () => {
      cancelled = true
    }
    // Reload when the library changes: a finished job may have replaced the score.
  }, [selected, songs])

  const current = song && song.slug === selected ? song : null
  const summary = songs.find((item) => item.slug === selected)

  return (
    <div className="flex h-full">
      <Sidebar
        songs={songs}
        jobs={jobs}
        selected={selected}
        onSelect={select}
        onRetranscribe={(scope) => {
          api
            .retranscribe(scope)
            .then((queued) => setJobs((previous) => [...queued.reverse(), ...previous]))
            .catch((cause: Error) => setError(cause.message))
        }}
      />
      <main className="flex min-w-0 flex-1 flex-col">
        <SearchBar
          onJob={(job) => {
            started.current.add(job.id)
            setJobs((previous) => [job, ...previous])
          }}
          onError={setError}
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
                .catch((cause: Error) => setError(cause.message))
            }}
            onError={setError}
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
                .catch((cause: Error) => setError(cause.message))
            }}
          />
        )}
      </main>
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
              : '음원은 이미 받아 두었습니다. 분석만 하면 되고 1분쯤 걸립니다.'
            : hasSongs
              ? '왼쪽에서 곡을 고르거나, 위에서 새 곡을 검색하세요. 곡 하나에 1분쯤 걸립니다.'
              : '위 검색창에 곡 이름이나 YouTube 주소를 넣으세요. 드럼을 분리해 악보 초안을 만들고, 영상에 맞춰 커서가 따라갑니다.'}
        </p>
      </div>
    </div>
  )
}
