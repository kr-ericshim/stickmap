import { useState } from 'react'
import { api, type Job, type SongSummary, type StageKey } from '../api'

const STAGE_LABELS: Record<StageKey, string> = {
  fetch: '음원 받기',
  separate: '드럼 분리',
  beats: '박자 추적',
  onsets: '타격 검출',
  score: '악보 만들기',
}

type Props = {
  songs: SongSummary[]
  jobs: Job[]
  selected: string | null
  onSelect: (slug: string | null) => void
  onRetranscribe: (scope: 'outdated' | 'all') => void
  onShutdown: () => void
}

export function Sidebar({ songs, jobs, selected, onSelect, onRetranscribe, onShutdown }: Props) {
  const active = jobs.filter((job) => job.status === 'queued' || job.status === 'running')
  const pending = new Set(active.map((job) => job.slug))
  const outdated = songs.filter((song) => song.outdated && !pending.has(song.slug)).length
  const failed = jobs.filter((job) => job.status === 'error').slice(0, 2)
  const [cancelling, setCancelling] = useState<Set<string>>(new Set())

  const cancel = (id: string) => {
    setCancelling((ids) => new Set(ids).add(id))
    // The job list poll removes it once the server has stopped it.
    api.cancelJob(id).catch((error: Error) => {
      setCancelling((ids) => {
        const next = new Set(ids)
        next.delete(id)
        return next
      })
      window.alert(error.message)
    })
  }

  return (
    <aside className="flex w-72 shrink-0 flex-col bg-lacquer text-paper">
      <button onClick={() => onSelect(null)} className="flex items-center gap-3 px-6 pt-6 pb-5 text-left">
        <Mark />
        <span className="font-display text-xl font-bold tracking-tight [font-stretch:85%]">stickmap</span>
      </button>

      {(active.length > 0 || failed.length > 0) && (
        <section className="mx-3 mb-3 space-y-3 rounded-lg bg-lacquer-soft p-3">
          {active.map((job) => (
            <div key={job.id}>
              <div className="flex items-start gap-2">
                <p className="min-w-0 flex-1 truncate text-sm font-medium">{job.title ?? job.source}</p>
                <button
                  onClick={() => cancel(job.id)}
                  disabled={cancelling.has(job.id)}
                  aria-label="취소"
                  title={job.status === 'running' ? '지금 단계가 끝나면 멈춥니다' : '대기열에서 뺍니다'}
                  className="-mt-0.5 shrink-0 rounded px-1 text-sm leading-5 text-paper/50 hover:bg-paper/10 hover:text-paper disabled:opacity-40"
                >
                  ×
                </button>
              </div>
              {job.eta_seconds != null && (
                <p className="mt-0.5 text-xs text-paper/50">약 {Math.max(1, Math.round(job.eta_seconds / 60))}분 남음</p>
              )}
              {cancelling.has(job.id) ? (
                <p className="mt-1 text-xs text-paper/60">취소하는 중…</p>
              ) : job.status === 'queued' ? (
                <p className="mt-1 text-xs text-paper/60">앞 작업이 끝나길 기다리는 중</p>
              ) : (
                <ol className="mt-2 space-y-1">
                  {job.stages.map((stage) => (
                    <li key={stage.key} className="flex items-center gap-2 text-xs">
                      <span
                        className={`size-1.5 rounded-full ${
                          stage.status === 'running'
                            ? 'animate-pulse bg-brass'
                            : stage.status === 'pending'
                              ? 'bg-paper/20'
                              : 'bg-paper/70'
                        }`}
                      />
                      <span className={stage.status === 'pending' ? 'text-paper/40' : ''}>
                        {STAGE_LABELS[stage.key]}
                      </span>
                      <span className="ml-auto font-mono text-paper/50">
                        {stage.status === 'cached' ? '저장됨' : stage.seconds != null ? `${Math.round(stage.seconds)}초` : ''}
                      </span>
                    </li>
                  ))}
                </ol>
              )}
            </div>
          ))}
          {failed.map((job) => (
            <div key={job.id} className="text-xs">
              <p className="truncate text-sm font-medium">{job.title ?? job.source}</p>
              <p className="mt-1 text-[#ffb4a3]" title={job.error_detail ?? undefined}>
                채보하지 못했습니다: {job.error}
              </p>
            </div>
          ))}
        </section>
      )}

      <div className="flex items-baseline justify-between gap-3 px-6 pt-2 pb-2">
        <h2 className="text-xs font-medium text-paper/50">채보한 곡</h2>
        {outdated > 0 ? (
          <button
            onClick={() => onRetranscribe('outdated')}
            title="이전 버전으로 만든 악보를 저장된 설정 그대로 다시 채보합니다"
            className="text-xs text-brass hover:text-paper"
          >
            이전 버전 {outdated}곡 다시 채보
          </button>
        ) : (
          songs.length > 0 && (
            <button
              onClick={() => {
                if (window.confirm(`${songs.length}곡을 모두 다시 채보할까요? 곡마다 저장된 설정을 그대로 쓰고, 캐시가 있으면 재사용합니다.`))
                  onRetranscribe('all')
              }}
              className="text-xs text-paper/50 hover:text-paper"
            >
              전체 다시 채보
            </button>
          )
        )}
      </div>
      <nav className="min-h-0 flex-1 overflow-y-auto px-3 pb-4">
        {songs.length === 0 && (
          <p className="px-3 py-2 text-sm text-paper/50">아직 없습니다. 오른쪽에서 곡을 검색해 보세요.</p>
        )}
        {songs.map((song) => (
          <button
            key={song.slug}
            onClick={() => onSelect(song.slug)}
            aria-current={song.slug === selected}
            className={`block w-full rounded-md px-3 py-2 text-left ${
              song.slug === selected ? 'bg-paper text-ink' : 'hover:bg-lacquer-soft'
            }`}
          >
            <span className="block truncate text-sm font-medium">{song.title}</span>
            <span className={`block truncate font-mono text-[11px] ${song.slug === selected ? 'text-mute' : 'text-paper/50'}`}>
              {song.transcribed && song.tempo != null
                ? `${Math.round(song.tempo)} BPM · ${song.bars}마디`
                : '채보 전'}
              {song.outdated && !pending.has(song.slug) && (
                <span className={song.slug === selected ? 'text-brass-deep' : 'text-brass'}> · 이전 버전</span>
              )}
            </span>
          </button>
        ))}
      </nav>
      <button
        onClick={onShutdown}
        title="stickmap 서버를 끕니다. 다시 쓰려면 바탕화면의 'stickmap 실행'을 여세요."
        className="flex items-center gap-2 border-t border-paper/10 px-6 py-3 text-left text-xs text-paper/50 hover:text-paper"
      >
        <svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round">
          <path d="M6 1.2v4.3" />
          <path d="M3.3 3a4.3 4.3 0 1 0 5.4 0" />
        </svg>
        엔진 끄기
      </button>
    </aside>
  )
}

/** Two bars of a backbeat, drawn the way the song map draws them. */
function Mark() {
  const hat = [0, 1, 2, 3, 4, 5, 6, 7]
  return (
    <svg width="30" height="22" viewBox="0 0 30 22" aria-hidden="true">
      {hat.map((i) => (
        <rect key={i} x={1 + i * 3.7} y="2" width="1.6" height="4" fill="#c9982f" />
      ))}
      {[2, 6].map((i) => (
        <rect key={i} x={1 + i * 3.7} y="9" width="1.6" height="4" fill="#e8edec" />
      ))}
      {[0, 3, 4].map((i) => (
        <rect key={i} x={1 + i * 3.7} y="16" width="1.6" height="4" fill="#e8edec" opacity="0.55" />
      ))}
    </svg>
  )
}
