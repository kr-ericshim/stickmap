import { useRef, useState } from 'react'
import { api, type Job, type SearchResult } from '../api'
import { formatClock } from '../timeline'

type Props = {
  onJob: (job: Job) => void
  onError: (message: string) => void
}

export function SearchBar({ onJob, onError }: Props) {
  const fileRef = useRef<HTMLInputElement>(null)
  const [query, setQuery] = useState('')
  const [results, setResults] = useState<SearchResult[] | null>(null)
  const [searching, setSearching] = useState(false)

  const start = async (work: Promise<Job>) => {
    try {
      onJob(await work)
      setResults(null)
      setQuery('')
    } catch (error) {
      onError(error instanceof Error ? error.message : String(error))
    }
  }

  const submit = async (event: React.FormEvent) => {
    event.preventDefault()
    const text = query.trim()
    if (!text) return
    if (/^https?:\/\//.test(text)) return start(api.createJob(text))
    setSearching(true)
    try {
      setResults(await api.search(text))
    } catch (error) {
      onError(error instanceof Error ? error.message : String(error))
    } finally {
      setSearching(false)
    }
  }

  return (
    <div className="relative border-b border-line px-8 py-3">
      <form onSubmit={submit} className="flex items-center gap-3">
        <input
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="곡 이름을 검색하거나 YouTube 주소를 붙여넣으세요"
          aria-label="곡 검색"
          className="h-10 min-w-0 flex-1 rounded-full border border-line bg-paper px-5 text-sm placeholder:text-mute/70"
        />
        <button
          type="submit"
          disabled={searching}
          className="h-10 rounded-full bg-lacquer px-5 text-sm font-medium text-paper hover:bg-lacquer-soft disabled:opacity-60"
        >
          {searching ? '찾는 중…' : '찾기'}
        </button>
        <button
          type="button"
          onClick={() => fileRef.current?.click()}
          className="h-10 rounded-full border border-line px-4 text-sm text-mute hover:text-ink"
        >
          파일 올리기
        </button>
        <input
          ref={fileRef}
          type="file"
          accept="audio/*,video/*"
          className="hidden"
          onChange={(event) => {
            const file = event.target.files?.[0]
            event.target.value = ''
            if (file) void start(api.upload(file))
          }}
        />
      </form>

      {results && (
        <div className="absolute inset-x-8 top-full z-20 mt-1 overflow-hidden rounded-lg border border-line bg-paper shadow-xl">
          {results.length === 0 ? (
            <p className="px-5 py-4 text-sm text-mute">검색 결과가 없습니다. 검색어를 바꿔 보세요.</p>
          ) : (
            results.map((result) => (
              <button
                key={result.video_id}
                onClick={() => start(api.createJob(result.url))}
                className="flex w-full items-baseline gap-4 px-5 py-2.5 text-left hover:bg-room"
              >
                <span className="min-w-0 flex-1 truncate text-sm font-medium">{result.title}</span>
                <span className="max-w-48 shrink-0 truncate text-xs text-mute">{result.uploader}</span>
                <span className="w-10 shrink-0 text-right font-mono text-xs text-mute">
                  {formatClock(result.duration)}
                </span>
              </button>
            ))
          )}
          <button
            onClick={() => setResults(null)}
            className="w-full border-t border-line px-5 py-2 text-left text-xs text-mute hover:text-ink"
          >
            닫기
          </button>
        </div>
      )}
    </div>
  )
}
