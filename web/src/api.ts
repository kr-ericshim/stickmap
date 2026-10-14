// Types and calls for the local groovemap server. See docs/api.md.

// The five families the detector tells apart, and what was played within each.
// 'tom' is the mid tom and any tom of unknown size; 'cymbal' is a crash and any cymbal not known to be the ride.
export type Family = 'kick' | 'snare' | 'hihat' | 'tom' | 'cymbal'
export type Inst =
  | 'kick'
  | 'snare' | 'sidestick'
  | 'hihat' | 'hihat_open' | 'hihat_pedal'
  | 'tom_high' | 'tom' | 'tom_low'
  | 'ride' | 'ride_bell' | 'cymbal'

export const FAMILY: Record<Inst, Family> = {
  kick: 'kick',
  snare: 'snare', sidestick: 'snare',
  hihat: 'hihat', hihat_open: 'hihat', hihat_pedal: 'hihat',
  tom_high: 'tom', tom: 'tom', tom_low: 'tom',
  ride: 'cymbal', ride_bell: 'cymbal', cymbal: 'cymbal',
}

export type Note = {
  tick: number; inst: Inst; velocity: number; ghost?: boolean
  raw_time?: number // source onset in the recording; independent of the displayed tick
  ghost_status?: 'ghost' | 'normal' | 'unknown'
}
export type BeatGrid = { beats: number[]; downbeats: number[] } // seconds in the original recording

export type Bar = {
  index: number
  beats: number // numerator
  denominator?: 4 | 8 | 16 // absent in legacy scores: 4
  beat_groups?: number[] // denominator notes per beaming/counting group
  beat_times: number[]
  subdivisions: (number | [number, number])[]
  tempo: number
  notes: Note[]
}

// A '+' joins bar lengths that share one beat unit; each bar picks one of them.
export type TimeSignature = `${number}/${4 | 8 | 16}` | `${number}/${4 | 8 | 16}+${string}`

export type TimeSignatureSuggestion = {
  time_signature: TimeSignature
  reasons: ('title_hint' | 'erratic_downbeats' | 'bar_period_mismatch')[]
  suggested_bar_seconds: number
  current_bar_seconds: number
  repetition: number
  current_repetition: number
  erratic_downbeat_fraction: number
}

export type Score = {
  title: string
  subtitle: string
  bars: Bar[]
  grid: number
  tempo: number
  divisions?: number // ticks per quarter; scores written before this field use 12
  merged_strokes?: number
  pulse?: {
    source: 'tracker' | 'rebuilt'
    denominator_period: number
    alignment: number
    repetition: number
    tracker_compatible: boolean
    tempo_unit: 'quarter'
    display_tempo_unit: 'quarter' | 'dotted-quarter'
  }
  time_signature_suggestion?: TimeSignatureSuggestion
}

// What the user named a sound group: an instrument to write it as, or 'ignore' to leave it out.
export type SoundName = Inst | 'ignore'

export type MeterChange = { time: number; time_signature: TimeSignature }

export type Options = {
  grid: 8 | 16
  triplets: 'auto' | 'off'
  meter: number | null
  time_signature: TimeSignature | null
  // From each audio time on, bars use that signature instead of the automatic choice.
  meter_changes?: MeterChange[]
  odd_bars: 'none' | 'auto' | 'detected'
  beat_shift: number
  tempo_scale: 0.5 | 1 | 2
  sensitivity: number
  voices: 1 | 2
  ride_runs: boolean // evenly spaced cymbal runs are written as ride, lone cymbals as crash
  sound_names: Record<string, SoundName> // sound group id -> name; {} means the detector's labels stand
}

// One stroke of a sound group that is good to watch and hear: `time` is on the shared timeline of
// audio.wav, the drum stem and video.mp4; `start`..`end` is a clip to loop (time - 0.6 .. time + 0.4).
export type SoundExample = { time: number; start: number; end: number }

export type SoundGroup = {
  id: string // stable id of this sound, e.g. 'kick-2' or 'extra-1'; survives re-quantizing and retranscription
  detected: Family | null // what the detector called it; null for attacks it left unexplained ('extra-*')
  count: number // strokes in the group
  mixed: number // of them struck together with another drum (their sound is a mixture; never used as examples)
  examples: SoundExample[] // up to 6, isolated where possible and spread across the song
  similar: Partial<Record<Family, number>> // similarity (0..1) to each detected class's main sound: description, not advice
  name: SoundName | null // the user's saved name, if any
}

export type SoundMap = {
  groups: SoundGroup[]
  has_video: boolean // video.mp4 exists, so the examples can be watched as well as heard
}

export type SongSummary = {
  slug: string
  title: string
  uploader: string
  duration: number
  kind: 'youtube' | 'file'
  video_id: string
  transcribed: boolean
  outdated: boolean // made by older detector/notation code
  bars: number | null
  tempo: number | null
  updated_at: number
  size_bytes: number
  cache_bytes: number
}

export type Song = SongSummary & {
  options: Options
  score: Score
  has_drums: boolean
  has_mr: boolean
  has_video: boolean
}

export type SearchResult = {
  video_id: string
  title: string
  uploader: string
  duration: number
  url: string
}

export type StageKey = 'fetch' | 'separate' | 'beats' | 'onsets' | 'score'

export type Stage = {
  key: StageKey
  status: 'pending' | 'running' | 'done' | 'cached'
  seconds: number | null
}

export type Job = {
  id: string
  status: 'queued' | 'running' | 'done' | 'error' | 'cancelled'
  source: string
  title: string | null
  slug: string | null
  stages: Stage[]
  /** Short Korean message; the raw tool output is in error_detail. */
  error: string | null
  error_detail: string | null
  created_at: number
  started_at: number | null
  finished_at: number | null
  audio_seconds: number | null
  /** Seconds left, from this server's finished jobs; null when there is nothing honest to say. */
  eta_seconds: number | null
  /** 1 for the next queued job; null unless queued. */
  queue_position: number | null
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, init)
  if (!response.ok) {
    const body = await response.json().catch(() => null)
    throw new Error(body?.detail ?? `${response.status} ${response.statusText}`)
  }
  if (response.status === 204) return undefined as T
  const type = response.headers.get('content-type') ?? ''
  return (type.includes('json') ? response.json() : response.text()) as Promise<T>
}

const json = (method: string, body: unknown): RequestInit => ({
  method,
  headers: { 'content-type': 'application/json' },
  body: JSON.stringify(body),
})

const song = (slug: string) => `/api/songs/${encodeURIComponent(slug)}`

export const api = {
  songs: () => request<SongSummary[]>('/api/songs'),
  song: (slug: string) => request<Song>(song(slug)),
  requantize: (slug: string, options: Partial<Options>) =>
    request<Song>(`${song(slug)}/score`, json('POST', options)),
  transcribe: (slug: string) => request<Job>(`${song(slug)}/transcribe`, { method: 'POST' }),
  retranscribe: (scope: 'outdated' | 'all') => request<Job[]>('/api/songs/retranscribe', json('POST', { scope })),
  musicxml: (slug: string) => request<string>(`${song(slug)}/musicxml`),
  beats: (slug: string) => request<BeatGrid>(`${song(slug)}/beats`),
  sounds: (slug: string) => request<SoundMap>(`${song(slug)}/sounds`),
  nameSounds: (slug: string, names: Record<string, SoundName>) =>
    request<Song>(`${song(slug)}/sounds`, json('POST', { names })),
  clearCache: (slug: string) => request<Song>(`${song(slug)}/cache`, { method: 'DELETE' }),
  deleteSong: (slug: string) => request<void>(song(slug), { method: 'DELETE' }),
  search: (query: string) => request<SearchResult[]>(`/api/search?q=${encodeURIComponent(query)}`),
  jobs: () => request<Job[]>('/api/jobs'),
  createJob: (source: string) => request<Job>('/api/jobs', json('POST', { source })),
  cancelJob: (id: string) => request<Job>(`/api/jobs/${encodeURIComponent(id)}`, { method: 'DELETE' }),
  shutdown: (force: boolean) =>
    request<{ stopping: boolean; interrupted: number }>('/api/shutdown', json('POST', { force })),
  upload: (file: File) => {
    const form = new FormData()
    form.append('file', file)
    return request<Job>('/api/upload', { method: 'POST', body: form })
  },
  audioUrl: (slug: string, stem: 'mix' | 'drums' | 'nodrums') => `${song(slug)}/audio?stem=${stem}`,
  videoUrl: (slug: string) => `${song(slug)}/video`,
  downloadUrl: (slug: string, kind: 'pdf' | 'musicxml' | 'midi') => `${song(slug)}/download/${kind}`,
}
