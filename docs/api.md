# groovemap HTTP API

Local, single-user. Served by `groovemap serve` on `127.0.0.1:8765`. All JSON. Errors are
`{"detail": "<message>"}` with a 4xx/5xx status.

## Types

```ts
type Options = {
  grid: 8 | 16            // default 16
  triplets: "auto" | "off" // default "auto"
  meter: number | null     // legacy N/4 regrouping; default null
  time_signature: string | null // explicit N/4, N/8 or N/16; N=1..15; default null
  odd_bars: "none" | "auto" | "detected" // default "none": regular bars; "auto": odd bars when
                            // the detections insist; "detected": follow every detected downbeat
  beat_shift: number        // move beat 1 later by this many beats, in steps of 0.25 (-11..11), default 0
  tempo_scale: 0.5 | 1 | 2  // count the beat half / as / twice as fast as detected, default 1
  sensitivity: number       // 0.5..5, default 1; 2 picks up hits half as strong
  voices: 1 | 2            // default 1
  sound_names: Record<string, SoundName> // sound group id -> name, default {} (see "Sound map" below)
}

type SoundName = Inst | "ignore"   // Inst: the 12 names of score.py's KIT

type SoundExample = {
  time: number              // seconds; one timeline for audio.wav, the drum stem and video.mp4
  start: number             // a clip to loop, time - 0.6 (never below 0)
  end: number               // time + 0.4
}

type SoundGroup = {
  id: string                // stable id of the sound, "kick-2", "extra-1"; see "Sound map"
  detected: "kick" | "snare" | "tom" | "hihat" | "cymbal" | null // detector's class; null: unexplained attacks
  count: number             // strokes in the group when the map was learned
  mixed: number             // of them struck together with another drum (within 30 ms)
  examples: SoundExample[]  // up to 6; isolated, typical, from different parts of the song
  similar: Record<string, number> // similarity 0..1 to each detected class's main sound in this song
  name: SoundName | null    // the saved name, if any
}

type SoundMap = { groups: SoundGroup[]; has_video: boolean }

type SongSummary = {
  slug: string
  title: string
  uploader: string
  duration: number          // seconds, 0 if unknown
  kind: "youtube" | "file"
  video_id: string          // "" for files
  transcribed: boolean      // score.json exists
  outdated: boolean         // transcribed, but by score-making code other than the server's (see below)
  bars: number | null
  tempo: number | null      // BPM
  updated_at: number        // unix seconds (mtime of score.json, else of meta.json)
  size_bytes: number        // everything in the song directory
  cache_bytes: number       // the part `DELETE .../cache` would free
}

type Song = SongSummary & {
  options: Options          // the options the current score was made with
  score: Score              // contents of score.json (see src/groovemap/score.py)
  has_drums: boolean        // a separated drum stem is available
  has_video: boolean        // video.mp4 was downloaded with the song
}

type SearchResult = { video_id: string; title: string; uploader: string; duration: number; url: string }

type Stage = {
  key: "fetch" | "separate" | "beats" | "onsets" | "score"
  status: "pending" | "running" | "done" | "cached"
  seconds: number | null    // elapsed when done
}

type Job = {
  id: string
  status: "queued" | "running" | "done" | "error"
  source: string
  title: string | null      // known once the source is resolved
  slug: string | null       // known once the song directory exists
  stages: Stage[]           // always all five, in order
  error: string | null
  created_at: number
}
```

## Endpoints

| Method | Path | Body / query | Returns |
|---|---|---|---|
| GET | `/api/songs` | | `SongSummary[]`, newest first |
| GET | `/api/songs/{slug}` | | `Song` |
| POST | `/api/songs/{slug}/score` | `Partial<Options>` | `Song` — re-quantizes from cached model outputs (no PDF), persists options. Songs transcribed before the tom and ghost-note refinement get their two extra caches made here, once (about 15 s) |
| POST | `/api/songs/{slug}/transcribe` | `Partial<Options>` (optional) | `Job` — (re)runs the pipeline on a song already in the library, without fetching |
| POST | `/api/songs/retranscribe` | `{scope?: "outdated" \| "all"}` (default `outdated`) | `Job[]` — queues `/transcribe` for every outdated song (or every song), each with its saved options; songs that already have a queued or running job are skipped |
| GET | `/api/songs/{slug}/sounds` | | `SoundMap` — the song's sound groups; learned from the caches on first request (about 0.2 s per song minute) and kept in `soundmap.json` after that; 409 when a refinement cache is missing |
| POST | `/api/songs/{slug}/sounds` | `{names: Record<string, SoundName>}` | `Song` — replaces the saved names (send `{}` to clear them) and re-quantizes with them, like `/score`; 409 before the first GET, 422 for an unknown group id or name |
| GET | `/api/songs/{slug}/musicxml` | | MusicXML text of the current score |
| GET | `/api/songs/{slug}/audio` | `?stem=mix\|drums` | WAV, supports HTTP Range |
| GET | `/api/songs/{slug}/video` | | `video.mp4` (H.264 + AAC, same timeline as `audio.wav`), supports HTTP Range; 404 if none |
| GET | `/api/songs/{slug}/download/{kind}` | kind = `musicxml` \| `midi` \| `pdf` | file as attachment; `pdf` is engraved on demand |
| DELETE | `/api/songs/{slug}/cache` | | `Song` — deletes regenerable model outputs (drum stem, beats, activations, snare loudness, preview, PDF); keeps audio, video and the score |
| DELETE | `/api/songs/{slug}` | | 204 — deletes the whole song directory |
| GET | `/api/search` | `?q=` | `SearchResult[]` (YouTube, up to 8) |
| POST | `/api/jobs` | `{source: string, options?: Partial<Options>}` | `Job` — source is a YouTube URL or an absolute local file path |
| POST | `/api/upload` | multipart `file`, optional `options` (JSON string) | `Job` — transcribe an uploaded audio file |
| GET | `/api/jobs` | | `Job[]` — queued, running, and the last few finished, newest first |
| GET | `/api/jobs/{id}` | | `Job` |

Outputs carry fingerprints of the code that made them (`src/groovemap/fingerprint.py`): the syntax
tree of a function and of everything in the package it reaches, so comments and unrelated code do
not count. Every score writes `code_version` (fingerprint of `pipeline.run`) into `options.json`;
every cached model output gets a stamp in `cache.json` (its producer's fingerprint, combined with
the stamps of the caches it was made from). `outdated` is true when either differs from the running
server's code, or a cache has no stamp (made before stamping). Jobs (`/jobs`, `/transcribe`,
`/retranscribe`) remake stale caches; `/score` never does, so a re-scored song stays `outdated`
until it is re-transcribed. A server whose source changed on disk after it started stamps what it
makes as unknown, since lazily imported modules may be newer than the rest.

Jobs run one at a time in submission order. A job for a song that is already transcribed re-runs with the
given options and reuses cached model outputs.

## Sound map: naming a song's sounds by ear

On an unusual kit the detector mislabels whole sounds, differently per recording (JD Beck at
NAMM: 5 of 42 notated snares found, 16 of the misses called kick). No automatic rule fixed this
without hurting ordinary kits, so the correction is opt-in and by ear (`src/groovemap/kit.py`):

- `GET .../sounds` groups the detected hits by detected class × timbre (a class splits into at
  most 4 groups, only while its strokes fall clearly in two along a spectral axis), plus
  `extra-*` groups of attacks in the stem that no detected hit explains (nothing within 50 ms).
  Typically 10–20 groups per song. Strokes struck together with another drum (`mixed`) have a
  mixed spectrum: they are assigned to the nearest group of their class but never learned from
  or shown as examples.
- `examples` are for the UI's listen/watch loop: isolated (no other stroke within 80 ms where
  possible), typical of the group, at least 3% of the song apart; play `start..end` of the
  drum stem (`/audio?stem=drums`) or of `video.mp4` at 0.25–0.5×, they share one timeline.
  `has_video` says whether the clip can be watched.
- `similar` is descriptive, not advice: on the JD Beck truth, naming a group after its nearest
  class was right 5 times and wrong 8 with the strictest rule, so the server offers no
  suggestion. Unnamed groups keep the detector's labels; unnamed `extra-*` groups add nothing.
- `POST .../sounds` (or `sound_names` in `/score`) replaces the whole name map and re-quantizes.
  Names apply after the fill toms and before tom sizing: a group named `tom` is sized by pitch
  like the detector's own toms (and is subject to the same stray rule), `tom_high`/`tom_low`
  keep their size; a renamed hit keeps its strength and keeps its ghost flag only within the
  snare family; an `extra-*` group named `snare`/`sidestick` gets ghost flags from its
  loudness against the group's typical stroke. `ignore` drops the group's hits.
- Names live in `options.json` (`sound_names`) and replay from `soundmap.json` without reading
  the audio again beyond one spectrogram, so re-quantizing with other options changed (grid,
  sensitivity, signature…) keeps them: every current hit goes to the nearest saved group of its
  class. With no names set the score is byte-identical to one made without a sound map, and no
  `soundmap.json` is written by `/score`.
- `soundmap.json` records fingerprints of the stem, activations and attacks it was learned
  from. "다시 채보" (`/transcribe`) that remakes none of those keeps the map and the names. When
  one is remade (a Demucs/ADTOF/attack code change, or `force`), the map is rebuilt on the
  next use; a rebuilt group inherits an old id when its centre is still the same sound
  (similarity ≥ 0.9), so its name survives, and new sounds get numbers no old group had. A
  name whose id no longer exists is ignored; `GET .../sounds` shows which names still apply.
  `DELETE .../cache` does not remove `soundmap.json`.

Measured on the JD Beck transcription truth (±50 ms, accepted regions only, micro-F1 over
kick/snare/tom/hat-cymbal), with an oracle naming each group by the majority truth label of
its members (at least 3 members in the truth regions): NAMM 0.556 → 0.737 (snare recall
0.12 → 0.90) with 4 renames, of which the first two give 73% of the gain; Zildjian full-band
0.895 → 0.904 (3), Zildjian drums-only, Two Shrimps and All Caps unchanged; VFJams unchanged
because its mislabelled strokes did not separate from their groups (purity 0.64 against
0.84–0.99 elsewhere). The ceiling is what a user who listens to every group can reach, 9–20
groups per song, and with no names set nothing changes.

## Correcting a time signature

`Options.time_signature` is a new nullable **string**, default `null`. Valid values
are `"N/4"`, `"N/8"`, or `"N/16"`, with integer **N = 1..15**, without whitespace or
leading zeroes. Invalid values return 422. For example:

```http
POST /api/songs/{slug}/score
Content-Type: application/json

{"time_signature":"7/8"}
```

The response is the usual `Song`, including the saved `options` and new `score`.
The same field is accepted by `/api/jobs`, `/transcribe`, and upload options.
Python: `Options(time_signature="7/8")`. CLI: `--time-signature 7/8`.

- An omitted field preserves the saved setting. `null` disables the explicit
  signature. To return fully to automatic mode, send
  `{"time_signature":null,"meter":null,"odd_bars":"none","beat_shift":0,"tempo_scale":1}`.
- The old `meter: 1..12 | null` remains supported **with its original behaviour**:
  count N tracked/repaired quarters and write N/4. An old `options.json` without
  `time_signature` loads it as `null`; no song folders are migrated in place.
- A non-null `time_signature` takes precedence over `meter` and `odd_bars`.
  It defines regular bars of the stated length. The legacy options stay saved
  and become active again when `time_signature` is cleared.
- Explicit correction estimates the denominator-note period from body-drum
  pattern repetition and onset comb alignment. It validates the tracker against
  that pulse and the actual onsets, subdividing/decimating compatible tracked
  intervals; otherwise it follows a new onset-derived grid. Local phase and
  period fitting follow gradual tempo drift. It does **not** just relabel
  tracked quarters as eighths or count seven old beats.
- `beat_shift` remains in quarter-unit steps (-11..11). With an explicit signature,
  its unit is **one denominator note**: `0.5` in 7/8 moves the downbeat later by
  one sixteenth; `1` by one eighth. In 7/16, `0.5` means one thirty-second. In
  6/8 this is still an eighth, not a dotted quarter. The grid is cyclically
  regrouped and a padded pickup bar preserves earlier notes; its first time can
  be negative. With automatic/legacy metre, the original quarter-beat meaning
  is unchanged.
- `tempo_scale` applies **after** pulse recovery. `2` halves denominator-note
  duration, doubles quarter BPM and roughly doubles the bar count; `0.5` does
  the inverse. It changes how the same recording is counted, not audio speed.
- `grid` and `triplets` still control notation. Explicit /4 reuses the existing
  quarter planner. /8 supports straight eighths, sixteenths, thirty-seconds and
  triplet sixteenths; /16 supports sixteenths and thirty-seconds. `grid:8` is the
  simplified mode (a /16 signature still needs its sixteenth-note units).
  Compound eighths are ordinary written eighths in groups of three, without
  artificial 3:2 tuplet brackets.

For an explicit signature, `/score` checks **all** required refinement/model
caches before doing work: `drums.htdemucs.wav`, `beats.json`,
`activations.drums-htdemucs.npy`, `activations.mix.npy`, `snare.htdemucs.npy`, and
`attacks.htdemucs.npy`. If one is absent, it returns **409**, leaving the score and
options alone. It never silently starts a missing slow model for this correction.
The user can explicitly request `/transcribe` to recreate missing caches. The
normal initial transcription job may of course run models. Existing complete
caches are read; the corrected pulse does not overwrite `beats.json`.

## Score JSON and frontend migration

The following fields are added or acquire a denominator-aware meaning. Everything
else, including `notes[].inst`, `velocity`, `ghost`, and `divisions`, is unchanged.
Fields marked optional can be absent in older/automatic scores.

```ts
type Bar = {
  index: number
  beats: number                    // signature NUMERATOR; always an integer
  denominator?: 4 | 8 | 16         // absent means 4
  beat_groups?: number[]           // group sizes in denominator notes; sums to beats
  beat_times: number[]             // denominator-note boundaries in audio seconds
                                   // length beats + 1; last entry is the bar end
  subdivisions: (number | [number, number])[] // one entry per denominator note
  tempo: number                    // QUARTER notes per minute, including in /8 and /16
  notes: {tick:number, inst:Inst, velocity:number, ghost?:boolean}[]
}

type ScorePulse = {
  source: "tracker" | "rebuilt"    // tracker means validated/resampled, not copied unchanged
  denominator_period: number      // median seconds per denominator note AFTER corrections
  alignment: number               // weighted onset-comb fit, 0..1, before scale/shift
  repetition: number              // normalized multi-lag correlation, before scale/shift
  tracker_compatible: boolean
  tempo_unit: "quarter"
  display_tempo_unit: "quarter" | "dotted-quarter"
}

type TimeSignatureSuggestion = {
  time_signature: string          // same N/D syntax as the option
  reasons: ("title_hint" | "erratic_downbeats" | "bar_period_mismatch")[]
  suggested_bar_seconds: number
  current_bar_seconds: number
  repetition: number              // candidate's normalized repeated-pattern evidence
  current_repetition: number      // correlation at the current bar period
  erratic_downbeat_fraction: number // tracked downbeat gaps inconsistent with four beats
}

// Additions to Score:
// pulse?: ScorePulse
// time_signature_suggestion?: TimeSignatureSuggestion
// Existing Score.tempo and SongSummary.tempo remain QUARTER BPM.
```

Frontend calculations (Q = `score.divisions ?? 12`, D = `bar.denominator ?? 4`):

| Purpose | Formula / field |
|---|---|
| Signature label | `${bar.beats}/${D}` |
| Ticks in a denominator note | `Q * 4 / D` |
| Total bar ticks | `bar.beats * Q * 4 / D` |
| Bar length in OSMD whole-note units | `bar.beats / D` |
| Time for note tick | Interpolate between the corresponding entries of `beat_times` using the denominator-note tick size |
| Thin pulse ticks | Every denominator note; times already supplied in `beat_times` |
| Strong beat/group ticks | Cumulative sums of `beat_groups`, multiplied by denominator-note tick size |
| Compound displayed BPM | `score.tempo * 2 / 3` for dotted-quarter marks |

A 7/8 bar has **84 ticks** at the current 24 ticks/quarter, seven subdivision
entries, eight `beat_times`, and `beat_groups: [2,2,3]`. Group starts are ticks
0, 24 and 48, with the final barline at 84. 6/8 uses `[3,3]`, 12/8 uses
`[3,3,3,3]`; those are dotted-quarter beats with three ordinary eighths each.
Other /8 and /16 metres use groups of two with a final three for odd numerators,
except multiples of three which use threes. /4 uses groups of one.

`subdivisions[i]` is the number of slots spanning denominator note i. For /4,
a pair still means the first-half and second-half grid densities from the legacy
planner (each number describes a full-quarter density). For explicit /8 the
values are 1, 2, 3 or 4; for /16 they are 1 or 2. `notes[].tick` always remains
quarter-based and does **not** change units when the denominator changes.

`beat_groups` is output metadata, not currently an input option. A future UI for
7/8 as **3+2+2** can add a validated `beat_groups: [3,2,2]` option whose positive
entries must sum to the numerator, then pass it to the Bar model. The MusicXML
writer already honours that order. Grouping alone would change beaming/strong
beat ticks, while `beat_shift` changes the first beat. Do not send that future
option today: unknown option fields are rejected.

MusicXML writes both `<beats>` and `<beat-type>` and beams by the group boundaries.
Compound metronome marks use `<beat-unit-dot/>`; `<sound tempo>` remains quarter
BPM, as required by the [MusicXML sound specification](https://www.w3.org/2021/06/musicxml40/musicxml-reference/elements/sound/).
MIDI also uses quarter-based tempo and ticks, writes the real time signature, and
emits local tempo events at each denominator-note boundary for explicit scores.
The preview uses the same piecewise time interpolation. MIDI time zero corresponds
to the score's first bar boundary; preview playback uses absolute audio times.

The minimal compatibility changes in `timeline.ts`, `ScoreView.tsx`, and
`SongView.tsx` update timing and labels. No new controls or restyling are included.

## Suggestions are opt-in corrections

A new automatic transcription/re-quantization can populate
`Song.score.time_signature_suggestion`. It never applies the suggested option,
changes the notes, or changes the automatically chosen bars. Existing score files
are not recomputed by GET; their suggestion is absent until re-quantized.
An explicit signature or legacy forced `meter` suppresses suggestions.

The check requires erratic downbeats and stronger repeated-pattern evidence for
another bar length; a signature in the title lowers the evidence threshold but
cannot produce a suggestion by itself. Without a title hint, ambiguous related
counting levels are suppressed. Correlation/alignment values are diagnostics,
**not calibrated confidence probabilities**. Scores where 4/4 fits retain their
previous JSON shape and bytes.

To offer `7/8로 바꿀까요?`, read the suggestion, and only after the user accepts,
POST `{"time_signature":"7/8"}`. Keep the beat-shift control available: audio
alone cannot always identify the intended first beat or distinguish half/double
counting. The initial search covers quarter equivalents of 60..250 BPM; local
tracking follows drift, and `tempo_scale` allows the adjacent counting levels.
Sparse, nonrepeating, rubato or changing-metre recordings can remain ambiguous.
