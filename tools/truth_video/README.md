# Ground truth from transcription videos

## Instrument-band phase check and audio sanity (2026-10-01, second pass)

Regular grooves let a whole-beat or subdivision shift agree with class-agnostic
attacks almost as well as the true alignment; the fitted/held-out/shifted-control
gates then accept truth that is one beat early (atthedrum p7J0T8SLumc had 35 of
144 notated kicks and 0 of 72 snares on the detector's kicks/snares although
142/72 were detected). YouTubers edit cursor and colour timing by hand, so the
prior can be a beat off for a whole video and every system inherits the alias
through the shared seed.

`bands.py` computes band-limited onset strength from the separated drum stem
(`drums.htdemucs.wav`, or the Opus after compaction): STFT log-magnitude flux
summed over 35–110 Hz (kick), 250–2500 Hz (snare) and 5–16 kHz (hats), each
scaled by its 99th percentile, plus peak envelopes and ±40 ms sliding maxima.
It is cached as `songs/<slug>/onsets.bands.htdemucs.npy` (kept by compaction).
The detector's drum labels are never consulted; the notation is trusted for
WHICH drum plays and the audio decides WHEN:

- A notated kick agrees when a low-band peak (>0.3) lies within ±40 ms and the
  low flux is at least 0.8× the mid flux there; a notated snare agrees when a
  mid-band peak (>0.4) lies within ±40 ms and the mid flux is at least 0.8× the
  low flux. Unisons pass both; a kick's click alone or a hat alone passes
  neither rule for a snare.
- `align.fit_prior` adds the pooled kick/snare agreement to its objective, so
  the video-level seed and per-system offsets prefer the phase where notated
  kicks sit on low onsets, among offsets that tie on attacks alone.
- After warping, `bands.phase_check` compares the fitted times with every shift
  in ±0.09..0.45 s that still lands ≥75% (and ≥50%) as many moments on attacks
  (a rhythmic alias). Instruments are judged separately, because pooled numbers
  dilute the instrument that carries the evidence (four-on-the-floor kicks agree
  with themselves a beat later; the snares do not). A shift is beaten when some
  instrument with ≥4 moments gains ≥15 points and no instrument trails by more
  than 15 points and two moments without another instrument compensating.
- A system is `usable` only if it has ≥6 notated kick/snare moments with ≥60%
  agreement, no alias contradicts it, and it beats every alias — or, when its
  own pattern is ambiguous or short, the song-level check backs it: the fitted
  times of all individually established systems of the same video (second
  pass; raw cursor priors before any system is established). Backing never
  rescues a system whose own kicks/snares prefer a shift. New reasons: `phase
  alias: ...`, `notated kicks/snares off their onset bands ...`, `fewer than 6
  notated kick/snare moments ...`, `instrument-band phase check unavailable (no
  drum stem)`. They apply to the scoring tier as well; no strict/scoring timing
  threshold was loosened. `alignment.json` records `phase` per system,
  `band_evidence` and the phase thresholds; `QUALITY.md` shows the agreement
  and the best alias per system.

Measured on the 49 read harvest songs (re-alignment of the existing
`strip.json`, detector used for reporting only): every song flagged by the
detector as phase-shifted (atthedrum p7J0T8SLumc, BL3H5AuJies, 0MiLvEDSN60,
Krigj_C0jug, YbMTrzQJTm4; yeol wXy7poDRtIA; mydrumsheet DKz7y71fYuE; jilju
m9n2i4kV21Q) is corrected; see `truth/harvest/VALIDATION.md` for the before/
after table. Tests: `tests/test_bands.py` (synthesized kick/snare/hat audio,
forced alias rejection, song-level backing, missing stem).

Channel wave 4 (taktak, drumclass, drumscribe, nakamura, swartz enabled with
audited kick/snare/tom positions; moyin, dama, leminhtri, dailychops, olliewoods,
christal stay disabled with recorded reasons) is documented in
`truth/harvest/VALIDATION.md`; its blind crops are `truth/harvest/validation/
audit-wave4*/` (rendered with the detected staff lines, labels in `manual.json`).

`sanity.py` adds detector-free per-song flags written to
`truth/audio-sanity.json`, `result.json` (`audio_flags`), the manifest
(`truth_audio_flags`) and `CHANNEL_RESULTS.md`: `drum stem nearly silent`
(stem RMS < −40 dBFS or < 5% of 100 ms blocks above −40 dBFS; the two excluded
drumless tracks measure −46/−50 dBFS with 1–2% active, real songs ≥ −30 dBFS
and ≥ 15%) and `audio attacks far fewer than notated moments` (attacks inside
the notated systems' intervals < 50% of their moments). Flags are diagnostic;
a silent stem also fails the phase check on its own.

## Offline harvest batches (2026-10-01)

Use the existing environment from the repository root. No download, separation,
detector inference, or connection to the application server is made:

```bash
# Process eligible fetched songs, then exit.
.venv/bin/python tools/truth_video/batch.py

# Keep watching while harvest.py appends new completed fetches.
.venv/bin/python tools/truth_video/batch.py --watch --poll 60

# Bounded run or selected channels.
.venv/bin/python tools/truth_video/batch.py --only atthedrum yeol --limit 20

# Reconsider failures after fixing a legend/reader; refresh changed read exports.
.venv/bin/python tools/truth_video/batch.py --retry-unreadable --refresh-read

# Explicit inspection of a disabled channel; its gate still prevents export.
.venv/bin/python tools/truth_video/batch.py --validate --only drumscribe --per-channel 3

.venv/bin/python tools/truth_video/batch.py --report

# Separate, destructive disk compaction: run only after reviewing the exports.
# The batch reader NEVER invokes this command.
.venv/bin/python tools/harvest/harvest.py compact
```

The latest per-channel yield and label counts are in
`truth/harvest/CHANNEL_RESULTS.md` and `channel-results.json`. Validation evidence,
native crops, blind sheets, manual decisions and test logs are in
`truth/harvest/validation/`. Original legacy and JD exports are not rebuilt.

Every configured channel has its own JSON in `legends/`. `enabled` controls
production scheduling; each rule additionally needs `verified: true` and a
minimum glyph score to export. The numbers are heuristic scores, not calibrated
probabilities. The manifest selects channel identity. Image analysis can locate
an unknown score layout but cannot establish an unknown channel's notation key;
that channel stays disabled. Candidate rules in disabled profiles are explicitly
unverified, not asserted legends. Channel name alone does not establish a future
video's readability: every staff and timing system still passes the gates.

`layouts.py` / `harvest_scan.py` locate a light score region and support bottom
strips, top notation and multiple page rows, with blue/green/pink/lavender/orange
playheads. Yellow bar shading is not treated as a cursor. Light-paper margins
retain high o/+ marks; row crops, staff position limits and the observed cursor
span suppress video insets, lyrics/stickings and unvisited NEXT material. Missing
or contaminated geometry fails closed. Inverted transparent notation still
requires separate channel validation. Scanning retains at most 24 source crops
per segment and emits compressed, modest-sized staff/overlay PNGs.
Cursor wraparound also splits a segment when the displayed bars look almost
identical; repeated pages must not be blended into one median staff. An inspected
`staff_y_min_fraction` can restrict a channel's search to the bottom score region
(used for Jilju's white-room footage), while preserving the native pixels.

`harvest_scroll.py` offers light-paper scrolling reconstruction for layouts that
request a motion probe. It uses image-only translation correlation, ambiguity
checks and temporally supported ink in four-second blocks. Discontinuous or
periodic/ambiguous matches reject the block. **Production scrolling exports are
also blocked until that channel's `scrolling_verified` flag is validated.** The
legacy VFJams path remains separate. Generic motion tests do not constitute a
real-video/channel validation.

`color_timing.py` measures head-local chromatic transitions, requiring an
achromatic baseline, stable local ink shape and two consecutive changed frames. It excludes pale cursor
backgrounds and repeated transitions. A visual timestamp retains its source
frame and one-frame threshold-crossing quantization (about 33 ms at 30 fps).
The field named `uncertainty_seconds` records this quantization only: gradual
color fades can begin before the threshold edge, and rendering/audio lag remains
separate. It is not a total onset-error bound or independent acoustic truth.
Sufficient monotone color anchors improve the timing
prior; the existing class-agnostic audio alignment, held-out checks and shifted
controls remain mandatory. No strict/scoring thresholds were loosened.

The 7–8 px decision is conservative: these harvest profiles **retain the 8 px
native gate**. Inspection upscaling does not create extra source evidence. Real
small-staff fixtures are tested; no channel-specific small-glyph calibration has
yet justified production export below that gate. Geometry may be supported while
the native glyphs or channel legend remain unverified.

Each completed attempt writes `songs/<slug>/truth/` with `truth.json`,
`truth.scoring.json`, `alignment.json`, `alignment.scoring.json`, `QUALITY.md`,
`strip.json`, `scan.json`, `legend.json`, `label-coverage.json`, `result.json`,
source staff crops and numbered overlays under `check/`. Color-enabled profiles
also write `color-timing.json`. Strict and scoring outputs are separate; scoring
includes strict. `read` means at least one scoring-accepted system and at least
one exported verified note. Attempted failures/no accepted truth become
`unreadable` with a reason; skipped channels/songs stay `fetched`.

This is **partial positive truth**, not exhaustive silence/negative labels.
Unknown cymbals are retained in the inspectable strip but excluded from timed
export. `label-coverage.json` records unverified heads and verified notation
rules. False ghost/accent/flam/roll flags are not verified technique negatives;
`flag_verification` identifies enabled positive readers, not proven absence.
The MY DRUM SHEET ghost reader uses a calibrated complete parenthesized-head
glyph; its positive check is spatial within a supplied frame and does not imply
independent audio validation. Unmatched upper marks
withhold closed-hat labels. Generic full-family detector precision over a partial
interval can incorrectly penalize an unlabelled instrument; use the coverage
mask or separately audit exhaustive systems before making that comparison.

Each song has an OS advisory lock shared with compaction. Workers run in isolated
attempt folders with a configurable timeout (default 1200 s), input/code/legend/
template fingerprints and a final completion record. Completed folders are
published by rename, prior outputs move to `truth-history/`, and interrupted
work remains inspectable. A matching completed export repairs a missing journal
status without re-running the reader. Per-song errors do not stop other songs.
Dataset consumers should follow the manifest to each canonical `songs/<slug>/truth/`
folder, not recursively ingest `truth-history/`, failed candidate exports or the
validation fixtures. Those are recovery/diagnostic evidence, not extra examples.
Journal updates append; the journal is never rewritten. A partial final line is
ignored by readers and cannot be appended onto until the writer completes or an
owner repairs it explicitly. The already-running fetcher may continue using its
old append implementation; completed fetch records remain the admission signal.

Compaction retains `truth/`, small overlays, caches, metadata, history and
interrupted attempt directories. It converts the drum WAV to Opus and then
removes the source video, full-mix WAV and WAV stem only for `read` songs by
default, skipping a locked song. It never descends into truth directories.
Compaction was tested on disposable fixtures, not run on the harvested sources
during reader validation. Do not use `compact --force` for unread material you
still need to inspect. Opus is lossy; preserve the WAV elsewhere first if later
training requires lossless source audio.

Additional checks and audit tooling:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/truth_video/tests/test_harvest.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/truth_video/tests/test_batch_safety.py
.venv/bin/python tools/truth_video/harvest_audit/reader.py prepare truth/harvest
.venv/bin/python tools/truth_video/harvest_audit/reader.py report truth/harvest
.venv/bin/python tools/truth_video/harvest_audit/reader.py recheck truth/harvest
```

`prepare` refuses to overwrite an existing audit. Enter labels by viewing the
blind crops and source overlays; `report` compares those decisions and `recheck`
separately reports retained labels after fixes. A recheck is not a fresh audit.
The harvest sample's results must not inherit the old 190-head audit statistic.

The reader exports only systems that pass staff and timing checks. It uses the
video cursor as a first estimate and the provided **class-agnostic** attack times
as audio anchors. It never uses detector drum labels to choose an alignment.
A fitted timestamp is not independent proof: held-out predictions and shifted
controls are saved alongside fitted agreement, and doubtful systems are dropped.

Run from the repository root with the existing environment:

```bash
.venv/bin/python tools/truth_video/build.py songs/<slug> truth/<name>
.venv/bin/python tools/truth_video/build.py songs/<slug> truth/<name> --legend tools/truth_video/legends/drumtabs.json --score
.venv/bin/python tools/truth_video/build.py songs/<slug> truth/<name> --reuse-scan
.venv/bin/python tools/truth_video/evaluate.py songs/<slug> truth/<name>/strip.json
.venv/bin/python tools/truth_video/evaluate.py songs/<slug> truth/<name>/strip.json --alignment-only
.venv/bin/python tools/truth_video/evaluate.py songs/<slug> truth/<name>/strip.json --reuse-truth
.venv/bin/python tools/truth_video/report.py truth
.venv/bin/python tools/truth_video/test_reader.py
```

The build needs `video.mp4` and `attacks.htdemucs.npy`. Scoring additionally needs
the existing detector activation/snare caches and drum stem. Building truth does
not run a separator, download a model, use the network, or contact the application
server. `numpy`, `scipy`, and `cv2` implement the reader; scoring retains the
project's existing detector dependencies. Do not use `uv run` in this sandbox.

## Files and passes

- `scan.py` finds the paper strip, separates score changes from highlighting,
  records cursor centers per frame, and writes a median image per staff. It
  detects blue or faint red cursors. Colored highlighting is removed by the
  brightest channel, rather than deleting the entire note-bearing column.
  Vertical closing reconnects a cursor interrupted by staff ink.
- `heads.py` finds five lines, measures their spacing, reads filled, cross,
  circled-cross and diamond heads, and checks parentheses and grace glyphs.
  Shapes do not hard-code an instrument name. It uses native-scale morphology
  plus normalized glyph correlation. Scores are **heuristic confidence values**,
  not calibrated probabilities.
- `strip.py --legend <json>` applies a layout, reads unambiguous marks and writes
  `strip.json`. KBdrums is the default. Every system gets
  `check/system_NNN.png`: five staff lines, numbered heads and a readable label /
  position / confidence table. `G/A/F/R` mean ghost/accent/flam/roll.
- `align.py` groups heads in the same notation column into one moment. Sweeping
  cursors are interpolated; stationary box positions are clustered and their
  first arrival is used directly. Interpolating between successive box stops
  would place a note before the box actually highlights it. Without a cursor,
  the on-screen interval and x extent supply the initial mapping.
- The alignment first fits a lag, using other systems where possible to resist
  rhythmic phase aliases. An ordered dynamic program assigns attacks one-to-one,
  allowing missing notes and extra attacks. Median-filtered offset anchors form
  a piecewise-linear warp; nearby assigned attacks refine final times. Any
  non-increasing result is rejected rather than silently reordered.
- Three interleaved held-out folds refit using only training moments. A held-out
  moment does not supply an offset-fit observation or warp anchor. These tests
  still use the same performance's attack stream; they are a prediction check,
  not external ground truth. Shifts of ±230 and ±370 ms measure accidental
  agreement in dense attacks.
- `evaluate.py` builds `truth.json` and `alignment.json`, then prints detector
  precision/recall, ghost recall, confusion rows, ghost agreement, tom-size
  correspondence and timing spread. `--alignment-only` needs no detector.
  `--reuse-truth` scores a saved export without fitting again.

`scan.json` version 2 retains `fps`, `cursor`, `systems`, `top` and `bottom`, and
adds records with start/end frames, page row, native staff spacing and cursor
observations. `strip.py` can still read the old scan format. `strip.json` remains
a list of systems with `cursor_x`, `cursor_t` and `notes`; new fields include the
interval, cursor model, legend name and staff checks.

`truth.json` remains a time-sorted list of note objects. Each includes `inst`,
`time`, `system`, `x`, `y`, `pos`, `shape`, `shape_confidence`, `confidence`,
`ghost`, `accent`, `flam`, `roll`, and mark evidence. Hats also have
`articulation`. `timing_confidence` is that system's held-out 40 ms fraction,
not a probability for an individual note. Small grace heads annotate the main
note's flam flag; the reader does not invent a separate grace onset time.

`alignment.json` contains every system, including rejected systems: initial,
calibrated, fitted and held-out 20/40 ms fractions; x/time knots; lag; control
scores; monotonicity; acceptance; reasons; and accepted scoring intervals.
An empty truth export is a valid outcome, not a failed or fabricated alignment.
`QUALITY.md` links every system overlay and lists the exact rejection reasons.
`SUMMARY.md` and `summary.json` contain aggregate timing, label counts, duration
and the manual reading audit. Duration is the union of accepted first-to-last
moment spans, excluding rejected gaps and lead-in/outro material.

## Acceptance

A system needs a valid staff, at least six moments, and strictly increasing times.
The fixed timing gates are:

| Measure | 20 ms | 40 ms |
|---|---:|---:|
| Fitted one-to-one attack agreement | ≥85% | ≥95% |
| Held-out prediction agreement | ≥55% | ≥80% |
| No-cursor held-out prediction agreement | ≥70% | ≥90% |

Held-out agreement must additionally exceed the shifted control by at least
15 percentage points at either tolerance. Dense streams can saturate the 40 ms
control, so the 20 ms contrast can establish evidence in that case. High fitted
agreement by itself never qualifies a system. The thresholds are intentionally
conservative; they do not make all downloaded videos usable.

Staff checks record five-line support, spacing residuals, the paired percussion
clef bars at the expected left margin, bar-line candidates and kick/snare
occupancy. Uneven/weak lines, an unconfirmed clef, or native spacing below 8 px
reject a system. Low kick/snare occupancy is a warning, since real tom solos
and side-stick grooves can legitimately lack those heads. Bar candidates can
include stems spanning the staff, so occupancy is a diagnostic, not proof of a
correct bar count or a rhythm parser. Inspect the overlay before treating an
unfamiliar layout as validated.

## Legends and marks

`legends/kbdrums.json` and `legends/drumtabs.json` are small, separate layout
maps. Positions count half-spaces upward from the bottom line (`0`): kick `1`,
snare `5`, cross in snare space = side stick, hat `9`, ride `8`, and crash crosses
at `10`/`11`. Crosses below the staff are pedal hats; diamond heads in the supported
upper positions are ride bells. Other cymbal positions stay `cymbal_unknown`.
Round filled tom positions retain `pos`; a floor/rack size is not inferred.

Each rule maps `shape:position` to `inst`, optional `articulation` and a mapping
confidence. For a verified alternate layout, change the specific rule, e.g.
`"filled:-1": {"inst": "tom", "confidence": 0.95}` for a floor tom below the
staff. Do not transfer that convention to an unrelated channel without looking
at its notation. The two supplied layouts share the conventions actually seen
in these downloads, and keep different left margins; the JSON is not a claim
that every future video from either channel uses the same notation.

Unmarked hats default to closed, circled hats are open, and local `o` / `+` marks
override articulation. State is not propagated through later unmarked notes.
Accent ownership is conservative and restricted by distance. Ghost detection
currently validates parentheses around filled heads; non-filled ghost heads,
triangle cymbals and all buzz/slash variations are not comprehensively supported.
One clear three-slash cymbal roll is read, but a tied roll is not expanded into
unseen individual strokes. A false flag means “not confidently read,” not a
verified absence of the musical technique.

Family evaluation maps side stick to snare, tom sizes to tom, and ride/bell/crash/
unknown-position cymbals to cymbal. Multiple fine cymbal labels at one moment
produce one family reference. Kick/snare/tom reporting is retained; hihat and
cymbal P/R are added. Fine labels/articulations and accent/flam/roll are printed
as **not yet detected**, not scored as failed detector classes. Predictions in
rejected gaps are never charged as false positives.

## Full pages and resolution

One page can contain several detected staves. Each gets its own image and row
number. The cursor's vertical span assigns observations to the active row;
visited rows use their active interval, and unvisited rows cannot qualify merely
because the page was visible. Native low-resolution rows are enlarged for
inspection but retain the rejection flag. The CJ Thompson 360×640 download has
about 4 px staff spacing. At the same framing, a genuine 720×1280 source reaches
the 8 px minimum; 1080×1920 is recommended for about 12 px spacing. Scaling the
existing pixels is not a substitute for a better source.

The tests include a two-staff page video with a faint red cursor moving down the
page and the real upscaled CJ artifact's rejection. Other checks cover one-to-one
matching, box arrival, chord grouping, a known nonlinear warp, no-cursor priors,
dense-attack rejection, malformed staff spacing, empty truth, family mappings
and exclusion of dropped gaps. Tests use `truth/` for temporary fixtures.

## Reading audit

`audit.py truth` creates prediction-blind crops with deterministic stratified
sampling. **Do not run it over an existing completed audit** unless starting a
new audit: it replaces the sample manifest and crop sheets. Labels must be
entered by actually viewing the crops and full overlays; the script does not
claim to verify its own predictions. `audit.py truth --report` compares an
existing `manual.json` with the sample. `report.py truth` additionally reports
flag rechecks against the current reader.

The completed run inspected 190 heads, including explicit open marks, pedal
hats, ride bells and side sticks. Its raw source coordinates, visual decisions,
37 flag positives, enlarged ambiguity image and before/after flag counts are
in `truth/audit/`. The statistic measures label correctness among detected heads;
it is not whole-score recall, calibrated confidence, a listening audit, or an
independent human review. Template development sources were excluded from the
head sample. Regression rechecks after fixes are identified as such.

`truth/` contains local derived material from other people's transcriptions and
videos. The rebuild does not alter source audio/video, application code, tests,
web assets, project configuration or Git state.

## JD Beck pass (2026-10-01)

Results are in `truth/JD_RESULTS.md` and `truth/jd-results.json`. The prior
`truth/SUMMARY.md` is preserved under `truth/jd-run/`; the updated summary links
the new results without applying the old 190-head audit statistic to them.

`build.py` now prefers `video.hi.mp4` whenever present. `--crop LEFT TOP RIGHT
BOTTOM` specifies an inspected native pixel region; `--no-cursor` prevents video
colors from becoming fake cursor observations. Neither cropping nor inspection
upscaling changes the native-spacing gate. The Growing Older source remains
below 8 px even at 1080p and produces no timed export.

The JD legends are separate JSON files under `legends/jd-*.json`, copied into
the relevant truth folders. A glyph's position/shape is retained independently
of its label. No comprehensive printed physical-kit key was verified for the
Zildjian, NAMM, All Caps or VFJams layouts. Their ambiguous upper crosses and
triangles remain `cymbal_unknown`, rather than being guessed as hi-hat, stack
or ride bell. Growing Older explicitly says `cymbal bell`, mapped to `bell`.
The Zildjian intro says `cross stick play over cymbal on head` and is not a
complete instrument key. Two Shrimps explicitly refers to hi-hat in local text.

JD reading additions:

- Staff line ink centroids correct even-width plateau rounding; all original
  spacing/support thresholds remain unchanged. The original strict exports for
  the 11 older sets are not rebuilt or overwritten.
- Layout-specific glyph templates support the two triangle styles and small
  regular filled heads. The ordinary reader's default behavior is unchanged
  when these legend options are absent. Development sources are listed in
  `jd_templates/README.md`.
- An omitted clef is accepted only with an explicit, visually verified layout
  note (`clef_omitted_verified`); this is not a timing-gate relaxation.
- `reading-corrections.json` contains actual visually added heads, including
  their source coordinates/evidence. `reading-rejections.json` can reject an
  otherwise well-aligned system for incomplete/ambiguous reading. These affect
  both tiers. Re-running `strip.py` reapplies them and redraws the overlays.
- `scrolling.py` measures image translation without reading audio. It reconstructs
  four-second VFJams blocks using 25% temporal white-ink support, then restricts
  heads to the span traversed by the cursor. A simple median erased thin ink and
  was rejected. `scroll-motion.json` retains all measured offsets/correlations.
- `prepare_zildjian.py` selects the visually verified single bottom staff,
  avoiding false staff candidates formed by beam rows. `beam_rhythm.py` only
  accepts complete quarter groups whose two/three beams sum to one quarter and
  whose moment assignments are exhaustive and unique. Short partial third
  beams are retained. Rest/tuplet patterns that this narrow parser cannot read
  stay on the less reliable interval prior. Independently read opening groups
  from OTHER systems, at the printed quarter=88 tempo, constrain the shared
  lag, avoiding an interleaved held-out fold choosing a one-stroke phase alias.
  This is still prediction against the same attack stream, not manual audio
  truth or proof that every transcription is accurate.

For example, after a generic Zildjian scan with crop `0 820 1920 1080`:

```bash
.venv/bin/python tools/truth_video/prepare_zildjian.py truth/jd-zildjian-drums
.venv/bin/python tools/truth_video/strip.py truth/jd-zildjian-drums --legend tools/truth_video/legends/jd-zildjian.json
.venv/bin/python tools/truth_video/beam_rhythm.py truth/jd-zildjian-drums
.venv/bin/python tools/truth_video/evaluate.py truth/songs/jd-beck-zildjian-live-transcription-drums-only-audio-H_xatJgX7V4 truth/jd-zildjian-drums/strip.json --alignment-only
.venv/bin/python tools/truth_video/tiers.py truth
```

For VFJams, run `scrolling.py <song> truth/jd-vfjams`, then `build.py <song>
truth/jd-vfjams --reuse-scan --legend tools/truth_video/legends/jd-vfjams.json`.
Its current reconstruction covers 12–204 s; other time is not exported.

### Separate scoring tier

`tiers.py truth` reads existing alignments without refitting or changing
`truth.json` or `alignment.json`. It writes:

- `truth.scoring.json`: strict plus scoring-only additions, with `tier` per note.
- `truth.additional.json`: only the additions.
- `alignment.scoring.json`: both statuses, exact reasons and scoring intervals.

Additional systems need fitted 40 ms agreement >=90%, held-out 40 ms >=85%
(cursor) or >=90% (no cursor, including rhythm priors), at least six moments,
strictly increasing times, valid staff/reading, and >=15 points over shifted
controls at either tolerance. There is no new 20 ms requirement. Existing strict
systems remain included even if their 40 ms held-out rate is below 85% but above
the original strict 80% gate. Strict thresholds were not loosened. This second
tier is a separately reported screening reference for +/-50 ms detector scoring.

`evaluate.py --reuse-truth --tier scoring` scores the union; omitting `--tier`
retains strict behavior. Confusion diagnostics now use the same 50 ms window
as P/R. A combined `hat/cymbal` row accommodates unresolved fine cymbal identity;
separate hat/cymbal family scores remain available but must not be mistaken for
validated physical-instrument distinctions in ambiguous layouts.

`transfer.py <source-truth-folder> <target-song> <target-truth-folder>` uses
200 Hz waveform envelopes in independent 20 s windows to measure offset/drift.
It requires correlation >=.8 and leave-one-window-out lag residual <=20 ms.
Zildjian drums-only/full-band transfer passed; the full-band reference is not
counted twice as independent truth. Its source-held-out quality is inherited,
not a new independent timing validation on the full mix.

### Detector reproducibility and diagnostics

Another worker was editing `src/`, so this pass copied it read-only into
`truth/jd-run/source/groovemap` before evaluation. `source-manifest.json` records
hashes and capture time; `source-drift.json` records subsequent unrelated edits.
No original detector source was edited. Use the frozen implementation with:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=truth/jd-run/source .venv/bin/python tools/truth_video/evaluate.py <song> <truth-folder>/strip.json --reuse-truth --tier scoring
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=truth/jd-run/source .venv/bin/python tools/truth_video/diagnose_jd.py <truth-folder>
.venv/bin/python tools/truth_video/test_reader.py
.venv/bin/python tools/truth_video/test_tiers.py
.venv/bin/python tools/truth_video/jd_report.py truth
```

`diagnose_jd.py` saves pre/post tom-stage correspondences, exact head shares,
actual `_pitch` calls and raw hertz values. For hits removed before `_pitch`, a
separately identified counterfactual call to the same function is made. The
code's head-share calculation sums spectral **amplitudes**, not squared energy.
A truth-label set within 50 ms is a coincidence diagnostic, not causal attribution;
coincident instruments can make the source of a detector hit ambiguous. `nothing`
means no exported note within the window, not independently verified silence.
P/R itself remains one-to-one. Ghost negatives mean no confirmed printed mark;
flag recall and underlying transcription fidelity were not exhaustively audited.

## Chops channels (2026-10-02)

`moyin`, `olliewoods`, `golly` and `ivannikov` have separate partial legends.
Only audited filled kick/snare/tom positions are verified; no cymbal or technique
mapping was enabled without channel-specific evidence. Blind native crops,
manual decisions (including failures), colour-edge comparisons and clef reviews
are under `truth/harvest/validation/*20261002/`. The measured yield and remaining
blocks are in `truth/harvest/validation/chops-20261002/REPORT.md`.

MoYin opts into `color_notes` and `page_color_notes`. The latter allows a page
row's visit to be established by at least six stable head-local colour edges,
covering >=70% of notated moments and >70% of their x span in increasing time.
A pale box background cannot establish that visit. Every other staff failure,
all audio timing gates and the instrument-band phase checks remain mandatory.
The sampled source heads turn blue or green inside the box. Their one-frame
threshold-crossing quantization is not a total acoustic onset error bound.

The four chops profiles opt into `strict_clef_pair`: a percussion clef is
searched relative to sustained five-line ink, before the actual first head's
left edge (including heads immediately after the staff start), requiring one
close pair of partial-height vertical strokes. Head and attached-stem columns
are excluded from the search. This also rejects the old
margin check's false confirmations from ordinary note stems. Existing profiles
retain their original clef check exactly. Missing/low-contrast clefs remain
rejected; no `clef_omitted_verified` waiver was added. S.O.A.R. remains rejected
at its retained native resolution.

Golly restricts its geometry search to the actual bottom score, excluding drum
footage that produced a false staff in the trial. Ivannikov rejects staff crops
obscured by large opaque chromatic promotional lettering. This additional
rejection is opt-in and applies to static/page scans and scrolling blocks.
Scrolling exports still require `scrolling_verified`; the promotional-overlay
check does not constitute channel-level scrolling validation.

Ivannikov also opts into `native_frame_fallback`. If a median fails staff
search, at least six retained native frames and >=75% agreeing five-line
geometries, with <=1.1 px line disagreement and stable notation ink, may supply
one actual source frame. The selected frame still passes every ordinary staff,
8 px, clef, head and timing/phase gate. Every detected head must recur with the
same shape and staff position, within 0.35 native spacing in x/y, in a strict
majority of the agreeing reservoir frames. A source frame with any unsupported
head is skipped; if none qualifies, the fallback is rejected. `scan.json`
records the source frame, required majority and supporting frames per head;
the failed median is retained. This restores Jojo Mayer's first staff
(the real native fixture is tested), but its timing/phase failure still prevents
export. No interpolation/upscaling is used.

Retained Golly/Ivannikov samples can be admitted without fetching anything:

```bash
.venv/bin/python tools/truth_video/import_samples.py --only golly ivannikov
.venv/bin/python tools/truth_video/batch.py --only golly ivannikov --retry-unreadable --refresh-read
```

The importer moves existing directories into `truth/harvest/songs/`, appends
`fetched` records with local provenance, refuses identity/destination conflicts,
and can recover a move interrupted before journal registration. It does not
copy/download audio or video. Missing retained video is recorded and the batch
fails closed. The ten samples from this pass now live in the canonical harvest
folders, rather than `truth/samples/`. Both new channel configurations use a
20-second sampling minimum so their 40-68 second chops are not silently skipped
by the fetcher's default 75-second filter; all truth acceptance gates stay fixed.

`result.json` now includes `timing_only_systems` and
`timing_only_system_ids`: strict-rejected systems whose every reason starts with
`held-out agreement`, `fitted attack agreement`, `ambiguous alignment` or
`no-cursor alignment`, matching `fill_seq.TIMING` / `tools/finetune/dense.py`.
These counts do not include phase, monotonicity, spacing, clef or reading failures.
They are candidates only; dense.py's assignment, coverage and run filters still
apply. The reader tests place temporary fixtures in `.eval-cache/channels/tmp/`.
