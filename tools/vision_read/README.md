# Gemini drum-notation vision pilot

## Codex GPT transport

`hybrid_gpt.py` attaches the same clean and numbered native images with
`codex exec -m gpt-6-luna -c model_reasoning_effort=xhigh
--skip-git-repo-check -s read-only -i original.png -i annotated.png -o <file>`
from the per-system job folder. It keeps the frozen musical prompt, rhythm
schema, CV-owned identity, merge/validation, and look policy. Models are
configurable (`gpt-6-luna`, `gpt-6.1-sol`), effort defaults to `xhigh`, and workers
default to six. Config isolation and ephemeral JSONL output retain local audit
logs. Rate-limit retries share an exponential cooldown, honor retry hints, and
retain every attempt; malformed visual output is retained as in `hybrid_run.py`.

```bash
.venv/bin/python -B tools/vision_read/test_hybrid_gpt.py
.venv/bin/python -B tools/vision_read/hybrid_gpt.py \
  --run-name hybrid-gpt-luna-xhigh --model gpt-6-luna \
  --effort xhigh --workers 6 --resume
.venv/bin/python -B tools/vision_read/hybrid_gpt_compare.py \
  --runs hybrid-gpt-luna-xhigh
```

Only if Luna xhigh's complete-gold joint is below 95%, evaluate Sol xhigh in a
new run and pass both run names to the comparator in that order. It uses the
same frozen `score.py` and `hybrid_compare.extra_metrics` on six complete plus
15 partial systems, writing `HYBRID_GPT_REPORT.md` and
`hybrid_gpt_summary.json`. Dense takeover requires all six complete results and
at least 95% for both XY and symbolic joint. A pass permits only scratch
candidates through the existing WSL adapter and timing gates. A failure keeps
the deferred Gemini coordinator. Medium/high runs cannot authorize takeover.

## Automatic dense-solo timing run

The `solos_*` entry points and `auto_timing.py` replace the manual anchor-sheet
step for the authorized dense-solo experiment. No person supplies bar origins,
staff approval or head-review flags. CV supplies native staff and verified
instrument identities; rational rhythm plus scan/cursor/registered-scroll
evidence supplies video-time priors. Unknown Gemini-only instruments remain
unknown. `align.py`, `bands.py` and `tiers.py` are imported unchanged.

On WSL, set `GROOVEMAP_VISION_DATA="$PWD/scratch/hybrid_solos"` and run
`solos_extract.py --workers 7`. It checks Vault membership before media access,
inventories the whole five dense videos, and prepares existing-strict visits
from non-vault Eric `9juoKgimeR4` and Larnell `N0PXFlgigjw` as controls.
The sealed `Ikg4P3F_Wj8` is excluded. Continuous horizontal scores are tiled
into disjoint visited four-second windows; source scanner fragments and
overlapping native context are not counted as separate playback occurrences.
Static repeated visits without a cursor reset are deduplicated.

Copy `scratch/hybrid_solos/hybrid/jobs/` to the Mac's
`.eval-cache/vision_pilot/hybrid/jobs/`. Run `solos_run.py --run-name
hybrid-solos-controls-v1 --videos 9juoKgimeR4 N0PXFlgigjw`. It uses the existing
hybrid runner (3.8 flash medium, look, four workers, shared `agy.lock`), cancels
all calls at the first 429, and waits for the reported reset plus 90 seconds.
An unknown reset deadline is recorded without blind quota retries.

After a complete batch, `auto_timing.py --run <run-name>` stages `candidates/`.
Transfer run artifacts and candidates to WSL's scratch tree. With the WSL data
environment above, `solos_gate.py --videos 9juoKgimeR4 N0PXFlgigjw` executes the
existing strict and scoring gates, reads band evidence with `cache=False`, and
writes the acceptance report. Agreement is one-to-one, exact instrument and
at most 20 ms; every hybrid-strict note is in the denominator. Missing reference
matches are not discarded. Empty strict output does not count as perfect.
Stop if either measured agreement is below 95%; otherwise run the five dense
video IDs through the same runner, adapter and gate commands. No canonical
truth, manifest, vault setting, training code or protected reader is written.

`solos_verify.py --cv-controls` checks that the final native CV implementation
reproduces the frozen control head/staff inputs and that registered geometry
retains the original continuation limits. `solos_verify.py` verifies the
protected file hashes. The Korean report and all failed or unresolved systems
remain under WSL `scratch/hybrid_solos/`; copy `REPORT.md` to the Mac's
`.eval-cache/vision_pilot/HYBRID_SOLOS_REPORT.md`.

`solos_finish.py` coordinates the Mac-to-WSL handoff, acceptance decision, all
five dense readings, audio gates and report mirroring. A completed control run
can be reused with `--resume-after-controls`. Check
`hybrid-solos-background.json`, `hybrid-solos-state.json` and the per-run
`*-state.json` before starting another coordinator. A live coordinator already
continues automatically after the recorded quota reset. A strict-empty control
is explicitly inconclusive; with no measured agreement below 95%, the task may
continue to scratch candidates without claiming overall acceptance. The runner
keeps a per-run lock during quota waits and honours the previous reset deadline
after process restart. The shared `agy.lock` still bounds actual inference.

All harness code lives here. All inputs, copied job images, logs, model readings,
scores and scratch timing candidates live under `.eval-cache/vision_pilot/`.
The harness never imports the benchmark reader or reads audio during vision
inference. Use the repository `.venv/bin/python -B`; no dependency installation
is needed (the scorer uses the existing NumPy/SciPy installation).

## Reproduce the pilot

```bash
.venv/bin/python -B tools/vision_read/test_harness.py
.venv/bin/python -B tools/vision_read/prepare.py --selection pilot

.venv/bin/python -B tools/vision_read/run.py \
  --selection gold --model gemini-3.8-flash-high --mode look \
  --workers 4 --timeout 900 --run-name high-look-new-4w
.venv/bin/python -B tools/vision_read/run.py \
  --selection gold --model gemini-3.8-flash-medium --mode look \
  --workers 4 --timeout 900 --run-name medium-look-new-4w
.venv/bin/python -B tools/vision_read/run.py \
  --selection gold --model gemini-3.8-flash-high --mode zoom \
  --workers 4 --timeout 900 --run-name high-zoom-budgeted-4w
.venv/bin/python -B tools/vision_read/run.py \
  --selection dense --model gemini-3.8-flash-high --mode look \
  --workers 4 --timeout 900 --run-name dense-high-look-resumed-4w

.venv/bin/python -B tools/vision_read/score.py --run high-look
.venv/bin/python -B tools/vision_read/score.py --run medium-look
.venv/bin/python -B tools/vision_read/score.py --run high-zoom-budgeted
.venv/bin/python -B tools/vision_read/score.py --run dense-high-look-resumed
.venv/bin/python -B tools/vision_read/report.py \
  --runs high-look medium-look high-zoom-budgeted --dense-run dense-high-look-resumed
```

Run those batches sequentially under the shared `agy.lock`, with at most four workers. There
are three comparison conditions, not a full 2x2 factorial: the model comparison
holds `look` fixed; the mode comparison holds `high` fixed. Medium/zoom was not
requested for the speed comparison. The same frozen 21 note-gold systems are
used in every comparison. Dense adds two deterministic, spread-out no-note-gold
systems from each of Steve Smith, Jojo, Weckl, Novak and Haake (ten total).

`prepare.py --systems <ids...>` selects exact systems; `--selection all` prepares
all 150 without running inference. Best frame is the manifest's manually
inspected sharpness-selected primary, copied first; other native frames follow.
It does not assert a new independent best-frame review. Channel variant rules
are merged exactly as in the benchmark. `legend.json` retains both notation
instrument candidates and verification flags; the prompt is not an export
approval. No GOLD annotations are copied into jobs or exposed to the model.

Each run has its own input copies and `INSTRUCTIONS.md`, `prompt-*.txt`,
`stdout-*.log`, `stderr-*.log`, `cli-*.log`, `reading.json`, and `result.json` per
system. `run.json` records per-system and batch wall time. The subprocess has
closed stdin, a total timeout shared by both attempts, and process-group
termination on timeout. There is one fresh inference retry for invalid schema
or JSON; CLI errors and timeouts are retained without extra inference retries.
The exact flag form is `agy --model <model> ... -p=<prompt>` using an argv array
(no shell interpolation). Every `gemini-3.1*` model is rejected before launch.

Use a fresh `--run-name` for another experiment. `--resume` only reuses successful
jobs whose source, legend, prompt generator and runner fingerprints match.
Failed jobs may be attempted again on explicit resume; existing readings are
never silently overwritten. A single successful CLI exit is not sufficient:
JSON and schema validation must pass. CLI diagnostics are retained locally.

`look` opens the primary with the image viewer once, optionally each alternate
once if obstructed, and prohibits shell/crop/script/search/agent tools.
`zoom` permits crops and inspection tools within that system folder. The final
protocol caps it at four new crops and six post-initial-view tool calls, and
prohibits exhaustive pixel sampling/head-detector scripts. Those are prompt
limits, not a tool sandbox. The first unbounded `high-zoom` attempt generated
many crops without final JSON and hit the account's 429 quota; it is retained
as an interrupted attempt, not scored as model accuracy zero. The final
JSON is saved by Python, so `look` needs no file-write tool. These are **prompt
restrictions, not a tool sandbox**. Text CLI output does not provide a complete
tool-call trace; do not interpret zero regex hits in CLI logs as proof of no
tool use. Model speed can differ from earlier single-system measurements because
the requested output now includes all visible heads, coordinates and beat checks.

## Reading contract

`reading.json` is a strict JSON object with `schema_version: 1`, `system_id`,
`time_signature`, `staff`, `bar_count`, `bars`, `beat_checks`, and `uncertain`.
Time signature is null unless visibly shown. A numerator/denominator object is
canonical; an equivalent string such as `"4/4"` is accepted and canonicalized.
The unambiguous notehead synonym `x` is canonicalized to `cross`.
The first batches had an unnecessarily narrow validator;
`recover.py --run high-look` can recover only those representation issues from
preserved responses, without changing notes or making model calls. Original
retry errors and actual elapsed time remain recorded. Recovery is idempotent
for successful jobs and never replaces an existing reading.

- `staff`: `line_count`, five ascending native `lines_y`, `clef_visible`.
- `bars`: sequential visible zero-based `index`, `partial_left`, `partial_right`,
  optional native `x_start`/`x_end`, and time-ordered `events`. Include clipped
  leading/trailing bars; printed bar numbers never replace local indices.
- `events`: rational string or null `onset` and `duration`, `voice`, and `heads`.
  Both fractions are **whole-note units**: quarter = `1/4`, sixteenth = `1/16`,
  sextuplet sixteenth in 6:4 = `1/24`. Chords consume one duration per voice.
- `heads`: unique `id`, legend `instrument` or null, integer staff `pos` or null,
  `notehead` (`filled`, `cross`, `circled`, `diamond`, `open`, or null), native
  `x`/`y`, `ghost`, `accent`, `grace`, `tie_continuation`, and `tuplet`.
  Boolean fields may be null for unresolved observations. Tuplet is null or
  `{ "ratio": "6:4", "group": "b0-v1-t0" }`.
- `beat_checks`: bar/voice, rational `beat_start`, `beat_length`,
  `sounded_duration`, `implied_rest_duration`, and status. The prompt requires
  every known beat/voice to sum correctly. The scorer independently checks
  reported arithmetic, same-voice overlaps and known-meter bar overflow; missing
  checks/unknown meter do not prove beat completeness.
- `uncertain`: explicit location/field/reason objects. A left-clipped bar with
  unknown origin must have null onsets rather than inventing zero. Grace duration
  is zero and is not a separately timed attack. Tie continuation heads are
  inventoried, but do not become new attacks. Rests are implied by event gaps.

Coordinates refer to the primary **native** image, undoing crop/zoom offsets.
Instrument labels describe channel notation, not physical tom sizes or verified
cymbal identity. Preserve exact staff position and notehead even when the
instrument is unknown/unverified.

## Scoring and its limits

Read `bench/REPORT.md`, `bench/bench.py` and the frozen reconciler before changing
scoring. `gold.json` is the finalized reconciliation of the two geometry/note
journals and supplements in `annotations/`; their SHA-256 values are recorded
in every score output. The 21 note-gold systems contain 330 heads; only six have
complete visible-note inventories. This is provisional, same-reviewer partial
GOLD, not independent expert transcription or general accuracy on 150 systems.

`scores.tsv` contains one row per selected system; `scores.json` includes the
summary, instrument/channel breakdowns, per-head errors, matches, provenance,
conditional metrics, positive tie/ghost/accent recall and geometry sensitivity.
For the six complete inventories, `full_symbolic_inventory`,
`full_symbolic_onset`, and `full_symbolic_onset_duration` also compare multisets
of (visible bar, staff position, head shape, tie state), adding exact onset and
duration respectively. These do not require model pixel coordinates and do not
perform gold-guided sequence matching. Their 159-head denominator and all-head
full-inventory precision are separate from partial-region geometry precision.
They help assess musical reading when VLM coordinate estimates are rough.
Timing includes real end-to-end wall time with retries and first-attempt time
separately, so representation retries do not masquerade as model slowness.
Missing/invalid/timed-out jobs count as missed known gold, not excluded successes.

The main matcher uses the benchmark's geometry-only one-to-one Hungarian rule:
x tolerance `max(4 px, 0.65 staff-space)`, y tolerance `max(3 px, 0.4 staff-space)`.
It never uses instrument, shape, position, bar or onset to choose matches. This
prevents rhythm/label accuracy from matching itself, but rough VLM coordinates
can depress all end-to-end metrics. A separately labeled relaxed sensitivity
uses x `max(8 px, 1.3 space)`, y `max(5 px, 0.6 space)`; it is not the headline
benchmark. Read conditional accuracy alongside recall to distinguish missing/
mislocated heads from errors on confidently matched heads.

Head recall uses reviewed positives. Precision counts unmatched predictions
**only inside exhaustive gold rectangles**, excluding uncertain regions. Per
instrument recall requires the correct verified gold label. Instrument precision
also excludes matched gold with null/unverified identity. Gold quarter-note
fractions are compared to model whole-note fractions multiplied by four, using
`Fraction`, and exact visible bar ordinal. Duration, onset, position, shape,
tuplet ratio/membership and ties are scored only where gold exists. Tie-positive
recall is separate from the many negative examples. Group boundaries themselves
lack independent GOLD IDs, so ratio/member success does not certify grouping.

Bar counts are scored only for the six complete visible-note systems, using
their known visible bar indices. Gold boundary coordinates are scored separately
for partial systems. Staff-found, exact five-line positions and clef presence
are scored on all selected systems with geometry GOLD. The ten additional dense
systems have geometry scores but **no note accuracy score**. Known-fields system
correctness excludes independent rest-gap correctness, which was not requested
as a metric and cannot be inferred from empty unreviewed areas.

## Quota interruption and automatic continuation

The initial high/zoom run and all ten additional dense attempts were refused
with `RESOURCE_EXHAUSTED` / HTTP 429. A medium dense probe confirmed the same
limit; changing high to medium cannot bypass it. The original CLI errors and
crop artifacts remain in `high-zoom/`, `dense-high-look/` and
`dense-medium-probe/`. Their missing-prediction scores are operational failures,
not measured model accuracy. The report explicitly censors quota-hit conditions.

`deferred.py` reads the CLI's reset duration and log timestamp and adds a
90-second margin. `--hybrid-run hybrid-medium-look` prioritizes the hybrid
comparison (six complete inventories first, then the 15 partial systems),
followed by dense/high/look and the bounded high/zoom pilot. All use the shared
`agy.lock` with at most four workers. Fresh baseline names are
`dense-high-look-resumed-4w` and `high-zoom-budgeted-4w`; interrupted runs remain
intact. First 429 cancels queued/in-flight calls. One further reset-based retry
is permitted by default (`--max-quota-retries 1`), after which quota failure is
retained explicitly. `--hybrid-only` defers only the hybrid comparison. The
15-minute timeout bounds each system; hybrid has one image-view pass and no
automatic schema reread. It rescores and rewrites local reports without audio
checks, account changes or model substitution.

The current scheduled process and its UTC due time/PID/heartbeat are recorded
in `.eval-cache/vision_pilot/deferred.json`; its log is `deferred.log`.
`complete` means all queued batches returned valid readings for every job;
`finished_with_failures` and `quota_blocked` explicitly retain incomplete work.
No chat notification is sent by this local process. Inspect the state/report
files for its eventual outcome. To preview a new schedule without model calls:

```bash
.venv/bin/python -B tools/vision_read/deferred.py --plan-only
.venv/bin/python -B tools/vision_read/deferred.py \
  --hybrid-run hybrid-medium-look --plan-only
```

Do not start a second scheduled process while the recorded PID is alive.

## Hybrid numbered geometry and rhythm experiment

The CV adapter in `hybrid.py` calls the current `layouts.ink`, `heads.heads`
and `strip.label_heads` exactly as the benchmark does, with the same channel
variant merge and frozen primary native frame. It imports protected code
read-only. It saves final heads plus raw detector heads, staff checks/reasons,
candidate barlines, coordinates, position, shape and legend instrument.
`Vault().is_vault` is checked before frame access in preparation, workers and
comparison; sealed material cannot enter development scoring.

`annotated.png` preserves native scale and adds blue numbered head labels in
padded lanes with thin leaders/outlines. Orange B labels mark candidate barlines,
including possible false-positive stems. The clean `original.png` is also
provided. Neither image contains gold annotations. Gemini chooses real bars
from notation rather than blindly trusting CV boundaries.

`hybrid_run.py` fixes `gemini-3.8-flash-medium` and look policy: original once,
annotation once, no other tools. Each numbered head must appear exactly once
as head/false_positive/uncertain, with bar, rational onset/duration, voice,
tie, tuplet group, ghost/accent and grace. The merger owns CV instruments,
positions, shapes and coordinates; model overrides are ignored. Unlabelled
misses retain optional approximate coordinates/position but always
`instrument=null`, `instrument_status=unverified-instrument`. Raw rhythm JSON
and false-positive decisions remain inspectable. Text CLI logs do not certify
an exhaustive tool trace; look is a prompt policy, not a tool sandbox.

```bash
.venv/bin/python -B -m unittest discover -s tools/vision_read -p 'test_*.py'
.venv/bin/python -B tools/vision_read/hybrid.py --selection selected
# Run only after the quota deadline, and never beside deferred.py's inference:
.venv/bin/python -B tools/vision_read/hybrid_run.py \
  --run-name hybrid-medium-look --workers 4 --timeout 900 --resume
.venv/bin/python -B tools/vision_read/score.py --run hybrid-medium-look
.venv/bin/python -B tools/vision_read/hybrid_compare.py --run hybrid-medium-look
```

`HYBRID_REPORT.md` and `hybrid_summary.json` compare fresh CV-only results with
the retained high/medium Gemini look baselines and hybrid using the unchanged
`score.py`. XY metrics preserve the benchmark's region-only FP rules; symbolic
multisets count every head on complete inventories, including unlocated
additions. The extra joint onset+instrument metric requires both fields, with
112 known-instrument heads in complete gold as its denominator. Whole-system
symbolic exactness also requires duration, position/shape, tie, ghost/accent,
positive tuplet ratios, known instruments and equal head inventory. Tuplet
group IDs are retained but gold does not independently annotate group borders.
Unreviewed gold fields remain unknown; implied rests are not certified.

Quota/missing jobs are censored, never reported as zero model accuracy.
The comparison appends the reviewed `HYBRID_WSL_PLAN.md` here only if all six
complete-gold readings finish and joint symbolic onset+instrument reaches 85%,
whole-system XY exactness beats all baselines, or symbolic onset recall beats
all baselines by at least 10 percentage points. The plan covers five dense
solos, xiangsong and actual Ollie Shorts, separate canonical vault ingestion,
native frame extraction, reviewed anchors, strip conversion and existing
strict timing/phase gates. No performance or rollout approval is inferred from
unit tests or head counts.

## Exact next step on WSL: audio timing gates

Do not treat `reading.json` as timed truth. This pilot does not access audio or
run this step. Transfer only `tools/vision_read/` and
`.eval-cache/vision_pilot/` into the corresponding WSL Groovemap checkout.
Keep candidate outputs beneath `.eval-cache/vision_pilot/wsl_handoff/`, outside
the canonical harvest truth and manifest.

1. Generate an anchor review sheet (already generated for the pilot when present):

   ```bash
   .venv/bin/python -B tools/vision_read/export_timing.py \
     --run dense-high-look-resumed --write-anchor-template \
     --out .eval-cache/vision_pilot/wsl_handoff/resumed
   ```

2. Copy `wsl_handoff/resumed/anchors.template.json` to `anchors.reviewed.json` within that data
   folder. For each system, inspect the source video/native staff and supply:
   `staff_reviewed: true`; the existing independently checked `staff` object
   (`lines`, `spacing`, `reasons`, `clef_confirmed`); native measured spacing;
   audited clef-omission evidence where applicable; one exact whole-note offset
   per visible bar; `time_origin_seconds`; `seconds_per_whole`; and the true
   playback `interval`. For quarter-note BPM, seconds per whole is `240/BPM`.
   Use visible cursor/color transitions or separately verified video-time anchors
   for origin/tempo, not the same audio attack points later used as held-out
   evidence. Scan start/end in the template are provenance seeds, not a claim
   that every visible score event occurs inside that interval. Retain every
   existing rejection reason; a five-line model claim is not a staff validation.
   Keep `verified_heads` empty until note identity/position is independently
   inspected; populate only reviewed head IDs. Record evidence paths and all
   unresolved bar origins as null. Unresolved onsets cannot be audio-gated yet.

3. Stage candidate strips, without audio access:

   ```bash
   .venv/bin/python -B tools/vision_read/export_timing.py \
     --run dense-high-look-resumed \
     --anchors .eval-cache/vision_pilot/wsl_handoff/anchors.reviewed.json \
     --out .eval-cache/vision_pilot/wsl_handoff/reviewed
   ```

   `handoff.json` lists candidates and blocked inputs; each staged video has
   `strip.json`. The adapter maps `bar offset + onset` to a synthetic rhythm
   axis (`x = whole-note position * 4096`) for `align.py::moments` and `warp`.
   `native_x`/`native_y` remain for overlays. `rhythm_x`/`rhythm_t` carry video
   time priors. This avoids treating engraving x-spacing as rhythm. Exact
   fractions are also retained as quarter-note `onset_q`/`duration_q`.
   Chord heads share an x; tie continuations are kept but excluded by existing
   `moments`; grace heads and unresolved onsets are omitted from timing and
   explicitly block automatic complete-system acceptance. Unverified mappings
   become `inst: unknown`; only independently reviewed, verified legend heads
   become exportable. Observed flags remain separate from unverified exports.

4. On WSL only, find the canonical song directory for each staged video from
   `job.json.video_path`. It must have `attacks.htdemucs.npy` and a separated
   `drums.htdemucs.wav` or `.opus`; band evidence is
   `onsets.bands.htdemucs.npy` plus its metadata. On that audio-owning WSL checkout,
   execute the **existing** gate function on a scratch candidate directory:

   ```bash
   SONG='truth/harvest/songs/<canonical-slug>'
   CANDIDATE='.eval-cache/vision_pilot/wsl_handoff/reviewed/<video-id>'
   PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=tools/truth_video \
     .venv/bin/python -B -c \
     'from pathlib import Path; import sys; from align import build_truth; build_truth(Path(sys.argv[1]), Path(sys.argv[2]), bands="auto")' \
     "$SONG" "$CANDIDATE"
   ```

   `bands="auto"` loads/computes real stem evidence on WSL; it may create the
   band's cache in the audio-owning song directory, so run it only as part of
   that authorized audio check. It writes scratch `alignment.json` and
   `truth.json`. No detector labels are used. Review every system's reasons,
   held-out results, controls and phase before any later publication.

Current gate mapping was verified from `tools/truth_video/align.py`, `bands.py`,
`tiers.py` and README: at least six distinct moments; fitted 20/40 ms at least
85%/95%; held-out 55%/80%, raised to 70%/90% for `rhythm` or interval priors;
held-out gain over shifted controls at either tolerance at least 15 points;
strictly increasing warped times; native spacing at least 8 px and existing
staff/clef/layout checks. Instrument phase needs at least six notated kick/snare
moments, at least 60% band agreement and a 15-point advantage over rhythmic
aliases (or established independent song-level evidence). Kick uses the low
band and snare the mid band. Missing stem/phase evidence rejects export.

The separate scoring tier in `tiers.py` keeps structural/phase/control reasons,
requires fitted 40 ms >=90% and held-out 40 ms >=85% (>=90% without cursor).
Do not silently switch to it to improve pilot yield. Zero accepted dense
systems is a valid outcome. Audio-held-out agreement validates timing within
these controls, not independent musical-note transcription correctness.

## Hybrid WSL rollout after measured improvement

This plan is appended to README only by `hybrid_compare.py` after a real hybrid
run completes all six complete-gold systems and meets the recorded improvement
rule. A quota/missing result cannot activate it. It is a rollout plan, not
permission to turn unreviewed notation into strict truth. Existing protected
reader/scanner/alignment/tier code remains unchanged.

1. **Extract native frames on WSL.** Use the audio-owning Groovemap checkout and
   its `.venv/bin/python -B`. For the five exact video IDs `jAMDUcKmjD4`,
   `qLi8dFevOF0`, `n3jYSzj9VX0`, `tK9MGMbPS_w`, `Ly7Pbg4EDug`, resolve the
   canonical song directory from `truth/harvest/manifest.jsonl`. For xiangsong
   and Ollie Shorts resolve actual manifest rows for channels `xiangsong` and
   `olliewoods_shorts`; do not substitute Ollie's `4VX2domLud8` longform. Retain
   video IDs, source path, native dimensions/fps, scan SHA and source size.
   Use existing canonical `truth/scan.json` or the reviewed canonical pilot's
   `notation/scan.json`. If absent, invoke the existing scanner API with the
   inspected channel profile into the candidate cache; do not create a new
   layout guess or median reconstruction. Import the existing scanner read-only.

   Add a WSL extraction entry point under `tools/vision_read/` that takes
   `(video_id, canonical video path, scan.json, legend path)` explicitly and
   emits a bench-compatible native manifest into
   `.eval-cache/vision_pilot/wsl_native/<video_id>/`. Iterate EVERY scan record,
   including rejected staff records; preserve row, start/end, native spacing,
   scroll/read rejection reasons, cursor frames/x, foreground and row crop.
   For each record extract unique frame indices
   `int(start+(end-start-1)*f)` for `f=0.15,0.50,0.85`. Native crop is
   `[scan.left, max(0,scan.top+record.crop[0]-40), scan.right,
   min(video_height,scan.top+record.crop[1]+50)]`, matching the pilot, with no
   enlargement. Store PNG SHA, frame/time and crop coordinates for all three.
   Rank by Laplacian sharpness as a candidate heuristic and visually select the
   unobstructed system: no cursor covering a head, cropped beams or overlapping
   video inset. Keep clips partial; scrolling viewport is not a complete system.
   Track `(video,row,bar/source interval)` to avoid duplicate overlapping exports.
   Never fill missing heads from prior strip predictions or audio.

2. **Run the fixed hybrid reader.** For non-vault development inputs the existing
   portable entry point already accepts the frame manifest:

   ```bash
   .venv/bin/python -B tools/vision_read/hybrid.py \
     --bench .eval-cache/vision_pilot/wsl_native/<video-id> \
     --systems <manifest-system-ids>
   .venv/bin/python -B tools/vision_read/hybrid_run.py \
     --run-name hybrid-wsl-<video-id> --systems <manifest-system-ids> \
     --workers 4 --timeout 900 --resume
   ```

   First run `Vault().is_vault(video)` before image access. Sealed xiangsong or
   Shorts must NOT be passed to these development entry points, any gold scorer,
   baseline comparison or threshold tuning. Add a separate canonical-only
   entry point with provenance `purpose=canonical_truth`, which reuses the same
   CV/annotation/merge functions and writes candidate `strip.json` into each
   canonical notation attempt. It must not expose a general development
   `allow-vault` override. Only run that entry point later as part of canonical
   reading; sealed material is excluded from every development selection and
   comparison. The current development commands deliberately reject it.

   Use `gemini-3.8-flash-medium`, original once plus annotated once, no other
   Gemini tools, frozen prompts and no gold/audio in jobs. Use the shared
   `agy.lock` across all batches, at most four workers, stop all pending and
   in-flight calls at the first 429. Schedule a retry after the CLI-reported
   reset plus the existing 90-second cushion using `deferred.py`; retain prior
   successes and immutable inputs. Never fall back to gemini-3.1. Retain every
   numbered disposition, raw response, native geometry, tuplet groups, flags,
   false-positive rejections and unlabelled misses. Gemini-only instruments
   remain unverified until independent native notation/legend review.

3. **Stage the existing strip contract.** For non-vault candidates:

   ```bash
   .venv/bin/python -B tools/vision_read/export_timing.py \
     --run hybrid-wsl-<video-id> --write-anchor-template \
     --out .eval-cache/vision_pilot/wsl_handoff/<video-id>
   .venv/bin/python -B tools/vision_read/export_timing.py \
     --run hybrid-wsl-<video-id> \
     --anchors .eval-cache/vision_pilot/wsl_handoff/<video-id>/anchors.reviewed.json \
     --out .eval-cache/vision_pilot/wsl_handoff/<video-id>/reviewed
   ```

   Independently review source staff lines/spacing/reasons and clef/omission
   evidence, one rational `bar_offsets_whole` per visible bar, actual playback
   `interval`, `time_origin_seconds`, `seconds_per_whole=240/quarter_BPM`, and
   reviewed head IDs. Record native frame/video evidence; do not derive anchors
   from the held-out acoustic attacks. Unresolved bar origins stay null.
   The canonical-only entry point calls the same `export_timing.convert` with
   reviewed anchors for sealed songs, without invoking any development scorer.

   Adapter fields are `x=(bar_offset+onset)*4096`, exact `onset_q=4*onset` and
   `duration_q=4*duration`, `native_x/native_y`, `rhythm_x/rhythm_t`,
   `tied_continuation`, `pos/shape`, instrument and verification provenance.
   Geometry spacing stays native, not 4096. Ties remain inventoried but are
   excluded by existing `align.moments`. Grace/unresolved heads block automatic
   coverage acceptance. Uncertain numbered heads and rejected CV heads require
   explicit review; unverified heads remain `inst=unknown`, `exportable=false`.
   Observed ghost/accent is not an export approval. Human-reviewed corrections
   are new candidate artifacts with provenance, never replacements of the raw
   model response or development gold.

4. **Let the existing timing/phase gates decide.** On WSL with real separated
   drums and `attacks.htdemucs.npy`, execute the existing gate on a new candidate
   directory, never overwrite published truth during the trial:

   ```bash
   PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=tools/truth_video \
     .venv/bin/python -B -c \
     'from pathlib import Path; import sys; from align import build_truth; build_truth(Path(sys.argv[1]), Path(sys.argv[2]), bands="auto")' \
     'truth/harvest/songs/<canonical-slug>' \
     '.eval-cache/vision_pilot/wsl_handoff/<video-id>/reviewed/<video-id>'
   ```

   Keep current staff/native-size/clef/layout gates, at least six distinct
   moments, monotone warp, fitted/held-out 20/40 ms tests, shifted controls and
   kick/snare band phase. `rhythm`/interval priors use the existing stricter
   held-out checks. Keep rejected systems and reasons in `alignment.json`.
   `tiers.py` may classify a scoring candidate separately; never relax or swap
   strict thresholds to raise yield. Compare notation/coverage only on unsealed
   development gold. Canonical sealed songs may produce accepted/rejected truth
   artifacts later, but cannot contribute to hybrid development scores.

   Independently verify instrument/flag evidence before adding a Gemini-only
   miss to reviewed/exportable heads. Audio gates validate timing under their
   controls; they do not independently prove note identity. Zero accepted dense
   systems remains a valid result. Publish only the strict subset the existing
   canonical pipeline admits, preserving source/model/prompt/image fingerprints.
