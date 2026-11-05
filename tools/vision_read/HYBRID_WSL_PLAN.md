## Hybrid WSL rollout after measured improvement

This plan is appended to README only by `hybrid_compare.py` after a real hybrid
run completes all six complete-gold systems and meets the recorded improvement
rule. A quota/missing result cannot activate it. It is a rollout plan, not
permission to turn unreviewed notation into strict truth. Existing protected
reader/scanner/alignment/tier code remains unchanged.

1. **Extract native frames on WSL.** Use the audio-owning Stickmap checkout and
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
