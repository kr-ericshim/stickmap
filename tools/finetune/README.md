# Fine-tuning ADTOF on transcription videos

The detector's misses in fast fills are frames where ADTOF's activation never rises, so no
peak-picking rule can recover them. Here ADTOF's Frame-RNN is fine-tuned on the harvested
ground truth (`tools/harvest`, 131 songs from 14 channels, 3.4 h inside accepted systems).
The result ships as `src/stickmap/adtof-stickmap-v2.pt` and is used on the drum stem only.

WSL GPU에서 학습·추론하고 Mac에서 채점하는 절차는 [docs/wsl-gpu.md](../../docs/wsl-gpu.md)에 있다.

```bash
.venv/bin/python tools/finetune/data.py                  # features + truth per song -> .eval-cache/ft/
.venv/bin/python tools/finetune/teacher.py               # the original model on those features
.venv/bin/python tools/finetune/train.py --fold 0 --tag v2 --doubt 0.3   # also folds 1, 2
.venv/bin/python tools/finetune/predict.py v2-fold0      # held-out songs' activations (each fold)
.venv/bin/python tools/finetune/solos.py v2              # drum-solo videos, never trained on
ONLY=(swartz allyeodrum drumclass mydrumsheet taktak drumshin drumplace nakamura chock yeol atthedrum jilju drummate drumscribe)
FT_TAG=v2 PYTHONPATH=tools/finetune .venv/bin/python tools/harvest/fill_score.py --detect ft_variants:tuned --only "${ONLY[@]}" --jobs 4
FT_TAG=v2 PYTHONPATH=tools/finetune .venv/bin/python tools/harvest/global_score.py --detect ft_variants:tuned --only "${ONLY[@]}"
FT_TAG=v2 PYTHONPATH=tools/finetune .venv/bin/python tools/truth_video/fill_seq.py --detect ft_variants:tuned --jobs 4
.venv/bin/python tools/finetune/train.py --fold all --tag v2 --doubt 0.3 # the shipped model
```

## Vault

`truth/harvest/vault.json` seals explicit `video_ids` and every video whose **first**
committed manifest row is newer than `cutoff` (UTC ISO timestamp or Unix seconds).
`vault.py init` persists the current time once; edit `cutoff` to choose another boundary.
Carry this file with the manifest when copying datasets; missing/invalid config fails closed.

```bash
.venv/bin/python tools/harvest/vault.py add VIDEO_ID OTHER_VIDEO_ID
.venv/bin/python tools/harvest/vault.py list
.venv/bin/python tools/harvest/vault.py stats
.venv/bin/python tools/harvest/fill_score.py --loose --jobs 4
# Only after model/settings are frozen: opens ONLY the vault and records this invocation.
.venv/bin/python tools/harvest/fill_score.py --loose --final --model-tag frozen-v5 --jobs 4
```

Data preparation, cached `Song` loading, and every fold (including `all`) exclude vault
songs, including existing caches. Routine scorers exclude them; `--final` selects only
vault songs, prints a banner, and appends timestamp, full command, and model tag to
`truth/harvest/vault_log.jsonl` before evaluation. This also applies to `score.py`,
`global_score.py`, and `fill_seq.py` (resolved through its song pointer/metadata).

`--loose` reuses `fill_seq`'s order DP (default ±60 ms) on strict and cleanly read
timing-only rejected systems, including manifest `unreadable`/`fetched` rows. Reading,
phase, unvisited-row and nonmonotone failures stay excluded; fitted 40 ms agreement
must be ≥75% with ≥6 drum moments. Very fast means nearest kick/snare/tom neighbour
gap <90 ms. These are approximate scoring references, never training truth. `stats`
reports video seconds and eligible fast moments per kick/snare/tom without a detector.

## Training

- Kick, snare and tom are taught by the truth inside accepted intervals; one frame either side
  of a written stroke is left alone (the truth is good to about 10 ms).
- Transcribers leave strokes out (ghost notes above all), so a frame nobody wrote anything at
  is taught silence only where the original model is quiet too. Where it hears a kick or tom
  nobody wrote, silence is taught with weight 0.3 (`--doubt`); such snares stay unknown.
- Everywhere else, and always for hi-hat and cymbal (many channels never write them), the
  model is held to the original model's output (distillation, weight 0.3), except right at
  written strokes, which is what is being fixed.
- 8 s crops, 40 % centred on written fill strokes; gain and spectral-tilt augmentation;
  BatchNorm statistics frozen; Adam 1e-4, 1500 steps of 12 crops (fixed in advance; the
  logged proxy scores never choose a checkpoint).
- Folds are by channel; a song another channel also transcribed is never trained on while
  the other is scored (`SAME_SONG`).

## v3 experiment: v2 + dense positives

Legacy v3 uses positive-only `dense.json` beside each song's strict cache, for cleanly
read dense passages rejected only for timing. `--dense W` (default 0, preserving v2)
adds `W * mean(BCE(max(class output in window), 1))` over every label whose **whole**
inclusive `round(time * fps) +/- tolerance_frames` window fits in a crop. Distillation
is disabled only for that class in those included windows. Dense data never changes
frame targets, accepted intervals, or negative truth weights; partial windows are ignored.
The channel and `SAME_SONG` exclusions apply to whole songs, including their dense labels.

When enabled and a song has dense labels, crop sampling is 40% fill-centred, 20% centred
on dense labels, 30% within accepted intervals, and 10% anywhere in the song. The original
anywhere bucket was only 10%, so the dense share comes from the non-fill, accepted-interval
bucket; the fill share and uniform coverage are retained. Songs without dense labels keep
the original 40% / 50% / 10% sampling.

```bash
# Cached dense labels and existing v2 held-out/solo predictions are prerequisites.
bash tools/finetune/run_v3.sh

# Synthetic tests; put all test scratch under the fine-tuning cache.
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider \
  --basetemp=.eval-cache/ft/test-tmp tools/finetune/test_train.py tools/finetune/test_dense.py
```

`run_v3.sh` trains the original 14-channel cohort for 1500 steps per fold with
`--tag v3 --doubt 0.3 --dense 1.0 --dense-mode window --dense-labels legacy --no-withheld-mask`:
fold 0 on MPS and folds 1–2 on CPU with four threads. It then predicts each fold's held-out
songs and runs `solos.py v3`. Finally it runs the three commands above with **both**
`FT_TAG=v3` and `FT_TAG=v2`, using the default thresholds and scorer settings. Their JSON
integer totals and logs go to `.eval-cache/ft/results/v3/`; `report_v3.py` writes the
comparison table to `.eval-cache/ft/results/v3.md`. This experiment does not train a
deployment (`fold all`) model. Reruns rebuild v3's derived snare responses after prediction.

## v4 experiment: local peak supervision

`--dense` without a weight enables 0.3; omitting it still disables dense supervision for v2.
`--dense-mode peak` (default) keeps the positive BCE at the window maximum, but selects
one detached argmax (first on ties) and masks KD and negative truth only within +/-1 frame
of that argmax, clipped to the bag. Other classes and verified positives are preserved.
All other frames retain their existing truth/KD weights. This penalizes broad elevated
windows relative to isolated peaks when the teacher is quiet, without inventing new
negatives outside accepted intervals. It encourages one peak; it is not a hard stroke-count
guarantee, and a 20-step smoke cannot establish the final precision/recall tradeoff.

`dense.py` now writes `dense-fold{0,1,2,all}.json` and `dense.folds.summary.json`.
Song/channel/cohort quality references are pooled only after the channel and same-song
exclusions for that fold. Held-out and duplicate songs receive empty labels. Training
requires the matching fold cache; it never silently falls back to legacy global criteria.
Existing `dense.json`, `dense.summary.json`, and v2/v3 checkpoints/results are preserved.
`--dense-labels legacy --dense-mode window --dense 1.0 --no-withheld-mask` retains the
historical v3 loss/label policy. The v3 runner also fixes its cohort to the original 14.

`data.py` records withheld heads from the usable strict systems' exact moment groups and
fitted times. The negative truth loss, including `--doubt`, is masked for their class within
+/-3 frames; verified positives and KD survive. An unclassified withheld head masks all
three learned families conservatively. These heads never become positive targets.

The original channel assignments stay fixed. New channels: fold 0 = Ollie Woods, Golly,
Ivannikov; fold 1 = MoYin; fold 2 unchanged. Readable manifest song-seconds are
10535 / 10730 / 12394 (title/ID audit on 2026-10-02; readable status is broader than nonempty
strict training truth). Ivannikov currently has no strict exports. The newly confirmed
cross-channel duplicate is Official HIGE DANdism **Mixed Nuts**: MoYin `SRKKu5mTeAE` /
Nakamura `DqSJyUZiHIk`, registered by IDs in `splits.SAME_SONG`. The other 19 new titles/IDs
have no confirmed cross-channel same-song match in the local harvested manifest;
generic Golly chop titles and anonymous solos do not establish underlying composition identity.

```bash
# CPU only, at most four workers; use retained stems/caches, never download.
.venv/bin/python -B tools/finetune/data.py --jobs 4  # cached old features reused; all truth metadata refreshed
.venv/bin/python -B tools/finetune/teacher.py --jobs 4 --only moyin olliewoods golly ivannikov
.venv/bin/python -B tools/finetune/dense.py

# Separate v4-smoke checkpoints, predictions and report; 20 steps per fold.
bash tools/finetune/run_v4.sh --smoke
# If MPS is unavailable in the execution environment, use CPU for the smoke only.
bash tools/finetune/run_v4.sh --smoke-cpu

# Full experiment, only when requested: 1500 steps per fold.
bash tools/finetune/run_v4.sh
```

The runner trains sequentially (fold 0 MPS; folds 1/2 CPU, four threads), predicts held-out
songs and averages the three folds for solos. Both v4 and v2 harvested scorers use explicit
`--only` filters for the original 14 channels so new songs never require v2 predictions.
The tracked `tools/harvest/global_score.py` forwards the same filter to `score.py`.
The full comparison is `.eval-cache/ft/results/v4.md`, generated by `report_v4.py` from
the integer scorer totals; smoke writes `v4-smoke.md` with an explicit 20-step limitation.
Derived snare responses are rebuilt by `prepare_score.py` on four CPU workers before
scoring, avoiding serial cache preparation in the global scorer.
New training data and the three data fixes also change this comparison: it does not isolate
the loss change alone. Neither runner builds or replaces the shipped model.

## Exclusive dense supervision

`--dense-mode exclusive` addresses the cross-instrument shortcut observed in
the GPU cover-fill diagnosis (`.eval-cache/ft/results/why/REPORT.md`): a dense
snare bag can otherwise place its positive on a nearby tom, or a tom bag on a kick.
The existing `peak` (still the default) and `window` paths retain their losses,
crop sampling, and label policies for reproduction.

Eligibility is computed once from original teacher activations and notation,
before cropping, between **kick/snare/tom**:

- Block other strict positive frames and the **entire original window** of each
  other-class dense label, expanded by `--dense-guard-frames` (default 2 = 20 ms).
- Also block other learned classes' teacher activations >=
  `--dense-teacher-threshold` (default 0.5), with the same guard. Hat/cymbal teacher
  activity keeps its KD but does not block candidates: this notation tier omits
  those instruments, and they often accompany legitimate kick/snare/tom strokes.
- Exempt an explicitly simultaneous dense label only when its exact `time`,
  `system`, and `moment` match (legacy labels without provenance use exact time).
  Overlapping windows or equal rounded frames are insufficient. Strict joint
  positives exempt their shared frame and +/-1-frame teacher neighborhood.
  Unrelated neighboring strokes still block candidates.
- Select one detached student argmax **only among eligible frames**. Give it
  positive BCE, and mask own-class KD/negative truth only at eligible frames
  within +/-1 of that peak. Blocked neighbors, other classes, and verified
  positives retain their losses. Empty bags receive no dense loss or dense
  sampling, with no fallback to an unsafe argmax.

Two optional controls isolate additional hypotheses without changing cached labels:
`--dense-radius N` intersects eligibility with +/-N frames of the original timing
estimate (never widens a bag or shrinks competitor protection).
`--dense-snare-weight W` scales each snare bag's BCE inside the mean over enabled,
eligible bags (`W=0` removes dense snare positives and sampling, teaching dense kick/tom only;
disabled snare labels still act as competitors). Both controls and guard/teacher
tuning require `exclusive`.

This chooses a candidate restriction over an additional hinge: an unsafe frame
receives no dense positive in the first place, while existing KD is preserved
there. Conservative eligibility may skip real strokes when teacher class
assignments or timing estimates are wrong; it is not a measured GPU improvement.
Training logs and checkpoint JSON record per-class `dense eligibility` counts
(`labels`, `usable`, `skipped`, `candidate_frames`), plus all CLI settings.

Run these three **separate tags**, each on folds 0, 1, 2, with v4's full cohort,
fold-local labels and withheld mask. Hold seed and all other settings fixed;
score the original 14 held-out channels and the same external solos.

```bash
# WSL: run only when starting the GPU experiment.
for fold in 0 1 2; do
  .venv/bin/python tools/finetune/train.py --device cuda --fold "$fold" \
    --tag v5ex0 --steps 1500 --doubt 0.3 --seed 0 \
    --dense 0.3 --dense-mode exclusive
  .venv/bin/python tools/finetune/train.py --device cuda --fold "$fold" \
    --tag v5ex_s025 --steps 1500 --doubt 0.3 --seed 0 \
    --dense 0.3 --dense-mode exclusive --dense-snare-weight 0.25
  .venv/bin/python tools/finetune/train.py --device cuda --fold "$fold" \
    --tag v5ex_r1 --steps 1500 --doubt 0.3 --seed 0 \
    --dense 0.3 --dense-mode exclusive --dense-radius 1
done
```

`dense.py` already records a timing estimate in `time`, plus source moment
provenance, so no label regeneration is needed for these flags. The narrower
variant treats that uncertain estimate more strongly and should be judged by
recall as well as precision. Use a second seed for a promising variant.
The existing [WSL tag launcher](../../docs/wsl-gpu.md) forwards these flags.

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider \
  --basetemp=.eval-cache/ft/test-exclusive-tmp tools/finetune
.venv/bin/python -m pytest -q
```

## Calibration and dense ablation

`ft_variants` accepts `FT_KICK_SCALE`, `FT_SNARE_SCALE`, `FT_TOM_SCALE`,
`FT_HIHAT_SCALE` and `FT_CYMBAL_SCALE` as peak-threshold multipliers (larger is
stricter). All default to 1. Snare-response and rescue thresholds stay fixed.

```bash
# Existing predictions/responses only; no downloads, inference or training.
# One class at a time, original 14 channels + 10 external solos, four CPU workers.
.venv/bin/python -B tools/finetune/calibrate.py

# v4's channels/folds/withheld masks, with dense loss and dense sampling disabled.
# Separate v4nd-smoke outputs; 20 steps per fold on CPU, then evaluation.
bash tools/finetune/run_v4_ablate.sh --smoke-cpu
# Prepared full ablation (1500 steps/fold); run only when requested.
bash tools/finetune/run_v4_ablate.sh
```

Calibration writes integer counts and operating points to
`.eval-cache/ft/results/calibration/sweep.json` and per-song files beside it.
It reuses the existing scorer functions, memoizes independent column peak
picking, and retains the tom stray filter while skipping pitch naming for
family metrics. Default kick/snare/tom totals must exactly reproduce retained
v2/v3/v4 reports. Matched precision means maximum observed recall with precision
at least v2's default precision; no interpolation. Other classes stay at scale 1.
These are exploratory threshold choices on evaluation data, not an independent
confirmation of a deployment threshold.

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m pytest -q tools/finetune
```

## Results (2026-10-01)

Every harvested song scored by the fold model that never saw its channel; solos (an external
test, never trained on) by the shipped model. Full pipeline, default thresholds, with the snare
response remade from each model's activations as the pipeline does (`ft_variants.snare`).

| tom P / R | original | fine-tuned v2 |
|---|---|---|
| all 131 songs (`score.py`) | .76 / .79 | .77 / .80 |
| cover fills, strokes < 90 ms apart (`fill_score.py`) | .86 / .90 | .88 / .90 |
| drum solos, strokes < 90 ms apart (`fill_seq.py`) | .46 / .72 | .47 / .75 |

Kick over all songs .94 / .96 → .93 / .98, snare .80 / .95 → .83 / .97; very fast cover-fill
kick .96 / .83 → .94 / .87 and snare .94 / .86 → .94 / .93; solo very fast kick recall
.67 → .73. Precision against this truth is a lower bound (unwritten strokes count as false).

v1 (no `--doubt`) found as many strokes but lost tom precision (.74 / .82 over all songs).
Running ADTOF on time-stretched audio, rejected earlier, did not help either.

The cached activations of the harvested and solo songs are the original model's; scoring the
shipped model on harvested songs would score it on its own training data.
