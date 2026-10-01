# Fine-tuning ADTOF on transcription videos

The detector's misses in fast fills are frames where ADTOF's activation never rises, so no
peak-picking rule can recover them. Here ADTOF's Frame-RNN is fine-tuned on the harvested
ground truth (`tools/harvest`, 131 songs from 14 channels, 3.4 h inside accepted systems).
The result ships as `src/groovemap/adtof-groovemap-v2.pt` and is used on the drum stem only.

```bash
uv run python tools/finetune/data.py                  # features + truth per song -> .eval-cache/ft/
uv run python tools/finetune/teacher.py               # the original model on those features
uv run python tools/finetune/train.py --fold 0 --tag v2 --doubt 0.3   # also folds 1, 2
uv run python tools/finetune/predict.py v2-fold0      # held-out songs' activations (each fold)
uv run python tools/finetune/solos.py v2              # drum-solo videos, never trained on
FT_TAG=v2 PYTHONPATH=tools/finetune uv run python tools/harvest/fill_score.py --detect ft_variants:tuned
uv run python tools/finetune/train.py --fold all --tag v2 --doubt 0.3 # the shipped model
```

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
