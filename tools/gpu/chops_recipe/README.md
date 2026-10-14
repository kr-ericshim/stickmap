# Durable A–D comparison recipes

Use `../chops_compare.py run --name NAME --dry-run` to inspect the default
16 jobs (A–D × seeds 0/1 × two grouped folds), the explicit TRAIN channel roster,
and D's promotion gate. Dry-run reads config/manifest metadata only; audit
eligibility is unverified until the real plan checks TRAIN masks. The real runner freezes cohort,
grouping, budget, code and annotation hashes in a plan, then trains sequentially
on one visible CUDA GPU. `--folds 3` retains the historical three-fold policy
before cross-channel performance grouping (24 jobs). v5b is opt-in via
`--arms A B C D v5b`, or select it alone to reproduce the synthetic comparison.

| Arm | Training membership | Recipe |
|---|---|---|
| A | Original 14 channels only, vault excluded | Current v2, dense off, withheld mask off |
| B | All `splits.FOLDS` channels plus `Vault.train_channels`, vault excluded | Same v2 |
| C | B | `stratified.py` hierarchical crops |
| D | C | `three_state.py` audited three-state labels and conflict-masked KD |
| E | Exactly A's strict songs and v2 sampler; separate non-vault weak windows | `weakorder.py`, v2 + KD-only weak crop + 0.05 × order CTC |

Every arm uses the same held-out references; A never trains on the new data.
MoYin, Ollie Woods, Golly, Ivannikov and future `splits.FOLDS` extensions remain
in the channel roster. This never unseals a song: `Vault.contains_song` decides
membership before truth/features are read. Each real plan records and prints
per-arm/fold strict fast positive frames, class-specific independent fast
performances, accepted seconds and original/expanded song and time counts.
The wrapper replaces the current trainer's split function with the frozen
train/held lists. Initialization uses the original ADTOF weights, never shipped
fold-all weights. Default budget is 1500 steps, batch 12, 8-second crops,
Adam 1e-4, KD 0.3, frozen BatchNorm, unchanged augmentation and picker.
Proxy validation is logged only; it never selects the checkpoint.

## E: v2 recipe + order supervision

After the canonical WSL reread has finished, send code from the Mac with
`bash tools/gpu/harvest_wsl.sh push --tools-only`. Then on WSL:

```bash
cd ~/groovemap
.venv/bin/python -B tools/gpu/chops_compare.py run --name chopround3 --arms A,E --folds 2 --steps 1500 --gpu 0
```

This creates a **fresh** A/E plan with eight sequential jobs (two arms × two
seeds × two folds). A is rerun; chopround2 is not reused because identical
cohort, folds, seeds, budget, data and code have not been established. E's
strict training membership, held references, teacher, ADTOF initialization,
frame loss, crop sampling/augmentation, optimizer, step count and RNG streams
match A. Additional weak-only channels use the same channel hash and transitive
same-performance grouping; all arms in this plan share the resulting folds.
E adds **one weak sample to every 12-sample strict optimizer step**. Its own
NumPy RNG and forked Torch RNG preserve the strict batch/dropout streams.
The default CTC coefficient is **lambda=0.05** from WEAK_ORDER.md. The single
preregistered lower-strength alternative is `--weak-lambda 0.02`, with a new
comparison name; do not select it after repeatedly inspecting confirmation.

The auxiliary crop contains the entire padded system window, at most eight
seconds. Weak crops get no frame truth; K/S/T teacher KD is masked only inside
the source window, with v2 KD on context and HH/C. CTC uses the existing five
sigmoid outputs and no new head. It runs on **CPU with differentiable copies**
from/to the CUDA forward, because weak_order.py's tested backend is CPU.
Loss and parameter gradients must be finite. One hundred consecutive
greedy-blank weak windows stop training. Normalized CTC/KD loss and class/song
draws are logged every 100 steps and frozen in checkpoint metadata. E–A tests
the auxiliary branch including its KD policy; a matched weak-crop/KD-mask
lambda=0 control remains necessary before attributing an improvement solely to CTC.

`tools/finetune/weak_set.py` builds `weak-order-fold{fold}.json` **inside the
comparison directory**, without changing canonical truth/dense exports.
Admission defaults to [WEAK_ORDER.md](../../finetune/WEAK_ORDER.md)'s automated
policy: reader-clean >=8 K/S/T moments, a plausible cursor-based whole-system
span (40–300 bpm given notated beats), acoustic onset counts within the notated
HH/C-only budget plus ceil(15% of core moments), and >=85% core coverage by
monotone one-to-one drum-stem attack assignment. Tie continuations are removed.
Rhythm/tuplet/duration faults alone do not block admission; missing or duplicated
heads and unresolved identities do. Band identity/phase gates are off. The old
hash-bound human review remains opt-in via `weak_set.py --manual-review`.
The system window is +/-0.65 seconds, floor/ceil at 100 fps, stop
exclusive. Source/fitted/padded spans cannot overlap strict intervals or
strict fitted spans, including rounding and +/-1 frame protection. Overlapping
weak windows are excluded deterministically in source system order. No partial
target/crop is synthesized, and no scoring event export is used for tokens.

Vault/config/manifest is checked before notation/features and checked again
at training load. Whole held-out songs/channels, SAME_SONG, transitive held-out
performances and **every external `truth/<solo>/song.txt` source component**
(including JD VFJams outside the routine SOLOS roster) are excluded before
notation is read. Their sidecar entries are empty, with exclusion reports.
The same external-peer exclusions also filter strict cache preparation and
routine strict/loose/global scoring before annotation or prediction access;
unused harvest reposts cannot re-enter through a channel-wide scorer.
Source hashes, missing reviews, feature stamps, seal fingerprint, folds and
policy are frozen; subsequent rereads require a new plan. The writer lock
prevents starting a comparison during canonical harvest publication.

An empty reviewed set is a correct outcome under the design's current data
limitations. Real preflight prints the missing requirements and stops **before
any A/E training**. Reader rereading alone does not establish rational rhythm
or padded-window completeness. Inspect the Mac mirror without audio/GPU or
canonical writes using:

```bash
.venv/bin/python -B tools/finetune/weak_set.py --output .eval-cache/weak-order-mirror
.venv/bin/python -B tools/gpu/chops_compare.py run --name chopround3_preflight --arms A,E --dry-run
```

### Independent review input (not automatically generated)

Each non-vault song can provide `truth/weak_order_review.json`. Top-level
`schema_version=1`, `source_hashes` maps **filenames** `strip.json`,
`alignment.json`, `alignment.scoring.json` to their SHA256s; `systems` maps
string system IDs to review objects. This is a new review contract, not a claim
that current reader output contains verified rhythms. A review object requires:

- `quality_fold` integer, `reference_scope="training-only"`, nonempty
  `reviewer`, `evidence`, `source_revision`. Evidence must establish the whole
  padded audio span and all its K/S/T strokes without teacher/student labels.
- All of `staff_clean`, `clef_verified`, `legend_verified`,
  `visited_span_verified`, `reader_complete`, `heads_complete`,
  `duplicates_removed`, `rhythm_verified`, `ties_verified`, `rests_verified`,
  `grace_roll_verified`, `system_location_verified`, `complete_window_sequence`,
  `padding_sequence_verified` set explicitly to true; `unresolved_rhythm=0`
  and `unresolved_identity=0`. Booleans alone cannot establish human evidence.
- `source_span_seconds` equal to the **whole** strip system interval,
  `audio_offset_seconds`, and `feature_origin_seconds=0` for full-song caches.
  Margins containing other strokes must be independently resolved or excluded.
- `moments` in verified rational onset order. Each has `onset`, `duration`,
  `chord_id`, `tuplet_ratio`, `tie_chain`, `grace=false`, `roll=false`,
  `repeat_count=1`, and `head_indices` referencing source strip notes.
  Rational values are strings such as `"1/8"` or integer `[1,8]` pairs.
  Expand verified rolls/repeats explicitly in source notation before admission;
  unresolved ones are excluded. Source core heads are covered exactly once;
  tie continuations are excluded, simultaneous tom positions merge within a
  moment, and repeated tom moments remain separate tokens.
- Optional positive finite `diagnostic_tempo_bpm` permits approximate fast
  K/S/T counts from rational beats (<90ms nearest distinct neighbour). These
  counts are diagnostic only; absence is recorded, not interpreted as zero
  verified fast content. No beat or per-note time becomes a frame-positive.

E is promotion-eligible alongside A; D's audited-negative gate does not apply
to it. Its **primary metric is external solo fast K/S/T macro F1**, printed in
comparison Markdown/JSON for both seeds. E is “promising” only if its gain
over A is **at least +2 percentage points in each seed**, with no strict
fast/slow/all K/S/T or global K/S/T/HH/C family F1 drop above **0.5 points**;
pooled all-five-family global F1 is guarded too. Missing denominators mean
inconclusive. This is a screening rule, not a replacement for WEAK_ORDER.md's
additional independent confirmation/support/bootstrap gates. E plans explicitly
reject `final`; no E workflow opens the vault or replaces deployment weights.

C/D select **bucket → performance group → song → K/S/T → position**.
Exact unisons are not additional draw targets. Bucket targets: 50% fast, 30% original ordinary replay,
20% slow/context. Fast means a strict written K/S/T moment with a nearest
distinct written neighbour <90 ms inside the same accepted interval. Rejected
gaps cannot connect fast moments; repeated/overlapping intervals are deduplicated. Slow crops centre
on non-fast moments and may include fast context in the eight-second crop.
Empty buckets are omitted and their probabilities renormalized, visibly in
checkpoint JSON. Available classes are uniform within the selected
song; this cannot balance a class missing from a performance. Original replay
uses the unchanged v2 crop routine. Logs record bucket/group/class draws,
fast-labelled frame exposures, unique fast-labelled frames seen, selected
centre labels, actual unison exposure, song draws and original/expanded draws.
Class balancing can still select an unison via either constituent class;
there is no extra combination target or additional unison weight.

D assigns positive only to strict written frames, with no dense bags or
teacher eligibility gate. K/S/T negatives require a canonical
`truth/harvest/songs/<slug>/truth/kst_audit.json`. No audit means no negative.
D without usable TRAIN negatives is labelled **positive-only stress test**,
warned about and excluded from promotion in plan/checkpoint/report metadata.
Audits on held-only, unrelated or sealed songs cannot clear this gate. Every
training fold must contain usable audited negative frames to clear it; an
empty audit is insufficient. Passing this gate is not itself promotion.
Mixed audit coverage remains a diagnostic stress test. D–C changes multiple
loss policies and must not be interpreted as the pure effect of three-state labels.
Unknown K/S/T has **zero supervised and KD loss**, including rejected gaps,
outside accepted intervals and withheld competing heads. Thus this strict
unknown rule also removes v2's unaudited K/S/T replay; hat/cymbal KD remains,
and audited K/S/T negatives retain KD away from conflicts. All K/S/T KD is
off within ±2 frames of any written K/S/T onset. Missing competing labels
do not become negatives. Exact written unisons remain multi-label positives.
Positive, negative and KD BCE terms each normalize by their own valid count;
negative coefficient is 1, KD coefficient 0.3. No onset auxiliary head is added.

Optional audit schema (schema support is implemented; this repository change
does not create or claim any human-audited anchors):

```json
{
  "schema": 1,
  "auditor": "person or independent audit identifier",
  "evidence": "source video and waveform review record",
  "complete_intervals": [
    {"start": 12.0, "end": 16.0, "classes": ["kick", "snare", "tom"]}
  ],
  "verified_negative": [
    {"start": 17.0, "end": 17.1, "class": "kick"}
  ]
}
```

Training audit intervals must lie inside one strict accepted interval.
Complete intervals must establish that the strict event list contains **all**
K/S/T strokes there; add missing positives through the authorized truth audit
workflow before claiming completeness. Negative-only intervals cannot overlap
a written positive. ±2 frames around own positives are not negative targets;
withheld heads mask ±3 frames. These checks validate the schema and masks;
they cannot verify an auditor's claim from JSON alone. Audits and their absence
are frozen in the routine plan so they cannot be added during a comparison.
Vault audits are read and hashed only after the single durable opening.

Routine loose/solo scoring uses `../chops_scoring.py`, preserving the legacy
label-dependent ±60 ms DP. Matched predictions inherit their matched reference's
fast/slow subset, including extra predicted classes. Unmatched predictions use
fixed nearest-reference time regions (earlier reference on ties, slow without
references). Strict scoring retains its fill-window definition and exposes
its legacy `fill` subset as `slow` too. All paths enforce `TP <= est`,
`TP <= ref` and `all = fast + slow` for each class and stroke count. Final
ensemble scoring uses the same corrected counter; final is still a separate
explicit vault opening. Existing plans must be replaced after these code changes.

## Historical selectable v5b

`train_v2frozen.py` and `train_synth.py` were copied from `.eval-cache/chops/`.
The former is the historical git-HEAD snapshot used by that experiment. It is
never regenerated from today's working `tools/finetune/train.py` on WSL.

Source SHA256 before relocating paths:

```text
train_v2frozen.py 4f45841105d3fe2de7a59d126bf5b050c47bf46f6b3e60bfef485fd79a093c02
train_synth.py    e17a269754952fc2d3889a23f33021abb7f4ab448529cc5a1d291bf604ef4daf
```

Only repository-root/import paths and the synthetic cohort log wording were changed in those copies. Use
`../chops_compare.py`, not either historical script directly: `train_recipe.py`
applies the vault API **before loading Song/features/teacher**, assigns new
train channels to held-out folds, retains the current shared SAME_SONG rules,
and puts checkpoints in `.eval-cache/ft/models/` for the existing detector.
The expanded `train.FOLDS` also expands the synthetic script's original
14-channel filter. Solos never enter training or the synthetic snippet bank.

v2 uses current `tools/finetune/train.py --steps 1500 --doubt 0.3 --dense 0
--no-withheld-mask`. v5b uses the frozen trainer, synthesis on, low-prior
probabilities `0.08/0.05/0.05`, the same steps/doubt, and no dense/solo/maskfix
options. Defaults for batch size, crop length, optimizer, learning rate,
distillation and validation remain those of each historical recipe. All source
hashes, cohort truth hashes and the byte-preserved manifest/config hashes are
recorded in each comparison's `plan.json`; seeds 0 and 1 are separate tags.
