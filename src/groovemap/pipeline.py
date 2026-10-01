"""Run the stages for one song directory, reusing cached model outputs."""

from __future__ import annotations

import json
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import fingerprint, kit, library, models, render
from .fetch import SongMeta
from .fills import attack_times, fill_snares, tom_runs
from .midi import to_midi
from .musicxml import to_musicxml
from .quantize import quantize
from .score import Hit, Score
from .toms import tom_sizes


@dataclass
class Options:
    grid: int = 16
    triplets: str = "auto"
    meter: int | None = None
    time_signature: str | None = None
    odd_bars: str = "none"  # none: regular bars; auto: odd bars when detections insist; detected
    beat_shift: float = 0  # move beat 1 later by this many beats, in steps of a quarter beat
    tempo_scale: float = 1  # 2 counts the beat twice as fast as the tracker did, 0.5 half
    voices: int = 1
    device: str = "auto"
    demucs_model: str = "htdemucs"
    onset_input: str = "drums"  # run ADTOF on the separated "drums" stem or the full "mix"
    thresholds: list[float] | None = None
    sensitivity: float = 1  # divides the default thresholds: 2 picks up hits half as strong
    sound_names: dict[str, str] = field(default_factory=dict)  # the user's names for the song's sound groups (kit.py)
    pdf: bool = True
    preview: bool = False
    force: bool = False
    refresh_stale: bool = False  # remake caches whose code changed; off keeps re-scoring from caches fast

    def __post_init__(self) -> None:
        if self.time_signature is not None:
            from .metre import Signature

            Signature.parse(self.time_signature)
        if any(name not in kit.NAMES for name in self.sound_names.values()):
            raise ValueError(f"sound names must be one of {kit.NAMES}")


@dataclass
class Result:
    score: Score
    files: dict[str, Path] = field(default_factory=dict)
    hit_counts: Counter = field(default_factory=Counter)
    note_counts: Counter = field(default_factory=Counter)


def cache_paths(song_dir: Path, opts: Options) -> dict[str, Path]:
    """The song's audio and cached model outputs under these options, by role."""
    onset_tag = "mix" if opts.onset_input == "mix" else f"drums-{opts.demucs_model}"
    return {
        "audio": song_dir / "audio.wav",
        "drums": song_dir / f"drums.{opts.demucs_model}.wav",
        "beats": song_dir / "beats.json",
        "activations": song_dir / f"activations.{onset_tag}.npy",
        # On the stem, two more looks at the song refine the hits: ADTOF on the full
        # mix (toms) and the snare's loudness (ghost notes); attacks complete the fills.
        "mix": song_dir / "activations.mix.npy",
        "snare": song_dir / f"snare.{opts.demucs_model}.npy",
        "attacks": song_dir / f"attacks.{opts.demucs_model}.npy",
    }


def detect_hits(song_dir: Path, opts: Options) -> tuple[list[Hit], np.ndarray, np.ndarray]:
    """The detector's hits from the cached model outputs, with the activations and attacks behind them.

    Before sound names and tom sizing: this is what the sound map groups. On
    the mix there is no stem to hear attacks in, so they are empty.
    """
    paths = cache_paths(song_dir, opts)
    thresholds = opts.thresholds or [t / opts.sensitivity for t in models.DEFAULT_THRESHOLDS]
    activations = np.load(paths["activations"])
    if opts.onset_input == "mix":
        return models.pick_hits(activations, thresholds), activations, np.empty(0)
    mix, attacks = np.load(paths["mix"]), np.load(paths["attacks"])
    hits = models.pick_hits(activations, thresholds, mix=mix, snare=np.load(paths["snare"]))
    return tom_runs(hits, activations, mix, attacks, thresholds), activations, attacks


def run(
    song_dir: Path,
    meta: SongMeta,
    opts: Options,
    log: Callable[[str], None] = print,
    progress: Callable[[str, str, float | None], None] | None = None,
) -> Result:
    device = models.pick_device(opts.device)
    paths = cache_paths(song_dir, opts)
    audio, drums, beats_json, activations_npy = paths["audio"], paths["drums"], paths["beats"], paths["activations"]
    refine = opts.onset_input != "mix"
    mix_npy, snare_npy, attacks_npy = paths["mix"], paths["snare"], paths["attacks"]

    def report(key: str, status: str, seconds: float | None = None) -> None:
        if progress is not None:
            progress(key, status, seconds)

    fingerprints = fingerprint.snapshot(check=True)
    stamps = library.cache_stamps(song_dir)

    def stale(output: Path) -> bool:
        """Made by other code than this, or by code from before caches were stamped."""
        return (
            opts.refresh_stale and fingerprints is not None
            and stamps.get(output.name) != fingerprint.cache_stamp(output.name, fingerprints)
        )

    def stage(key: str, name: str, *steps: tuple[Path, Callable[[Path], None]]) -> None:
        """Make each output that is missing or stale; a stage with nothing to make is cached."""
        todo = [(output, work) for output, work in steps if opts.force or not output.exists() or stale(output)]
        if not todo:
            log(f"{name}: cached")
            report(key, "cached")
            return
        report(key, "running")
        started = time.monotonic()
        for output, work in todo:
            # Written under a temporary name so a failed run never leaves a half-made cache.
            part = output.with_name(f".part.{output.name}")
            try:
                work(part)
                part.replace(output)
                stamps[output.name] = fingerprint.cache_stamp(output.name, fingerprints)
                library.stamp_cache(song_dir, output.name, stamps[output.name])
            finally:
                part.unlink(missing_ok=True)
        seconds = time.monotonic() - started
        log(f"{name}: {seconds:.0f}s")
        report(key, "done", seconds)

    def separate(target: Path) -> None:
        models.separate_drums(audio, target, device=device, model=opts.demucs_model)

    def beats(target: Path) -> None:
        found_beats, found_downbeats = models.track_beats(audio, device=device)
        target.write_text(json.dumps({"beats": found_beats, "downbeats": found_downbeats}))

    def onsets(target: Path) -> None:
        if opts.onset_input == "mix":
            np.save(target, models.drum_activations(audio, device=device))
        else:
            np.save(target, models.stem_activations(drums, device=device))

    def onsets_on_mix(target: Path) -> None:
        np.save(target, models.drum_activations(audio, device=device))

    def snare_loudness(target: Path) -> None:
        from .dynamics import snare_response

        np.save(target, snare_response(drums, np.load(activations_npy)))

    def attacks(target: Path) -> None:
        np.save(target, attack_times(drums))

    if opts.onset_input != "mix" or opts.preview:
        stage("separate", f"drum separation ({opts.demucs_model}, {device})", (drums, separate))
    stage("beats", "beat tracking", (beats_json, beats))
    refinements = [(mix_npy, onsets_on_mix), (snare_npy, snare_loudness), (attacks_npy, attacks)] if refine else []
    stage("onsets", f"drum transcription (on {opts.onset_input})", (activations_npy, onsets), *refinements)

    report("score", "running")
    started = time.monotonic()
    grid = json.loads(beats_json.read_text())
    hits, activations, attacks = detect_hits(song_dir, opts)
    if refine:
        if opts.sound_names:
            # The user's names for this song's sounds, before the toms are sized: a group
            # named "tom" is sized by pitch like the detector's own, a specific size stays.
            sound_map = kit.load_or_learn(song_dir, drums, activations_npy, attacks_npy, hits, attacks)
            named = kit.rename(hits, sound_map, opts.sound_names, drums, attacks)
            log(f"sound map: {len(opts.sound_names)} names applied, {len(hits)} hits -> {len(named)}")
            hits = named
        else:
            # Quiet snares in fills only where the user has not named the song's sounds: an
            # attack they sorted into a group of its own must not come back as a snare.
            hits = fill_snares(hits, activations, np.load(snare_npy), attacks)
        hits = tom_sizes(drums, hits)
    common = dict(title=meta.title, subtitle=meta.uploader, grid=opts.grid,
                  triplets=opts.triplets, beat_shift=opts.beat_shift, tempo_scale=opts.tempo_scale)
    if opts.time_signature is not None:
        from .signature_quantize import quantize_signature

        score = quantize_signature(
            hits, grid["beats"], grid["downbeats"], activations=activations,
            attacks=np.load(attacks_npy) if attacks_npy.exists() else [],
            time_signature=opts.time_signature, **common,
        )
    else:
        score = quantize(hits, grid["beats"], grid["downbeats"], meter=opts.meter, odd_bars=opts.odd_bars, **common)
        if opts.meter is None:
            from .meter_suggestion import suggest_signature

            score.time_signature_suggestion = suggest_signature(
                score, activations, np.load(attacks_npy) if attacks_npy.exists() else [],
                grid["beats"], grid["downbeats"],
            )

    if score.merged_strokes:
        log(f"quantization: {score.merged_strokes} strokes merged into occupied slots")

    name = song_dir.name
    files = {
        "json": song_dir / "score.json",
        "musicxml": song_dir / f"{name}.musicxml",
        "midi": song_dir / f"{name}.mid",
    }
    score.save(files["json"])
    files["musicxml"].write_text(to_musicxml(score, voices=opts.voices), encoding="utf-8")
    to_midi(score, files["midi"])

    if opts.pdf:
        pdf = song_dir / f"{name}.pdf"
        if render.render_pdf(files["musicxml"], pdf):
            files["pdf"] = pdf
        else:
            log("pdf: skipped (MuseScore not found or failed to engrave)")
    if opts.preview:
        files["preview"] = song_dir / "preview.wav"
        render.render_preview(score, audio, files["preview"])

    report("score", "done", time.monotonic() - started)
    return Result(
        score=score,
        files=files,
        hit_counts=Counter(h.inst for h in hits),
        note_counts=Counter(n.inst for bar in score.bars for n in bar.notes),
    )
