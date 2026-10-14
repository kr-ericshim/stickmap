"""Optional outputs: PDF engraving through MuseScore, and an ear-check preview mix."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

from .score import Score

MUSESCORE_APP = Path("/Applications/MuseScore 4.app/Contents/MacOS/mscore")
# House style for the PDF: flat beams, so a beam's height does not jump with the drums under it.
STYLE = Path(__file__).with_name("engraving.mss")


def find_musescore() -> str | None:
    if MUSESCORE_APP.exists():
        return str(MUSESCORE_APP)
    return shutil.which("mscore") or shutil.which("musescore")


def render_pdf(musicxml: Path, pdf: Path) -> bool:
    """Engrave with MuseScore. Returns False when MuseScore is missing or fails."""
    mscore = find_musescore()
    if not mscore:
        return False
    proc = subprocess.run([mscore, "-S", str(STYLE), "-o", str(pdf), str(musicxml)], capture_output=True, text=True)
    return proc.returncode == 0 and pdf.exists()


def _voice(inst: str, sr: int, rng: np.random.Generator) -> np.ndarray:
    """A crude but distinguishable one-shot per instrument."""

    def tone(freq: float, dur: float, decay: float) -> np.ndarray:
        t = np.arange(int(sr * dur)) / sr
        return np.sin(2 * np.pi * freq * t) * np.exp(-t * decay)

    def noise(dur: float, decay: float, highpass: bool) -> np.ndarray:
        n = rng.standard_normal(int(sr * dur))
        if highpass:
            n = np.diff(n, prepend=0.0)
        return n * np.exp(-np.arange(len(n)) / sr * decay) * 0.5

    if inst == "kick":
        return tone(55, 0.18, 22)
    if inst == "snare":
        return 0.6 * tone(190, 0.14, 35) + noise(0.14, 30, False)
    if inst == "sidestick":
        return tone(900, 0.04, 90)
    if inst.startswith("tom"):
        return tone({"tom_high": 180, "tom": 130, "tom_low": 85}[inst], 0.22, 16)
    if inst in ("hihat", "hihat_pedal"):
        return noise(0.05, 90, True)
    if inst == "hihat_open":
        return noise(0.3, 12, True)
    if inst == "ride":
        return 0.6 * noise(0.35, 9, True) + 0.2 * tone(3200, 0.35, 9)
    if inst == "ride_bell":
        return 0.5 * tone(2400, 0.4, 8) + 0.2 * noise(0.2, 20, True)
    return noise(0.5, 7, True)  # crash, and any other cymbal


def render_preview(score: Score, audio_wav: Path, out_wav: Path) -> None:
    """Original song on the left channel, the transcription re-synthesised on the right."""
    audio, sr = sf.read(audio_wav, dtype="float32", always_2d=True)
    original = audio.mean(axis=1)
    synth = np.zeros_like(original)
    rng = np.random.default_rng(0)
    voices = {}
    for bar in score.bars:
        for note in bar.notes:
            shot = voices.setdefault(note.inst, _voice(note.inst, sr, rng))
            # A padded pickup bar can start before the audio does.
            start = max(0, int(bar.note_time(note) * sr))
            if start >= len(synth):
                continue
            end = min(start + len(shot), len(synth))
            synth[start:end] += shot[: end - start] * (note.velocity / 127)
    peak = float(np.abs(synth).max()) or 1.0
    sf.write(out_wav, np.stack([original * 0.8, synth / peak * 0.8], axis=1), sr)
