"""Training data for fine-tuning ADTOF on the harvested transcription truth.

For every harvested song with strict truth this caches, under
.eval-cache/ft/<slug>/:

- feat.npy     ADTOF's input (100 fps log-frequency spectrogram of the drum stem), float16
- teacher.npy  the original model's activations on the same stem (copied from the song's cache)
- truth.json   channel, accepted intervals and the notated kick/snare/tom/hihat/cymbal times

    uv run python tools/finetune/data.py [--jobs 6]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "tools" / "truth_video"), str(ROOT / "tools" / "harvest")]

import harvest  # noqa: E402
from evaluate import family  # noqa: E402
from score import load, songs  # noqa: E402

OUT = ROOT / ".eval-cache" / "ft"
CLASSES = ("kick", "snare", "tom", "hihat", "cymbal")  # ADTOF's output columns


def stem(song: Path) -> Path:
    wav = song / "drums.htdemucs.wav"
    return wav if wav.exists() else wav.with_suffix(".opus")


def one(row: dict) -> str | None:
    from adtof_pytorch.audio import process_audio_file

    song = harvest.SONGS / row["slug"]
    try:
        truth, intervals = load(song / "truth", "strict")
    except FileNotFoundError:
        return None
    if not truth or not (song / "activations.drums-htdemucs.npy").exists():
        return None
    out = OUT / row["slug"]
    out.mkdir(parents=True, exist_ok=True)
    if not (out / "feat.npy").exists():
        feat, _ = process_audio_file(str(stem(song)))
        np.save(out / "feat.npy", feat[:, :, 0].astype(np.float16))
    shutil.copyfile(song / "activations.drums-htdemucs.npy", out / "teacher.npy")
    events = {c: sorted({round(e["time"], 4) for e in truth if family(e["inst"]) == c}) for c in CLASSES}
    (out / "truth.json").write_text(json.dumps({
        "slug": row["slug"], "channel": row["channel"], "intervals": intervals, "events": events,
    }))
    return row["slug"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=6)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(args.jobs) as pool:
        done = [s for s in pool.map(one, songs(None)) if s]
    print(len(done), "songs cached in", OUT)


if __name__ == "__main__":
    main()
