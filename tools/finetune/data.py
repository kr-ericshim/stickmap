"""Training data for fine-tuning ADTOF on the harvested transcription truth.

For every harvested song with strict truth this caches, under
.eval-cache/ft/<slug>/:

- feat.npy     ADTOF's input (100 fps log-frequency spectrogram of the drum stem), float16
- teacher.npy  the original model's activations on the same stem (copied from the song's cache)
- truth.json   channel, accepted intervals and the notated kick/snare/tom/hihat/cymbal times

    .venv/bin/python tools/finetune/data.py [--jobs 4] [--only moyin olliewoods golly ivannikov]
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
import vault  # noqa: E402
from evaluate import family  # noqa: E402
from score import load, songs  # noqa: E402

OUT = ROOT / ".eval-cache" / "ft"
CLASSES = ("kick", "snare", "tom", "hihat", "cymbal")  # ADTOF's output columns


def withheld_heads(strip: list[dict], alignment: dict) -> list[dict]:
    """Use the exact strict export's moment groups/times, retaining withheld heads.

    An unclassified head masks all learned families conservatively. These are
    uncertainty masks, not additional positives or accepted intervals.
    """
    systems = {s["system"]: s for s in strip}
    out = []
    for r in alignment["systems"]:
        if not r.get("usable"):
            continue
        s = systems[r["system"]]
        times, groups = r["times"], r["moment_groups"]
        if len(times) != len(groups):
            raise ValueError(f"system {r['system']}: times / moment_groups mismatch")
        for t, group in zip(times, groups):
            for index in group:
                head = s["notes"][index]
                if head.get("exportable") is not False:
                    continue
                cls = family(head.get("inst", "unknown"))
                for c in (cls,) if cls in CLASSES else CLASSES[:3]:
                    out.append({"time": round(float(t), 4), "class": c,
                                "system": r["system"], "head": index,
                                "inst": head.get("inst", "unknown"),
                                "pos": head.get("pos"), "shape": head.get("shape")})
    return sorted(out, key=lambda q: (q["time"], q["class"]))


def stem(song: Path) -> Path:
    wav = song / "drums.htdemucs.wav"
    return wav if wav.exists() else wav.with_suffix(".opus")


def one(row: dict) -> str | None:
    if vault.is_vault_song(row):
        return None
    import torch
    from adtof_pytorch.audio import process_audio_file

    torch.set_num_threads(1)

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
    withheld = withheld_heads(json.loads((song / "truth" / "strip.json").read_text()),
                              json.loads((song / "truth" / "alignment.json").read_text()))
    (out / "truth.json").write_text(json.dumps({
        "slug": row["slug"], "video_id": row["video"], "channel": row["channel"], "intervals": intervals, "events": events,
        "withheld_heads": withheld, "withheld_tolerance_frames": 3,
    }))
    return row["slug"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--only", nargs="+", help="cache only these channels (no downloads)")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(args.jobs) as pool:
        done = [s for s in pool.map(one, songs(args.only)) if s]
    print(len(done), "songs cached in", OUT)


if __name__ == "__main__":
    main()
