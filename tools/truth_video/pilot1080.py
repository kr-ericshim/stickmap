"""Offline scratch pilots: separation and notation alignment, never drum inference.

All outputs stay beside the supplied video/audio under .eval-cache/pilot1080.
No harvest manifest or canonical song is written. A missing local Demucs weight
fails instead of downloading it. The same detector-free path serves test-only
sources; no transcription/scoring CLI is invoked.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
from media import video_path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PILOTS = ROOT / '.eval-cache/pilot1080'
TEST_ONLY = {'SHvfn-bM_rg', '8Eh6uT0j_D4'}
ENV = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '1',
       'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
       'VECLIB_MAXIMUM_THREADS': '1'}


def metadata(video):
    channel = 'woshudiue' if video in TEST_ONLY else 'dailychops'
    rows = (ROOT / f'truth/harvest/lists/{channel}.txt').read_text().splitlines()
    for row in rows:
        vid, seconds, title = row.split('|', 2)
        if vid == video:
            return dict(video=video, video_id=video, channel=channel,
                        seconds=float(seconds), title=title, test_only=video in TEST_ONLY,
                        provenance='local 1080p pilot; offline separation and notation only')
    raise ValueError(f'pilot not present in retained channel list: {video}')


def prepare(folder, meta=None):
    import cv2
    import numpy as np
    import torch
    from demucs.api import Separator
    import soundfile as sf
    sys.path.insert(0, str(ROOT / 'src'))
    from stickmap.fills import attack_times

    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    # Even if a weight is absent, this entry point cannot access the network.
    def offline(*args, **kwargs):
        raise RuntimeError('offline pilot: required Demucs weights are not cached')
    torch.hub.download_url_to_file = offline
    meta = metadata(folder.name) if meta is None else dict(meta)
    cap = cv2.VideoCapture(str(video_path(folder)))
    meta['native_video'] = dict(width=int(cap.get(3)), height=int(cap.get(4)),
                                fps=cap.get(5), frames=int(cap.get(7)))
    cap.release()
    (folder / 'meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2) + '\n')
    mix, stem = folder / 'audio.wav', folder / 'drums.htdemucs.wav'
    if not stem.exists():
        subprocess.run(['ffmpeg', '-loglevel', 'error', '-threads', '1', '-y',
                        '-i', str(folder / 'audio.m4a' if (folder / 'audio.m4a').exists()
                                  else video_path(folder)), '-ar', '44100', '-ac', '2',
                        str(mix)], check=True)
        audio, sr = sf.read(mix, dtype='float32', always_2d=True)
        separator = Separator(model='htdemucs', device='cpu', progress=False)
        _, stems = separator.separate_tensor(torch.from_numpy(audio.T.copy()), sr)
        part = folder / 'drums.htdemucs.part.wav'
        sf.write(part, stems['drums'].cpu().numpy().T, separator.samplerate)
        part.replace(stem)
        mix.unlink()  # supplied compressed mix remains available
    attacks = folder / 'attacks.htdemucs.npy'
    if not attacks.exists():
        np.save(attacks, attack_times(stem))
    from bands import load
    load(folder)
    print(folder.name, 'prepared', len(np.load(attacks)), 'class-agnostic attacks', flush=True)


def run(phase, videos, jobs=3, output='truth'):
    if not 1 <= jobs <= 4:
        raise ValueError('jobs must be between one and four')
    if Path(output).name != output:
        raise ValueError('output must be one folder name')
    def attempt(video):
        folder = PILOTS / video
        if phase == 'prepare':
            args = [sys.executable, str(HERE / 'pilot1080.py'), '--prepare-worker', video]
            log = folder / 'prepare.log'
        else:
            channel = metadata(video)['channel']
            out = folder / output
            if (out / 'result.json').exists():
                raise FileExistsError(f'preserve existing read: {out}; choose --output')
            out.mkdir(exist_ok=True)
            args = [sys.executable, str(HERE / 'batch.py'), '--worker', str(folder),
                    str(out), str(HERE / f'legends/{channel}.json')]
            log = out / 'run.log'
        with log.open('w') as stream:
            result = subprocess.run(args, env=ENV, stdout=stream, stderr=subprocess.STDOUT)
        print(video, phase, result.returncode, str(log.relative_to(ROOT)), flush=True)
        return dict(video=video, phase=phase, returncode=result.returncode)
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        return list(pool.map(attempt, videos))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', nargs='?', choices=('prepare', 'read'))
    parser.add_argument('--prepare-worker')
    parser.add_argument('--videos', nargs='+')
    parser.add_argument('--jobs', type=int, default=3)
    parser.add_argument('--output', default='truth')
    args = parser.parse_args()
    if args.prepare_worker:
        prepare(PILOTS / args.prepare_worker)
    else:
        videos = args.videos or sorted(p.name for p in PILOTS.iterdir() if (p / 'video.mp4').is_file())
        results = run(args.phase, videos, args.jobs, args.output)
        raise SystemExit(any(r['returncode'] for r in results))
