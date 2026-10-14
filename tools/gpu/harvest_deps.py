"""Generate pip inputs from pyproject, pin the installed CUDA stack, or verify.

Does not install packages or download model weights. Run on WSL, never uv sync.
"""
from __future__ import annotations

import argparse
from importlib import import_module, metadata
import json
from pathlib import Path
import shutil
import subprocess
import tomllib

ROOT = Path(__file__).resolve().parents[2]


def freeze(destination: Path) -> None:
    import torch
    if not torch.cuda.is_available() or not torch.__version__.endswith('+cu128'):
        raise RuntimeError('expected the existing working torch cu128 CUDA venv')
    destination.mkdir(parents=True, exist_ok=True)
    protected = {}
    for name in ('torch', 'torchaudio', 'torchvision', 'torchcodec'):
        try:
            protected[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    # The beat tracker needs matching torchaudio. Install its CUDA wheel separately.
    protected.setdefault('torchaudio', torch.__version__)
    (destination / 'cuda-constraints.txt').write_text(''.join(f'{n}=={v}\n' for n, v in protected.items()))
    project = tomllib.loads((ROOT / 'pyproject.toml').read_text())
    requirements = []
    for dependency in project['project']['dependencies']:
        if dependency.startswith('adtof-pytorch'):
            source = project['tool']['uv']['sources']['adtof-pytorch']
            dependency = f"adtof-pytorch @ git+{source['git']}@{source['rev']}"
        requirements.append(dependency)
    requirements += [d for d in project['dependency-groups']['dev'] if d.startswith('opencv-python-headless')]
    requirements += ['yt-dlp[default]']
    (destination / 'harvest-requirements.txt').write_text('\n'.join(requirements) + '\n')
    (destination / 'cuda-before.json').write_text(json.dumps({'torch': torch.__version__, 'cuda': torch.version.cuda}))
    print((destination / 'harvest-requirements.txt').read_text(), end='')
    print('Protected:', protected)


def verify() -> None:
    import torch
    if not torch.cuda.is_available() or not torch.__version__.endswith('+cu128'):
        raise RuntimeError('CUDA/cu128 unavailable; no CPU fallback')
    before = ROOT / 'logs/harvest/deps/cuda-before.json'
    if before.exists() and json.loads(before.read_text())['torch'] != torch.__version__:
        raise RuntimeError('torch changed during dependency installation')
    for name in ('torchaudio', 'demucs.api', 'beat_this.inference', 'adtof_pytorch.audio',
                 'cv2', 'librosa', 'soundfile', 'scipy', 'groovemap.cli'):
        import_module(name)
        print('import OK:', name)
    for name in ('ffmpeg', 'ffprobe', 'yt-dlp', 'deno', 'flock', 'setsid', 'rsync'):
        path = shutil.which(name)
        if not path:
            raise RuntimeError(f'{name} missing from PATH')
        print(name, path)
    # Verify libopus, needed by compact, without touching the dataset.
    encoders = subprocess.check_output(['ffmpeg', '-hide_banner', '-encoders'], text=True, stderr=subprocess.STDOUT)
    if 'libopus' not in encoders:
        raise RuntimeError('ffmpeg lacks libopus')
    sample = torch.ones(8, device='cuda')
    print('CUDA allocation OK:', float(sample.sum()), torch.__version__, torch.cuda.get_device_name())


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('freeze', 'verify'))
    p.add_argument('--out', type=Path, default=ROOT / 'logs/harvest/deps')
    a = p.parse_args()
    freeze(a.out) if a.command == 'freeze' else verify()
