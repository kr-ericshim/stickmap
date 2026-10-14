"""Select a retained native video without transcoding or extension guessing."""
from pathlib import Path
import json

VIDEO_NAMES = ('video.mp4', 'video.webm', 'video.mkv', 'video.mov')


def video_path(folder):
    paths = [Path(folder) / name for name in VIDEO_NAMES
             if (Path(folder) / name).is_file()]
    if len(paths) != 1:
        raise ValueError(f'expected one retained native video, found {len(paths)}: {folder}')
    return paths[0]


def source_video_id(video):
    """Use retained source identity, never infer identity from notation pixels."""
    meta=Path(video).parent/'meta.json'
    if not meta.exists():return None
    value=json.loads(meta.read_text()).get('video_id')
    return value if isinstance(value,str) else None
