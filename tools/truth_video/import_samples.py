"""Register retained sample videos for the offline batch, without downloading.

Moves each existing song directory into the canonical harvest songs directory.
The append-only journal records provenance; a rerun also recovers a directory
already moved before its fetched record was committed. No source is deleted.
"""
import argparse
import json
from pathlib import Path
import re
from batch import ROOT, manifest_read, manifest_append


def register(channel, root=ROOT/'truth/harvest', samples=ROOT/'truth/samples'):
    config=json.loads((ROOT/'tools/harvest/channels.json').read_text())['channels']
    profile=next((c for c in config if c['name']==channel),None)
    if profile is None or channel not in ('golly','ivannikov'):
        raise ValueError('only configured golly/ivannikov samples may be imported')
    source=samples/channel
    destination=root/'songs';destination.mkdir(parents=True,exist_ok=True)
    existing=manifest_read(root/'manifest.jsonl');registered=[]
    # Canonical candidates permit recovery after a move interrupted the append.
    candidates={p.name:p for p in destination.iterdir() if p.is_dir() and (p/'meta.json').is_file()}
    candidates.update({p.name:p for p in source.iterdir() if p.is_dir()} if source.exists() else {})
    for slug,song in sorted(candidates.items()):
        if song.is_symlink():raise ValueError(f'symlink sample is not admitted: {song}')
        meta_path=song/'meta.json'
        if not meta_path.is_file():continue
        meta=json.loads(meta_path.read_text());title=meta.get('title','')
        uploader='Vincent Golly' if channel=='golly' else 'Eugeniy Ivannikov'
        if meta.get('uploader')!=uploader or not re.search(profile['include'],title,re.I):continue
        video=meta.get('video_id')
        if not isinstance(video,str) or not re.fullmatch(r'[A-Za-z0-9_-]{11}',video):
            raise ValueError(f'invalid video identity: {song}')
        target=destination/slug
        if video in existing:
            row=existing[video]
            if row.get('channel')!=channel or row.get('slug')!=slug:
                raise ValueError(f'journal identity conflict: {video}')
            if song!=target:raise FileExistsError(f'already registered source is still outside harvest: {song}')
            continue
        for name in ('attacks.htdemucs.npy',):
            if not (song/name).is_file():raise FileNotFoundError(song/name)
        origin=str(source/slug)
        if song!=target:
            if target.exists():raise FileExistsError(target)
            song.rename(target)
        row=dict(video=video,channel=channel,layout=profile['layout'],title=title,
                 seconds=meta.get('duration'),slug=slug,status='fetched',
                 imported_from=origin,import_method='retained local directory move; no download')
        row['retained_missing_inputs']=[name for name in ('video.mp4','attacks.htdemucs.npy')
                                        if not (target/name).is_file()]
        manifest_append(root/'manifest.jsonl',**row);existing[video]=row
        registered.append(row)
    return registered


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--only',nargs='+',choices=('golly','ivannikov'),default=['golly','ivannikov'])
    a=p.parse_args()
    for channel in a.only:
        for row in register(channel):print(f'{channel}: {row["video"]} fetched (retained local files)',flush=True)
