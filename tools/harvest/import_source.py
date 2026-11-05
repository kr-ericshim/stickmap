"""Import detector-free sources + offline reads into the canonical harvest (WSL only).

    .venv/bin/python tools/harvest/import_source.py --channel olliewoods_shorts \
        --allow .eval-cache/chops_split/allow_olliewoods_shorts_train.json [--dry-run] [--device cuda]

For every allowed (train-side) video: sources/<channel>/<id>/ media -> truth/harvest/songs/<slug>/,
the original model's activations (activations.mix.npy, activations.drums-htdemucs.npy) computed from
the SAME drum stem/audio that the offline read was aligned to, the channel_read 'cv' truth tiers copied
unchanged into truth/, and manifest rows appended (fetched -> read/unreadable) under the shared harvest
writer lock. Refuses explicit vault IDs, IDs already in the manifest and existing song folders: nothing
is overwritten. Vault-side videos are never imported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/harvest'))
sys.path.insert(0, str(ROOT / 'tools/gpu'))
import vault  # noqa: E402
from manifest_io import append  # noqa: E402

SOURCES = ROOT / 'sources'
READS = ROOT / '.eval-cache/vision_pilot/channel_read'
TRUTH_FILES = ('truth.json', 'truth.scoring.json', 'truth.additional.json', 'alignment.json',
               'alignment.scoring.json', 'strip.json')
REVIEWED_FILES = ('legend.json', 'QUALITY.md', 'audio-sanity.json', 'label-coverage.json')


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def slugify(text: str) -> str:
    sys.path.insert(0, str(ROOT / 'src'))
    from stickmap.fetch import slugify as pipeline_slugify  # the pipeline's own folder naming
    return pipeline_slugify(text)


def plan(channel: str, allow: list[str]) -> list[dict]:
    seal = vault.Vault()
    rows = []
    for vid in allow:
        src = SOURCES / channel / vid
        fetch = json.loads((src / 'fetch.json').read_text())
        meta = json.loads((src / 'meta.json').read_text())
        if fetch.get('status') != 'ok' or not meta.get('detector_free'):
            raise ValueError(f'{vid}: source not fetched/prepared')
        if vid in seal.explicit or seal.is_vault(vid):
            raise ValueError(f'{vid}: sealed; refusing to import a vault video')
        title = meta.get('title') or fetch['title']
        slug = f'{slugify(title)}-{vid}'
        resume = False
        if vid in seal.first:
            mine = [r for r in vault.records(vault.MANIFEST) if r['video'] == vid]
            ours = mine[0].get('imported_from') == str((SOURCES / channel / vid).relative_to(ROOT)) and mine[0].get('slug') == slug
            if ours and mine[-1].get('status') in ('read', 'unreadable') and (vault.SONGS / slug / 'import.json').exists():
                continue  # finished by an earlier run
            if not (ours and all(r.get('status') == 'fetched' for r in mine)):
                raise ValueError(f'{vid}: already in the manifest from elsewhere; refusing to append again')
            resume = True  # an earlier run stopped between its fetched row and the truth import
        if (vault.SONGS / slug).exists() or any(p.name.endswith('-' + vid) and not p.name.startswith('.') for p in vault.SONGS.iterdir()):
            raise ValueError(f'{vid}: a song folder already exists; refusing to overwrite')
        read = READS / channel / vid
        summary = read / 'cv' / 'summary.json'
        rows.append(dict(resume=resume, video=vid, slug=slug, title=title, seconds=meta.get('duration'), source=src,
                         read=read, summary=json.loads(summary.read_text()) if summary.exists() else None,
                         meta=meta, fetch=fetch))
    return rows


def import_one(row: dict, channel: str, layout: str, device: str) -> dict:
    import numpy as np
    sys.path.insert(0, str(ROOT / 'src'))
    from stickmap import models
    src, folder = row['source'], vault.SONGS / row['slug']
    work = vault.SONGS / f'.import-{row["video"]}'
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir()
    video = next(src / n for n in ('video.mp4', 'video.webm') if (src / n).exists())
    shutil.copy2(video, work / 'video.mp4' if video.suffix == '.mp4' else work / video.name)
    for name in ('drums.htdemucs.wav', 'attacks.htdemucs.npy', 'onsets.bands.htdemucs.npy'):
        shutil.copy2(src / name, work / name)
    # Copied pilot media keep their audio in a separate audio.m4a (no audio stream in the video).
    audio_source = src / 'audio.m4a' if (src / 'audio.m4a').exists() else video
    subprocess.run(['ffmpeg', '-loglevel', 'error', '-y', '-i', str(audio_source), '-vn', '-ar', '44100', '-ac', '2',
                    str(work / 'audio.wav')], check=True)
    info = json.loads((src / 'info.json').read_text()) if (src / 'info.json').exists() else {}
    (work / 'meta.json').write_text(json.dumps(dict(
        kind='youtube', source=row['fetch']['url'] if 'url' in row['fetch'] else f'https://www.youtube.com/shorts/{row["video"]}',
        title=row['title'], uploader=info.get('uploader') or row['meta'].get('uploader'),
        duration=row['seconds'], video_id=row['video'], fingerprint=''), ensure_ascii=False, indent=1) + '\n')
    np.save(work / 'activations.drums-htdemucs.npy', models.stem_activations(work / 'drums.htdemucs.wav', device=device))
    np.save(work / 'activations.mix.npy', models.drum_activations(work / 'audio.wav', device=device))
    result = dict(status='unreadable', strict=0, scoring=0, reason=None, signature=None)
    summary = row['summary']
    if summary is not None:
        cv = row['read'] / 'cv'
        truth = work / 'truth'
        truth.mkdir()
        for name in TRUTH_FILES:
            shutil.copy2(cv / name, truth / name)
        notation = row['read'] / 'work' / 'notation' / 'scan.json'
        if notation.exists():
            shutil.copy2(notation, truth / 'scan.json')
        reviewed = row['read'] / 'work' / 'truth-reviewed'
        for name in REVIEWED_FILES:
            if (reviewed / name).exists():
                shutil.copy2(reviewed / name, truth / name)
        signature = hashlib.sha256(json.dumps({n: sha(truth / n) for n in TRUTH_FILES}, sort_keys=True).encode()).hexdigest()
        alignment = json.loads((cv / 'alignment.json').read_text())
        reasons = {}
        for s in alignment.get('systems', []):
            for r in s.get('reasons', []):
                reasons[r] = reasons.get(r, 0) + 1
        provenance = dict(
            reader='tools/vision_read/channel_read.py representative-frame CV (cv variant): CV strip cursor/colour '
                   'evidence kept, notes from the best of three native frames; unchanged align/tiers gates',
            reader_sha256={p: sha(ROOT / p) for p in ('tools/vision_read/channel_read.py',
                                                       'tools/vision_read/channel_read_core.py',
                                                       'tools/vision_read/channel_pilot.py',
                                                       'tools/truth_video/align.py', 'tools/truth_video/tiers.py',
                                                       'tools/truth_video/bands.py')},
            representative_frames=str((row['read'] / 'work' / 'native').relative_to(ROOT)),
            read_output=str(cv.relative_to(ROOT)),
            source=str(src.relative_to(ROOT)), source_fetch=row['fetch'],
            encoding='H.264 (avc1) video, opus audio; AV1 avoided because WSL OpenCV cannot decode it',
            native_video=row['meta'].get('native_video'), detector_used_for_truth=False,
            cv_only_reference=summary.get('cv_only'))
        (truth / 'result.json').write_text(json.dumps(dict(
            complete=True, signature=signature, channel=channel, enabled=True,
            strict_systems=summary['strict_pass'], scoring_systems=summary['scoring_pass'],
            strict_notes=summary['strict_notes'], scoring_notes=summary['scoring_notes'],
            rejection_counts=reasons, provenance=provenance), ensure_ascii=False, indent=1) + '\n')
        result.update(status='read' if summary['scoring_pass'] > 0 else 'unreadable', strict=summary['strict_pass'],
                      scoring=summary['scoring_pass'], signature=signature, reason=None if summary['scoring_pass'] else reasons)
    else:
        result['reason'] = {'channel_read failed before gating (see channel_read run.log)': 1}
    (work / 'import.json').write_text(json.dumps(dict(video=row['video'], channel=channel, method='fetch_source + channel_read',
                                                       source=str(src.relative_to(ROOT)), result=result), indent=1) + '\n')
    work.rename(folder)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--channel', required=True)
    p.add_argument('--allow', required=True, type=Path)
    p.add_argument('--device', default='cuda')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args(argv)
    if not (vault.OUT / '.wsl-authority').exists():
        raise SystemExit('import_source runs only on the WSL manifest writer (.wsl-authority missing)')
    from harvest_transfer import lock
    channels = {c['name']: c for c in json.loads((ROOT / 'tools/harvest/channels.json').read_text())['channels']}
    layout = channels[args.channel]['layout']
    allow = json.loads(args.allow.read_text())['train']
    with lock():
        rows = plan(args.channel, allow)
        print(f'{len(rows)} videos to import; {sum(r["summary"] is not None for r in rows)} with offline reads', flush=True)
        if args.dry_run:
            for r in rows:
                print(r['video'], r['slug'], 'read' if r['summary'] else 'no-read')
            return
        for r in rows:
            if not r['resume']:
                append(vault.MANIFEST, video=r['video'], channel=args.channel, layout=layout, title=r['title'],
                       seconds=r['seconds'], slug=r['slug'], status='fetched', error=None,
                       import_method='fetch_source + channel_read', imported_from=str(r['source'].relative_to(ROOT)))
            if vault.Vault().is_vault(r['video']):
                raise RuntimeError(f'{r["video"]} became vault after its first row; stopping before any truth import')
            out = import_one(r, args.channel, layout, args.device)
            append(vault.MANIFEST, video=r['video'], slug=r['slug'], channel=args.channel, status=out['status'],
                   truth_signature=out['signature'], truth_reason=out['reason'],
                   truth_strict_systems=out['strict'], truth_scoring_systems=out['scoring'], truth_audio_flags=[])
            print(json.dumps({'video': r['video'], **{k: out[k] for k in ('status', 'strict', 'scoring')}}), flush=True)


if __name__ == '__main__':
    main()
