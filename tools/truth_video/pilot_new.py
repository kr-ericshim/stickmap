"""Offline, detector-free pilots from channel/video/{video.*,audio.m4a,title.txt}.

Only supplied media is enumerated; no online listing is implied. Derived files
stay in .eval-cache/pilot_new. Worker/thread limits match pilot1080.py.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys

from pilot1080 import ENV, HERE, ROOT, prepare
from media import VIDEO_NAMES, video_path

PILOTS = ROOT / '.eval-cache/pilot_new'
PROFILE_CHANNELS = {'ollieshorts': 'olliewoods_shorts'}


def profile_channel(folder):
    return PROFILE_CHANNELS.get(folder.parent.name, folder.parent.name)


def metadata(folder):
    title_path = folder / 'title.txt'
    if title_path.exists():
        title = title_path.read_text().strip()
    else:
        rows = (ROOT / f'truth/harvest/lists/{profile_channel(folder)}.txt').read_text().splitlines()
        matches = [row.split('|', 2) for row in rows if row.split('|', 1)[0] == folder.name]
        if len(matches) != 1:
            raise ValueError(f'pilot needs a title or unique retained list row: {folder}')
        _, duration, title = matches[0]
        if duration not in ('', 'NA', 'None'):
            title += '|' + duration
    text, sep, seconds = title.rpartition('|')
    try:
        seconds = float(seconds) if sep else None
    except ValueError:
        seconds = None  # A literal | may belong to the supplied title.
    if seconds is not None:
        title = text
    else:
        import cv2
        cap = cv2.VideoCapture(str(video_path(folder)))
        try:
            fps, count = cap.get(5), cap.get(7)
            if fps <= 0 or count <= 0:
                raise ValueError(f'missing native duration: {folder}')
            seconds = count/fps
        finally:
            cap.release()
    profile_path = HERE / f'legends/{profile_channel(folder)}.json'
    profile = json.loads(profile_path.read_text()) if profile_path.exists() else {}
    return dict(video=folder.name, video_id=folder.name, channel=profile_channel(folder),
                title=title, seconds=float(seconds), test_only=bool(profile.get('test_only')),
                provenance='supplied local pilot; offline notation-only processing')


def folders(channels=None):
    return sorted({p.parent for name in VIDEO_NAMES for p in PILOTS.glob(f'*/*/{name}')
                   if not channels or p.parent.parent.name in channels
                   or profile_channel(p.parent) in channels})


def survey(folder):
    import cv2
    import numpy as np
    cv2.setNumThreads(1)
    cap = cv2.VideoCapture(str(video_path(folder)))
    count, fps = int(cap.get(7)), cap.get(5)
    fractions = (.03, .2, .5, .8, .95, .99)
    sheet = np.full((3 * 294, 2 * 480, 3), 255, np.uint8)
    for i, fraction in enumerate(fractions):
        frame_no = min(count-1, round(count*fraction))
        cap.set(1, frame_no)
        ok, frame = cap.read()
        if not ok:
            continue
        image = cv2.resize(frame, (480, 270))
        row, col = divmod(i, 2)
        sheet[row*294+24:row*294+294, col*480:(col+1)*480] = image
        cv2.putText(sheet, f'{frame_no/fps:.2f}s', (col*480+6, row*294+18),
                    0, .5, (0, 0, 0), 1)
    cap.release()
    cv2.imwrite(str(folder / 'survey.jpg'), sheet)
    print(folder.parent.name, folder.name, 'survey', flush=True)


def run(phase, selected, jobs, output):
    if not 1 <= jobs <= 4:
        raise ValueError('jobs must be 1..4')
    if Path(output).name != output:
        raise ValueError('output must be one folder name')
    def attempt(folder):
        if phase in ('prepare', 'survey', 'notation'):
            args = [sys.executable, str(HERE / 'pilot_new.py'), '--worker', phase,
                    str(folder)]
            log = folder / f'{phase}.log'
        else:
            out = folder / output
            if (out / 'result.json').exists():
                return dict(video=folder.name, phase=phase, status='already complete')
            out.mkdir(exist_ok=True)
            args = [sys.executable, str(HERE / 'pilot_new.py'), '--worker', 'read',
                    str(folder), '--output', output]
            log = out / 'run.log'
        with log.open('w') as stream:
            result = subprocess.run(args, env=ENV, stdout=stream, stderr=subprocess.STDOUT)
        row = dict(channel=folder.parent.name, video=folder.name, phase=phase,
                   returncode=result.returncode, log=str(log.relative_to(ROOT)))
        print(json.dumps(row), flush=True)
        return row
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        return list(pool.map(attempt, selected))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', nargs='?', choices=('survey', 'prepare', 'notation', 'read'))
    parser.add_argument('--worker', nargs=2, metavar=('PHASE', 'FOLDER'))
    parser.add_argument('--channels', nargs='+')
    parser.add_argument('--jobs', type=int, default=3)
    parser.add_argument('--output', default='truth')
    args = parser.parse_args()
    if args.worker:
        phase, name = args.worker
        folder = Path(name).resolve()
        if folder.parent.parent != PILOTS:
            raise ValueError('worker must use supplied pilot directory')
        if phase == 'survey':
            survey(folder)
        elif phase == 'prepare':
            prepare(folder, metadata(folder))
        elif phase == 'notation':
            import harvest_scan
            import strip
            from layouts import detect_region, detect
            profile_path=HERE / f'legends/{profile_channel(folder)}.json'
            profile=json.loads(profile_path.read_text())
            out=folder / 'notation'
            video=video_path(folder)
            geometry=(detect_region(video,profile) if profile.get('score_region')
                      else detect(video,profile['layout']))
            reject=profile.get('reject_chromatic_overlays',False)
            fallback=profile.get('native_frame_fallback',False)
            harvest_scan.write(video,out,profile['layout'],geometry=geometry,
                               reject_overlays=reject,native_fallback=fallback)
            strip.read(out,profile_path)
            from batch import scan_signature, scan_approvals, region_scan_signature
            sig=scan_signature(folder,profile['layout'],False,
                               profile.get('staff_y_min_fraction',0),reject,fallback,
                               approval_state=scan_approvals(profile))
            sig=region_scan_signature(sig,profile)
            (out/'scan-signature.json').write_text(json.dumps(dict(scan_signature=sig)))
        elif phase == 'read':
            from batch import worker
            worker(folder,folder/args.output,HERE/f'legends/{profile_channel(folder)}.json',
                   notation_scan=folder/'notation')
        else:
            raise ValueError('invalid worker phase')
        return
    rows = run(args.phase, folders(args.channels), args.jobs, args.output)
    raise SystemExit(any(r.get('returncode', 0) for r in rows))


if __name__ == '__main__':
    main()
