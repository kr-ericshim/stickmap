"""Whole-video native viewport inventory, CV jobs, and image-only time priors.

For continuous scrolls a system is a disjoint four-second *visited* score
window, not every translated frame or every rejected scanner fragment. The
full native viewport is supplied for rhythm context; only its owned cursor
span can later export. Static systems use the existing scanner's page visits.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
import sys
from pathlib import Path
sys.dont_write_bytecode = True
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[key] = '1'
from common import DATA, ROOT, dump, load, sha

DENSE = ['jAMDUcKmjD4', 'qLi8dFevOF0', 'n3jYSzj9VX0', 'tK9MGMbPS_w', 'Ly7Pbg4EDug']
CONTROLS = ['9juoKgimeR4', 'N0PXFlgigjw']


def source(video):
    sys.path.insert(0, str(ROOT))
    from tools.harvest.vault import Vault
    if Vault().is_vault(video):
        raise ValueError('sealed video cannot be read: '+video)
    rows = [json.loads(line) for line in (ROOT/'truth/harvest/manifest.jsonl').read_text().splitlines()]
    row = [r for r in rows if r['video'] == video][-1]
    song = ROOT/'truth/harvest/songs'/row['slug']
    scan = load(song/'truth/scan.json')
    return row, song, scan


def static_knots(record, fps):
    import numpy as np
    xs = np.asarray(record.get('cursor_x', []), float)
    ts = np.asarray(record.get('cursor_frames', []), float)/fps
    mode = 'sweep'
    if len(xs) > 2 and np.mean(abs(np.diff(xs)) <= 1) > .55:
        mode = 'box'
        groups, start = [], 0
        for j in range(1, len(xs)+1):
            if j == len(xs) or abs(xs[j]-xs[start]) > record['native_space']*.6:
                if j-start >= 2:
                    groups.append((float(np.median(xs[start:j])), float(ts[start])))
                start = j
        xs = np.asarray([x for x, t in groups]); ts = np.asarray([t for x, t in groups])
    if len(xs):
        keep = np.r_[True, np.diff(np.maximum.accumulate(xs)) > .5]
        xs, ts = xs[keep], ts[keep]
    return xs.tolist(), ts.tolist(), mode


def windows(video, song, scan, control):
    """Selection is frozen before inference; control selection uses IDs only."""
    records = scan['records']; fps = scan['fps']
    if control:
        accepted = {r['system'] for r in load(song/'truth/alignment.json')['systems'] if r['usable']}
        return [{'start': r['start'], 'end': r['end'], 'records': [r], 'source_system': r['system']}
                for r in records if r['system'] in accepted]
    if not scan.get('reconstructed_scroll'):
        return [{'start': r['start'], 'end': r['end'], 'records': [r], 'source_system': r['system']} for r in records]
    step = round(fps*4)
    return [{'start': a, 'end': min(scan['frames'], a+step),
             'records': [r for r in records if r['start'] < a+step and r['end'] > a]}
            for a in range(0, scan['frames'], step)]


def extract_video(video):
    import cv2
    import numpy as np
    sys.path.insert(0, str(ROOT/'tools/truth_video'))
    from layouts import ink, cursor
    from harvest_scroll import motion_image, shift
    from native_cv import measured_lines
    from hybrid import cv_prediction, prepare
    cv2.setNumThreads(1)
    row, song, scan = source(video)
    folder = DATA/'native'/video
    folder.mkdir(parents=True, exist_ok=True)
    if (folder/'inventory.json').exists():
        print(video+' already extracted', flush=True)
        return load(folder/'inventory.json')
    control = video in CONTROLS
    selected = windows(video, song, scan, control)
    cap = cv2.VideoCapture(str(song/'video.mp4'))
    fps, width, height = cap.get(5), int(cap.get(3)), int(cap.get(4))
    scroll = bool(scan.get('reconstructed_scroll'))
    items, dispositions = [], []
    legend_path = 'tools/truth_video/legends/'+row['channel']+'.json'
    legend = load(ROOT/legend_path)
    prior_lines = np.median(scan.get('sampled_native_lines', []), axis=0) if scroll else None
    native_space = float(np.median(scan['native_spacing']))
    stride = max(1, round(fps/15))
    for wi, win in enumerate(selected):
        sid = f'{video}_auto_{wi:04d}'
        start, end = win['start'], win['end']
        records = win['records']
        source_record = max(records, key=lambda r: r['end']-r['start']) if records else {}
        fg = source_record.get('foreground', scan.get('foreground'))
        y0 = max(0, scan['top']+(0 if scroll else source_record['crop'][0])-40)
        y1 = min(height, scan['bottom']+50 if scroll else scan['top']+source_record['crop'][1]+50)
        left, right = scan.get('left', 0), scan.get('right', width)
        samples, edges = [], []
        candidates = sorted(set(int(start+(end-start-1)*f) for f in (.15, .5, .85)))
        frame_numbers = sorted(set(range(start, end, stride)) | set(candidates)) if scroll else candidates
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        current = start
        for frame_no in frame_numbers:
            while current < frame_no:
                if not cap.grab():
                    break
                current += 1
            ok, frame = cap.read(); current += 1
            if not ok:
                break
            photo = frame[y0:y1, left:right]
            gray = ink(photo, fg)
            lines, spacing, geom_error = None, None, None
            if scroll:
                try:
                    lines, spacing = measured_lines(gray, prior_lines-y0)
                except ValueError as error:
                    geom_error = str(error)
            box = cursor(frame[scan['top']:scan['bottom']], native_space, foreground=fg) if scroll else None
            sample = dict(frame=frame_no, photo=photo, gray=gray, lines=lines, spacing=spacing,
                          cursor=box, geometry_error=geom_error, offset=0, chain=0, geometry_origin=lines)
            if scroll and samples:
                prev = samples[-1]
                delta = None
                if (lines and prev['lines'] and abs(spacing-prev['spacing']) <= 1.1
                        and max(abs(np.asarray(lines)-prev['geometry_origin'])) <= spacing*.5):
                    # Keep the canonical translation thresholds. Sampling at
                    # 15 Hz avoids its 200-ms / 80-px fast-scroll displacement cap.
                    images = []
                    for s in (prev, sample):
                        g = ink(s['photo'], fg, motion=True)
                        im = motion_image(g, round(native_space*10), photo.shape[1]-round(native_space*4), s['lines'])
                        images.append(im)
                    delta = shift(*images)
                    if delta is not None and delta['dx'] > 2:
                        delta = None
                if delta:
                    sample['offset'] = prev['offset']+delta['dx']
                    sample['chain'] = prev['chain']
                    sample['geometry_origin'] = prev['geometry_origin']
                else:
                    sample['chain'] = prev['chain']+1
                edges.append(dict(before=prev['frame'], after=frame_no, match=delta))
            samples.append(sample)
        if not samples:
            dispositions.append(dict(id=sid, status='no_frame', start=start, end=end))
            continue
        # Rank three native frames with the same CV checks used in the jobs.
        variants, cvs, evaluated = [], [], []
        chosen_samples = [min(samples, key=lambda s: abs(s['frame']-f)) for f in candidates]
        for vi, sample in enumerate(chosen_samples):
            path = folder/f'{sid}_{vi}.png'
            cv2.imwrite(str(path), sample['photo'])
            variants.append(dict(path=path.name, sha256=sha(path), frame=sample['frame'],
                                 time=sample['frame']/fps, crop=[left,y0,right,y1]))
            item = dict(id=sid, video=video, channel=row['channel'], system=wi, record=source_record,
                        fps=fps, video_path=str(song/'video.mp4'), legend=legend_path, foreground=fg,
                        legend_variant=fg, reconstructed_scroll=scroll, primary=vi, variants=variants.copy(),
                        retain_left_margin_heads=scroll)
            if scroll:
                item['native_expected_lines'] = (prior_lines-y0).tolist()
            cv = cv_prediction(item, folder); cvs.append(cv); evaluated.append(item)
        def rank(i):
            cv, s = cvs[i], chosen_samples[i]
            reasons = [r for r in cv['staff'].get('reasons', []) if r != 'percussion clef / first bar not confirmed']
            group = [q for q in samples if q['chain'] == s['chain'] and q['cursor']] if scroll else samples
            traversed = (max(q['offset'] for q in group)-min(q['offset'] for q in group)) if scroll and group else 0
            sharp = cv2.Laplacian(s['gray'], cv2.CV_64F).var()
            return (bool(cv['error']), len(reasons), -traversed, -len(cv['notes']), -sharp)
        primary = min(range(len(cvs)), key=rank)
        item, cv, sample = evaluated[primary], cvs[primary], chosen_samples[primary]
        item['variants'] = variants; item['primary'] = primary
        if cv['error'] or not cv['staff'].get('lines') or not cv['notes']:
            dispositions.append(dict(id=sid, status='no_readable_notation', start=start, end=end,
                                     error=cv['error'], staff=cv['staff'], scan_records=[r['system'] for r in records]))
            continue
        anchor = dict(method='scan interval; no audio input', interval=[start/fps,end/fps],
                      source_scan_sha256=sha(song/'truth/scan.json'),
                      source_record_ids=[r['system'] for r in records],
                      source_rejections=[{'system':r['system'],'start':r['start'],'end':r['end'],
                                          'reasons':r.get('scroll_reasons',[])+r.get('reading_reasons',[])} for r in records],
                      cursor_x=[], cursor_t=[], cursor_mode='sweep', owned_x=None,
                      source_frame=sample['frame'], native_crop=[left,y0,right,y1],
                      staff_continuation=None, frame_selection='best of three native frames by CV checks, traversed span, heads, sharpness',
                      selection_scores=[list(rank(i)) for i in range(len(cvs))])
        if scroll:
            group = [s for s in samples if s['chain'] == sample['chain'] and s['cursor'] and s['lines']]
            cx = [s['cursor'][0]-s['offset']+sample['offset']-left for s in group]
            ct = [s['frame']/fps for s in group]
            if len(cx) >= 5 and cx[-1]-cx[0] >= 30:
                keep = np.r_[True,np.diff(np.maximum.accumulate(cx)) > .5]
                anchor.update(method='native image registration at 15 Hz plus observed red cursor',
                              cursor_x=np.asarray(cx)[keep].tolist(), cursor_t=np.asarray(ct)[keep].tolist(),
                              owned_x=[cx[0],cx[-1]-.01])
                if legend.get('white_overlay_clef_omitted_verified'):
                    anchor['staff_continuation'] = dict(method='registered native staff continuation',
                        profile_evidence=legend['white_overlay_clef_omitted_verified'],
                        supporting_frames=[s['frame'] for s in group],
                        native_lines=[s['lines'] for s in group],
                        source='automatic image evidence; no human review flag')
            anchor['registered_fraction'] = len(group)/len(samples)
            anchor['registration_edges'] = edges
        else:
            cx, ct, mode = static_knots(source_record, fps)
            anchor.update(method='existing native scan cursor', cursor_x=cx, cursor_t=ct, cursor_mode=mode)
            if len(cx) >= 4:
                anchor['owned_x'] = [cx[0]-cv['staff']['spacing'], cx[-1]+cv['staff']['spacing']]
        # No invented cursor or tempo when image registration is unavailable.
        # The interval fallback is explicit and goes through existing stricter
        # no-cursor timing gates; it does not inherit canonical staff rejections.
        item.update(record={**source_record,'start':start,'end':end,'native_space':cv['staff']['spacing']},
                    automatic_anchor=anchor)
        if win.get('source_system') is not None:
            item['record']['source_system'] = win['source_system']
        # Exact repeats at consecutive visits without cursor reset are one job.
        fingerprint = hashlib.sha256(json.dumps([(round(n['x']/2),round(n['y']/2),n['shape']) for n in cv['notes']]).encode()).hexdigest()
        duplicate = bool(items and not scroll and items[-1].get('geometry_fingerprint') == fingerprint
                         and start <= items[-1]['record']['end'] and anchor['cursor_x']
                         and items[-1]['automatic_anchor']['cursor_x']
                         and anchor['cursor_x'][0] >= items[-1]['automatic_anchor']['cursor_x'][-1]-cv['staff']['spacing'])
        if duplicate:
            previous = items[-1]; previous['record']['end'] = end
            previous['automatic_anchor']['interval'][1] = end/fps
            previous['automatic_anchor']['cursor_x'].extend(anchor['cursor_x'])
            previous['automatic_anchor']['cursor_t'].extend(anchor['cursor_t'])
            previous['automatic_anchor']['source_record_ids'].extend(anchor['source_record_ids'])
            if previous['automatic_anchor']['owned_x'] and anchor['owned_x']:
                previous['automatic_anchor']['owned_x'][1] = anchor['owned_x'][1]
            dispositions.append(dict(id=sid,status='duplicate_visit',deduplicated_into=previous['id']))
            continue
        item['geometry_fingerprint'] = fingerprint
        items.append(item)
        dispositions.append(dict(id=sid,status='found',cv_heads=len(cv['notes']),staff_reasons=cv['staff']['reasons'],
                                 cursor_knots=len(anchor['cursor_x']),registered_fraction=anchor.get('registered_fraction')))
        print(json.dumps(dict(video=video,window=wi,total=len(selected),found=len(items),heads=len(cv['notes']),knots=len(anchor['cursor_x']))), flush=True)
    cap.release()
    manifest = dict(version=1,video=video,source_video=str(song/'video.mp4'),source_size=(song/'video.mp4').stat().st_size,
                    source_scan_sha256=sha(song/'truth/scan.json'),fps=fps,width=width,height=height,systems=items,
                    policy='disjoint visited scroll windows; native contextual images; static visits deduplicated; no audio',
                    control_selection='canonical strict system IDs, never reference note times' if control else 'whole video')
    dump(folder/'manifest.json',manifest)
    if items:
        prepare([it['id'] for it in items],folder)
    inventory = dict(video=video,canonical_song=str(song),vault=False,control=control,scan_records=len(scan['records']),
                     windows_examined=len(selected),systems_found=len(items),dispositions=dispositions,
                     systems=[it['id'] for it in items],manifest_sha256=sha(folder/'manifest.json'))
    dump(folder/'inventory.json',inventory)
    print(json.dumps(dict(video=video,done=True,systems_found=len(items))),flush=True)
    return inventory


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--videos',nargs='+',default=CONTROLS+DENSE)
    p.add_argument('--workers',type=int,default=7)
    args=p.parse_args()
    if not 1 <= args.workers <= 8:
        p.error('workers must be 1..8')
    # Check all membership before opening any media.
    for video in args.videos:
        source(video)
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        results=list(pool.map(extract_video,args.videos))
    dump(DATA/'extraction.json',{'videos':[load(p) for p in sorted((DATA/'native').glob('*/inventory.json'))],
                                 'completed_this_run':[r['video'] for r in results]})


if __name__ == '__main__':
    main()
