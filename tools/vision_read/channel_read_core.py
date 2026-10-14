"""Path-parameterized channel-pilot CV extraction; all gates stay upstream.

Only the pilot's path-bound extraction/preparation/staging orchestration is
adapted here. Selection, geometry, CV reading and cursor conversion are reused.
Each caller runs in its own video process, so temporary module configuration
cannot affect another video or another session.
"""
from __future__ import annotations

import collections
import copy
from unittest.mock import patch

import channel_pilot as pilot
from common import ROOT, dump, load, sha

VERSION = 'channel-read-v1'


def relative(path):
    return pilot.relative(path)


def portable(folder):
    """Normalize upstream absolute provenance after its last local consumer."""
    prefix = str(ROOT) + '/'

    def visit(value):
        if isinstance(value, dict):
            return {key: visit(item) for key, item in value.items()}
        if isinstance(value, list):
            return [visit(item) for item in value]
        if isinstance(value, str) and value.startswith(prefix):
            return value[len(prefix):]
        return value

    for path in folder.rglob('*.json'):
        value = load(path)
        normalized = visit(value)
        if normalized != value:
            dump(path, normalized)


def notation(song, out, legend):
    """Exactly pilot_new's notation phase, with a caller-owned destination."""
    import harvest_scan
    import strip
    from layouts import detect_region, detect
    from media import video_path
    from batch import scan_signature, scan_approvals, region_scan_signature

    profile = load(legend)
    video = video_path(song)
    geometry = (detect_region(video, profile) if profile.get('score_region')
                else detect(video, profile['layout']))
    reject = profile.get('reject_chromatic_overlays', False)
    fallback = profile.get('native_frame_fallback', False)
    harvest_scan.write(video, out, profile['layout'], geometry=geometry,
                       reject_overlays=reject, native_fallback=fallback)
    strip.read(out, legend)
    sig = scan_signature(song, profile['layout'], False,
                         profile.get('staff_y_min_fraction', 0), reject, fallback,
                         approval_state=scan_approvals(profile))
    dump(out / 'scan-signature.json',
         dict(scan_signature=region_scan_signature(sig, profile)))
    portable(out)


def read_cv(song, out, legend, scan):
    """Unchanged batch.worker with its automatic band load made read-only."""
    import align
    import bands
    from batch import worker

    original = align.build_truth

    def build_readonly(song, folder, bands='auto'):
        if isinstance(bands, str) and bands == 'auto':
            from bands import load as load_bands
            bands = load_bands(song, cache=False)
        return original(song, folder, bands=bands)

    # worker imports build_truth at call time. No gate or reader is replaced.
    with patch.object(align, 'build_truth', build_readonly):
        worker(song, out, legend, notation_scan=scan)
    portable(out)


def extract(song, channel, folder, read_folder, scan_folder, system_id):
    """channel_pilot.extract_video's native algorithm with explicit paths/IDs.

    Keep the scanner's exact crop, 15/50/85% candidates, rank, eligibility,
    geometry check, and static cursor anchors identical to channel_pilot.
    """
    import cv2
    from layouts import ink
    from hybrid import cv_prediction
    from solos_extract import static_knots
    from media import video_path

    cv2.setNumThreads(1)
    video = song.name
    video_file = video_path(song)
    scan_path, strip_path = scan_folder / 'scan.json', read_folder / 'strip.json'
    scan, systems = load(scan_path), load(strip_path)
    folder.mkdir(parents=True, exist_ok=True)
    fingerprint = dict(version=VERSION, scan=sha(scan_path), strip=sha(strip_path),
                       legend=sha(ROOT / f'tools/truth_video/legends/{channel}.json'))
    records = {r['system']: r for r in scan['records']}
    cap = cv2.VideoCapture(str(video_file))
    fps, width, height = cap.get(5), int(cap.get(3)), int(cap.get(4))
    if fps <= 0 or abs(fps - scan['fps']) > .01:
        cap.release()
        raise ValueError('video/scan fps mismatch: ' + video)
    items, dispositions = [], []
    try:
        for s in systems:
            index = s['system']; sid = system_id(video, index)
            record = records.get(index)
            reasons = pilot.skip_reasons(s)
            if record is None:
                reasons.append('no matching native scan record')
            if reasons:
                dispositions.append(dict(id=sid, system=index, status='skipped', reasons=reasons))
                continue
            if scan.get('reconstructed_scroll'):
                raise ValueError('representative-frame read requires static native scan records')
            left, right = scan.get('left', 0), scan.get('right', width)
            y0, y1 = [scan['top'] + y for y in record['crop']]
            if not (0 <= left < right <= width and 0 <= y0 < y1 <= height):
                raise ValueError('native crop outside video: ' + sid)
            saved = read_folder / f'system_{index:03d}.png'
            if not saved.exists():
                saved = scan_folder / saved.name
            reference_image = cv2.imread(str(saved), 0)
            if reference_image is None or reference_image.shape != (y1 - y0, right - left):
                dispositions.append(dict(id=sid, system=index, status='skipped',
                                         reasons=['saved scanner crop does not match native crop bounds']))
                continue
            variants, predictions, scores, checks = [], [], [], []
            fg = record.get('foreground', scan.get('foreground'))
            frame_numbers = [int(record['start'] + (record['end'] - record['start'] - 1) * f)
                             for f in (.15, .50, .85)]
            base = dict(id=sid, video=video, channel=channel, system=index,
                        record={**record, 'source_system': index}, fps=fps,
                        video_path=relative(video_file),
                        legend=f'tools/truth_video/legends/{channel}.json', foreground=fg,
                        legend_variant=fg, reconstructed_scroll=False)
            for vi, frame_no in enumerate(frame_numbers):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
                ok, frame = cap.read()
                if not ok:
                    break
                photo = frame[y0:y1, left:right]
                path = folder / f'{sid}_{vi}.png'
                if not cv2.imwrite(str(path), photo):
                    raise ValueError('native image write failed: ' + sid)
                variants.append(dict(path=path.name, sha256=sha(path), frame=frame_no,
                                     time=frame_no / fps, crop=[left, y0, right, y1]))
                cv = cv_prediction({**base, 'primary': vi, 'variants': variants}, folder)
                predictions.append(cv)
                check = pilot.match_geometry(cv, s); checks.append(check)
                sharpness = float(cv2.Laplacian(ink(photo, fg), cv2.CV_64F).var())
                staff_reasons = [r for r in cv['staff'].get('reasons', [])
                                 if r != 'percussion clef / first bar not confirmed']
                scores.append([bool(cv['error']), len(staff_reasons), -len(cv['notes']), -sharpness])
            eligible = [i for i, cv in enumerate(predictions) if not cv['error']
                        and len(cv['staff'].get('lines', [])) == 5 and cv['notes']
                        and not checks[i]['reasons']]
            if len(variants) != 3 or not eligible:
                dispositions.append(dict(id=sid, system=index, status='skipped',
                                         reasons=['no coordinate-verified native candidate frame'],
                                         candidate_geometry=checks,
                                         candidate_errors=[c['error'] for c in predictions]))
                continue
            primary = min(eligible, key=lambda i: scores[i])
            cv = predictions[primary]; cx, ct, mode = static_knots(record, fps)
            anchor = dict(method='existing native scan cursor; no audio input',
                          interval=[record['start'] / fps, record['end'] / fps],
                          source_scan_sha256=sha(scan_path), source_record_ids=[index],
                          cursor_x=cx, cursor_t=ct, cursor_mode=mode,
                          owned_x=[cx[0] - cv['staff']['spacing'], cx[-1] + cv['staff']['spacing']]
                          if len(cx) >= 4 else None,
                          source_frame=frame_numbers[primary], native_crop=[left, y0, right, y1],
                          staff_continuation=None, frame_selection='CV errors, staff reasons, heads, sharpness',
                          selection_scores=scores)
            items.append({**base, 'primary': primary, 'variants': variants, 'automatic_anchor': anchor,
                          'channel_geometry': checks[primary], 'candidate_geometry': checks})
            dispositions.append(dict(id=sid, system=index, status='found', reasons=[],
                                     cv_heads=len(cv['notes']), strip_heads=len(s['notes']),
                                     frame=frame_numbers[primary], offset_xy=checks[primary]['offset_xy'],
                                     matched=checks[primary]['matched'],
                                     residual_max_px=checks[primary]['residual_max_px']))
    finally:
        cap.release()
    dump(folder / 'manifest.json', dict(version=1, channel_read=VERSION, video=video,
         source_video=relative(video_file), source_size=video_file.stat().st_size,
         source_scan_sha256=sha(scan_path), source_strip_sha256=sha(strip_path),
         fps=fps, width=width, height=height, systems=items,
         policy='one job per eligible CV strip system; exact native scanner row crop; no audio'))
    inventory = dict(video=video, channel=channel, source=relative(song), vault=False,
                     inputs=fingerprint, scan_records=len(scan['records']), strip_systems=len(systems),
                     systems_found=len(items), systems=[it['id'] for it in items], dispositions=dispositions,
                     native=relative(folder), source_strip=relative(strip_path),
                     manifest_sha256=sha(folder / 'manifest.json'))
    dump(folder / 'inventory.json', inventory)
    return inventory


def prepare(inventory, work):
    import hybrid
    folder = ROOT / inventory['native']
    items = load(folder / 'manifest.json')['systems']
    # hybrid.prepare has only a bench parameter. Its destination is isolated
    # per video here, including prepared.json; no shared pilot state is touched.
    with patch.object(hybrid, 'HYBRID', work / 'hybrid'):
        hybrid.prepare(inventory['systems'], folder)
    for item in items:
        job = work / 'hybrid/jobs' / item['id']
        meta, cv = load(job / 'job.json'), load(job / 'cv.json')
        meta.update(bench=relative(folder), job_dir=relative(job), channel_read=VERSION,
                    channel_geometry=item['channel_geometry'], source_strip=inventory['source_strip'],
                    source_strip_sha256=inventory['inputs']['strip'])
        dump(job / 'job.json', meta)
        (job / 'INSTRUCTIONS.md').write_text(hybrid.instructions(meta, cv))


def assert_job(job, inventory):
    meta = load(job / 'job.json')
    if meta.get('channel_read') != VERSION or meta['system_id'] not in inventory['systems']:
        raise ValueError('not a frozen channel read job')
    if sha(ROOT / meta['source_strip']) != meta['source_strip_sha256']:
        raise ValueError('CV strip changed after extraction')
    if sha(job / 'cv.json') != meta['cv_sha256'] or sha(job / 'annotation.json') != meta['annotation_sha256']:
        raise ValueError('CV job integrity mismatch')
    if sha(ROOT / meta['bench'] / 'manifest.json') != meta['manifest_sha256']:
        raise ValueError('native manifest changed after extraction')
    for image in meta['images']:
        if sha(job / image['file']) != image['sha256']:
            raise ValueError('job image integrity mismatch')
    return meta


def stage(inventory, work, out):
    from hybrid import cv_reading
    originals = load(ROOT / inventory['source_strip'])
    dispositions = {d['system']: d for d in inventory['dispositions']}
    systems, records = [], []
    for original in originals:
        index = original['system']; disposition = dispositions[index]; sid = disposition['id']
        system = copy.deepcopy(original)
        if disposition['status'] != 'found':
            # Preserve the pilot's exact rejection text and blocked contextual
            # rows, which still inform canonical song-phase/lag estimation.
            system['staff'].setdefault('reasons', []).append('channel pilot excluded: ' + '; '.join(disposition['reasons']))
            records.append(dict(id=sid, system=index, status='skipped', reasons=disposition['reasons']))
        else:
            job = work / 'hybrid/jobs' / sid
            meta = assert_job(job, inventory)
            cv, legend = load(job / 'cv.json'), load(job / 'legend.json')
            try:
                system = pilot.convert_cursor(cv_reading(sid, cv), meta, cv, legend, original, cv_sanity=True)
                status = 'staged'
            except (ValueError, TypeError, KeyError, OverflowError, ZeroDivisionError) as error:
                system['notes'] = []
                system['staff'].setdefault('reasons', []).append('hybrid conversion failed: ' + str(error))
                status = 'conversion_failed'
            records.append(dict(id=sid, system=index, status=status, reading_ok=True,
                                notes=len(system['notes']), reasons=system['staff']['reasons'], reading_sha256=None))
        systems.append(system)
    dump(out / 'strip.json', systems)
    record = dict(video=inventory['video'], variant='cv', run=None,
                  systems_found=inventory['systems_found'], systems_read=sum(r.get('reading_ok', False) for r in records),
                  systems=records, strip_sha256=sha(out / 'strip.json'),
                  policy='retain excluded CV rows as blocked context for canonical song-phase')
    dump(out / 'staging.json', record)
    return record


def counts(folder):
    from solos_gate import fast_counts
    alignment, scoring = load(folder / 'alignment.json'), load(folder / 'alignment.scoring.json')
    strict_events, scoring_events = load(folder / 'truth.json'), load(folder / 'truth.scoring.json')
    reasons = collections.Counter(r for s in alignment['systems'] for r in s['reasons'])
    return dict(systems_gated=len(alignment['systems']),
                strict_pass=sum(s['usable'] for s in alignment['systems']),
                scoring_pass=sum(s['scoring_usable'] for s in scoring['systems']),
                strict_notes=len(strict_events), scoring_notes=len(scoring_events),
                fast_strict=fast_counts(strict_events), fast_scoring=fast_counts(scoring_events),
                reasons=dict(reasons), top_reasons=reasons.most_common(5))


def gate(song, out):
    from align import build_truth
    import bands
    from tiers import write
    if sha(out / 'strip.json') != load(out / 'staging.json')['strip_sha256']:
        raise ValueError('staged strip changed')
    build_truth(song, out, bands=bands.load(song, cache=False))
    write(out)
    portable(out)
    return counts(out)
