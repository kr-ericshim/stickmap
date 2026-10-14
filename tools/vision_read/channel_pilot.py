"""Offline cursor-channel hybrid jobs and unchanged canonical timing gates.

Run with .venv/bin/python -B from any directory. Only common.DATA is written.
The cv variant is explicitly geometry-only: hybrid.cv_reading has no rhythm,
so its null onsets are not treated as failed GPT readings or invented beats.
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import copy
import datetime
import math
import os
import re
import sys
from fractions import Fraction
from pathlib import Path

sys.dont_write_bytecode = True
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[_key] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, ROOT, confined, dump, flatten, load, sha, validate

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools/truth_video'))
PILOT = DATA / 'channel_pilot'
VIDEOS = {
    'DMPkSQZyj76': 'xiangsong', 'DNz6yi-QMWp': 'xiangsong',
    'Dd8wm3ThGTL': 'xiangsong', 'Dd_S08chzqQ': 'xiangsong',
    '9akE2QqpnCs': 'ollieshorts', 'MPoZ-2nOESc': 'ollieshorts',
    'VhdBLkFMmWs': 'ollieshorts', 'aYgtt3EhBEg': 'ollieshorts',
    'mwLnCnc60b4': 'ollieshorts',
}
VERSION = 'channel-pilot-v1'
UNRESOLVED = 'hybrid contains unresolved heads inside the visited score span'
COLLISION = 'distinct rational moments collide on rhythm axis'
FEW = 'fewer than six resolved notated moments'


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def safe_name(value):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', value):
        raise ValueError('unsafe name: ' + value)
    return value


def source(video):
    from tools.harvest.vault import Vault
    if video not in VIDEOS:
        raise ValueError('not one of the nine pilot videos: ' + video)
    if Vault().is_vault(video):
        raise ValueError('vault input forbidden: ' + video)
    song = ROOT / '.eval-cache/pilot_new' / VIDEOS[video] / video
    from pilot_new import profile_channel
    return song, profile_channel(song)


def skip_reasons(system):
    staff = system.get('staff', {})
    reasons = [r for r in staff.get('reasons', []) if r != 'no verified channel labels']
    if len(staff.get('lines', [])) != 5:
        reasons.append('no usable native five-line staff')
    if staff.get('spacing', 0) < 8:
        reasons.append('native staff spacing below 8 px')
    if not system.get('notes'):
        reasons.append('no heads in CV strip')
    return list(dict.fromkeys(reasons))


def match_geometry(cv, system, expected=(0.0, 0.0)):
    """Measure one-to-one same-pos/shape translation, never a scale or warp.

    Scanner and native crops have identical bounds (expected offset zero).
    A four-pixel search distinguishes nearby raster centroid drift from a
    different head; EVERY such correspondence must fit one offset within 2px.
    Missing/spurious detections are counted separately, not forced to match.
    """
    import numpy as np
    reference = system['notes'] + system.get('unvisited_heads', [])
    candidates = []
    for i, n in enumerate(cv['notes']):
        for j, ref in enumerate(reference):
            if (n['pos'], n['shape']) != (ref['pos'], ref['shape']):
                continue
            dx, dy = ref['x'] - n['x'], ref['y'] - n['y']
            distance = max(abs(dx - expected[0]), abs(dy - expected[1]))
            if distance <= 4:
                candidates.append((distance, i, j, dx, dy))
    used_cv, used_ref, matches = set(), set(), []
    for _, i, j, dx, dy in sorted(candidates):
        if i not in used_cv and j not in used_ref:
            used_cv.add(i); used_ref.add(j)
            matches.append(dict(label=cv['notes'][i]['label'], strip_index=j, delta_xy=[dx, dy]))
    offset = np.median([m['delta_xy'] for m in matches], axis=0).tolist() if matches else list(expected)
    residuals = [max(abs(m['delta_xy'][k] - offset[k]) for k in (0, 1)) for m in matches]
    minimum = max(1, math.ceil(.6 * min(len(cv['notes']), len(reference))))
    reasons = []
    if len(matches) < minimum:
        reasons.append('insufficient CV/strip head correspondences to verify crop coordinates')
    if max(abs(offset[k] - expected[k]) for k in (0, 1)) > 2:
        reasons.append('measured head offset disagrees with known native crop offset')
    if residuals and max(residuals) > 2:
        reasons.append('CV/strip head offset is not constant within 2 px')
    return dict(expected_offset_xy=list(expected), offset_xy=offset, matches=matches,
                matched=len(matches), cv_heads=len(cv['notes']), strip_heads=len(reference),
                unmatched_cv=len(cv['notes']) - len(matches), unmatched_strip=len(reference) - len(matches),
                residual_max_px=max(residuals, default=None),
                residual_p95_px=float(np.percentile(residuals, 95)) if residuals else None,
                minimum_matches=minimum, reasons=reasons)


def extract_video(video):
    import cv2
    from layouts import ink
    from hybrid import cv_prediction
    from solos_extract import static_knots
    cv2.setNumThreads(1)
    song, channel = source(video)  # Vault before opening media.
    from media import video_path
    video_file = video_path(song)
    scan_path, strip_path = song / 'notation/scan.json', song / 'truth-reviewed/strip.json'
    scan, systems = load(scan_path), load(strip_path)
    folder = confined(DATA / 'native' / video)
    folder.mkdir(parents=True, exist_ok=True)
    fingerprint = dict(version=VERSION, scan=sha(scan_path), strip=sha(strip_path),
                       legend=sha(ROOT / f'tools/truth_video/legends/{channel}.json'))
    if (folder / 'manifest.json').exists():
        if load(folder / 'manifest.json').get('channel_pilot') != VERSION:
            raise ValueError('refusing to replace unrelated native manifest: ' + video)
        if (folder / 'inventory.json').exists() and load(folder / 'inventory.json').get('inputs') == fingerprint:
            inventory = load(folder / 'inventory.json')
            for item in load(folder / 'manifest.json')['systems']:
                for variant in item['variants']:
                    if sha(folder / variant['path']) != variant['sha256']:
                        raise ValueError('native frame changed: ' + item['id'])
            print(video + ': reuse ' + str(inventory['systems_found']) + ' native jobs', flush=True)
            return inventory
    records = {r['system']: r for r in scan['records']}
    cap = cv2.VideoCapture(str(video_file))
    fps, width, height = cap.get(5), int(cap.get(3)), int(cap.get(4))
    if fps <= 0 or abs(fps - scan['fps']) > .01:
        cap.release()
        raise ValueError('video/scan fps mismatch: ' + video)
    items, dispositions = [], []
    try:
        for s in systems:
            index = s['system']; sid = f'{video}_chan_{index:04d}'
            record = records.get(index)
            reasons = skip_reasons(s)
            if record is None:
                reasons.append('no matching native scan record')
            if reasons:
                dispositions.append(dict(id=sid, system=index, status='skipped', reasons=reasons))
                continue
            if scan.get('reconstructed_scroll'):
                raise ValueError('this pilot requires static native scan records')
            left, right = scan.get('left', 0), scan.get('right', width)
            y0, y1 = [scan['top'] + y for y in record['crop']]
            if not (0 <= left < right <= width and 0 <= y0 < y1 <= height):
                raise ValueError('native crop outside video: ' + sid)
            # Exact scanner crop, WITHOUT solos_extract's extra -40/+50 margin.
            saved = song / 'truth-reviewed' / f'system_{index:03d}.png'
            if not saved.exists():
                saved = song / 'notation' / saved.name
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
                check = match_geometry(cv, s); checks.append(check)
                sharpness = float(cv2.Laplacian(ink(photo, fg), cv2.CV_64F).var())
                # Same static rank as solos_extract; coordinate sanity is an
                # eligibility check, not a truth/audio-based selection score.
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
            item = {**base, 'primary': primary, 'variants': variants, 'automatic_anchor': anchor,
                    'channel_geometry': checks[primary], 'candidate_geometry': checks}
            items.append(item)
            dispositions.append(dict(id=sid, system=index, status='found', reasons=[],
                                     cv_heads=len(cv['notes']), strip_heads=len(s['notes']),
                                     frame=frame_numbers[primary], offset_xy=checks[primary]['offset_xy'],
                                     matched=checks[primary]['matched'],
                                     residual_max_px=checks[primary]['residual_max_px']))
    finally:
        cap.release()
    dump(folder / 'manifest.json', dict(version=1, channel_pilot=VERSION, video=video,
         source_video=relative(video_file), source_size=video_file.stat().st_size,
         source_scan_sha256=sha(scan_path), source_strip_sha256=sha(strip_path),
         fps=fps, width=width, height=height, systems=items,
         policy='one job per eligible CV strip system; exact native scanner row crop; no audio'))
    inventory = dict(video=video, channel=channel, pilot=relative(song), vault=False,
                     inputs=fingerprint, scan_records=len(scan['records']), strip_systems=len(systems),
                     systems_found=len(items), systems=[it['id'] for it in items], dispositions=dispositions,
                     manifest_sha256=sha(folder / 'manifest.json'))
    dump(folder / 'inventory.json', inventory)
    print(video + ': found ' + str(len(items)) + '/' + str(len(systems)), flush=True)
    return inventory


def prepare_video(inventory):
    """Use hybrid.prepare unchanged, then make its scratch metadata portable."""
    from hybrid import prepare, instructions
    folder = DATA / 'native' / inventory['video']
    items = {it['id']: it for it in load(folder / 'manifest.json')['systems']}
    pending = []
    for sid, item in items.items():
        job = DATA / 'hybrid/jobs' / sid
        if (job / 'job.json').exists():
            meta = load(job / 'job.json')
            if (meta.get('channel_pilot') == VERSION
                    and meta.get('manifest_sha256') == inventory['manifest_sha256']
                    and meta.get('channel_geometry') == item['channel_geometry']):
                assert_job(job, meta, {'systems': list(items)})
                continue  # Keep frozen CV timings/digests and GPT resume identity.
        pending.append(sid)
    prepared = DATA / 'hybrid/prepared.json'
    previous = prepared.read_bytes() if prepared.exists() else None
    try:
        if pending:
            prepare(pending, folder)
    finally:
        # Preserve unrelated batches' shared default selection.
        if previous is not None:
            prepared.write_bytes(previous)
        elif prepared.exists():
            prepared.unlink()
    for sid in pending:
        item = items[sid]
        job = DATA / 'hybrid/jobs' / sid
        meta, cv = load(job / 'job.json'), load(job / 'cv.json')
        meta.update(bench=relative(folder), job_dir=relative(job), channel_pilot=VERSION,
                    channel_geometry=item['channel_geometry'],
                    source_strip=relative(ROOT / inventory['pilot'] / 'truth-reviewed/strip.json'),
                    source_strip_sha256=inventory['inputs']['strip'])
        dump(job / 'job.json', meta)
        (job / 'INSTRUCTIONS.md').write_text(instructions(meta, cv))


def extract(workers):
    for video in VIDEOS:
        source(video)
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        inventories = list(pool.map(extract_video, VIDEOS))
    for inventory in inventories:
        prepare_video(inventory)
    result = dict(version=VERSION, systems=[sid for inv in inventories for sid in inv['systems']],
                  videos=inventories)
    dump(PILOT / 'systems.json', result)
    write_readme()
    return result


def convert_cursor(reading, meta, cv, legend, original, *, cv_sanity=False):
    """Variant A: replace heads, preserve canonical cursor/colour/staff evidence."""
    reading = validate(copy.deepcopy(reading), meta['system_id'])
    if original['system'] != meta['system']:
        raise ValueError('strip system index mismatch')
    result = copy.deepcopy(original)
    result.pop('rhythm_x', None); result.pop('rhythm_t', None)
    staff = result['staff']; reasons = staff.setdefault('reasons', [])
    spacing = staff['spacing']
    if len(staff.get('lines', [])) != 5:
        raise ValueError('CV did not establish a native five-line staff')
    dx, dy = meta['channel_geometry']['offset_xy']
    native = {f"cv-{n['label']}": n for n in cv['notes']}
    heads = flatten(reading); owned = original.get('read_x')
    timed, omitted, unvisited = [], [], []
    for h in heads:
        x = h['x'] + dx if h['x'] is not None else None
        if owned is not None and x is not None and not owned[0] <= x <= owned[1]:
            unvisited.append({**copy.deepcopy(h), 'native_x': h['x'], 'native_y': h['y'], 'x': x,
                              'y': h['y'] + dy if h['y'] is not None else None,
                              'exportable': False, 'reason': 'outside CV strip visited score span'})
        elif h.get('grace'):
            omitted.append(dict(id=h['id'], reason='resolved grace, no separate attack', blocking=False))
        elif x is None or (not cv_sanity and
                (h['onset'] is None or h['tie_continuation'] is None or h.get('rhythm_uncertain'))):
            omitted.append(dict(id=h['id'], reason='unresolved owned onset/tie/rhythm or location', blocking=True))
        else:
            timed.append(h)
    if any(m['blocking'] for m in omitted):
        reasons.append(UNRESOLVED)
    groups = collections.defaultdict(list)
    if not cv_sanity:
        for h in timed:
            groups[h['bar'], Fraction(h['onset'])].append(h)
    means = {key: sum(h['x'] for h in values) / len(values) + dx for key, values in groups.items()}
    attack_keys = {key for key, values in groups.items() if any(not h['tie_continuation'] for h in values)}
    positions = sorted(means[key] for key in attack_keys)
    if any(b - a < .55 * spacing for a, b in zip(positions, positions[1:])):
        reasons.append(COLLISION)
    if not cv_sanity and len(attack_keys) < 6:
        reasons.append(FEW)
    notes = []
    reference = original['notes'] + original.get('unvisited_heads', [])
    matched = {f"cv-{m['label']}": reference[m['strip_index']]
               for m in meta['channel_geometry'].get('matches', [])}
    for h in timed:
        orig = native.get(h['id'], {})
        rule = legend['rules'].get(f"{h['notehead']}:{h['pos']}", {})
        inst = (orig.get('inst') if h.get('source') == 'cv' and orig.get('verified') and rule.get('verified')
                and h['instrument'] == orig.get('inst') == rule.get('inst') else 'unknown')
        exportable = bool(inst != 'unknown' and orig.get('exportable') and not h['tie_continuation'])
        n = {**matched.get(h['id'], {}), **orig, 'id': h['id'],
             'x': h['x'] + dx if cv_sanity else means[h['bar'], Fraction(h['onset'])],
             'y': h['y'] + dy if h['y'] is not None else None,
             'native_x': h['x'], 'native_y': h['y'], 'pos': h['pos'], 'shape': h['notehead'],
             'inst': inst, 'exportable': exportable, 'tied_continuation': h['tie_continuation'],
             'bar': h['bar'], 'onset_q': str(Fraction(h['onset']) * 4) if h['onset'] is not None else None,
             'duration_q': str(Fraction(h['duration']) * 4) if h['duration'] is not None else None,
             'tuplet': h['tuplet'], 'observed_ghost': h['ghost'], 'observed_accent': h['accent'],
             'ghost': bool(orig.get('ghost', False)), 'accent': bool(orig.get('accent', False)),
             'flam': bool(orig.get('flam', False)), 'roll': bool(orig.get('roll', False)),
             'confidence': orig.get('confidence', 0), 'shape_confidence': orig.get('shape_confidence', 0),
             'head_independently_reviewed': False,
             'identity_provenance': 'native CV and existing verified channel legend' if inst != 'unknown'
                                    else 'unknown; no CV instrument approval',
             'source_system': meta['system']}
        notes.append(n)
    result.update(notes=notes, vision_system_id=meta['system_id'], vision_omitted_heads=omitted,
                  unvisited_head_ids=[h['id'] for h in unvisited], hybrid_unvisited_heads=unvisited,
                  hybrid_rejected_cv_heads=reading.get('hybrid', {}).get('rejected_cv_heads', []),
                  hybrid_geometry=meta['channel_geometry'],
                  axis='native strip x; CV geometry only' if cv_sanity else 'native strip x; chord means by (bar, rational onset)',
                  color_policy='original CV head colour knots retained inside their observed span; align.prior unchanged',
                  cv_sanity=cv_sanity)
    return result


def assert_job(job, meta, inventory):
    if meta.get('channel_pilot') != VERSION or meta['system_id'] not in inventory['systems']:
        raise ValueError('not a frozen channel pilot job')
    song, _ = source(meta['video'])
    if sha(song / 'truth-reviewed/strip.json') != meta['source_strip_sha256']:
        raise ValueError('CV strip changed after extraction: ' + meta['system_id'])
    if sha(job / 'cv.json') != meta['cv_sha256']:
        raise ValueError('CV job integrity mismatch: ' + meta['system_id'])
    for image in meta['images']:
        if sha(job / image['file']) != image['sha256']:
            raise ValueError('job image integrity mismatch: ' + meta['system_id'])


def stage(variant, run=None):
    from hybrid import cv_reading
    from auto_timing import convert_auto
    inventory = load(PILOT / 'systems.json')
    all_ids = set(inventory['systems'])
    run_folder = DATA / 'runs' / safe_name(run) if run else None
    outcomes = {}
    if variant != 'cv':
        if run_folder is None:
            raise ValueError('--run is required for A/B')
        config = load(run_folder / 'config.json')
        if not set(config['systems']) <= all_ids:
            raise ValueError('run contains non-channel-pilot systems')
        if (run_folder / 'run.json').exists():
            outcomes = {r['id']: r for r in load(run_folder / 'run.json')['results']}
    summaries = []
    for inv in inventory['videos']:
        video = inv['video']; song, _ = source(video)
        originals = load(song / 'truth-reviewed/strip.json')
        dispositions = {d['system']: d for d in inv['dispositions']}
        systems, records = [], []
        for original in originals:
            index = original['system']; disposition = dispositions[index]; sid = disposition['id']
            system = copy.deepcopy(original)
            if disposition['status'] != 'found':
                system['staff'].setdefault('reasons', []).append('channel pilot excluded: ' + '; '.join(disposition['reasons']))
                if variant == 'B':
                    system['notes'] = []
                records.append(dict(id=sid, system=index, status='skipped', reasons=disposition['reasons']))
            else:
                prepared = DATA / 'hybrid/jobs' / sid
                job = prepared if variant == 'cv' else run_folder / sid
                meta = load(job / 'job.json') if (job / 'job.json').exists() else load(prepared / 'job.json')
                # Verify both frozen prepared inputs and the per-run copies.
                assert_job(prepared, load(prepared / 'job.json'), inventory)
                outcome = (dict(status='ok') if variant == 'cv' else
                           load(job / 'result.json') if (job / 'result.json').exists() else
                           outcomes.get(sid, dict(status='not_run')))
                if variant != 'cv' and sid not in config['systems']:
                    outcome = dict(status='not_in_run')
                if outcome['status'] != 'ok':
                    system['staff'].setdefault('reasons', []).append('hybrid reading unavailable: ' + outcome['status'])
                    if variant == 'B':
                        system['notes'] = []
                    records.append(dict(id=sid, system=index, status=outcome['status'], reading_ok=False))
                else:
                    assert_job(job, meta, inventory)
                    cv, legend = load(job / 'cv.json'), load(job / 'legend.json')
                    if variant == 'cv':
                        reading = cv_reading(sid, cv)
                    else:
                        if sha(job / 'reading.json') != outcome['reading_sha256']:
                            raise ValueError('reading changed after inference: ' + sid)
                        reading = validate(load(job / 'reading.json'), sid)
                        identity = outcome.get('identity', {}).get('inputs', {})
                        for name, digest in identity.items():
                            if sha(prepared / name) != digest:
                                raise ValueError('prepared inference input changed: ' + sid + '/' + name)
                        if identity.get('legend.json') and sha(job / 'legend.json') != identity['legend.json']:
                            raise ValueError('run legend changed after inference: ' + sid)
                    try:
                        system = (convert_auto(reading, meta, cv, legend) if variant == 'B' else
                                  convert_cursor(reading, meta, cv, legend, original, cv_sanity=variant == 'cv'))
                        status = 'staged'
                    except (ValueError, TypeError, KeyError, OverflowError, ZeroDivisionError) as error:
                        system['notes'] = []
                        system['staff'].setdefault('reasons', []).append('hybrid conversion failed: ' + str(error))
                        status = 'conversion_failed'
                    records.append(dict(id=sid, system=index, status=status, reading_ok=True,
                                        notes=len(system['notes']), reasons=system['staff']['reasons'],
                                        reading_sha256=sha(job / 'reading.json') if variant != 'cv' else None))
            systems.append(system)
        folder = PILOT / variant / video
        dump(folder / 'strip.json', systems)
        record = dict(video=video, variant=variant, run=run, systems_found=inv['systems_found'],
                      systems_read=sum(r.get('reading_ok', False) for r in records), systems=records,
                      strip_sha256=sha(folder / 'strip.json'),
                      policy='A/cv retain excluded CV rows as blocked context for canonical song-phase; B excluded rows have no heads')
        dump(folder / 'staging.json', record); summaries.append(record)
        print(f"{video}: {variant} read {record['systems_read']}/{record['systems_found']}", flush=True)
    dump(PILOT / variant / 'staging.json', dict(variant=variant, run=run, videos=summaries))
    return summaries


def reference_counts(video):
    from solos_gate import fast_counts
    song, _ = source(video); folder = song / 'truth-reviewed'
    alignment, scoring, result = [load(folder / p) for p in
                                  ('alignment.json', 'alignment.scoring.json', 'result.json')]
    strict_events, scoring_events = load(folder / 'truth.json'), load(folder / 'truth.scoring.json')
    return dict(strict_pass=sum(s['usable'] for s in alignment['systems']),
                scoring_pass=sum(s['scoring_usable'] for s in scoring['systems']),
                strict_notes=len(strict_events), scoring_notes=len(scoring_events),
                fast_strict=fast_counts(strict_events), fast_scoring=fast_counts(scoring_events),
                result_counts={k: result[k] for k in ('strict_systems', 'scoring_systems', 'strict_notes', 'scoring_notes')})


def portable_alignment(folder):
    # build_truth emits resolved source paths; only its scratch provenance is
    # normalized after tiers.write, without altering gate results or inputs.
    for name in ('alignment.json', 'alignment.scoring.json'):
        value = load(folder / name)
        value['song'] = relative(value['song'])
        if value.get('band_evidence') and value['band_evidence'].get('source'):
            value['band_evidence']['source'] = relative(value['band_evidence']['source'])
        dump(folder / name, value)


def gate_video(task):
    video, variant = task
    from align import build_truth
    import bands
    from tiers import write
    from solos_gate import fast_counts
    song, _ = source(video); folder = PILOT / variant / video
    staging = load(folder / 'staging.json')
    if sha(folder / 'strip.json') != staging['strip_sha256']:
        raise ValueError('staged strip changed: ' + video)
    truth, alignment = build_truth(song, folder, bands=bands.load(song, cache=False))
    scoring = write(folder)
    scored = load(folder / 'truth.scoring.json')
    reasons = collections.Counter(r for s in alignment['systems'] for r in s['reasons'])
    result = dict(video=video, channel=VIDEOS[video], variant=variant, run=staging['run'],
                  systems_found=staging['systems_found'], systems_read=staging['systems_read'],
                  systems_gated=len(alignment['systems']),
                  strict_pass=sum(s['usable'] for s in alignment['systems']),
                  scoring_pass=sum(s['scoring_usable'] for s in scoring['systems']),
                  strict_notes=len(truth), scoring_notes=len(scored),
                  fast_strict=fast_counts(truth), fast_scoring=fast_counts(scored),
                  reasons=dict(reasons), top_reasons=reasons.most_common(5), cv_only=reference_counts(video),
                  gate_sha256={p: sha(ROOT / 'tools/truth_video' / p) for p in ('align.py', 'bands.py', 'tiers.py')},
                  attacks_sha256=sha(song / 'attacks.htdemucs.npy'), band_sha256=sha(song / bands.CACHE),
                  strip_sha256=staging['strip_sha256'])
    portable_alignment(folder)
    dump(folder / 'summary.json', result)
    print(f"{video}: {variant} systems {result['strict_pass']}/{result['scoring_pass']}; "
          f"notes {result['strict_notes']}/{result['scoring_notes']}", flush=True)
    return result


def gate(variant, workers):
    for video in VIDEOS:
        source(video)
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(gate_video, [(v, variant) for v in VIDEOS]))
    dump(PILOT / variant / 'summary.json', dict(variant=variant, videos=results))
    return results


def decision(positive, notes, baseline):
    if positive >= 4 and notes >= 2 * baseline:
        return '전체 수집 진행 권고'
    if positive <= 1:
        return '커서 있어도 실패 — 원인 분석 필요'
    return '중간'


def diagnose_cv():
    """Separate adapter contract changes from native-frame/CV input changes.

    For identical final-strip geometry, require identical moments, time priors,
    kick/snare phase inputs and exportable identities. Optional scratch replay
    of the untouched original strips is reported separately, never as cv yield.
    """
    import numpy as np
    from align import moments, prior, moment_instruments
    from hybrid import cv_reading
    inventory = load(PILOT / 'systems.json')
    diagnostics = []
    for inv in inventory['videos']:
        video = inv['video']; song, _ = source(video)
        original = load(song / 'truth-reviewed/strip.json')
        folder = PILOT / 'cv' / video
        if not (folder / 'summary.json').exists():
            continue
        staged = load(folder / 'strip.json')
        old_by = {s['system']: s for s in original}; new_by = {s['system']: s for s in staged}
        invariants, delta_counts = [], collections.Counter()
        for d in inv['dispositions']:
            if d['status'] != 'found':
                continue
            s = old_by[d['system']]; new = new_by[d['system']]
            job = DATA / 'hybrid/jobs' / d['id']
            cv, meta, legend = [load(job / p) for p in ('cv.json', 'job.json', 'legend.json')]
            exact_cv = {**cv, 'staff': s['staff'], 'notes': [{**n, 'label': i + 1} for i, n in enumerate(s['notes'])]}
            exact_meta = {**meta, 'channel_geometry': dict(offset_xy=[0, 0], matches=[])}
            exact = convert_cursor(cv_reading(d['id'], exact_cv), exact_meta, exact_cv, legend, s, cv_sanity=True)
            x, groups = moments(s); xe, ge = moments(exact)
            same_moments = np.array_equal(x, xe)
            same_prior = True
            if len(x) >= 6 and same_moments:
                t, mode = prior(s, x); te, me = prior(exact, xe)
                same_prior = np.array_equal(t, te) and mode == me
            same_core = ([q & {'kick', 'snare'} for q in moment_instruments(s, groups)] ==
                         [q & {'kick', 'snare'} for q in moment_instruments(exact, ge)])
            identity = lambda system: [(n['x'], n['pos'], n['inst'], n.get('articulation'))
                                       for n in system['notes'] if n.get('exportable', True)
                                       and not n.get('tied_continuation')]
            same_exports = identity(s) == identity(exact)
            invariants.append(dict(system=s['system'], moments=same_moments, priors=same_prior,
                                   phase_core_sets=same_core, export_identities=same_exports))
            xn, _ = moments(new)
            delta_counts['original_owned_heads'] += len(s['notes'])
            delta_counts['native_owned_heads'] += len(new['notes'])
            delta_counts['head_count_changed_systems'] += len(s['notes']) != len(new['notes'])
            delta_counts['moment_count_changed_systems'] += len(x) != len(xn)
            delta_counts['native_geometry_changed_systems'] += not np.array_equal(x, xn)
        old_alignment = load(song / 'truth-reviewed/alignment.scoring.json')['systems']
        new_alignment = load(folder / 'alignment.scoring.json')['systems']
        old_al = {r['system']: r for r in old_alignment}
        transitions = []
        for r in new_alignment:
            prev = old_al[r['system']]
            if (prev['usable'], prev['scoring_usable']) != (r['usable'], r['scoring_usable']):
                transitions.append(dict(system=r['system'], old_strict=prev['usable'], strict=r['usable'],
                                        old_scoring=prev['scoring_usable'], scoring=r['scoring_usable'],
                                        old_moments=prev['moments'], moments=r['moments'],
                                        old_held_out=prev.get('held_out'), held_out=r.get('held_out'),
                                        old_reasons=prev['reasons'], reasons=r['reasons'],
                                        song_phase_source=(r.get('phase') or {}).get('song_level', {}).get('source')))
        replay = PILOT / 'cv-original-replay' / video / 'summary.json'
        replay_equal = None
        if replay.exists():
            replay_result = load(replay)
            replay_strip = replay.parent / 'strip.json'
            if replay_result['strip_sha256'] == sha(replay_strip) and load(replay_strip) == original:
                current_reference = reference_counts(video)
                replay_equal = all(replay_result[k] == current_reference[k] for k in
                                   ('strict_pass', 'scoring_pass', 'strict_notes', 'scoring_notes'))
        diagnostics.append(dict(video=video, same_geometry_systems_checked=len(invariants),
                                same_geometry_adapter_exact=all(all(r[k] for k in
                                   ('moments', 'priors', 'phase_core_sets', 'export_identities')) for r in invariants),
                                invariants=invariants, native_deltas=dict(delta_counts),
                                tier_transitions=transitions, original_replay_equal=replay_equal))
    result = dict(videos=diagnostics, same_geometry_systems_checked=sum(r['same_geometry_systems_checked'] for r in diagnostics),
                  same_geometry_adapter_exact=all(r['same_geometry_adapter_exact'] for r in diagnostics),
                  original_replay_videos_equal=sum(r['original_replay_equal'] is True for r in diagnostics))
    dump(PILOT / 'cv-diagnostics.json', result)
    return result


def report():
    inventory = load(PILOT / 'systems.json')
    refs = {v: reference_counts(v) for v in VIDEOS}
    baseline = {k: sum(r[k] for r in refs.values()) for k in
                ('strict_pass', 'scoring_pass', 'strict_notes', 'scoring_notes')}
    variants = {}
    for variant in ('A', 'B', 'cv'):
        path = PILOT / variant / 'summary.json'
        if path.exists():
            rows = load(path)['videos']
            if any(r.get('strip_sha256') != sha(PILOT / variant / r['video'] / 'strip.json') for r in rows):
                continue  # A newer stage requires a new gate, not stale metrics.
            totals = {k: sum(r[k] for r in rows) for k in
                      ('systems_found', 'systems_read', 'strict_pass', 'scoring_pass', 'strict_notes', 'scoring_notes')}
            totals['strict_positive_videos'] = sum(r['strict_notes'] > 0 for r in rows)
            for key in ('fast_strict', 'fast_scoring'):
                totals[key] = {kind: sum(r[key][kind] for r in rows) for kind in ('K', 'S', 'T')}
            totals['decision'] = decision(totals['strict_positive_videos'], totals['strict_notes'], baseline['strict_notes'])
            variants[variant] = dict(videos=rows, totals=totals)
    primary = variants.get('A')
    verdict = primary['totals']['decision'] if primary else 'GPT A/B 미실시 — 전체 수집 판정 대기'
    lines = ['# 커서 채널 hybrid 파일럿', '', '**결론: ' + verdict + '.**', '',
             f"9개 영상에서 {len(inventory['systems'])}개 native hybrid job을 준비했다. "
             '이 도구는 오프라인 준비·staging·게이트만 수행한다. GPT reading은 기존 hybrid_gpt.py로 별도 실행한다.', '',
             f"CV-only 기준은 strict {baseline['strict_notes']} / scoring {baseline['scoring_notes']} notes, "
             f"strict/scoring 시스템 {baseline['strict_pass']}/{baseline['scoring_pass']}다. "
             'Xiangsong strict 0 / scoring 47, Ollie strict 193 (2/5 영상)이라는 참조값을 현재 JSON에서 다시 확인한다.', '',
             '판정 규칙: strict-positive 영상 ≥4/9 **그리고** strict notes ≥ CV-only total의 2배 → '
             '“전체 수집 진행 권고”; strict-positive 영상 ≤1 → “커서 있어도 실패 — 원인 분석 필요”; 그 외 “중간”. '
             'cv는 어댑터 진단이며 GPT 성능이나 수집 승인으로 사용하지 않는다.', '']
    for variant, label in [('A', 'cursor prior, 주 실험'), ('B', 'rhythm prior, 참조'), ('cv', 'CV geometry sanity')]:
        lines += ['## ' + variant + ': ' + label, '',
                  '| 영상 | found/read | strict/scoring 시스템 | strict/scoring notes | strict fast K/S/T | scoring fast K/S/T | CV-only 시스템 | CV-only notes | 주요 거부 사유 |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|---|']
        data = variants.get(variant)
        rows = {r['video']: r for r in data['videos']} if data else {}
        for video in VIDEOS:
            ref = refs[video]; r = rows.get(video)
            inv = next(i for i in inventory['videos'] if i['video'] == video)
            refcell = f"{ref['strict_pass']}/{ref['scoring_pass']} | {ref['strict_notes']}/{ref['scoring_notes']}"
            if r:
                fast = '/'.join(str(r['fast_strict'][k]) for k in ('K', 'S', 'T'))
                scoring_fast = '/'.join(str(r['fast_scoring'][k]) for k in ('K', 'S', 'T'))
                top = '; '.join(f'{reason} ({count})' for reason, count in r['top_reasons'][:3])
                lines.append(f"| {video} | {r['systems_found']}/{r['systems_read']} | {r['strict_pass']}/{r['scoring_pass']} | "
                             f"{r['strict_notes']}/{r['scoring_notes']} | {fast} | {scoring_fast} | {refcell} | {top} |")
            else:
                lines.append(f"| {video} | {inv['systems_found']}/대기 | 미실시 | 미실시 | 미실시 | 미실시 | {refcell} | GPT/게이트 대기 |")
        if data:
            t = data['totals']
            lines += ['', f"합계: found/read {t['systems_found']}/{t['systems_read']}, "
                      f"strict/scoring 시스템 {t['strict_pass']}/{t['scoring_pass']}, notes {t['strict_notes']}/{t['scoring_notes']}, "
                      f"strict-positive {t['strict_positive_videos']}/9. 규칙상 {t['decision']}" +
                      (' (진단 전용).' if variant == 'cv' else '.'), '']
    lines += ['## 추출과 좌표', '', '| 영상 | strip 시스템 | jobs | 제외 사유 (중복 집계) |', '|---|---:|---:|---|']
    for inv in inventory['videos']:
        excluded = collections.Counter(r for d in inv['dispositions'] if d['status'] != 'found' for r in d['reasons'])
        lines.append(f"| {inv['video']} | {inv['strip_systems']} | {inv['systems_found']} | " +
                     '; '.join(f'{r} ({n})' for r, n in excluded.most_common()) + ' |')
    geometry = [load(DATA / 'hybrid/jobs' / sid / 'job.json')['channel_geometry'] for sid in inventory['systems']]
    lines += ['', '스캐너 record.crop과 scan.left/right/top을 그대로 사용한다. 15/50/85% 원본 프레임을 '
              'CV 오류·staff 사유·head 수·선명도로 순위화하고, head 좌표의 상수 오프셋을 검사한다. '
              '원본·최종 strip의 같은 pos/shape head를 일대일 대응하여 각 축 최대 잔차 2px를 요구한다. '
              '미검출/추가 후보는 별도 기록하며 CV와 median strip의 동일 head inventory를 가정하지 않는다.', '',
              f"좌표 검사: {sum(g['matched'] for g in geometry)} 대응, "
              f"최대 잔차 {max((g['residual_max_px'] or 0 for g in geometry), default=0):.3f}px. "
              '개별 오프셋·잔차·미대응 수는 native manifest와 job.json에 있다.', '',
              '## 게이트·비교 해석', '',
              'A는 최종 CV strip을 깊은 복사하고 notes만 hybrid reading으로 교체한다. read_x 바깥은 '
              'unvisited로 보존하며 내보내지 않는다. 같은 (bar, rational onset)의 heads는 평균 native x를 공유한다. '
              '미해결 onset/tie/rhythm·위치와 서로 다른 순간의 충돌은 구조 거부 사유다. CV/legend가 함께 '
              '검증한 악기만 exportable이며 GPT-only head는 unknown이다. Ghost/accent 등의 export flag는 CV를 따른다.', '',
              'A/cv의 color_x/color_t는 CV head 좌표에 연결된 원래 knots를 그대로 유지한다. '
              'align.prior가 관측 span 안에서만 colour를 쓰고 나머지는 기존 cursor/interval로 처리한다. '
              'rhythm_x/rhythm_t를 만들지 않는다. B는 같은 scan record의 static_knots 앵커로 '
              'auto_timing.convert_auto를 그대로 호출한다. rhythm/interval 모드에만 기존 70%/90% no-cursor 조건이 적용된다.', '',
              'cv_reading은 모든 onset/duration이 null이다. cv-sanity는 동일 A 함수의 명시적인 geometry-only 모드로 '
              'native x와 CV tie를 보존하고 유리수 rhythm·chord 보정·unresolved-onset 판정을 적용하지 않는다. '
              '실제 GPT A의 미해결 head 검사는 유지한다. cv 결과 차이는 선택한 원본 프레임과 scanner median의 '
              '검출 head/위치 차이, 검증되지 않은 악기를 unknown으로 바꾸는 공통 export 정책, '
              'visited span/원본보다 강한 grace 제외 정책에서 발생할 수 있다.', '',
              'A/cv는 job 제외 행도 원래 CV notes·prior와 구조 거부 사유를 남겨 build_truth의 다른 시스템 '
              'song-phase/lag 처리와 문맥을 유지한다. 누락 reading과 좌표 거부 행은 추가 구조 사유로 export를 막는다. '
              'B는 job 제외/미읽은 행의 notes를 비워 automatic adapter reference를 유지한다.', '',
              'build_truth(song, output, bands=bands.load(song, cache=False)) 다음 tiers.write(output)를 호출한다. '
              'strict/scoring은 기존 채널 batch 경로와 동일 함수이며 scoring은 strict를 포함한다. '
              'Fast K/S/T는 solos_gate.fast_counts의 같은 시스템 내부 core 순간 간격 <90ms 정의다. '
              '검출 정확도·완전한 연주 truth·청취 일치는 이 게이트만으로 증명되지 않는다.', '',
              '## 재현', '', '[정확한 실행 명령](README.md). 시스템 목록은 systems.json, '
              'native 프레임은 ../native/<video>, GPT job은 ../hybrid/jobs/<sid>, '
              'staging·alignment·truth·요약은 <variant>/<video>에 있다. 파일 경로는 ROOT 기준 상대 경로다. ', '']
    diagnostics = diagnose_cv() if 'cv' in variants else None
    if diagnostics:
        lines += ['## CV sanity 차이 분리 검증', '',
                  f"동일한 최종-strip geometry를 A 어댑터에 넣은 {diagnostics['same_geometry_systems_checked']}개 시스템에서 "
                  f"moments/prior/K·S phase 입력/export identity 동등성: {diagnostics['same_geometry_adapter_exact']}. "
                  f"원본 strip 그대로의 scratch 게이트 재실행은 {diagnostics['original_replay_videos_equal']}/9개 영상의 "
                  'strict/scoring 시스템·notes를 기존 truth-reviewed와 정확히 재현했다. '
                  '선택된 원본 프레임의 cv-sanity 수치는 위 표에 별도로 유지한다.', '',
                  '| 영상 | 원본/native owned heads | head 수 변동 시스템 | moment 수 변동 시스템 | tier 변동 시스템 |',
                  '|---|---:|---:|---:|---|']
        for d in diagnostics['videos']:
            delta = d['native_deltas']
            lines.append(f"| {d['video']} | {delta.get('original_owned_heads', 0)}/{delta.get('native_owned_heads', 0)} | "
                         f"{delta.get('head_count_changed_systems', 0)} | {delta.get('moment_count_changed_systems', 0)} | "
                         ', '.join(str(r['system']) for r in d['tier_transitions']) + ' |')
        lines += ['', 'Ollie의 차이는 선택 원본 프레임과 scanner median의 head inventory/centroid 변화에 민감하다. '
                  'VhdBLkFMmWs system 44는 원래 held-out 20ms 7/13 (53.8%)에서 native 8/13 (61.5%)로 바뀌며 '
                  'strict timing을 통과한다. 이 시스템의 phase가 성립하자 기존 build_truth의 2차 song-phase 처리가 '
                  '다른 짧은 시스템을 뒷받침한다. mwLnCnc60b4도 다른 시스템의 pooled phase가 60% 경계를 넘으면서 '
                  '통과가 바뀐다. MPoZ-2nOESc system 62는 moment가 10→11로 늘어 기존 timing 통과를 잃는다. '
                  '이는 GPT 개선이 아니며, 단일 프레임 CV의 안정성을 제한한다. 개별 전후 사유·검사는 cv-diagnostics.json에 있다.', '']
    integrity = PILOT / 'integrity.json'
    if integrity.exists():
        check = load(integrity)
        lines += ['보호 파일 검사: ' + str(check.get('unchanged')) + '/' + str(check.get('files_checked')) +
                  ' unchanged. 변경: ' + str(check.get('changed', [])) + '. 추가/삭제: ' + str(check.get('added', [])) +
                  '/' + str(check.get('removed', [])) + '.', '']
    summary = dict(version=VERSION, created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   jobs=len(inventory['systems']), cv_only=refs, baseline=baseline,
                   variants=variants, conclusion=verdict, cv_diagnostics=diagnostics,
                   gpt_invocations_by_this_tool=0,
                   decision_rule=dict(min_strict_positive_videos=4, strict_note_multiplier=2, failure_max_positive_videos=1))
    dump(PILOT / 'summary.json', summary)
    (confined(PILOT / 'REPORT.md')).write_text('\n'.join(lines) + '\n')
    write_readme()
    return summary


def write_readme():
    text = '''# Channel pilot handoff

Run from the repository root. All input pilots are read-only. Default DATA is
`.eval-cache/vision_pilot`; common.DATA also supports its existing configured
`scratch/hybrid_solos` location. Keep repo-relative pilot/DATA paths when rsyncing.
Transfer the whole DATA tree, the nine `.eval-cache/pilot_new/{channel}/{video}`
folders, this script and unchanged reader/gate dependencies. Native jobs contain
repo-relative metadata; hybrid_gpt rewrites its own run job_dir on that machine.
Vault config/manifest must also be available and membership must remain false.

```bash
.venv/bin/python -B tools/vision_read/channel_pilot.py extract --workers 3
.venv/bin/python -B tools/vision_read/test_channel_pilot.py
.venv/bin/python -B tools/vision_read/channel_pilot.py stage --variant cv
.venv/bin/python -B tools/vision_read/channel_pilot.py gate --variant cv --workers 2
```

GPT invocation below is for the later, network-enabled machine; it was NOT run
in this offline task. Use a new run name for a new inference experiment.

```bash
.venv/bin/python -B tools/vision_read/hybrid_gpt.py \\
  --run-name channel-pilot-sol-high-v1 --model gpt-6.1-sol --effort high \\
  --schema-retries 1 --workers 32 --timeout 1800 --lock channelpilot \\
  --codex ~/.local/bin/codex \\
  --systems $(jq -r '.systems[]' .eval-cache/vision_pilot/channel_pilot/systems.json)
.venv/bin/python -B tools/vision_read/channel_pilot.py stage --run channel-pilot-sol-high-v1 --variant A
.venv/bin/python -B tools/vision_read/channel_pilot.py stage --run channel-pilot-sol-high-v1 --variant B
.venv/bin/python -B tools/vision_read/channel_pilot.py gate --variant A --workers 2
.venv/bin/python -B tools/vision_read/channel_pilot.py gate --variant B --workers 2
.venv/bin/python -B tools/vision_read/channel_pilot.py report
```

Gate workers are CPU workers (max 4); GPT workers are separate and controlled
by the existing runner. Failed/missing readings are inventoried and block export;
a reading SHA mismatch aborts staging. No gate threshold is changed. Report
tables mark A/B pending until their summaries exist. cv is geometry-only sanity,
not a GPT score, because hybrid.cv_reading intentionally has null rhythm.
'''
    confined(PILOT / 'README.md').parent.mkdir(parents=True, exist_ok=True)
    (PILOT / 'README.md').write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    ext = commands.add_parser('extract'); ext.add_argument('--workers', type=int, default=3)
    st = commands.add_parser('stage'); st.add_argument('--run'); st.add_argument('--variant', choices=('A', 'B', 'cv'), required=True)
    gt = commands.add_parser('gate'); gt.add_argument('--variant', choices=('A', 'B', 'cv'), required=True)
    gt.add_argument('--workers', type=int, default=2)
    commands.add_parser('report')
    args = parser.parse_args()
    if hasattr(args, 'workers') and not 1 <= args.workers <= 4:
        parser.error('Mac CPU workers must be 1..4')
    if args.command == 'extract':
        result = extract(args.workers); print('Total jobs:', len(result['systems']))
    elif args.command == 'stage':
        if args.variant != 'cv' and not args.run:
            parser.error('--run is required for A/B')
        stage(args.variant, args.run)
    elif args.command == 'gate':
        gate(args.variant, args.workers)
    else:
        result = report(); print(result['conclusion'])


if __name__ == '__main__':
    main()
