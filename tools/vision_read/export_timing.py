"""Stage rational readings for existing WSL timing gates; never load audio here."""
import argparse
import collections
import json
import sys
sys.dont_write_bytecode = True
from fractions import Fraction
from pathlib import Path
from common import DATA, confined, dump, flatten, load, sha, validate


def convert(reading, meta, legend, anchor):
    """Use rhythmic distance as align.py's x, preserving native_x for inspection.

    Existing moments() and warp() operate on x. A synthetic exact-rhythm axis
    avoids pretending engraving distance is elapsed musical time. 4096 units
    per whole note separates even very short moments at native staff tolerance.
    """
    staff = anchor.get('staff')
    if not isinstance(staff, dict) or not anchor.get('staff_reviewed'):
        raise ValueError('independently reviewed existing staff checks required')
    if not {'lines', 'spacing', 'reasons', 'clef_confirmed'} <= staff.keys() or len(staff['lines']) != 5:
        raise ValueError('staff needs five lines, spacing, reasons and clef_confirmed')
    if type(anchor.get('native_spacing')) not in (int, float) or anchor['native_spacing'] <= 0:
        raise ValueError('native_spacing must be measured on the source, not a zoomed image')
    staff = {**staff, 'reasons': list(staff['reasons'])}
    if anchor['native_spacing'] < 8:
        staff['reasons'].append('native staff spacing below 8 px')
    if not staff['clef_confirmed'] and not anchor.get('clef_omission_evidence'):
        staff['reasons'].append('percussion clef / first bar not confirmed')
    offsets = anchor.get('bar_offsets_whole')
    if not isinstance(offsets, list) or len(offsets) != reading['bar_count']:
        raise ValueError('one independently established rational offset per visible bar required')
    origin = anchor.get('time_origin_seconds')
    rate = anchor.get('seconds_per_whole')
    interval = anchor.get('interval')
    if type(origin) not in (float, int) or type(rate) not in (float, int) or rate <= 0:
        raise ValueError('need a video-time origin and positive seconds_per_whole (240/quarter BPM)')
    if not isinstance(interval, list) or len(interval) != 2 or not interval[0] < interval[1]:
        raise ValueError('need actual video-time interval [start,end]')
    reviewed_heads = set(anchor.get('verified_heads', []))
    hybrid = reading.get('hybrid', {})
    unresolved = set(hybrid.get('uncertain_heads', []))-reviewed_heads
    if unresolved:
        staff['reasons'].append('hybrid numbered heads remain uncertain; independent review required')
    if hybrid.get('rejected_cv_heads') and not anchor.get('rejected_cv_heads_reviewed'):
        staff['reasons'].append('hybrid false-positive rejections require independent coverage review')
    notes, omitted, point_times = [], [], {}
    for head in flatten(reading):
        # Retain tie heads in notes; align.moments excludes them. Grace heads do
        # not acquire invented acoustic times and remain in the original reading.
        if head.get('grace') or head['onset'] is None or offsets[head['bar']] is None or head['tie_continuation'] is None:
            omitted.append({'id': head['id'], 'reason': 'grace, unresolved onset/bar origin, or unresolved tie state'})
            continue
        coordinate = Fraction(offsets[head['bar']])+Fraction(head['onset'])
        x = float(coordinate*4096)
        rule = legend['rules'].get(f"{head['notehead']}:{head['pos']}", {})
        instrument = rule.get('inst') if rule.get('verified') and head['instrument'] == rule.get('inst') else 'unknown'
        exportable = head['id'] in reviewed_heads and instrument != 'unknown' and head['tie_continuation'] is False
        notes.append({'id': head['id'], 'x': x, 'native_x': head['x'], 'native_y': head['y'], 'pos': head['pos'], 'shape': head['notehead'],
                      'inst': instrument, 'candidate_instrument': head['instrument'], 'exportable': exportable,
                      'tied_continuation': head['tie_continuation'], 'bar': head['bar'], 'onset_q': str(Fraction(head['onset'])*4),
                      'duration_q': str(Fraction(head['duration'])*4) if head['duration'] is not None else None,
                      'tuplet': head['tuplet']['ratio'] if head['tuplet'] else None,
                      'observed_ghost': head['ghost'], 'observed_accent': head['accent'], 'ghost': False, 'accent': False,
                      'legend_verified': rule.get('verified', False), 'head_independently_reviewed': head['id'] in reviewed_heads})
        if not head['tie_continuation']:
            point_times[x] = origin+float(coordinate)*rate
    xs = sorted(point_times)
    if any(b-a <= staff['spacing']*.55 for a, b in zip(xs, xs[1:])):
        raise ValueError('rhythm-axis scale would merge distinct moments; increase scale before timing use')
    if len(xs) < 6:
        staff['reasons'].append('fewer than six resolved notated moments')
    if omitted:
        # A subset may omit attacks and inflate precision. Preserve the result
        # for inspection, but never silently qualify a complete-system export.
        staff['reasons'].append('vision reading contains unresolved/grace heads; review coverage before export')
    system = {'system': meta['system'], 'vision_system_id': meta['system_id'], 'notes': notes, 'staff': staff,
              'interval': interval, 'cursor_x': [], 'cursor_t': [], 'rhythm_x': xs, 'rhythm_t': [point_times[x] for x in xs],
              'moment_tolerance': .55, 'vision_omitted_heads': omitted, 'axis': 'exact whole-note rhythm *4096; native pixels retained separately',
              'time_anchor_evidence': anchor.get('evidence'), 'native_spacing': anchor['native_spacing']}
    return system


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', default='dense-high-look')
    ap.add_argument('--anchors', type=Path)
    ap.add_argument('--out', type=Path, default=DATA/'wsl_handoff')
    ap.add_argument('--write-anchor-template', action='store_true')
    a = ap.parse_args()
    run = confined(DATA/'runs'/a.run)
    out = confined(a.out)
    config = load(run/'config.json')
    if a.write_anchor_template:
        template = {}
        for sid in config['systems']:
            meta = load(run/sid/'job.json')
            result = load(run/sid/'result.json') if (run/sid/'result.json').exists() else {}
            reading = load(run/sid/'reading.json') if result.get('status') == 'ok' else None
            record = meta['record']
            template[sid] = {'staff_reviewed': False, 'staff': None, 'native_spacing': record.get('native_space'), 'clef_omission_evidence': None,
                             'bar_offsets_whole': [None]*(reading['bar_count'] if reading else 0), 'time_origin_seconds': None, 'seconds_per_whole': None,
                             'interval': [record['start']/meta['fps'], record['end']/meta['fps']],
                             'verified_heads': [], 'evidence': None, 'source_video_path': meta['video_path']}
        target = out/'anchors.template.json'
        if target.exists():
            ap.error('anchor template exists; choose a fresh --out')
        dump(target, template)
        print(target)
        return
    if a.anchors is None:
        ap.error('--anchors or --write-anchor-template is required')
    anchors = load(a.anchors)
    staged = collections.defaultdict(list)
    provenance = []
    for sid in config['systems']:
        job = run/sid
        result = load(job/'result.json')
        if result['status'] != 'ok':
            provenance.append({'id': sid, 'status': 'reading_failed'})
            continue
        reading = validate(load(job/'reading.json'), sid)
        if sha(job/'reading.json') != result['reading_sha256']:
            raise ValueError(f'reading modified since run: {sid}')
        meta, legend = load(job/'job.json'), load(job/'legend.json')
        try:
            system = convert(reading, meta, legend, anchors[sid])
        except (KeyError, ValueError, TypeError) as error:
            provenance.append({'id': sid, 'status': 'needs_anchor_review', 'reason': str(error)})
            continue
        staged[meta['video']].append(system)
        provenance.append({'id': sid, 'status': 'candidate_only', 'reading_sha256': sha(job/'reading.json'), 'staff_reasons': system['staff']['reasons']})
    for video, systems in staged.items():
        target = out/video/'strip.json'
        if target.exists():
            ap.error(f'candidate exists: {target}; choose a fresh --out')
        dump(target, systems)
    dump(out/'handoff.json', {'run': a.run, 'anchors_sha256': sha(a.anchors), 'videos': sorted(staged), 'systems': provenance,
                             'audio_checked': False, 'scope': 'scratch candidates only; call existing build_truth on WSL with real attacks and drum stem; preserve gates'})
    print(json.dumps({'videos_staged': len(staged), 'systems': provenance}, indent=2))


if __name__ == '__main__':
    main()
