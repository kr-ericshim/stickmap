"""CV geometry + numbered-head Gemini rhythm; never reads gold or audio."""
import argparse
import copy
import datetime
import json
import os
import re
import shutil
import sys
import time
from fractions import Fraction
from pathlib import Path
sys.dont_write_bytecode = True
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
from common import BENCH, DATA, ROOT, confined, dump, load, rational, sha, validate

HERE = Path(__file__).resolve().parent
HYBRID = DATA/'hybrid'
MODEL = 'gemini-3.8-flash-medium'
VERSION = 'hybrid-numbered-v1'


def unsealed(video):
    sys.path.insert(0, str(ROOT))
    from tools.harvest.vault import Vault
    if Vault().is_vault(video):
        raise ValueError('vault input forbidden for development: '+video)


def cv_prediction(item, bench=BENCH):
    """Same ink -> heads / label_heads calls and variant merge as bench.py."""
    unsealed(item['video'])
    sys.path.insert(0, str(ROOT/'tools/truth_video'))
    import cv2
    from layouts import ink
    from heads import heads
    from strip import label_heads
    cv2.setNumThreads(1)
    variant = item['variants'][item['primary']]
    image = bench/variant['path']
    if sha(image) != variant['sha256']:
        raise ValueError('native image integrity mismatch: '+item['id'])
    frame = cv2.imread(str(image))
    if frame is None:
        raise ValueError('unreadable image: '+str(image))
    source_legend = ROOT/item['legend']
    legend = load(source_legend)
    kind = item.get('legend_variant') or ('white_overlay' if item.get('foreground') == 'white_overlay' else
            'static_light' if item['channel'] == 'dailychops' and not item.get('reconstructed_scroll') else None)
    overrides = legend.get('variants', {}).get(kind, {})
    legend = {**legend, **overrides, 'rules': {**legend['rules'], **overrides.get('rules', {})}}
    if item.get('retain_left_margin_heads'):
        legend = {**legend, 'min_x_fraction': 0}
    if item.get('native_expected_lines'):
        from native_cv import bound_readers
        heads, label_heads = bound_readers(item['native_expected_lines'])
    gray = ink(frame, item.get('foreground'))
    start = time.monotonic()
    raw, raw_staff, raw_error = [], {}, None
    try:
        raw, lines, spacing = heads(gray, ROOT/'tools/truth_video'/legend['template_dir'] if legend.get('template_dir') else None,
                                    prefer_line_span=legend.get('prefer_staff_line_span', False))
        raw_staff = {'lines': lines, 'spacing': spacing}
    except Exception as error:
        raw_error = type(error).__name__+': '+str(error)
    try:
        notes, staff = label_heads(gray, legend)
        error = None
    except Exception as exc:
        notes, staff, error = [], {}, type(exc).__name__+': '+str(exc)
    # IDs are deterministic, independent of gold and original detector ordering.
    numbered = [{**n, 'label': index+1} for index, n in enumerate(sorted(notes, key=lambda n: (n['x'], n['y']))) ]
    bars = [{'label': 'B'+str(i+1), 'x': float(x)} for i, x in enumerate(sorted(staff.get('bar_x', [])))]
    return {'version': VERSION, 'system_id': item['id'], 'notes': numbered, 'staff': staff, 'barlines': bars,
            'raw_notes': raw, 'raw_staff': raw_staff, 'error': error, 'raw_error': raw_error,
            'wall_seconds': time.monotonic()-start, 'image_sha256': sha(image), 'legend_sha256': sha(source_legend),
            'reader_sha256': {str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT/'tools/truth_video').glob('*.py'))},
            'adapter': 'layouts.ink -> heads.heads and strip.label_heads; frozen primary native frame',
            'width': frame.shape[1], 'height': frame.shape[0], 'variant': kind}


def annotate(original, cv, target):
    """Keep original pixels at native resolution; labels in padded empty lanes."""
    import cv2
    import numpy as np
    im = cv2.imread(str(original))
    h, w = im.shape[:2]
    margin = 92
    canvas = np.full((h+2*margin, w, 3), 255, dtype=np.uint8)
    canvas[margin:margin+h] = im
    # Small drawing adapter keeps all color arguments in RGB.
    class Draw:
        def textbbox(self, xy, text, font=None):
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, .43, 1)
            return (0, 0, tw, th)
        def line(self, xy, fill, width=1):
            a, b = (xy[:2], xy[2:]) if len(xy) == 4 else xy
            cv2.line(canvas, tuple(map(round, a)), tuple(map(round, b)), fill[::-1], width, cv2.LINE_AA)
        def ellipse(self, box, outline, width=1):
            a,b,c,d = box
            cv2.ellipse(canvas, (round((a+c)/2), round((b+d)/2)), (round((c-a)/2), round((d-b)/2)), 0, 0, 360, outline[::-1], width, cv2.LINE_AA)
        def rectangle(self, box, fill, outline, width=1):
            a,b,c,d = map(round, box)
            cv2.rectangle(canvas, (a,b), (c,d), (255,255,255), -1)
            cv2.rectangle(canvas, (a,b), (c,d), outline[::-1], width)
        def text(self, xy, text, fill, font=None):
            cv2.putText(canvas, text, (round(xy[0]), round(xy[1])+13), cv2.FONT_HERSHEY_SIMPLEX, .43, fill[::-1], 1, cv2.LINE_AA)
    draw, font = Draw(), None
    occupied = []
    placements = []
    mid = sum(cv['staff'].get('lines', [h/2]))/max(1, len(cv['staff'].get('lines', [])))
    for note in cv['notes']:
        x, y = round(note['x']), round(note['y'])+margin
        text = str(note['label'])
        box = draw.textbbox((0, 0), text, font=font)
        tw, th = box[2]-box[0]+6, 20
        above = note['y'] <= mid
        choices = []
        for lane in range(4):
            ly = (margin-25-lane*20) if above else (margin+h+6+lane*20)
            for dx in (0, -tw, tw, -2*tw, 2*tw):
                lx = min(w-tw-1, max(1, x-tw//2+dx))
                rect = (lx, ly, lx+tw, ly+th)
                conflicts = sum(not (rect[2] < b[0] or rect[0] > b[2] or rect[3] < b[1] or rect[1] > b[3]) for b in occupied)
                choices.append((conflicts*10000+abs(dx)+lane*8, rect))
        _, rect = min(choices)
        occupied.append(rect)
        lx, ly, rx, ry = rect
        end = ((lx+rx)//2, ry if above else ly)
        draw.line([(x, y-5 if above else y+5), end], fill=(150, 185, 225), width=1)
        rw, rh = max(5, note.get('w', 10)/2+2), max(4, note.get('h', 8)/2+2)
        draw.ellipse((x-rw, y-rh, x+rw, y+rh), outline=(20, 110, 215), width=1)
        placements.append({'label': note['label'], 'box': list(rect), 'head_native_xy': [note['x'], note['y']]})
    # Draw boxes last so leader lines do not obscure any numbered text.
    for place in placements:
        lx, ly, rx, ry = place['box']
        draw.rectangle((lx, ly, rx, ry), fill='white', outline=(150, 185, 225), width=1)
        draw.text((lx+3, ly+1), str(place['label']), fill=(0, 65, 150), font=font)
    for index, bar in enumerate(cv['barlines']):
        x = round(bar['x'])
        lines = cv['staff'].get('lines', [])
        if lines:
            draw.line((x, round(lines[0])+margin, x, round(lines[-1])+margin), fill=(225, 110, 35), width=1)
        draw.text((max(0, min(w-45, x-12)), index%2*19), bar['label'], fill=(180, 70, 0), font=font)
    if not cv2.imwrite(str(confined(target)), canvas):
        raise ValueError('annotation write failed')
    return {'original_offset_xy': [0, margin], 'size': [w, h+2*margin], 'head_labels': placements,
            'barline_labels': cv['barlines'], 'policy': 'native pixels, thin outlines and leaders; clean original also supplied'}


def base_head(note):
    return {'id': 'cv-'+str(note['label']), 'instrument': note.get('inst'), 'pos': note['pos'], 'notehead': note['shape'],
            'x': note['x'], 'y': note['y'], 'ghost': note.get('ghost', False), 'accent': note.get('accent', False),
            'grace': note.get('grace', False), 'tuplet': None, 'tie_continuation': bool(note.get('tied_continuation', False)),
            'source': 'cv', 'cv_label': note['label'], 'instrument_verified': bool(note.get('verified')),
            'legend_evidence': note.get('legend_evidence'), 'cv_exportable': note.get('exportable', False)}


def cv_reading(sid, cv):
    xs = [0]+[b['x'] for b in cv['barlines'] if 2 < b['x'] < cv['width']-2]+[cv['width']]
    bars = [{'index': i, 'partial_left': False, 'partial_right': False, 'x_start': a, 'x_end': b, 'events': []}
            for i, (a, b) in enumerate(zip(xs, xs[1:]))]
    for note in cv['notes']:
        index = min(len(bars)-1, sum(note['x'] >= x for x in xs[1:-1]))
        bars[index]['events'].append({'onset': None, 'duration': None, 'voice': '1', 'heads': [base_head(note)]})
    lines = cv['staff'].get('lines') or None
    return validate({'schema_version': 1, 'system_id': sid, 'time_signature': None,
                     'staff': {'line_count': len(lines) if lines else None, 'lines_y': lines,
                               'clef_visible': bool(cv['staff'].get('clef_confirmed', False))},
                     'bar_count': len(bars), 'bars': bars, 'beat_checks': [], 'uncertain': []}, sid)


def instructions(meta, cv):
    table = [{'label': n['label'], 'x': round(n['x'], 2), 'y': round(n['y'], 2), 'pos': n['pos'], 'notehead': n['shape']} for n in cv['notes']]
    return f'''Read the visible drum notation for system {meta['system_id']}, using computer-vision head IDs.
LOOK MODE: Open {meta['job_dir']}/original.png with the image viewer EXACTLY ONCE and then {meta['job_dir']}/annotated.png EXACTLY ONCE. Only these two image-view calls are permitted. No terminal, file-reading, search, crop, code, agent, browser, write or other tools. The Python harness saves your final response. Return JSON only.
The clean original is authoritative for notation. The annotated image retains native pixels, offset down by 92 px, with BLUE numbered head labels and ORANGE B-numbered candidate barlines. Trace each thin blue leader to its head. The CV table supplies native geometry. Instruments and head positions of labelled heads are owned by CV; DO NOT output or reinterpret instrument/position for those heads.
HEADS: {json.dumps(table)}
CANDIDATE BARLINES (may contain stems falsely detected as boundaries): {json.dumps(cv['barlines'])}
Identify actual bars from the original, zero-based left to right including clipped edge bars; a leading clef's line is not an empty bar. For each bar x_start/x_end, select the corresponding B label when valid or estimate native x if no candidate. Do not trust barline candidates blindly.
Assign EVERY numbered label exactly once in heads, including false positives. status is head, false_positive, or uncertain. For a false_positive set bar/onset/duration/voice/flags/tuplet to null and explain reason. For uncertain keep identifiable information and null only genuinely unknown fields. Do not omit difficult heads. Read beams, flags, dots, rests and voices; NEVER infer rhythm from uniform x spacing.
onset and duration are exact rational STRINGS in WHOLE-NOTE units (quarter=1/4; sixteenth=1/16; 6:4 sixteenth=1/24), or null if unresolved. onset is relative to the true bar origin. A clipped left bar with unknown origin has null onset, not a fabricated zero. Include tie_continuation (same-pitch tie, not slur), tuplet ratio AND stable group, ghost (parentheses only), accent, grace, voice, and duration. Tied continuation remains inventoried but is not a new attack. Grace duration=0.
Report unlabelled REAL heads in misses with approximate bar, rational onset/duration, integer staff pos (bottom line=0, upwards half-space=+1), notehead, flags, tuplet, voice, and optional approximate native x,y or null. Their instruments are unverified and MUST NOT be guessed. Do not duplicate an existing label as a miss.
Check beat/voice sums including implied rests when meter/origin is known; no chord double counting. Time signature only when shown; otherwise null. Tuplet weighted 5:3 does not require five heads. Count beams and durations. Provide unknowns explicitly.
Return exactly this contract (format examples are NOT image content):
{{"schema_version":1,"system_id":"{meta['system_id']}","time_signature":null,
"bars":[{{"index":0,"partial_left":false,"partial_right":false,"x_start":0,"x_end":1000,"start_label":null,"end_label":"B1"}}],
"heads":[{{"label":1,"status":"head","bar":0,"onset":"0","duration":"1/16","voice":"1","tie_continuation":false,"tuplet":null,"ghost":false,"accent":false,"grace":false,"reason":null}}],
"misses":[],"beat_checks":[],"uncertain":[]}}
Every miss has id (unique string), bar, onset, duration, voice, pos, notehead, x, y, ghost, accent, grace, tie_continuation, tuplet, reason. Tuplet format is {{"ratio":"6:4","group":"b0-v1-t0"}}. beat_checks: bar, voice, beat_start, beat_length, sounded_duration, implied_rest_duration, status; rational whole-note strings. Empty arrays are permitted when nothing is missing/uncertain.
'''


def merge(answer, cv, sid):
    """Strict identity coverage; the model can supply rhythm but never CV labels."""
    if answer.get('schema_version') != 1 or answer.get('system_id') != sid:
        raise ValueError('wrong hybrid schema or system_id')
    fields = {'bars', 'heads', 'misses', 'beat_checks', 'uncertain', 'time_signature'}
    if not fields <= answer.keys() or any(not isinstance(answer[k], list) for k in fields-{'time_signature'}):
        raise ValueError('missing hybrid arrays')
    notes = {n['label']: n for n in cv['notes']}
    labels = [n.get('label') for n in answer['heads']]
    if any(type(n) is not int for n in labels) or len(labels) != len(set(labels)) or set(labels) != set(notes):
        raise ValueError('every CV label must occur exactly once, no invented labels')
    bounds = {b['label']: b['x'] for b in cv['barlines']}
    bars = []
    for index, b in enumerate(answer['bars']):
        if b.get('index') != index:
            raise ValueError('nonsequential bars')
        bar = copy.deepcopy(b)
        for side in ('start', 'end'):
            label = bar.get(side+'_label')
            if label is not None:
                if label not in bounds:
                    raise ValueError('unknown barline label')
                bar['x_'+side] = bounds[label]
            x = bar.get('x_'+side)
            if x is not None and (type(x) not in (int, float) or not 0 <= x <= cv['width']):
                raise ValueError('bar boundary outside native image')
        bar['events'] = []
        bars.append(bar)
    if not bars:
        raise ValueError('at least one visible bar required')
    rejected, uncertain_heads = [], []
    def add(record, head):
        required = {'bar', 'onset', 'duration', 'voice', 'tie_continuation', 'tuplet', 'ghost', 'accent', 'grace'}
        if not required <= record.keys():
            raise ValueError('missing rhythm fields')
        bar = record['bar']
        if type(bar) is not int or not 0 <= bar < len(bars):
            raise ValueError('head needs a valid bar')
        rational(record['onset']); rational(record['duration'])
        head.update({k: record[k] for k in ('tie_continuation', 'tuplet', 'ghost', 'accent', 'grace')})
        bars[bar]['events'].append({'onset': record['onset'], 'duration': record['duration'], 'voice': record['voice'], 'heads': [head]})
    for record in answer['heads']:
        state = record.get('status')
        if state == 'false_positive':
            if not record.get('reason'):
                raise ValueError('false-positive rejection needs evidence/reason')
            rejected.append({'label': record['label'], 'reason': record['reason']})
            continue
        if state not in ('head', 'uncertain'):
            raise ValueError('unknown head disposition')
        head = base_head(notes[record['label']])
        head['rhythm_source'] = 'gemini-numbered'
        head['rhythm_uncertain'] = state == 'uncertain'
        if state == 'uncertain':
            uncertain_heads.append(head['id'])
        add(record, head)
    for miss in answer['misses']:
        needed = {'id', 'pos', 'notehead', 'x', 'y', 'reason'}
        if not needed <= miss.keys() or not isinstance(miss['id'], str):
            raise ValueError('missing unlabelled-head fields')
        head = {k: miss[k] for k in ('pos', 'notehead', 'x', 'y')}
        head.update(id='miss-'+miss['id'], instrument=None, source='gemini-only-miss', instrument_verified=False,
                    instrument_status='unverified-instrument', rhythm_source='gemini-numbered', reason=miss['reason'])
        add(miss, head)
    for bar in bars:
        bar['events'].sort(key=lambda e: (e['onset'] is None, Fraction(e['onset']) if e['onset'] is not None else 0))
    reading = cv_reading(sid, cv)
    reading.update(bars=bars, bar_count=len(bars), time_signature=answer['time_signature'], beat_checks=answer['beat_checks'], uncertain=answer['uncertain'],
                   hybrid={'version': VERSION, 'rejected_cv_heads': rejected, 'uncertain_heads': uncertain_heads, 'miss_count': len(answer['misses']),
                           'cv_sha256': sha_from_json(cv), 'instrument_policy': 'CV only; all Gemini-only additions unverified-instrument'})
    return validate(reading, sid)


def sha_from_json(value):
    import hashlib
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def prepare(systems, bench=BENCH):
    manifest = load(bench/'manifest.json')
    items = {it['id']: it for it in manifest['systems']}
    for sid in systems:
        if not re.fullmatch(r'[A-Za-z0-9_-]+', sid) or sid not in items:
            raise ValueError('unknown/unsafe system: '+sid)
        item = items[sid]
        cv = cv_prediction(item, bench)
        job = confined(HYBRID/'jobs'/sid)
        job.mkdir(parents=True, exist_ok=True)
        shutil.copy2(bench/item['variants'][item['primary']]['path'], job/'original.png')
        dump(job/'cv.json', cv)
        render_start = time.monotonic()
        layout = annotate(job/'original.png', cv, job/'annotated.png')
        dump(job/'annotation.json', layout)
        meta = {'system_id': sid, 'video': item['video'], 'channel': item['channel'], 'system': item['system'],
                'record': item['record'], 'fps': item['fps'], 'video_path': item['video_path'],
                'source_legend': item['legend'], 'bench': str(bench), 'manifest_sha256': sha(bench/'manifest.json')}
        if item.get('automatic_anchor'):
            meta['automatic_anchor'] = item['automatic_anchor']
        meta.update(job_dir=str(job), images=[{'file': f, 'sha256': sha(job/f), 'width': cv['width'], 'height': cv['height'] if f == 'original.png' else layout['size'][1]} for f in ('original.png', 'annotated.png')],
                    cv_sha256=sha(job/'cv.json'), annotation_sha256=sha(job/'annotation.json'), hybrid_version=VERSION,
                    annotation_seconds=time.monotonic()-render_start)
        dump(job/'job.json', meta)
        source_legend = load(ROOT/item['legend'])
        overrides = source_legend.get('variants', {}).get(cv['variant'], {})
        rules = {**source_legend['rules'], **overrides.get('rules', {})}
        dump(job/'legend.json', {'channel': item['channel'], 'variant': cv['variant'], 'position_convention': 'bottom line=0, upward half-space=+1',
                               'rules': {k: {f: v[f] for f in ('inst', 'verified', 'articulation') if f in v} for k, v in rules.items()}})
        (job/'INSTRUCTIONS.md').write_text(instructions(meta, cv))
        print(json.dumps({'id': sid, 'heads': len(cv['notes']), 'barline_candidates': len(cv['barlines']), 'cv_seconds': round(cv['wall_seconds'], 3)}), flush=True)
    dump(HYBRID/'prepared.json', {'version': VERSION, 'systems': systems, 'manifest_sha256': sha(bench/'manifest.json'), 'generator_sha256': sha(__file__)})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--systems', nargs='+')
    ap.add_argument('--selection', choices=['gold', 'dense', 'selected'], default='gold')
    ap.add_argument('--bench', type=Path, default=BENCH, help='native-frame manifest directory; outputs remain confined to vision_pilot')
    a = ap.parse_args()
    prepare(a.systems or load(DATA/'selection.json')[a.selection], a.bench.resolve())


if __name__ == '__main__':
    main()
