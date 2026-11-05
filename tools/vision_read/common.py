"""Shared, audio-independent contracts for the vision pilot."""
import hashlib
import json
import math
import os
import threading
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get('STICKMAP_VISION_DATA', ROOT / '.eval-cache/vision_pilot')).resolve()
if DATA not in ((ROOT / '.eval-cache/vision_pilot').resolve(), (ROOT / 'scratch/hybrid_solos').resolve()):
    raise ValueError('vision data must be vision_pilot or scratch/hybrid_solos')
BENCH = DATA / 'bench'
VERSION = 'vision-pilot-v1'


def confined(path, base=DATA):
    path = Path(path).resolve()
    if not path.is_relative_to(base.resolve()):
        raise ValueError(f'output must be inside {base}: {path}')
    return path


def dump(path, value):
    path = confined(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+f'.{os.getpid()}.{threading.get_ident()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def rational(value, nullable=True):
    if value is None and nullable:
        return None
    if not isinstance(value, str):
        raise ValueError('rational values must be strings or null')
    answer = Fraction(value)
    if answer < 0:
        raise ValueError('negative rational')
    return answer


def validate(reading, system_id):
    if not isinstance(reading, dict) or reading.get('system_id') != system_id:
        raise ValueError('wrong or missing system_id')
    required = {'schema_version', 'system_id', 'time_signature', 'staff', 'bar_count', 'bars', 'uncertain', 'beat_checks'}
    if not required <= reading.keys() or reading['schema_version'] != 1:
        raise ValueError('missing top-level fields or unsupported schema_version')
    bars = reading['bars']
    if not isinstance(bars, list) or not isinstance(reading['uncertain'], list) or not isinstance(reading['beat_checks'], list):
        raise ValueError('bars, uncertain, beat_checks must be lists')
    count = reading['bar_count']
    if type(count) is not int or count < 0 or count != len(bars):
        raise ValueError('bar_count must equal len(bars)')
    staff = reading['staff']
    if not isinstance(staff, dict) or not {'line_count', 'lines_y', 'clef_visible'} <= staff.keys():
        raise ValueError('missing staff fields')
    if staff['line_count'] is not None and type(staff['line_count']) is not int:
        raise ValueError('staff line_count must be integer/null')
    if staff['clef_visible'] is not None and type(staff['clef_visible']) is not bool:
        raise ValueError('clef_visible must be boolean/null')
    lines = staff['lines_y']
    if lines is not None and (not isinstance(lines, list) or len(lines) != 5 or lines != sorted(lines)):
        raise ValueError('lines_y must be five ascending native y coordinates/null')
    if lines is not None and any(type(y) not in (int, float) or not math.isfinite(y) for y in lines):
        raise ValueError('staff coordinates must be finite numbers')
    signature = reading['time_signature']
    # A meter written "4/4" carries the same information as numerator=4,
    # denominator=4. Canonicalize only this representation, never note content.
    if isinstance(signature, str):
        parts = signature.split('/')
        if len(parts) != 2 or not all(p.isdigit() and int(p) > 0 for p in parts):
            raise ValueError('invalid time_signature string')
        signature = {'numerator': int(parts[0]), 'denominator': int(parts[1])}
        reading['time_signature'] = signature
    if signature is not None and (not isinstance(signature, dict) or type(signature.get('numerator')) is not int or type(signature.get('denominator')) is not int or signature['numerator'] <= 0 or signature['denominator'] <= 0):
        raise ValueError('invalid time_signature')
    ids = set()
    for index, bar in enumerate(bars):
        if not isinstance(bar, dict) or bar.get('index') != index or not {'partial_left', 'partial_right', 'events'} <= bar.keys():
            raise ValueError('bars must have sequential zero-based index and partial flags/events')
        if type(bar['partial_left']) is not bool or type(bar['partial_right']) is not bool:
            raise ValueError('partial flags must be boolean')
        previous = Fraction(-1)
        for event in bar['events']:
            if not {'onset', 'duration', 'voice', 'heads'} <= event.keys() or not isinstance(event['heads'], list) or not event['heads']:
                raise ValueError('event needs onset, duration, voice, nonempty heads')
            onset = rational(event['onset'])
            duration = rational(event['duration'])
            if onset is not None:
                if onset < previous:
                    raise ValueError('events must be in onset order')
                previous = onset
            if duration is not None and duration == 0 and not all(h.get('grace') is True for h in event['heads']):
                raise ValueError('zero duration requires grace heads')
            for head in event['heads']:
                fields = {'id', 'instrument', 'pos', 'notehead', 'x', 'y', 'ghost', 'accent', 'grace', 'tuplet', 'tie_continuation'}
                if not fields <= head.keys():
                    raise ValueError(f'missing head fields: {sorted(fields - head.keys())}')
                if not isinstance(head['id'], str) or head['id'] in ids:
                    raise ValueError('head IDs must be unique strings')
                ids.add(head['id'])
                if head['notehead'] == 'x':
                    head['notehead'] = 'cross'  # unambiguous notation synonym
                if head['notehead'] not in ('filled', 'cross', 'circled', 'diamond', 'open', None):
                    raise ValueError('unknown notehead type')
                if head['instrument'] is not None and not isinstance(head['instrument'], str):
                    raise ValueError('instrument must be string/null')
                if head['pos'] is not None and type(head['pos']) is not int:
                    raise ValueError('pos must be integer/null')
                for field in ('ghost', 'accent', 'grace', 'tie_continuation'):
                    if head[field] is not None and type(head[field]) is not bool:
                        raise ValueError(f'{field} must be boolean/null')
                for field in ('x', 'y'):
                    value = head[field]
                    if value is not None and (type(value) not in (int, float) or not math.isfinite(value)):
                        raise ValueError('head coordinates must be finite numbers/null')
                tuplet = head['tuplet']
                if tuplet is not None and (not isinstance(tuplet, dict) or not {'ratio', 'group'} <= tuplet.keys() or not isinstance(tuplet['group'], str)):
                    raise ValueError('tuplet must contain ratio and group')
                if tuplet is not None:
                    a, b = tuplet['ratio'].split(':')
                    if int(a) <= 0 or int(b) <= 0:
                        raise ValueError('invalid tuplet ratio')
    # Beat checks are evidence reported by the model; independent coverage is
    # computed by the scorer, never accepted just because the model says "ok".
    return reading


def flatten(reading):
    return [{**head, 'bar': bar['index'], 'onset': event['onset'], 'duration': event['duration'], 'voice': event['voice']}
            for bar in reading['bars'] for event in bar['events'] for head in event['heads']]
