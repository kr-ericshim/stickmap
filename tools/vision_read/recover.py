"""Recover equivalent notation representations from saved responses, no model calls.

The first validator omitted meter strings and the x/cross head synonym.
Canonicalizing these is a representation repair, not a changed musical
prediction. Actual wall time and all original retry errors remain recorded.
"""
import argparse
import sys
sys.dont_write_bytecode = True
from common import DATA, confined, dump, load, sha, validate
from run import parse_response


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', required=True)
    a = ap.parse_args()
    run = confined(DATA/'runs'/a.run)
    config = load(run/'config.json')
    recovered = []
    for sid in config['systems']:
        job = run/sid
        if not (job/'result.json').exists():
            ap.error('run still incomplete; recover only after all systems finish')
        result = load(job/'result.json')
        if result['status'] != 'invalid_json':
            continue
        for attempt in result['attempts']:
            if attempt['returncode'] != 0 or attempt.get('error') not in ('invalid time_signature', 'unknown notehead type'):
                continue
            raw = job/f'stdout-{attempt["attempt"]}.log'
            try:
                parsed = parse_response(raw.read_text())
                operations = []
                if isinstance(parsed.get('time_signature'), str):
                    operations.append('time_signature: meter string to numerator/denominator object')
                if any(h.get('notehead') == 'x' for b in parsed.get('bars', []) for e in b.get('events', []) for h in e.get('heads', [])):
                    operations.append('notehead: x to cross synonym')
                if not operations:
                    continue
                reading = validate(parsed, sid)
            except (ValueError, TypeError, KeyError, ZeroDivisionError):
                continue
            if (job/'reading.json').exists():
                ap.error(f'refuse to replace existing reading: {sid}')
            dump(job/'reading.json', reading)
            result['original_status'] = result['status']
            result['status'] = 'ok'
            result['reading_sha256'] = sha(job/'reading.json')
            result['format_recovery'] = {'operations': operations,
                                         'source_attempt': attempt['attempt'], 'source_sha256': sha(raw), 'recovery_sha256': sha(__file__), 'new_model_calls': 0}
            dump(job/'result.json', result)
            recovered.append(sid)
            break
    batch = load(run/'run.json')
    batch['results'] = [load(run/sid/'result.json') for sid in config['systems']]
    batch['format_recovered_systems'] = [r['id'] for r in batch['results'] if r.get('format_recovery')]
    dump(run/'run.json', batch)
    print({'format_recovered': recovered})


if __name__ == '__main__':
    main()
