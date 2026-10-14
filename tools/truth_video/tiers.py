"""Derive an explicitly separate 50-ms scoring selection; never rewrite strict truth."""
import argparse
import json
from pathlib import Path
from align import moments,unique_events


THRESHOLDS = dict(fitted_40ms=.90, held_out_40ms=.85,
                  no_cursor_held_out_40ms=.90, gain_either=.15)
TIMING_REASONS = ('fitted attack agreement', 'held-out agreement', 'no-cursor alignment')


def scoring_reasons(r):
    reasons = [v for v in r['reasons'] if not v.startswith(TIMING_REASONS)]
    if not r.get('times'):
        return reasons or ['no fitted times']
    if r.get('moments',0)<6:
        reasons.append('scoring requires six moments')
    if not r.get('strictly_monotone',False):
        reasons.append('scoring requires strictly monotone times')
    if max(r.get('held_out_gain_20ms',0),r.get('held_out_gain_40ms',0))<.15:
        reasons.append('scoring gain over shifted controls below 15 points')
    if r['after']['within_40ms'] + 1e-9 < THRESHOLDS['fitted_40ms']:
        reasons.append('scoring fitted 40 ms below 90%')
    minimum = .90 if r['mode'] in ('interval','rhythm') else .85
    if r['held_out']['within_40ms'] + 1e-9 < minimum:
        reasons.append(f'scoring held-out 40 ms below {minimum:.0%}')
    return reasons


def write(folder):
    folder = Path(folder)
    r = json.loads((folder/'alignment.json').read_text())
    systems = {s['system']: s for s in json.loads((folder/'strip.json').read_text())}
    events, intervals = [], []
    for s in r['systems']:
        s['strict_usable'] = s['usable']
        s['scoring_reasons'] = [] if s['usable'] else scoring_reasons(s)
        s['scoring_usable'] = not s['scoring_reasons']
        s['tier'] = 'strict' if s['usable'] else 'scoring' if s['scoring_usable'] else 'rejected'
        if not s['scoring_usable']:
            continue
        _, groups = moments(systems[s['system']])
        intervals.append([s['times'][0], s['times'][-1]])
        for time, group in zip(s['times'], groups):
            for i in group:
                if (systems[s['system']]['notes'][i].get('exportable') is False or
                        systems[s['system']]['notes'][i].get('tied_continuation')):
                    continue
                e = {**systems[s['system']]['notes'][i], 'time': time, 'system': s['system'],
                     'timing_confidence': round(s['held_out']['within_40ms'], 3), 'tier': s['tier']}
                events.append(e)
    events=unique_events(events)
    extra=[event for event in events if event['tier']=='scoring']
    merged = []
    for a,b in sorted(intervals):
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(b, merged[-1][1])
        else:
            merged.append([a,b])
    r['strict_intervals'] = r['intervals']
    r['intervals'] = merged
    r['strict_seconds'] = r['trustworthy_seconds']
    r['trustworthy_seconds'] = sum(b-a for a,b in merged)
    r['scoring_thresholds'] = THRESHOLDS
    for name, value in [('truth.scoring.json', sorted(events,key=lambda e:e['time'])),
                        ('truth.additional.json', sorted(extra,key=lambda e:e['time'])),
                        ('alignment.scoring.json', r)]:
        (folder/name).write_text(json.dumps(value,indent=2))
    return r


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    a=p.parse_args()
    for f in sorted(a.root.glob('*/alignment.json')):
        r=write(f.parent)
        print(f'{f.parent.name}: strict {sum(s["usable"] for s in r["systems"])}, additional {sum(s["tier"]=="scoring" for s in r["systems"])}, {r["trustworthy_seconds"]:.2f}s total')
