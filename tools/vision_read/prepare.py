"""Prepare isolated image/legend jobs without exposing benchmark answers."""
import argparse
import json
import shutil
import struct
import sys
sys.dont_write_bytecode = True
from common import BENCH, DATA, ROOT, VERSION, confined, dump, load, sha

DENSE = ['jAMDUcKmjD4', 'qLi8dFevOF0', 'n3jYSzj9VX0', 'tK9MGMbPS_w', 'Ly7Pbg4EDug']


def instructions(meta, legend, mode):
    images = '\n'.join(f'- {meta["job_dir"]}/{v["file"]} ({v["width"]} x {v["height"]} native pixels)' for v in meta['images'])
    policy = ('LOOK MODE: Open the FIRST image ONCE using the file/image viewer. If an obstruction prevents reading, you may view each alternate once. Do NOT run shell commands, scripts, crops, searches, agents, or other tools. Do not read INSTRUCTIONS/legend with another tool: all needed text is embedded below. Return JSON as your final response; the harness writes reading.json for you.'
              if mode == 'look' else
              'ZOOM MODE: View the first image. You may use tools ONLY to crop/resize images and inspect the crops/alternates. Budget: at most FOUR new crop images and SIX tool calls after the first image view. Do not run head detectors, exhaustive pixel/column sampling, or iterative analysis scripts. Make a few targeted enlargements, then return the full JSON with unresolved observations in uncertain; stop tools when the budget is reached. Store every temporary file ONLY in this job folder. Use the project .venv if needed. Do not inspect any other project data or code. Return JSON as your final response; the harness writes reading.json for you.')
    return f'''Read the visible percussion notation for system {meta['system_id']}, independently of audio or any answer key.
{policy}
Workspace boundary: {ROOT}. Only the listed job inputs are authorized. Do not read benchmark gold, annotations, other readings, repository instructions, audio/video, or training data. Do not modify project code.
Images (best native frame FIRST; alternatives show the SAME viewport, never concatenate them):
{images}
Channel legend (bottom staff line pos=0; each half-space upward +1):
{json.dumps(legend, ensure_ascii=False)}
Use exactly the legend instrument names. Missing mappings => instrument null with uncertainty. Unverified mappings are notation candidates, NOT confirmed physical drum identities. Preserve pos and notehead independently. Do not invent tom size or cymbal identity.

Count visible bars left to right, zero based, including clipped bars at either edge. True barlines cross the staff; stems are not barlines. The leftmost visible bar is index 0 even if its printed number is 42. Record printed number optionally. Record time signature only if SHOWN; otherwise null (an inferred meter may be recorded separately).
Read beams/flags/dots/voices/rests, not horizontal spacing as rhythm. Within each bar list events in time order. ALL onset and duration fractions are WHOLE-NOTE UNITS: 3/16 means three sixteenths from that bar's start; quarter note=1/4, sextuplet sixteenths in 6:4=1/24. Use rational strings, never decimals. If the bar starts off-screen and its exact origin cannot be known, onset=null, partial_left=true, and explain uncertainty; do not reset a mid-bar fragment to time zero.
Group simultaneous heads only when they share onset, duration and voice; otherwise separate events. Note every head, including grace and tie continuations. Grace duration=0, not a separately timed acoustic attack. Tie continuations are tie_continuation=true and MUST NOT be counted as new onsets/attacks. A slur is not a tie. Mark ghost ONLY for head parentheses; mark accents separately. Tuplet membership needs a ratio like 6:4 and a stable group ID; do not confuse a weighted 5:3 group with five heads. Rests are implied by gaps, not head events.
Estimate each head's native-image x,y centre (undo any zoom/crop offset), pos, and the five staff line y coordinates top to bottom. Coordinates are for locating heads, NOT for inferring onsets. If genuinely uncertain use null and list the reason instead of guessing.
Verify EACH beat and EACH voice sums to its beat length, including implied rests, dotted values and tuplet scaling. Do not add simultaneous heads together. For partial bars or unknown meter mark checks uncertain. Explain any mismatch in uncertain. beat_checks contains bar, voice, beat_start, beat_length, sounded_duration, implied_rest_duration, status (ok/uncertain/mismatch). All lengths use rational whole-note strings. Do not force durations to fill a bar.

Output ONLY one strict JSON object, no Markdown or narrative. Output contract for {meta['job_dir']}/reading.json (the harness saves your final JSON there):
{{"schema_version":1,"system_id":"{meta['system_id']}","time_signature":null,
 "staff":{{"line_count":5,"lines_y":[10,20,30,40,50],"clef_visible":false}},
 "bar_count":1,"bars":[{{"index":0,"partial_left":false,"partial_right":false,"x_start":0,"x_end":1900,
 "events":[{{"onset":"3/16","duration":"1/16","voice":"1","heads":[
 {{"id":"h000","instrument":"snare","pos":5,"notehead":"filled","x":200,"y":25,
 "ghost":false,"accent":false,"grace":false,"tuplet":null,"tie_continuation":false}}]}}]}}],
 "beat_checks":[],"uncertain":[]}}
This is a FORMAT EXAMPLE only, not image content. Tuplet example: {{"ratio":"6:4","group":"b0-v1-t0"}}. Unknown fields may be null. uncertain items are objects with location/field/reason. Preserve all events you can see rather than omit whole difficult bars.
'''


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--systems', nargs='+')
    ap.add_argument('--selection', choices=['gold', 'dense', 'pilot', 'all'], default='pilot')
    ap.add_argument('--mode', choices=['look', 'zoom'], default='look')
    ap.add_argument('--bench', type=__import__('pathlib').Path, default=BENCH)
    a = ap.parse_args()
    bench = a.bench.resolve()
    manifest = load(bench/'manifest.json')
    gold = load(bench/'gold.json')['systems']
    items = manifest['systems']
    gold_ids = [it['id'] for it in items if gold[it['id']].get('notes')]
    dense_ids = []
    for video in DENSE:
        candidates = [it for it in items if it['video'] == video and not gold[it['id']].get('notes')]
        for index in (len(candidates)//3, 2*len(candidates)//3):
            if candidates and candidates[index]['id'] not in dense_ids:
                dense_ids.append(candidates[index]['id'])
    choices = {'gold': gold_ids, 'dense': dense_ids, 'pilot': gold_ids+dense_ids, 'all': [it['id'] for it in items]}
    selected = a.systems or choices[a.selection]
    unknown = set(selected)-{it['id'] for it in items}
    if unknown:
        ap.error(f'unknown systems: {sorted(unknown)}')
    for item in items:
        if item['id'] not in selected:
            continue
        job = confined(DATA/'jobs'/item['id'])
        job.mkdir(parents=True, exist_ok=True)
        original = load(ROOT/item['legend'])
        kind = 'white_overlay' if item.get('foreground') == 'white_overlay' else 'static_light' if item['channel'] == 'dailychops' and not item.get('reconstructed_scroll') else None
        override = original.get('variants', {}).get(kind, {})
        rules = {**original['rules'], **override.get('rules', {})}
        legend = {'channel': item['channel'], 'variant': kind, 'position_convention': 'bottom line=0, upward half-space=+1',
                  'rules': {k: {f: v[f] for f in ('inst', 'verified', 'articulation') if f in v} for k, v in rules.items()}}
        dump(job/'legend.json', legend)
        order = [item['primary']] + [i for i in range(len(item['variants'])) if i != item['primary']]
        images = []
        for rank, index in enumerate(order):
            variant = item['variants'][index]
            source = bench/variant['path']
            if sha(source) != variant['sha256']:
                raise ValueError(f'image integrity mismatch: {source}')
            target = job/f'{rank:02d}_{source.name}'
            shutil.copy2(source, target)
            width, height = struct.unpack('>II', target.read_bytes()[16:24])
            images.append({**variant, 'file': target.name, 'width': width, 'height': height, 'variant_index': index})
        meta = {'version': VERSION, 'system_id': item['id'], 'video': item['video'], 'channel': item['channel'], 'system': item['system'],
                'job_dir': str(job), 'images': images, 'source_legend': item['legend'], 'legend_sha256': sha(ROOT/item['legend']),
                'manifest_sha256': sha(bench/'manifest.json'), 'bench': str(bench), 'record': item['record'], 'fps': item['fps'], 'video_path': item['video_path']}
        dump(job/'job.json', meta)
        (job/'INSTRUCTIONS.md').write_text(instructions(meta, legend, a.mode))
    dump(DATA/'selection.json', {'version': VERSION, 'gold': gold_ids, 'dense': dense_ids, 'selected': selected, 'manifest_sha256': sha(bench/'manifest.json')})
    print(json.dumps({'prepared': len(selected), 'note_gold': len(gold_ids), 'dense_no_gold': len(dense_ids), 'mode': a.mode}))


if __name__ == '__main__':
    main()
