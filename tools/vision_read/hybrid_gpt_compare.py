"""Compare Codex runs with Gemini on the identical frozen 21-system gold set."""
import argparse
import json
import sys
sys.dont_write_bytecode = True
from common import BENCH, DATA, confined, dump, load, sha
from hybrid import HERE
from hybrid_compare import count, evaluate


def qualifies(condition):
    rows = [row for row in condition['systems'] if row['complete']]
    metrics = condition['complete_gold']['metrics']
    return (len(rows) == 6 and all(row['status'] == 'ok' for row in rows)
            and all((metrics[key]['value'] or 0) >= .95
                    for key in ('onset_instrument', 'symbolic_onset_instrument')))


def compare(runs):
    systems = load(DATA/'selection.json')['gold']
    gold = load(BENCH/'gold.json')['systems']
    items = {item['id']: item for item in load(BENCH/'manifest.json')['systems']}
    if len(systems) != 21 or sum(gold[sid].get('complete', False) for sid in systems) != 6:
        raise ValueError('expected the frozen 6 complete + 15 partial gold systems')
    conditions = {}
    configs = {}
    for name in ['hybrid-medium-look', *runs]:
        config = load(DATA/'runs'/name/'config.json')
        if set(config['systems']) != set(systems):
            raise ValueError('comparison run must contain exactly the same gold: '+name)
        if name in runs and (config.get('effort') not in ('high', 'xhigh', 'max') or config['model'] not in ('gpt-6-luna', 'gpt-6.1-sol')):
            raise ValueError('only the requested GPT configurations are eligible')
        conditions[name] = evaluate('hybrid', systems, items, gold, name)
        configs[name] = config
    chosen = next((name for name in runs if qualifies(conditions[name])), None)
    result = dict(version='hybrid-gpt-comparison-v1', systems=systems, configs=configs,
                  conditions=conditions, chosen_run=chosen, threshold=.95,
                  rule='all six complete gold results ok; both XY and symbolic joint >=95%; first listed qualifying run is chosen (user picked Sol high after Luna xhigh/max failed)',
                  gold_sha256=sha(BENCH/'gold.json'), manifest_sha256=sha(BENCH/'manifest.json'),
                  scorer_sha256=sha(HERE/'score.py'), metric_sha256=sha(HERE/'hybrid_compare.py'),
                  scope='provisional same-reviewer gold; 6 complete + 15 partial viewports; no audio/vault; partial labels are unknown outside annotations')
    dump(DATA/'hybrid_gpt_summary.json', result)
    lines = ['# GPT hybrid evaluation', '', result['scope'], '',
             '| Run / model / effort | Complete joint XY | Complete joint symbolic | Coordinate-free onset | Tuplets | Ties + | Accents + | Mean min/system |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name, condition in conditions.items():
        full = condition['complete_gold']['metrics']; all_metrics = condition['all']['metrics']
        sec = condition['completed_only']['time_seconds']['mean']
        config = configs[name]
        label = name+' / '+config['model']+' / '+config.get('effort', 'medium')
        positives = lambda key: f"{all_metrics[key].get('positive_correct', 0)}/{all_metrics[key].get('positive_total', 0)}"
        lines.append('| '+' | '.join([label, count(full['onset_instrument']), count(full['symbolic_onset_instrument']),
                     count(full['full_symbolic_onset']), count(all_metrics['tuplet_ratio']), positives('tie'), positives('accent'),
                     f'{sec/60:.3f}' if sec is not None else 'N/A'])+' |')
    lines += ['', 'Joint uses the 112 heads with known instrument and onset in complete gold; missing heads count as errors. Coordinate-free onset uses all 159 complete-gold heads. Tuplets/ties/accents use positive annotations across all 21 systems. Timing includes frozen CV/annotation preparation and every attempt/backoff; elapsed batch time benefits from six workers. Failed or pending jobs are explicitly listed below and cannot authorize a full run.', '',
              '| Run | Completed | Status counts | XY onset, all | XY instrument, all | Whole systems symbolic | Batch minutes |',
              '|---|---:|---|---:|---:|---:|---:|']
    for name, condition in conditions.items():
        batch = load(DATA/'runs'/name/'run.json')
        mm = condition['all']['metrics']; full = condition['complete_gold']['metrics']
        lines.append(f"| {name} | {condition['completed_only']['systems']}/21 | {condition['all']['statuses']} | {count(mm['onset'])} | {count(mm['instrument'])} | {count(full['symbolic_known_fields_system_exact'])} | {batch['batch_wall_seconds']/60:.3f} |")
    lines += ['', '## Complete systems', '', '| Run | System | Joint XY | Joint symbolic | Coordinate-free onset | Status |', '|---|---|---:|---:|---:|---|']
    for name, condition in conditions.items():
        for row in condition['systems']:
            if row['complete']:
                mm = row['metrics']
                lines.append(f"| {name} | {row['id']} | {count(mm['onset_instrument'])} | {count(mm['symbolic_onset_instrument'])} | {count(mm['full_symbolic_onset'])} | {row['status']} |")
    lines += ['', '## Decision', '', 'Chosen GPT run: `'+str(chosen)+'`.', '',
              'Luna xhigh and Luna max both failed the complete-gold criterion (Luna marked fully visible bars as left-truncated and left onsets null). The user then chose Sol high; the first listed qualifying run is used.', '',
              'Only image delivery differs: Codex receives original.png and annotated.png as attachments, with cwd set to each job folder. All musical instructions and the rhythm.json contract are the same. CV instrument/position ownership, unverified model-only misses, raw fractions, false positives, merge/validation and scoring remain unchanged. As in the Gemini hybrid, malformed visual responses are retained rather than reopened automatically; rate-limit calls back off across workers and retry. CLI runs use read-only sandbox, isolated config, and no project instructions; look/no-tools is a prompt policy, not exhaustive tool certification.', '',
              'A gold pass authorizes scratch dense candidates only. Existing WSL timing/phase/band and strict/scoring gates still decide yield; no canonical writes, vault inputs or training are permitted.', '',
              'Artifacts: `runs/<run>/<system>/{rhythm.json,reading.json,result.json,final-*.txt,*log}` and `hybrid_gpt_summary.json`.']
    confined(DATA/'HYBRID_GPT_REPORT.md').write_text('\n'.join(lines)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', required=True)
    args = parser.parse_args()
    result = compare(args.runs)
    print(json.dumps(dict(chosen_run=result['chosen_run'], report=str(DATA/'HYBRID_GPT_REPORT.md'))))
