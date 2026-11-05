"""First-pass cache-only score diagnosis. No LLM, model inference, or audio reads.

Run: PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B tools/score_eval/score_eval.py --workers 4
All writes are confined to .eval-cache/score_eval; vault is filtered before song I/O.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
from scipy.optimize import linear_sum_assignment

OUT = ROOT / '.eval-cache/score_eval'
FAMILIES = ('kick', 'snare', 'tom', 'hihat', 'cymbal')


def family(inst):
    if inst.startswith('tom'):
        return 'tom'
    if inst in ('ride', 'ride_bell', 'bell', 'crash', 'cymbal_unknown'):
        return 'cymbal'
    if inst in ('hihat_closed', 'hihat_open', 'pedal', 'hihat_pedal'):
        return 'hihat'
    if inst == 'side_stick':
        return 'snare'
    return inst


def time_pairs(ref, est, tolerance=.05):
    """Maximum-cardinality, then minimum total time distance, one-to-one.

    Partition disconnected time clusters to avoid whole-song dense matrices.
    Dummy columns allow every truth item to remain unmatched.
    """
    combined = sorted([(float(t), 0, i) for i, t in enumerate(ref)] +
                      [(float(t), 1, i) for i, t in enumerate(est)])
    clusters = []
    for item in combined:
        if not clusters or item[0] - clusters[-1][-1][0] > tolerance + 1e-10:
            clusters.append([])
        clusters[-1].append(item)
    pairs = []
    for cluster in clusters:
        ri = [i for _, side, i in cluster if side == 0]
        ei = [i for _, side, i in cluster if side == 1]
        if not ri or not ei:
            continue
        delta = np.abs(np.asarray(ref)[ri, None] - np.asarray(est)[None, ei])
        # A single lost pair costs more than every possible timing difference.
        penalty = len(ri) + len(ei) + 1
        costs = np.full((len(ri), len(ei) + len(ri)), float(penalty))
        costs[:, :len(ei)] = np.where(delta <= tolerance + 1e-10,
                                              delta / tolerance, penalty * 3)
        ii, jj = linear_sum_assignment(costs)
        pairs.extend((ri[i], ei[j]) for i, j in zip(ii, jj)
                     if j < len(ei) and delta[i, j] <= tolerance + 1e-10)
    return sorted(pairs)


def match_heads(ref, est, tolerance=.05):
    """Reserve same-family matches first, then residual cross-family matches."""
    same = []
    for fam in FAMILIES:
        ri = [i for i, e in enumerate(ref) if family(e['inst']) == fam]
        ei = [i for i, e in enumerate(est) if family(e['inst']) == fam]
        same.extend((ri[i], ei[j]) for i, j in time_pairs(
            [ref[i]['time'] for i in ri], [est[j]['time'] for j in ei], tolerance))
    used_r, used_e = {i for i, _ in same}, {j for _, j in same}
    ri = [i for i in range(len(ref)) if i not in used_r]
    ei = [i for i in range(len(est)) if i not in used_e]
    wrong = [(ri[i], ei[j]) for i, j in time_pairs(
        [ref[i]['time'] for i in ri], [est[j]['time'] for j in ei], tolerance)]
    used_r.update(i for i, _ in wrong)
    used_e.update(j for _, j in wrong)
    return same, wrong, [i for i in range(len(ref)) if i not in used_r], [
        j for j in range(len(est)) if j not in used_e]


def display_threshold(quarter_seconds):
    return max(.040, quarter_seconds / 12)


def load(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=1) + '\n')


def eligible_rows():
    from tools.harvest.vault import Vault
    seal = Vault()
    exclusions = {r['video'] for r in load(ROOT / 'truth/harvest/vault_final_exclusions.json')['exclusions']}
    rows, rejected = [], 0
    for vid, row in sorted(seal.rows.items()):
        ids = seal.song_ids(row) | {vid}
        if any(seal.is_vault(v) for v in ids) or ids & exclusions:
            rejected += 1
            continue
        if row.get('slug'):
            rows.append({**row, 'video': vid})
    return rows, rejected


def rebuild(song, options):
    """Current quantizer + cached family detector; intentionally no audio stages.

    tom_sizes can remove detections and split tom voices; excluded here because
    it needs decoded audio. Cymbal subtype/veto and sound-map work also omitted.
    This is a reproducible family-level replay, not an exact app pipeline replay.
    """
    from groovemap import models
    from groovemap.fills import tom_runs, fill_snares
    from groovemap.dynamics import annotate_confidence
    from groovemap.quantize import quantize
    act = np.load(song / 'activations.drums-htdemucs.npy', allow_pickle=False)
    mix = np.load(song / 'activations.mix.npy', allow_pickle=False)
    snare = np.load(song / 'snare.htdemucs.npy', allow_pickle=False)
    attacks = np.load(song / 'attacks.htdemucs.npy', allow_pickle=False)
    thresholds = options.get('thresholds') or [v / options.get('sensitivity', 1) for v in models.DEFAULT_THRESHOLDS]
    hits = models.pick_hits(act, thresholds, mix=mix, snare=snare)
    hits = fill_snares(tom_runs(hits, act, mix, attacks, thresholds), act, snare, attacks)
    hits = annotate_confidence(hits, act, snare)
    grid = load(song / 'beats.json')
    common = {k: options[k] for k in ('grid', 'triplets', 'beat_shift', 'tempo_scale') if k in options}
    if options.get('time_signature'):
        from groovemap.signature_quantize import quantize_signature
        score = quantize_signature(hits, grid['beats'], grid['downbeats'], activations=act,
            attacks=attacks, time_signature=options['time_signature'],
            meter_changes=options.get('meter_changes', []), **common)
    else:
        score = quantize(hits, grid['beats'], grid['downbeats'], attacks=attacks,
            meter=options.get('meter'), odd_bars=options.get('odd_bars', 'none'), **common)
    return hits, score


def proxy_sample(ref, score, systems):
    """Descriptive ratios only. Dominant IOI is NOT independent beat truth."""
    samples = []
    for sid in sorted(systems):
        times = sorted({round(e['time'], 5) for e in ref if e['system'] == sid})
        if len(times) < 8:
            continue
        gaps = np.diff(times)
        gaps = gaps[(gaps >= .04) & (gaps <= 1.5)]
        bars = [b for b in score.bars if b.start <= times[-1] and b.beat_times[-1] >= times[0]]
        if not len(gaps) or not bars:
            continue
        bins = Counter(int(round(g / .01)) for g in gaps)
        center, count = bins.most_common(1)[0]
        near = gaps[np.abs(gaps - center * .01) <= .015]
        ioi = float(np.median(near))
        period = float(np.median([float(g) * b.denominator / 4 for b in bars for g in np.diff(b.beat_times)]))
        downbeats = [b.start for b in bars if times[0] <= b.start <= times[-1]]
        support = sum(min(abs(t - d) for t in times) <= .05 for d in downbeats)
        samples.append(dict(system=sid, time=times[0], truth_heads=sum(e['system'] == sid for e in ref),
            quarter_ms=round(period*1000, 1), dominant_ioi_ms=round(ioi*1000, 1),
            quarter_to_ioi=round(period/ioi, 2), modal_ioi_share=round(count/len(gaps), 3),
            downbeats=len(downbeats), supported_downbeats=support))
    return samples


def one(row):
    start = time.monotonic()
    song = ROOT / 'truth/harvest/songs' / row['slug']
    base = dict(slug=row['slug'], video=row['video'], channel=row.get('channel', 'unknown'))
    truth_file, alignment_file = song/'truth/truth.json', song/'truth/alignment.json'
    if not truth_file.exists() or not alignment_file.exists():
        return {**base, 'status': 'no_strict_truth'}
    systems = {s['system']: s for s in load(alignment_file)['systems'] if s.get('usable') is True}
    ref = [e for e in load(truth_file) if e.get('system') in systems and
           e.get('exportable', True) and e.get('verified', False) and family(e['inst']) in FAMILIES]
    if not ref:
        return {**base, 'status': 'no_strict_truth'}
    ref.sort(key=lambda e: (e['time'], e['inst']))
    base.update(truth_heads=len(ref), systems=len({e['system'] for e in ref}))
    required = ['beats.json', 'activations.drums-htdemucs.npy', 'activations.mix.npy',
                'snare.htdemucs.npy', 'attacks.htdemucs.npy']
    missing = [p for p in required if not (song/p).exists()]
    if missing:
        return {**base, 'status': 'missing_cache', 'missing': missing}
    options = load(song/'options.json') if (song/'options.json').exists() else {}
    if options.get('sound_names') or options.get('onset_input', 'drums') != 'drums':
        return {**base, 'status': 'unsupported_options'}
    hits, score = rebuild(song, options)
    dest = OUT/'songs'/row['slug']
    dest.mkdir(parents=True, exist_ok=True)
    score.save(dest/'score.json')
    # Score hit extras only between first/last strict head per system, ±50 ms.
    intervals = [(min(e['time'] for e in ref if e['system'] == sid)-.05,
                  max(e['time'] for e in ref if e['system'] == sid)+.05)
                 for sid in sorted({e['system'] for e in ref})]
    est = [dict(time=h.time, inst=h.inst) for h in hits if
           any(a <= h.time <= b for a, b in intervals)]
    notes = []
    for b in score.bars:
        for n in b.notes:
            beat = min(int(n.tick // b.unit_ticks), b.beats - 1)
            quarter = (b.beat_times[beat+1]-b.beat_times[beat])*b.denominator/4
            notes.append(dict(time=n.raw_time, inst=n.inst, displayed=b.tick_time(n.tick),
                              bar=b.index, tick=n.tick, threshold=display_threshold(quarter)))
    # raw_time is source provenance: do NOT rematch by displayed time.
    note_lookup = defaultdict(list)
    for n in notes:
        note_lookup[(round(n['time'], 7), family(n['inst']))].append(n)
    same, wrong, missed, extra = match_heads(ref, est)
    records = []
    matched = {i: j for i, j in same}
    misclassified = {i: j for i, j in wrong}
    totals = Counter(truth=len(ref), raw_est=len(est))
    instruments = defaultdict(Counter)
    for i, e in enumerate(ref):
        rec = dict(system=e['system'], time=e['time'], inst=family(e['inst']))
        if i in matched:
            h = est[matched[i]]
            ns = note_lookup.get((round(h['time'], 7), family(h['inst'])), [])
            rec['raw_time'] = h['time']
            totals['raw_correct'] += 1
            if ns:
                n = ns[0]
                error = abs(n['displayed']-e['time'])
                rec.update(displayed_time=n['displayed'], error_ms=error*1000,
                           threshold_ms=n['threshold']*1000, bar=n['bar'], tick=n['tick'])
                rec['category'] = '3' if error > n['threshold']+1e-10 else 'ok'
                rec['reason'] = 'displayed_time' if rec['category'] == '3' else 'within_threshold'
                totals['displayed_tested'] += 1
                totals['display_gt40'] += error > .04
                totals['display_gt50'] += error > .05
                totals['display_gt80'] += error > .08
            else:
                # Do not call a raw detection a miss because notation merged it.
                rec.update(category='notation_loss', reason='raw_hit_has_no_surviving_note')
        elif i in misclassified:
            h = est[misclassified[i]]
            rec.update(category='2', detected_inst=family(h['inst']), raw_time=h['time'])
        else:
            rec.update(category='1', reason='no_residual_hit_within_50ms')
        totals[rec['category']] += 1
        instruments[rec['inst']]['truth'] += 1
        instruments[rec['inst']][rec['category']] += 1
        records.append(rec)
    for j in extra:
        h = est[j]
        totals['5'] += 1
        instruments[family(h['inst'])]['5'] += 1
        records.append(dict(category='5', time=h['time'], inst=family(h['inst'])))
    assert sum(totals[k] for k in ('1','2','3','ok','notation_loss')) == len(ref)
    assert totals['raw_correct'] + totals['2'] + totals['5'] == len(est)
    result = {**base, 'status': 'ok', 'counts': dict(totals), 'instruments': dict(instruments),
              'options': options, 'seconds': round(time.monotonic()-start, 2),
              'merged_strokes': score.merged_strokes, 'records': records,
              'proxy': proxy_sample(ref, score, systems),
              'inputs': {p: {'bytes': (song/p).stat().st_size,
                         'mtime_ns': (song/p).stat().st_mtime_ns} for p in required}}
    write(dest/'result.json', result)
    return result


def tally(rows):
    c = Counter()
    for r in rows:
        c.update(r['counts'])
    return c


def cell(c, key):
    return f"{c[key]:,} ({100*c[key]/max(1,c['truth']):.1f}%)"


def table_row(name, c):
    return '| ' + ' | '.join([name, f"{c['truth']:,}", cell(c,'1'), cell(c,'2'), cell(c,'3'),
                              'N/A·표본 proxy', cell(c,'5'), cell(c,'notation_loss'),cell(c,'ok')]) + ' |'


def report(results, excluded, elapsed, fingerprints):
    ok = [r for r in results if r['status'] == 'ok']
    strict = [r for r in results if r.get('truth_heads')]
    skipped = [r for r in strict if r['status'] != 'ok']
    total = tally(ok)
    lines = ['# Score-level first-pass diagnosis', '',
        f"- 평가 {len(ok)}곡 / strict truth 보유 {len(strict)}곡, {len({r['channel'] for r in ok})} channels, {sum(r['systems'] for r in ok):,} systems, {total['truth']:,} heads. 캐시/옵션 부족 등 {len(skipped)}곡 제외 ({sum(r['truth_heads'] for r in skipped):,} heads; {dict(Counter(r['channel'] for r in skipped))}).",
        f'- Vault 및 final exclusions: manifest 단계에서 {excluded} videos 제외 후에만 song 파일 접근. 실행 {elapsed/60:.1f}분, 최대 4 processes. 원본 수정 없음.',
        '- 범위: cached ADTOF → pick_hits → tom_runs/fill_snares → 현재 quantizer의 family replay. 기존 score는 raw_time이 없어 재생성. 음원·LLM·모델 추론 없음. 재생성 score/result는 본 디렉터리 songs/에 보존.',
        '- 방법: ±50 ms, family 우선 최대 개수/최소 시간차 1:1 매칭 후 잔여 악기 혼동. ③은 동일-family 검출 성공 중 |bar.tick_time(tick) − truth| > max(40 ms, local quarter / 12). ④ 원인은 ③과 겹칠 수 있어 합산하지 않음. 검출은 됐으나 최종 note가 사라진 경우는 별도 손실 열.',
        '', '| 범위 | Truth heads | ① 검출 누락 | ② 악기 혼동 | ③ 표시 위치 오류 | ④ grid | ⑤ 추가 검출 | 기보 단계 손실 | 시간 기준 통과 |',
        '|---|---:|---:|---:|---:|---|---:|---:|---:|', table_row('전체', total)]
    for ch in sorted({r['channel'] for r in ok}):
        lines.append(table_row(ch, tally([r for r in ok if r['channel']==ch])))
    for fam in FAMILIES:
        c = Counter()
        for r in ok:
            c.update(r['instruments'].get(fam, {}))
        lines.append(table_row('악기: '+fam, c))
    lines += ['', '모든 %의 분모는 해당 행 truth heads. ⑤는 raw-hit extras이며 부분 악보의 미표기 소리도 포함할 수 있음. ①+②+③+손실+통과=truth; ⑤는 별도.',
        f"표시 검사 {total['displayed_tested']:,} heads: >40 ms {total['display_gt40']:,}, >50 ms {total['display_gt50']:,}, >80 ms {total['display_gt80']:,}. 같은-family 검출 {total['raw_correct']:,} 중 ③ {100*total['3']/max(1,total['raw_correct']):.1f}%.",
        '', '## 10곡 청취 우선순위', '',
        '순위 = (①+②+③+기보손실) / truth, truth ≥30. ⑤는 불완전한 기보 때문에 순위에서 제외. 링크는 오류 직전 약 2초.', '',
        '| 곡 / channel | heads | ① | ② | ③ | 손실 | 오류율 | 예시 |',
        '|---|---:|---:|---:|---:|---:|---:|---|']
    worst = sorted([r for r in ok if r['counts']['truth'] >= 30],
        key=lambda r: sum(r['counts'].get(k,0) for k in ('1','2','3','notation_loss'))/r['counts']['truth'], reverse=True)[:10]
    for r in worst:
        c=Counter(r['counts']); errors=[e for e in r['records'] if e['category'] in ('1','2','3','notation_loss')]
        samples=[]
        for cat in ('3','1','2','notation_loss'):
            e=next((e for e in errors if e['category']==cat),None)
            if e:
                label=f"{cat}@{e['time']:.2f}s"
                if cat=='3':label+=f" Δ{e['error_ms']:.0f}ms"
                samples.append(f"[{label}](https://www.youtube.com/watch?v={r['video']}&t={max(0,int(e['time'])-2)})")
            if len(samples)==2:break
        rate=sum(c[k] for k in ('1','2','3','notation_loss'))/c['truth']
        lines.append(f"| {r['slug']} / {r['channel']} | {c['truth']} | {c['1']} | {c['2']} | {c['3']} | {c['notation_loss']} | {rate:.1%} | {' · '.join(samples)} |")
    lines += ['', '## ④ quick sample — tempo/downbeat proxy', '',
        '기존 vision reads는 strict canonical systems와의 겹침이 적고 좌표 변환이 필요하므로 이번 범위에서는 결합하지 않았다. truth의 고유 onset 간격을 10 ms bin으로 집계한 dominant IOI와 현재 quarter period를 비교한다. ratio≈2/4/8은 정상 8/16/32분음표도 만들므로 half/double-tempo 판정 수로 쓰지 않는다. downbeat support는 system 내 downbeat ±50 ms에 truth onset이 있는 비율이며, downbeat 정답률이 아니다.', '',
        '| video / system / channel | quarter ms | dominant IOI ms | ratio | half/double 후보 | downbeat support |',
        '|---|---:|---:|---:|---|---:|']
    chop={'olliewoods','olliewoods_shorts','golly','drumscribe','swartz','moyin'}
    sample_rows=[]
    sampled_channels=set()
    for r in sorted(ok, key=lambda r:(r['channel'] not in chop,r['slug'])):
        if r['channel'] in sampled_channels or not r['proxy']:
            continue
        sampled_channels.add(r['channel'])
        for s in r['proxy'][:2]:
            sample_rows.append((r,s))
        if len(sample_rows)>=12:break
    for r,s in sample_rows[:12]:
        ratio=s['quarter_to_ioi']
        flag='2× period · 8분과 모호' if abs(ratio-2)<=.2 else ('½ period · 배속 후보' if abs(ratio-.5)<=.05 else '—')
        lines.append(f"| [{r['video']} / {s['system']}](https://www.youtube.com/watch?v={r['video']}&t={max(0,int(s['time'])-2)}) / {r['channel']} | {s['quarter_ms']} | {s['dominant_ioi_ms']} | {ratio} | {flag} | {s['supported_downbeats']}/{s['downbeats']} |")
    detection=total['1']+total['2'];notation=total['3']+total['notation_loss']
    verdict=("시간 기반 관측에서는 검출 누락·악기 혼동이 더 큰 오류량이다. quantize/beat tracking이 검출보다 큰 다음 개선 지점이라는 근거는 없다." if detection>notation else
             "시간 기반 관측에서는 표시 위치·기보 손실이 더 크므로 quantize/beat tracking을 다음 개선 후보로 우선 조사할 근거가 있다.")
    lines += ['', '## 판정', '',
        f"{verdict} ①+② {detection:,} ({detection/max(1,total['truth']):.1%}) 대 ③ {total['3']:,} ({total['3']/max(1,total['truth']):.1%}), 기보 손실까지 포함하면 {notation:,} ({notation/max(1,total['truth']):.1%})다. 단, ④의 half/double tempo·마디 시작 오류는 시간 오차가 0이어도 존재할 수 있으므로 이 결과로 기각할 수 없다. 우선 위 청취 링크의 ③과 손실을 확인하고, grid 문제는 별도 소규모 마디/박 정답으로 확인하는 것이 합리적이다.",
        '', '## 한계', '',
        '- 최초 진단용 family replay: 음원 기반 tom_sizes(일부 tom 제거 포함), 사용자 sound map, cymbal subtype/veto를 실행하지 않았다. 최신 전체 앱 결과·MusicXML engraving 정답률을 뜻하지 않는다.',
        '- strict 통과 시스템만 평가해 어려운 chop의 선택 편향이 있다. Ollie Shorts 15곡은 beats/snare 캐시가 없어 제외됐다. truth는 CV/오디오 정렬 산출물이며 일부 악기가 미표기/미검증이다. 최악 1위는 제목에 drumless가 있어 오디오에 기보된 드럼이 실제 존재하는지도 청취 확인해야 한다. ⑤와 동시타의 ② 매칭은 청취 검토 후보다.',
        '- ③은 표시 시각 오류이며 subdivision 원인과 beat-grid 원인을 분리하지 못한다. threshold 이내 오기보·duration/tuplet spelling·ghost/tie·마디 위상 오류는 검출하지 않는다. 정상 expressive timing도 ③ 후보가 될 수 있다.',
        '- raw hit가 quantizer의 병합/범위/reading-mode 처리로 사라지면 기보 손실로 별도 집계한다. 해당 hit를 검출 누락으로 다시 세지 않았다. ⑤는 strict 첫/마지막 head ±50 ms의 합집합 안에서만 계산한다.',
        '- ④ proxy는 독립 beat/bar truth가 없어 N/A다. per-system downbeat support의 차이와 IOI ratio는 점검 힌트이고 오류 개수가 아니다.',
        '', '재현: `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B tools/score_eval/score_eval.py --workers 4`',
        '세부: `summary.json`, `songs/<slug>/result.json` (head별 분류/시각), `songs/<slug>/score.json` (재생성 악보). 소스 SHA-256은 summary.json에 기록.', '']
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    write(OUT/'summary.json', dict(counts=dict(total), evaluated=len(ok), strict_songs=len(strict),
        systems=sum(r['systems'] for r in ok), vault_excluded=excluded, skipped=skipped,
        statuses=dict(Counter(r['status'] for r in results)), fingerprints=fingerprints,
        elapsed_seconds=elapsed, rows=[{k:v for k,v in r.items() if k not in ('records','proxy')} for r in ok]))
    print(json.dumps(dict(evaluated=len(ok),strict=len(strict),counts=dict(total)),ensure_ascii=False),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=4,choices=range(1,5))
    parser.add_argument('--limit',type=int)
    args=parser.parse_args()
    started=time.monotonic()
    OUT.mkdir(parents=True,exist_ok=True)
    rows,excluded=eligible_rows()
    if args.limit:rows=rows[:args.limit]
    fingerprints={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [*sorted((ROOT/'src/groovemap').glob('*.py')),Path(__file__)]}
    results=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(one,r):r for r in rows}
        for future in as_completed(futures):
            r=futures[future]
            try:
                result=future.result()
            except Exception as error:
                result=dict(slug=r['slug'],video=r['video'],channel=r.get('channel'),
                            status='error',error=f'{type(error).__name__}: {error}')
                print('ERROR',r['slug'],result['error'],flush=True)
            results.append(result)
            if result['status']=='ok':
                print('OK',len([x for x in results if x['status']=='ok']),r['slug'],result['seconds'],flush=True)
    results.sort(key=lambda r:r['slug'])
    write(OUT/'inventory.json',[{k:v for k,v in r.items() if k not in ('records','proxy','inputs','options','instruments','counts')} for r in results])
    report(results,excluded,time.monotonic()-started,fingerprints)


if __name__=='__main__':
    main()
