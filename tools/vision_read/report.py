"""Write a comparable pilot report from completed, frozen score artifacts."""
import argparse
import collections
import json
import statistics
import sys
sys.dont_write_bytecode = True
from common import DATA, confined, dump, load


def percent(value):
    return 'N/A' if value is None else f'{100*value:.1f}%'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--runs', nargs='+', default=['high-look', 'medium-look', 'high-zoom'])
    ap.add_argument('--dense-run', default='dense-high-look')
    a = ap.parse_args()
    docs = {name: load(DATA/'runs'/name/'scores.json') for name in a.runs}
    reference = next(iter(docs.values()))
    artifact_counts = {}
    for name, doc in docs.items():
        if set(doc['config']['systems']) != set(reference['config']['systems']):
            raise ValueError(f'different pilot systems: {name}')
        for field in ('manifest_sha256', 'gold_sha256', 'scoring_version', 'scorer_sha256', 'common_sha256', 'annotation_sha256'):
            if doc[field] != reference[field]:
                raise ValueError(f'incomparable {field}: {name}; rescore with the same scorer')
        batch = load(DATA/'runs'/name/'run.json')
        crop_counts = {}
        for sid in doc['config']['systems']:
            job = DATA/'runs'/name/sid
            inputs = {v['file'] for v in load(job/'job.json')['images']}
            crop_counts[sid] = sum(p.suffix == '.png' and p.name not in inputs for p in job.rglob('*') if p.is_file())
        artifact_counts[name] = crop_counts
        if batch['completed'] != len(doc['config']['systems']):
            raise ValueError(f'incomplete run: {name}')
    dense = load(DATA/'runs'/a.dense_run/'scores.json')
    rows = []
    for name, doc in docs.items():
        summary = doc['summary']
        m = summary['metrics']
        batch = load(DATA/'runs'/name/'run.json')
        rows.append({'run': name, 'model': doc['config']['model'], 'mode': doc['config']['mode'],
                     'head_recall': m['head']['value'], 'head_precision': m['head']['precision'], 'instrument_accuracy': m['instrument']['value'],
                     'onset_accuracy': m['onset']['value'], 'onset_conditional_accuracy': m['onset']['conditional_correct']/m['onset']['conditional_total'] if m['onset']['conditional_total'] else None,
                     'full_symbolic_onset_recall': m['full_symbolic_onset']['value'], 'full_symbolic_onset_precision': m['full_symbolic_onset']['precision'],
                     'duration_accuracy': m['duration']['value'], 'tuplet_ratio_accuracy': m['tuplet_ratio']['value'],
                     'tie_positive_recall': m['tie']['positive_correct']/m['tie']['positive_total'] if m['tie']['positive_total'] else None,
                     'bar_count': m['bar_count'], 'staff_lines': m['staff_lines'], 'clef': m['clef'],
                     'mean_minutes_actual': summary['time_seconds']['mean']/60, 'median_minutes_actual': summary['time_seconds']['median']/60,
                     'mean_minutes_first_attempt': summary['first_attempt_time_seconds']['mean']/60, 'batch_minutes_actual': doc['batch_wall_seconds']/60,
                     'statuses': summary['statuses'], 'retries': sum(len(r.get('attempts', []))-1 for r in batch['results']),
                     'format_recovered': [r['id'] for r in batch['results'] if r.get('format_recovery')],
                     'by_instrument': summary['by_instrument'], 'relaxed_head_recall': doc['relaxed_geometry_sensitivity']['metrics']['head']['value'],
                     'relaxed_onset_accuracy': doc['relaxed_geometry_sensitivity']['metrics']['onset']['value'],
                     'beat_claim_mismatches': sum(r['beat_diagnostics']['claim_sum_mismatches'] for r in doc['systems'] if r['beat_diagnostics']),
                     'same_voice_overlaps': sum(r['beat_diagnostics']['same_voice_overlaps'] for r in doc['systems'] if r['beat_diagnostics'])})
        rows[-1]['additional_png_artifacts'] = artifact_counts[name]
        rows[-1]['quota_failures'] = sum(any('RESOURCE_EXHAUSTED' in p.read_text(errors='replace') for p in (DATA/'runs'/name/sid).glob('stderr-*.log')) for sid in doc['config']['systems'])
        rows[-1]['accuracy_censored'] = rows[-1]['quota_failures'] > 0
    timing_candidates = []
    for row in dense['systems']:
        job = DATA/'runs'/a.dense_run/row['id']
        if (job/'reading.json').exists():
            reading = load(job/'reading.json')
            heads = [h for b in reading['bars'] for e in b['events'] for h in e['heads']]
            null_onsets = sum(len(e['heads']) for b in reading['bars'] for e in b['events'] if e['onset'] is None)
            timing_candidates.append({'id': row['id'], 'reading': str((job/'reading.json').relative_to(DATA)), 'heads': len(heads),
                                      'null_onset_heads': null_onsets, 'uncertain': len(reading['uncertain']), 'audio_checked': False})
    result = {'comparisons': rows, 'dense': dense['summary'], 'dense_readings': timing_candidates,
              'scope': '21 provisional note-gold systems (330 heads), six complete inventories (159 heads), ten additional dense systems without note gold; actual reading completion below; no audio check',
              'all_time_minutes': sum(r['batch_minutes_actual'] for r in rows)+dense.get('batch_wall_seconds', 0)/60,
              'mode_policy': 'Prompt-only; complete tool traces unavailable in text output. A high/zoom assignment is not proof every job cropped.',
              'representation_recovery': 'Meter strings and x/cross synonyms only; raw responses and wasted retry time retained.'}
    dump(DATA/'pilot_summary.json', result)
    text = ['# Vision pilot 결과', '', result['scope'], '',
            '| 조건 | 머리 recall / precision (21개) | exact onset (21개) | 좌표 독립 onset recall / precision (완전 6개) | 평균 분/시스템 (실제 / 첫 시도) |',
            '|---|---:|---:|---:|---:|']
    for row in rows:
        if row['accuracy_censored']:
            text.append(f'| {row["run"]} | quota로 미측정 | quota로 미측정 | quota로 미측정 | 비교 불가 |')
        else:
            text.append(f'| {row["run"]} | {percent(row["head_recall"])} / {percent(row["head_precision"])} | {percent(row["onset_accuracy"])} | {percent(row["full_symbolic_onset_recall"])} / {percent(row["full_symbolic_onset_precision"])} | {row["mean_minutes_actual"]:.2f} / {row["mean_minutes_first_attempt"]:.2f} |')
    text += ['', '좌표 독립 onset은 (마디, 오선 위치, 머리 모양, tie 상태, exact onset) multiset의 교집합이며 159개 머리를 사용한다. native 좌표 기반 onset은 280개 알려진 GOLD onset을 사용한다. 서로 분모와 질문이 다르다.',
             '머리 precision은 완전 열거된 사각형 안에서만 계산하며, 미검수 구간은 FP가 아니다. GOLD 악기명 222개만 verified instrument accuracy에 포함한다.',
             'look/zoom은 프롬프트 정책이다. text 로그만으로 뷰어 횟수나 도구 미사용을 인증하지 않는다.', '', '## 악기·기법·구조', '']
    for row in rows:
        if row['accuracy_censored']:
            text += [f'### {row["run"]}', '', f'RESOURCE_EXHAUSTED 429로 {row["quota_failures"]}개 작업이 중단됐다. 상태 {row["statuses"]}. Batch 시도 시간 {row["batch_minutes_actual"]:.2f}분.',
                     '채점 원자료의 missing-prediction 0점은 운영 실패 기록이며 모델 정확도 0%를 뜻하지 않는다. 완료된 조건과 시간/정확도를 비교할 수 없다.',
                     f'추가 PNG가 생성된 시스템 {sum(n > 0 for n in row["additional_png_artifacts"].values())}개. crop 도구 사용은 관측됐지만 최종 판독 JSON은 미완료다.', '']
            continue
        text += [f'### {row["run"]}', '', f'음가 {percent(row["duration_accuracy"])}, 악기 {percent(row["instrument_accuracy"])}, 검출된 머리의 조건부 onset {percent(row["onset_conditional_accuracy"])}, tuplets {percent(row["tuplet_ratio_accuracy"])}, tie 양성 recall {percent(row["tie_positive_recall"])}.',
                 f'마디 수 {row["bar_count"]["correct"]}/{row["bar_count"]["total"]}, 정확 오선 좌표 {row["staff_lines"]["correct"]}/{row["staff_lines"]["total"]}, clef {row["clef"]["correct"]}/{row["clef"]["total"]}.',
                 f'완료 상태 {row["statuses"]}; 재시도 {row["retries"]}개; 표현만 복구 {len(row["format_recovered"])}개. Batch {row["batch_minutes_actual"]:.2f}분, 실제 중앙값 {row["median_minutes_actual"]:.2f}분/시스템.',
                 f'좌표 허용오차 확대 보조 결과: head recall {percent(row["relaxed_head_recall"])}; onset {percent(row["relaxed_onset_accuracy"])}.',
                 f'beat 체크 산술 불일치 {row["beat_claim_mismatches"]}, 같은 voice 겹침 {row["same_voice_overlaps"]}. 박 길이 합이 맞아도 기보 정답을 보장하지 않는다.', '',
                 f'입력 사본 외 추가 PNG가 생성된 시스템 {sum(n > 0 for n in row["additional_png_artifacts"].values())}/{len(row["additional_png_artifacts"])} (확대 artifact 관측; 완전한 도구 trace는 아님).', '',
                 '| 악기 | recall | precision | TP / FN / FP |', '|---|---:|---:|---:|']
        for instrument, counts in row['by_instrument'].items():
            text.append(f'| {instrument} | {percent(counts["recall"])} | {percent(counts["precision"])} | {counts["tp"]} / {counts["fn"]} / {counts["fp"]} |')
        text += ['', '완전 GOLD별 좌표 독립 onset:', '', '| 시스템 | exact / GOLD | 예측 heads / GOLD heads |', '|---|---:|---:|']
        for system in docs[row['run']]['systems']:
            if system['complete']:
                mm = system['metrics']['full_symbolic_onset']
                text.append(f'| {system["id"]} | {mm["correct"]}/{mm["total"]} | {system["predicted_heads"]}/{system["gold_notes"]} |')
        text.append('')
    text += ['## 실패 해석과 권고', '',
             'Native head 좌표와 오선 y 추정 오차가 end-to-end 머리/리듬 점수를 함께 낮춘다. 좌표를 제거한 완전 GOLD에서도 틀린 staff position/head shape 또는 onset이 남으므로 좌표 문제만으로 모든 오류를 설명할 수 없다.',
             'Dense viewport의 부분 마디 origin은 null로 남는 경우가 있다. 이것은 오디오 시간으로 환산할 수 없으므로 WSL 전에 영상/마디 anchor 검수가 필요하다. Tuplet ratio와 tie의 음성 다수 점수를 양성 recall로 오해하지 않는다.',
             '단순 head 수, 박 합, 높은 조건부 음가 점수만으로 자동 truth를 승인하지 않는다. 비교 표의 좌표 독립 musical 정확도와 실제 시간을 함께 보고 후보 판독용 모델을 고르고, 오선 위치/shape/flags 검수 및 기존 audio gate를 모두 거친다. 이 소규모 잠정 GOLD로 dense 솔로 전체의 성능이나 모델 우열을 일반화하지 않는다.', '',
             '## Dense WSL 대기 자료', '', f'저장된 readings: {len(timing_candidates)}/10. 음표 GOLD 없음; 오디오 검사 없음.', '',
             '| 시스템 | heads | null onset heads | uncertain |', '|---|---:|---:|---:|']
    text += [f'| {r["id"]} | {r["heads"]} | {r["null_onset_heads"]} | {r["uncertain"]} |' for r in timing_candidates]
    text += ['', '정확한 후속 명령, anchors 필드, 기존 strict/phase/control gate 연결은 `tools/vision_read/README.md`의 WSL 절차를 따른다.',
             '읽기 파일: `.eval-cache/vision_pilot/runs/dense-high-look/<system>/reading.json`.',
             '정량 원자료: 각 run의 `scores.tsv`, `scores.json`, `run.json`; 비교 요약 `pilot_summary.json`.', '']
    confined(DATA/'PILOT_REPORT.md').write_text('\n'.join(text))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
