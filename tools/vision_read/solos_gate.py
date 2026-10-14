"""Run the unchanged canonical audio gates on scratch hybrid candidates."""
import argparse
import collections
import concurrent.futures
import datetime
import os
import sys
from pathlib import Path
sys.dont_write_bytecode=True
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[key]='1'
from common import DATA, ROOT, dump, load, sha
from solos_extract import DENSE, CONTROLS, source


def fast_counts(events):
    counts=collections.Counter()
    groups=collections.defaultdict(list)
    for n in events:
        inst=n['inst']
        kind='T' if inst=='tom' or inst.startswith('tom_') else {'kick':'K','snare':'S'}.get(inst)
        if kind:
            groups[n['system']].append((n['time'],kind))
    for values in groups.values():
        times=sorted({t for t,k in values})
        fast=set()
        for a,b in zip(times,times[1:]):
            if 0<b-a<.09:
                fast.update((a,b))
        counts.update(k for t,k in set(values) if t in fast)
    return {k:counts[k] for k in ('K','S','T')}


def agreement(estimated,reference):
    # The predeclared denominator is ALL hybrid-strict notes in the selected
    # control visits. New/missing reference notes are failures, not exclusions.
    from align import match
    pairs=[];unmatched=[]
    for inst in sorted({n['inst'] for n in estimated}):
        est=sorted([(i,n) for i,n in enumerate(estimated) if n['inst']==inst],key=lambda p:p[1]['time'])
        ref=sorted([(i,n) for i,n in enumerate(reference) if n['inst']==inst],key=lambda p:p[1]['time'])
        matched=match([n['time'] for i,n in ref],[n['time'] for i,n in est],.020)
        used=set()
        for a,b in matched:
            ri,r=ref[a];ei,e=est[b];used.add(b)
            pairs.append(dict(hybrid_index=ei,reference_index=ri,inst=inst,delta_ms=(e['time']-r['time'])*1000))
        for b,(ei,e) in enumerate(est):
            if b not in used:
                nearest=min(ref,key=lambda p:abs(p[1]['time']-e['time'])) if ref else None
                unmatched.append(dict(hybrid_index=ei,time=e['time'],inst=inst,system=e['system'],
                                      native_x=e.get('native_x'),pos=e.get('pos'),
                                      nearest_reference_time=nearest[1]['time'] if nearest else None,
                                      nearest_delta_ms=(e['time']-nearest[1]['time'])*1000 if nearest else None))
    ratio=len(pairs)/len(estimated) if estimated else None
    return dict(matched=len(pairs),hybrid_strict_notes=len(estimated),reference_strict_notes=len(reference),
                agreement=ratio,reference_recall=len(pairs)/len(reference) if reference else None,
                passed=ratio is not None and ratio>=.95,tolerance_ms=20,instrument_exact=True,
                denominator='all hybrid-strict notes; one-to-one time+instrument matching; no overlap-only filtering',
                pairs=pairs,unmatched=unmatched)


def gate(video):
    sys.path.insert(0,str(ROOT/'tools/truth_video'))
    from align import build_truth
    import bands
    from tiers import write
    row,song,scan=source(video)
    folder=DATA/'candidates'/video
    # cache=False prohibits a missing/invalid band cache from writing canonically.
    evidence=bands.load(song,cache=False)
    truth,alignment=build_truth(song,folder,bands=evidence)
    scoring=write(folder)
    scored=load(folder/'truth.scoring.json')
    summary=dict(video=video,systems_read=len(alignment['systems']),
                 strict_pass=sum(r['usable'] for r in alignment['systems']),
                 scoring_pass=sum(r['scoring_usable'] for r in scoring['systems']),
                 strict_notes=len(truth),scoring_notes=len(scored),fast_strict=fast_counts(truth),fast_scoring=fast_counts(scored),
                 reasons=dict(collections.Counter(r for s in alignment['systems'] for r in s['reasons'])),
                 gates={p:sha(ROOT/'tools/truth_video'/p) for p in ('align.py','bands.py','tiers.py')},
                 attack_sha256=sha(song/'attacks.htdemucs.npy'),
                 band_cache_sha256=sha(song/bands.CACHE) if (song/bands.CACHE).exists() else None)
    if video in CONTROLS:
        summary['acceptance']=agreement(truth,load(song/'truth/truth.json'))
    dump(folder/'summary.json',summary)
    return summary


def report():
    names=dict(zip(DENSE,['Steve Smith','Jojo Mayer','Dave Weckl','Gary Novak','Tomas Haake']))
    summaries={p.parent.name:load(p) for p in (DATA/'candidates').glob('*/summary.json')}
    model_results={}
    for run_path in sorted((DATA/'runs').glob('hybrid-solos-*/run.json')):
        for result in load(run_path).get('results',[]):
            model_results[result['id']]=result
    controls={v:summaries[v]['acceptance'] for v in CONTROLS if v in summaries}
    passed=len(controls)==len(CONTROLS) and all(v['passed'] for v in controls.values())
    measured_failure=any(v['agreement'] is not None and v['agreement']<.95 for v in controls.values())
    state=('acceptance_passed' if passed else 'acceptance_failed' if measured_failure else
           'acceptance_inconclusive' if len(controls)==len(CONTROLS) else 'acceptance_pending')
    acceptance=dict(status=state,passed=passed,measured_failure=measured_failure,threshold=.95,controls=controls,
                    dense_pipeline_complete=all(v in summaries for v in DENSE))
    dump(DATA/'acceptance.json',acceptance)
    lines=['# Hybrid CV + Gemini dense solos', '',
           '작성 UTC: '+datetime.datetime.now(datetime.timezone.utc).isoformat(), '',
           '**상태: '+state+'**. Gemini 모델은 `gemini-3.8-flash-medium`, look, 최대 4 workers다. '+
           'CV/추출은 WSL 최대 7 processes, 정렬은 기존 `align.build_truth`와 `tiers.write`를 호출했다. '+
           '오디오 공격점·기존 truth 시간·detector 출력은 Gemini 입력이나 자동 앵커 생성에 사용하지 않았다.', '',
           '| 영상 | 시스템 found / read | strict / scoring 통과 | strict / scoring notes | strict fast K/S/T |',
           '|---|---:|---:|---:|---:|']
    for video in DENSE:
        ip=DATA/'native'/video/'inventory.json';inv=load(ip) if ip.exists() else {}
        s=summaries.get(video)
        if s:
            f=s['fast_strict'];cells=[f"{inv.get('systems_found','?')} / {s['systems_read']}",f"{s['strict_pass']} / {s['scoring_pass']}",f"{s['strict_notes']} / {s['scoring_notes']}",f"{f['K']} / {f['S']} / {f['T']}"]
        else:
            read_count=sum(r.get('status')=='ok' for sid,r in model_results.items() if sid.startswith(video+'_auto_'))
            cells=[f"{inv.get('systems_found','미측정')} / {read_count}",'미실시','미실시','미실시']
        lines.append('| '+names[video]+' (`'+video+'`) | '+' | '.join(cells)+' |')
    lines += ['', 'Scoring 통과/notes는 strict를 포함한 별도 scoring selection이다. Fast는 수락한 같은 시스템 내부의 서로 다른 K/S/T onset 사이 간격이 90ms 미만인 class/time 수다. 동시에 친 K/S/T는 각각 세며 같은 순간의 tom 위치는 한 T로 합친다.', '',
              '| Acceptance 영상 | matched / hybrid-strict notes | 20ms + 악기 일치 | 기존 strict notes | 판정 |',
              '|---|---:|---:|---:|---|']
    for v in CONTROLS:
        a=controls.get(v)
        if a:
            pct=f"{a['agreement']:.2%}" if a['agreement'] is not None else '산출 불가 (strict 0)'
            verdict='미확정' if a['agreement'] is None else 'PASS' if a['passed'] else 'FAIL'
            lines.append(f"| {v} | {a['matched']} / {a['hybrid_strict_notes']} | {pct} | {a['reference_strict_notes']} | {verdict} |")
        else:
            lines.append(f'| {v} | 미측정 | 미측정 | 미측정 | 대기 |')
    lines += ['', 'Acceptance 분모는 선택한 기존 strict 방문에서 나온 **모든 hybrid-strict 음표**다. 20ms 이내·동일 악기를 일대일로 대응하며, 기존 truth에 없는 음표도 분모에서 제거하지 않는다. Hybrid strict가 0이면 100%로 처리하지 않고 acceptance 미충족으로 기록한다. 검증용 시스템 선택에는 canonical strict 시스템 ID만 사용했고, 그 시간이나 음표를 앵커로 넘기지 않았다.', '',
              '연속 가로 스크롤에는 원래의 줄바꿈 system 경계가 없다. 이 실행의 system은 기존 scanner의 4초 블록과 동일한 **서로 겹치지 않는 재생 방문 구간**이다. 같은 구간의 수십 ms 실패 records는 합치고, Gemini에는 문맥을 위해 전체 native viewport를 보여주되 출력에는 커서가 실제 방문한 x 범위만 사용한다. 정지형 페이지는 기존 scan 방문을 사용하고 커서 reset 없는 동일 geometry 반복만 합친다. 따라서 found는 native notation이 확인된 방문 창 수이며, 원본 인쇄 악보의 줄 수가 아니다.', '',
              '자동 프레임 선택은 각 창의 15/50/85% native 프레임 중 CV 구조 검사, 등록된 이동 범위, 검출 head 수, 선명도로 결정했다. 15Hz 영상 등록은 기존 translation correlation ≥0.88, uniqueness ≥0.025와 오른쪽 이동 거부를 유지한다. Canonical의 200ms 샘플 간 80px 이동 한계로 쪼개지던 구간을 더 짧은 프레임 간격으로 처리한다. 오선 후보는 scan의 native drum-row geometry로 선택하고 기존 head/label 함수와 8px·line support·spacing 검사에 넘긴다. Native 이미지는 합성·확대하지 않는다.', '',
              'White overlay의 고정 min-x 비율로 방문 범위의 왼쪽 head를 버리던 처리는 이 CV adapter에서 해제했다. 악기 legend·confidence·exportable 조건은 기존 CV 결과를 따른다. Gemini-only head는 `inst=unknown`, `exportable=false`다. False-positive disposition과 uncertain head는 보존하며, 방문 범위 안 unresolved onset/tie 또는 위치 없는 head는 구조상 거부한다.', '',
              '자동 bar origin/rate는 유리수 onset과 영상 cursor time의 기울기로 구한다. 커서가 부족한 경우 scan interval fallback을 명시하며 기존 rhythm/no-cursor held-out 70%/90% 조건을 적용한다. 수동 검토나 `staff_reviewed=true`를 꾸며 넣지 않는다. Fitted, held-out, shifted controls, dense ambiguity, phase/bands, monotonicity의 코드와 임계값은 그대로다.', '']
    if state=='acceptance_failed':
        lines += ['**사용자가 지정한 중단 조건에 따라 dense 솔로의 Gemini 및 audio gate 실행을 중단했다.** CV/추출 inventory는 남겨 두었다. 아래 실패 원인은 수락 결과의 실제 unmatched 음표와 gate reasons이며, 참조 agreement를 올리기 위해 게이트나 앵커를 다시 맞추지 않았다.', '']
    if state=='acceptance_inconclusive':
        lines += ['**전체 acceptance는 미확정이다.** Eric의 strict 출력이 0이라 일치율을 계산할 수 없다. Larnell의 측정 일치율은 95% 이상이므로 명시된 “측정 일치율 <95%이면 중단” 조건은 발생하지 않았다. 추가 질문 없이 안전한 선택으로 scratch 후보 생성·게이트 실행을 계속하되, 이 결과를 전체 acceptance 통과나 검증된 truth 증가로 승격하지 않는다. 어느 게이트도 완화하지 않았고 Eric 실패를 참조 truth에 맞춰 재조정하지 않았다.', '']
    for video,s in summaries.items():
        lines += [f"## {video}", '', f"Read {s['systems_read']}, strict {s['strict_pass']}, scoring {s['scoring_pass']}; strict notes {s['strict_notes']}, scoring notes {s['scoring_notes']}.", '', '주요 거부 사유: '+str(sorted(s['reasons'].items(),key=lambda p:-p[1])[:8]), '']
        if s.get('acceptance'):
            lines += ['불일치 상세는 `candidates/'+video+'/summary.json`의 `acceptance.unmatched`에 있다.', '']
    lines += ['## 산출물과 제한', '',
              '`native/<video>/manifest.json`: 원본 프레임·scan 해시·좌표·이동 edge·커서 priors. `hybrid/jobs/<system>/`: original/annotated/CV. `runs/<run>/<system>/`: raw Gemini 응답·reading·로그. `candidates/<video>/`: strip/truth/alignment 및 scoring JSON. `acceptance.json`: 안전성 판정. 보호 파일 해시 검사는 `integrity.json`에 기록한다.', '',
              '검증한 vault 제외 목록은 다섯 dense solo, Eric `9juoKgimeR4`, Larnell `N0PXFlgigjw`다. `Ikg4P3F_Wj8`는 봉인 유지하며 읽지 않았다. Canonical truth·manifest·vault 설정과 보호 코드에는 쓰지 않았고 training은 실행하지 않았다. Band 캐시가 없거나 유효하지 않아도 `cache=False`로 계산하여 canonical 쓰기를 방지한다.', '']
    runtime=DATA/'runtime-status.json'
    if runtime.exists() and not acceptance['dense_pipeline_complete']:
        q=load(runtime)
        lines += ['## 현재 실행 상태', '',
                  'Gemini 상태: `'+q.get('status','unknown')+'`. '+
                  '성공한 dense 읽기 '+str(q.get('ok',0))+'/325개. 첫 429에서 모든 진행/대기 호출을 취소했다. '+
                  '재개 하한 UTC: `'+str(q.get('not_before_utc'))+'` (CLI reset + 90초). '+
                  'Mac의 분리 실행 coordinator가 성공 결과를 재사용하고 남은 읽기, WSL 자동 변환/게이트, 보고서 복사를 순서대로 수행한다. '+
                  '전체 실행은 아직 완료되지 않았다. Quota 취소는 악보 정확도 0으로 세지 않는다.', '']
    (DATA/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return acceptance


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--videos',nargs='+',default=CONTROLS)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--report-only',action='store_true')
    a=p.parse_args()
    if a.report_only:
        report();return
    if not 1<=a.workers<=8:
        p.error('workers must be 1..8')
    for video in a.videos:
        source(video)
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as pool:
        results=list(pool.map(gate,a.videos))
    for r in results:
        print({k:r[k] for k in ['video','strict_pass','scoring_pass','strict_notes','fast_strict']},flush=True)
    report()


if __name__=='__main__':
    main()
