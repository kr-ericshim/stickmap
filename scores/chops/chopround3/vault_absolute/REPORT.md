**결론: 동결한 arm A의 단일 vault 절대 평가를 완료했다.** 레시피·seed 선택이나 학습은 수행하지 않았다.

Opening ID: `e07d4c643ea545fc8deca1fdec374f64`. Ensemble: seeds 0/1 평균을 fold별로 저장하고 두 fold를 평균했다.

| 집단 | 고려한 곡 | strict truth가 있는 곡 | loose notation이 있는 곡 | strict fill 점수에 기여한 곡 | 입력 불량(채점 제외) |
|---|---:|---:|---:|---:|---:|
| manifest | 46 | 19 | 24 | 13 | 0 |
| ollie | 30 | 9 | 8 | 7 | 0 |
| combined | 76 | 28 | 32 | 20 | 0 |

P/R/F는 백분율이며 곡별 평균이 아니라 정수 TP/est/ref를 합산했다. 분모 0은 —다.

| 지표 / 집단 | 부분집합 | Kick P/R/F | Snare P/R/F | Tom P/R/F |
|---|---|---|---|---|
| strict / manifest | all | 85.43/78.66/81.90 | 80.58/85.50/82.96 | 83.71/78.06/80.79 |
| strict / manifest | fast | 90.00/80.60/85.04 | 76.27/88.24/81.82 | 87.27/82.76/84.96 |
| strict / ollie | all | 98.51/95.65/97.06 | 80.82/95.16/87.41 | 85.71/82.11/83.87 |
| strict / ollie | fast | 97.62/95.35/96.47 | 81.58/91.18/86.11 | 84.51/85.71/85.11 |
| strict / combined | all | 89.45/83.69/86.47 | 80.66/88.60/84.44 | 84.29/79.22/81.68 |
| strict / combined | fast | 93.14/86.36/89.62 | 78.35/89.41/83.52 | 86.19/83.87/85.01 |
| loose manifest / complete | all | —/—/— | —/—/— | —/—/— |
| loose manifest / complete | fast | —/—/— | —/—/— | —/—/— |
| loose manifest / partial | all | 68.68/77.25/72.71 | 54.71/82.51/65.79 | 60.16/72.82/65.89 |
| loose manifest / partial | fast | 75.46/75.00/75.23 | 67.04/82.79/74.09 | 73.99/77.11/75.52 |
| loose manifest / all_notation | all | 68.68/77.25/72.71 | 54.71/82.51/65.79 | 60.16/72.82/65.89 |
| loose manifest / all_notation | fast | 75.46/75.00/75.23 | 67.04/82.79/74.09 | 73.99/77.11/75.52 |

| A 비교 기준: strict held-out fast F1 | Kick | Snare | Tom |
|---|---:|---:|---:|
| 개발 지표 (모델 고정 시점) | 83.03 | 82.42 | 82.89 |
| vault manifest | 85.04 | 81.82 | 84.96 |
| vault ollie | 96.47 | 86.11 | 85.11 |
| vault combined | 89.62 | 83.52 | 85.01 |

추론 장치: cuda (14.6 GiB free; capped at 2 GiB). GPU를 썼다면 일부 곡을 CPU로 다시 계산해 최대 절대 차이 0.0006111562252044678를 기록했다 (device.json).

개발 채널: moyin, dailychops, olliewoods, alexblack, olliewoods_shorts.

Strict는 기존 chops_scoring --scope strict와 동일하다: strict truth를 load하고 gap ≤140ms, ≥4 K/S/T 순간, tom을 포함하는 fill 중 median gap <90ms가 fast다. all은 이 fill들의 전체이며 곡 전체의 모든 strict note를 뜻하지 않는다.

Loose는 fill_seq.notated(0.75), nearest K/S/T neighbour <90ms, order DP ±60ms다. complete/partial/all_notation은 기존 vault scorer와 동일한 whole-system audit 범위를 사용한다. strict와 loose의 fast 수치를 직접 비교하지 않는다.

제외: final_exclusions()의 mixed-performance 목록과 수동 Eric Moore 목록을 모두 적용했다. 기존 scorer는 vault_final_exclusions.json을 읽지 않으므로 이 스크립트가 직접 적용했다.
- mixed_performance: 6CLJH8YJtFM: excluded: train peer K_AtZoNIoKo N0PXFlgigjw ZsMtIfjkubo
- mixed_performance: GAA07M1brJw: excluded: train peer qLi8dFevOF0
- mixed_performance: LAdCiQAb-C0: excluded: train peer bd17mxbLaNE
- mixed_performance: ROdoBlEcfcM: excluded: train peer qLi8dFevOF0
- manual: YytKWBpYIyM: audio-confirmed same performance as cDqw_3-F0Ag, grouped with xH-pMgRZWCA by the ambiguous-together rule; same train-peer risk
- manual: cDqw_3-F0Ag: audio-confirmed same performance as YytKWBpYIyM; same train-peer risk
- manual: xH-pMgRZWCA: Eric Moore II Short; audio inconclusive vs train 9juoKgimeR4 (and vault Ikg4P3F_Wj8, 8AHvVFQ0_To, JG2xMCBwL1Y); may be the same performance as a train song

CV gate 통과는 독립적으로 검증한 악기 정답이나 전체 악보 coverage를 보장하지 않는다. partial notation의 실제 미기보 연주는 FP로 계산될 수 있다. 반복·겹치는 clip은 상관된 표본이다. truth가 없거나 strict fill이 없는 곡도 coverage에 포함되며 유효 분모가 없으면 결과는 미정이다.

단일 A 절대 측정이며 bootstrap, arm 비교, 승격 판정은 하지 않았다. 입력 truth/체크포인트/코드/오디오 해시를 예측 전 저장하고 결과 작성 전에 다시 확인했다.

이전 opening 1개 (점수 산출 전 실패, 결과 미확인):
```json
{"timestamp": "2026-10-03T16:15:49.131027+00:00", "command": "/home/ericshim/groovemap/.venv/bin/python tools/gpu/vault_absolute.py run --plan scores/chops/chopround3/plan.json --arm A", "model_tag": "chopround3_A_absolute", "opening_id": "cd068aac75184c65ae983babf3e5a373", "pid": 184058, "outcome": "failed after truth generation, before inference/scoring: the write guard blocked numba JIT cache writes from librosa (\"cannot cache function ... no locator available\"); fixed by NUMBA_CACHE_DIR in a guard-allowed folder (and feature() tmp-file path allowed). Outputs preserved in scores/chops/chopround3/vault_absolute.attempt1-failed and vault_truth.attempt1/. Rerun approved."}
```

`truth/harvest/vault_log.jsonl`의 이번 durable row:
```json
{
  "timestamp": "2026-10-03T16:51:26.799718+00:00",
  "command": "/home/ericshim/groovemap/.venv/bin/python tools/gpu/vault_absolute.py run --plan scores/chops/chopround3/plan.json --arm A",
  "model_tag": "chopround3_A_absolute",
  "opening_id": "e07d4c643ea545fc8deca1fdec374f64",
  "pid": 226642
}
```

Scanner가 native 악보 영역/staff를 증명하지 못한 영상은 unknown으로 유지했다. truth event를 생성하지 않았고 점수 분모에 넣지 않았다. 고려한 곡 수에는 포함된다.

- 5JKuX18DXXo: unstable native Shorts page boundary
- 80S4pKQNfBw: unstable native Shorts page boundary
- LOlbqboyiHw: unstable native Shorts page boundary
- LYTPLv8a5Hk: unstable native Shorts page boundary
- O4KnrQzMFok: unstable native Shorts page boundary
- RXPA93tRp5U: unstable native Shorts page boundary
- T7uw_Y2lWfI: unstable native Shorts page boundary
- TVvo_UJT-3U: unstable native Shorts page boundary
- TdU43ovgyW0: unstable native Shorts page boundary
- VDhEWOZsHtQ: unstable native Shorts page boundary
- XwVZ0IWlLZ0: unstable native Shorts page boundary
- YJRkJQ-E1EE: unstable native Shorts page boundary
- d70zGTjtVdQ: unstable native Shorts page boundary
- gwQF1xZCaeY: unstable native Shorts page boundary
- ieO_oUmCOEo: unstable native Shorts page boundary
- kb1pUl2zXOk: unstable native Shorts page boundary
- tKWGuuI5TU4: unstable native Shorts page boundary