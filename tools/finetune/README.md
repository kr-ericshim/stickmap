# ADTOF 미세조정

빠른 필에서 놓치는 타격을 보면 ADTOF의 활성값 자체가 올라오지 않는 프레임이 대부분이었다. 이런 건 피크 검출 규칙을 아무리 바꿔도 살릴 수 없어서, 채보 영상에서 모은 정답(`tools/harvest`, 14개 채널 131곡, 채택된 구간 약 3.4시간)으로 ADTOF의 Frame-RNN을 미세조정했다. 결과는 `src/stickmap/adtof-stickmap-v2.pt`로 들어가 있고, 드럼 스템에만 쓴다. 믹스 쪽 활성값은 원래 모델 그대로다.

학습과 추론은 집 PC의 GPU(WSL2)에서 돌렸고 채점은 Mac에서 했다. GPU 머신 준비는 [docs/wsl-gpu.md](../../docs/wsl-gpu.md)에 적어 두었다.

## 학습 방식

- 킥·스네어·탐은 채택된 구간 안의 정답으로 가르친다. 정답 타이밍이 10 ms 정도 오차가 있어서 적힌 타격 앞뒤 한 프레임은 손대지 않는다.
- 채보자는 고스트 노트 같은 걸 자주 빼먹는다. 그래서 아무것도 안 적힌 프레임은 원래 모델도 조용할 때만 무음으로 가르친다. 원래 모델이 킥이나 탐을 듣는데 적힌 게 없으면 가중치 0.3으로만 무음을 가르치고(`--doubt`), 스네어는 모르는 것으로 둔다.
- 그 밖의 구간과 하이햇·심벌(안 적는 채널이 많다)은 원래 모델 출력을 따라가게 한다(증류, 가중치 0.3). 망각을 막기 위해서다.
- 8초 크롭 중 40%는 적힌 필 타격을 가운데에 둔다. 게인과 스펙트럼 기울기로 증강하고, BatchNorm 통계는 고정한다. Adam 1e-4, 크롭 12개씩 1500 step으로 미리 정해 두었고, 로그에 찍히는 검증 점수로 체크포인트를 고르지 않는다.
- fold는 채널 단위로 나눈다. 다른 채널이 같은 곡을 채보한 경우에는 한쪽을 채점하는 동안 다른 쪽으로 학습하지 않는다(`splits.SAME_SONG`).

## 실행

```bash
.venv/bin/python tools/finetune/data.py --jobs 4        # 곡별 특징량과 정답 -> .eval-cache/ft/
.venv/bin/python tools/finetune/teacher.py --jobs 4     # 같은 특징량에 대한 원래 모델 출력
.venv/bin/python tools/finetune/train.py --fold 0 --tag v2 --steps 1500 --doubt 0.3 --no-withheld-mask   # fold 1, 2도
.venv/bin/python tools/finetune/predict.py v2-fold0     # fold마다 빠진 곡의 활성값
.venv/bin/python tools/finetune/solos.py v2             # 학습에 안 쓴 드럼 솔로 영상

ONLY=(swartz allyeodrum drumclass mydrumsheet taktak drumshin drumplace nakamura chock yeol atthedrum jilju drummate drumscribe)
FT_TAG=v2 PYTHONPATH=tools/finetune .venv/bin/python tools/harvest/fill_score.py --detect ft_variants:tuned --only "${ONLY[@]}" --jobs 4
FT_TAG=v2 PYTHONPATH=tools/finetune .venv/bin/python tools/harvest/global_score.py --detect ft_variants:tuned --only "${ONLY[@]}"
FT_TAG=v2 PYTHONPATH=tools/finetune .venv/bin/python tools/truth_video/fill_seq.py --detect ft_variants:tuned --jobs 4

.venv/bin/python tools/finetune/train.py --fold all --tag v2 --steps 1500 --doubt 0.3 --no-withheld-mask   # 배포용 모델
```

GPU 머신에서는 `train.py`에 `--device cuda`를 붙인다. 기본값은 Mac의 `mps`다. 테스트는 합성 데이터만 쓴다.

```bash
.venv/bin/python -m pytest -q tools/finetune
```

## 결과

곡마다 그 채널을 본 적 없는 fold 모델로 채점했다. 솔로 영상은 학습에 한 번도 쓰지 않은 외부 테스트라 배포 모델로 채점했다. 전체 파이프라인을 기본 임계값으로 돌렸고, 스네어 응답도 각 모델의 활성값으로 다시 만들었다(`ft_variants.snare`).

| 탐 P / R | 원래 ADTOF | 미세조정 v2 |
|---|---|---|
| 131곡 전체 (`score.py`) | .76 / .79 | .77 / .80 |
| 커버 영상 필, 타격 간격 90 ms 미만 (`fill_score.py`) | .86 / .90 | .88 / .90 |
| 드럼 솔로, 타격 간격 90 ms 미만 (`fill_seq.py`) | .46 / .72 | .47 / .75 |

131곡 전체에서 킥은 .94 / .96에서 .93 / .98로, 스네어는 .80 / .95에서 .83 / .97로 바뀌었다. 커버 영상의 아주 빠른 필에서 킥은 .96 / .83에서 .94 / .87, 스네어는 .94 / .86에서 .94 / .93이 됐다. 솔로의 아주 빠른 킥 재현율은 .67에서 .73으로 올랐다. 채보자가 안 적은 타격도 오답으로 세기 때문에 정밀도는 실제보다 낮게 나온다.

`--doubt` 없이 학습한 v1은 찾는 타격 수는 비슷했지만 탐 정밀도가 떨어졌다(전체 .74 / .82). 시간 늘이기한 오디오에 ADTOF를 돌리는 방법도 해 봤지만 도움이 안 됐다.

## 봉인 세트

실험을 반복하면 개발 세트에 맞춰지기 마련이라, 어느 시점(`truth/harvest/vault.json`의 `cutoff`) 이후에 처음 들어온 곡은 봉인해 둔다. 봉인된 곡은 데이터 준비, 캐시 로딩, 모든 fold 학습(`all` 포함), 평소 채점에서 빠진다. 모델과 설정을 고정한 뒤 채점 스크립트에 `--final`을 줘야 봉인된 곡만 한 번 열리고, 그 실행은 `truth/harvest/vault_log.jsonl`에 기록된다.

```bash
.venv/bin/python tools/harvest/vault.py list
.venv/bin/python tools/harvest/vault.py stats
.venv/bin/python tools/harvest/fill_score.py --loose --final --model-tag NAME --jobs 4
```

## 그 밖의 실험

`--dense`(음표는 깨끗하게 읽혔지만 타이밍 검증에서 떨어진 빽빽한 구간에서 "이 창 안에 이 악기 타격이 하나 이상 있다"는 약한 정답을 쓰는 방식), `calibrate.py`(악기별 피크 임계값 스윕), `weak_order.py`·`weak_set.py`(타이밍 없이 순서만 쓰는 CTC 학습)도 이 폴더에 있다. 배포 모델은 여전히 v2다.

## 한계

- 하이햇과 심벌은 정답을 적는 채널이 적어서 사실상 원래 모델 출력에 묶여 있다.
- 캐시된 활성값은 원래 모델 것이다. 배포 모델을 수집한 곡으로 채점하면 자기 학습 데이터로 채점하는 셈이라, 수집 곡 점수는 fold 모델로만 낸다.
- 정답이 채보 채널에서 오기 때문에 채널마다 빼먹는 악기와 표기 습관이 그대로 섞여 있다.

자세한 실험 기록은 로컬에만 두고 공개하지 않는다.
