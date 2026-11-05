# truth_video: 채보 영상에서 정답 타격 만들기

유튜브에는 드럼 연주와 함께 악보를 띄워 주는 채보 영상이 많다. 이 폴더의 도구는 그런 영상에서 악보를 읽고, 음원과 맞춰서 "어느 시각에 어떤 드럼을 쳤는지"를 정답 데이터로 만든다. 공개 데이터셋 점수만으로는 실제 드럼 찹(chops) 영상에서의 성능을 알기 어려워서, 평가용 정답을 직접 만들 방법이 필요했다.

## 동작 방식

1. 스캔: 영상 프레임에서 악보 영역과 재생 커서(파란색, 빨간색 등)를 찾고, 보표마다 중간값 이미지를 만든다. 한 화면에 여러 줄이 있는 페이지형, 아래쪽 띠형, 가로 스크롤형 레이아웃을 지원한다(`scan.py`, `harvest_scan.py`, `layouts.py`, `harvest_scroll.py`).
2. 읽기: 오선 다섯 줄과 간격을 재고, 채운 머리, 크로스, 동그라미 크로스, 다이아몬드 머리와 괄호(고스트), 꾸밈음을 찾는다(`heads.py`). 어떤 위치의 어떤 모양이 어떤 악기인지는 채널마다 `legends/*.json`에 따로 적어 두었다. 같은 위치라도 채널마다 표기가 달라서, 직접 확인한 규칙만 `verified`로 표시하고 그것만 내보낸다(`strip.py`).
3. 정렬: 같은 세로줄의 머리를 한 순간으로 묶고, 커서 위치로 대략의 시각을 잡은 뒤, 분리한 드럼 스템에서 뽑은 타격 시각(악기 구분 없음)과 일대일로 맞춘다(`align.py`). 악기 종류는 악보를 믿고, 시각은 오디오가 정한다. 검출기의 악기 라벨은 정렬에 쓰지 않는다.
4. 검증: 맞춘 결과를 그대로 믿지 않고, 일부 순간을 빼고 다시 맞춰서 예측이 맞는지(held-out), 일부러 시간을 어긋나게 했을 때보다 확실히 나은지(shifted control)를 본다. 규칙적인 그루브는 한 박 밀려도 타격과 잘 맞기 때문에, 킥은 저역, 스네어는 중역 온셋과 맞는지를 따로 확인해서 박 단위로 밀린 정렬을 걸러낸다(`bands.py`).

보표 하나(system)가 통과하려면 오선이 온전하고, 퍼커션 음자리표가 보이고, 오선 간격이 원본 픽셀로 8px 이상이고, 순간이 6개 이상이어야 한다. 시간 기준은 아래와 같다.

| 기준 | 20 ms | 40 ms |
|---|---:|---:|
| 맞춘 뒤 타격 일치율 | 85% 이상 | 95% 이상 |
| held-out 예측 일치율 | 55% 이상 | 80% 이상 |
| 커서 없는 영상의 held-out 일치율 | 70% 이상 | 90% 이상 |

held-out 일치율은 shifted control보다 15%p 이상 높아야 한다. 이 기준을 통과한 것이 strict 정답이고, ±50ms 검출기 채점용으로 기준을 조금 낮춘 scoring 정답은 `tiers.py`가 따로 만든다. strict 기준은 건드리지 않는다. 통과한 보표가 하나도 없는 것도 정상적인 결과로 본다.

## 주요 스크립트

곡 하나를 읽는다. 곡 폴더에 `video.mp4`와 `attacks.htdemucs.npy`가 있어야 한다.

```bash
.venv/bin/python tools/truth_video/build.py songs/<slug> truth/<name> --legend tools/truth_video/legends/drumtabs.json
```

수집한 영상 전체를 채널 범례에 따라 일괄로 읽는다. 다운로드나 분리, 검출기 추론은 하지 않는다.

```bash
.venv/bin/python tools/truth_video/batch.py --only atthedrum yeol --limit 20
```

저장된 정답으로 검출기를 채점한다.

```bash
.venv/bin/python tools/truth_video/evaluate.py songs/<slug> truth/<name>/strip.json --reuse-truth
```

새 채널은 검출기 없이 따로 시험해 본다(`pilot_new.py`). `survey`, `prepare`, `notation`, `read` 단계를 차례로 돌리고, 결과는 `.eval-cache/pilot_new/` 아래에만 쓴다.

```bash
.venv/bin/python tools/truth_video/pilot_new.py read --channels ollieshorts --jobs 2
```

결과 폴더에는 `truth.json`, `truth.scoring.json`, `alignment.json`, 보표별 판정 이유를 적은 `QUALITY.md`, 그리고 머리마다 번호와 라벨을 그려 넣은 `check/system_NNN.png`가 생긴다. 탈락한 보표도 이유와 함께 남겨서 직접 열어 볼 수 있게 했다.

## 한계

- 부분 정답이다. 범례에서 확인한 악기만 내보내고, 확인 못 한 심벌은 `cymbal_unknown`으로 남긴 채 시간 정답에서 뺀다. 여기 없는 타격이 "치지 않았다"는 뜻은 아니다.
- 고스트, 악센트, 플램, 롤 표시가 false면 확실히 읽지 못했다는 뜻이다. 그 기법을 쓰지 않았다는 근거로 쓰면 안 된다.
- 대부분의 채널에서 확인한 것은 킥, 스네어, 탐 위치다. 탐 크기와 심벌 종류는 대체로 구분하지 않는다.
- 해상도가 낮아 오선 간격이 8px 미만이면 확대해도 통과시키지 않는다. 360×640 세로 영상은 오선 간격이 4px 정도라 읽지 못했고, 같은 구도의 1080×1920 원본이면 12px 정도 나온다.
- held-out 검사도 같은 연주의 타격을 쓰므로 외부 정답은 아니다. 시간이 맞는다는 근거일 뿐, 음표를 바르게 읽었다는 독립적인 증거는 아니다.

## 테스트

```bash
.venv/bin/python -m pytest tools/truth_video/tests
```

테스트 일부는 다른 사람의 채보 영상에서 캡처한 프레임을 fixture로 쓴다. 이 프레임은 공개 저장소에 넣지 않았으므로, 새로 clone한 상태에서는 fixture를 읽는 테스트가 건너뛰어진다.

자세한 실험 기록과 채널별 결과는 로컬에만 두고 공개하지 않는다.
