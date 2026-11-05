# vision_read: OpenCV와 비전 모델을 섞은 악보 읽기

`tools/truth_video`의 규칙 기반 리더는 머리 위치와 모양은 잘 찾지만 음표 길이는 읽지 않는다. 가로 위치와 커서로 시간을 잡기 때문에, 잇단음과 붙임줄이 섞인 빽빽한 솔로처럼 음표 간격이 리듬과 맞지 않는 악보에서는 한계가 있다. 반대로 GPT/Gemini 비전 모델은 박과 음표 길이를 읽을 수 있지만 좌표가 거칠고 악기 판단을 믿기 어렵다. 그래서 두 쪽이 잘하는 일을 나눠 맡기는 하이브리드 리더를 만들었다.

## 동작 방식

1. OpenCV 쪽은 `truth_video`의 오선, 머리, 범례 코드를 그대로 불러 쓴다(`hybrid.py`, `native_cv.py`). 원본 해상도 프레임에서 오선과 머리를 찾고, 머리마다 위치, 모양, 채널 범례상의 악기를 정한다.
2. 원본 이미지와 함께, 머리마다 파란 번호를 붙이고 마디선 후보에 주황색 B 표시를 한 이미지를 만든다. 마디선 후보에는 음표 기둥이 섞여 있을 수 있어서, 진짜 마디는 모델이 악보를 보고 고른다.
3. 비전 모델은 두 이미지를 한 번씩만 보고, 번호 붙은 머리 하나하나를 진짜 머리, 오검출, 불확실 중 하나로 판정한다. 진짜 머리에는 마디, 박 안의 시작 위치와 길이(온음표 단위 유리수, 예: 16분음표 = `1/16`), 성부, 붙임줄, 잇단음 묶음, 고스트, 악센트를 적는다. 출력은 고정된 JSON 스키마로 받고 검증을 통과하지 못하면 버린다.
4. 합칠 때 악기, 위치, 모양, 좌표는 CV 결과만 쓴다. 모델이 악기를 바꿔 말해도 무시하고, CV가 놓친 머리를 모델이 추가하면 악기를 비워 둔 채 미확인으로 남긴다.

모델 호출은 두 가지 경로가 있다. `hybrid_run.py`는 `agy` CLI로 Gemini를 부르고, `hybrid_gpt.py`는 `codex exec`로 GPT에 같은 이미지와 프롬프트를 보낸다. 둘 다 작업 폴더별로 원문 응답과 로그를 남기고, `--resume`은 입력과 프롬프트 지문이 같은 성공 작업만 재사용한다. 요청 한도(429)에 걸리면 진행 중인 호출을 모두 멈추고, CLI가 알려 준 리셋 시각 뒤에 이어서 돌린다(`deferred.py`, `solos_run.py`).

읽은 리듬은 아직 시간 정답이 아니다. `export_timing.py`와 `auto_timing.py`가 마디 위치와 박을 영상 시간 축으로 옮겨 `truth_video` 형식의 `strip.json`을 만들고, 그 뒤에는 `truth_video/align.py`의 타격 정렬, held-out 검사, 킥/스네어 대역 검사를 기준 그대로 통과해야 정답이 된다.

## 주요 스크립트

벤치마크 시스템으로 작업 폴더를 준비하고 CV 결과를 만든다.

```bash
.venv/bin/python -B tools/vision_read/hybrid.py --selection selected
```

Gemini로 리듬을 읽고 채점한다.

```bash
.venv/bin/python -B tools/vision_read/hybrid_run.py --run-name hybrid-medium-look --workers 4 --timeout 900 --resume
.venv/bin/python -B tools/vision_read/score.py --run hybrid-medium-look
```

같은 작업을 GPT로 돌리고, 이전 결과와 비교한다.

```bash
.venv/bin/python -B tools/vision_read/hybrid_gpt.py --run-name hybrid-gpt-xhigh --effort xhigh --workers 6 --resume
.venv/bin/python -B tools/vision_read/hybrid_gpt_compare.py --runs hybrid-gpt-xhigh
```

CV 없이 모델 혼자 읽는 기준선은 `run.py`, 이미 받아 둔 채널 영상을 대표 프레임으로 읽는 것은 `channel_read.py`, 빽빽한 솔로 영상 전체를 처리하는 것은 `solos_extract.py`부터 `solos_gate.py`까지의 스크립트다. 모델 입력과 출력, 점수, 시간 후보는 모두 `.eval-cache/vision_pilot/` 아래에 쓴다.

```bash
.venv/bin/python -B -m unittest discover -s tools/vision_read -p 'test_*.py'
```

테스트는 모델 호출이나 오디오 없이 돈다.

## 채점과 한계

- 정답 세트가 작다. 직접 맞춰 본 21개 보표, 330개 머리이고, 그중 보이는 음표를 빠짐없이 적은 보표는 6개(159개 머리)뿐이다. 내가 혼자 만든 부분 정답이라 전문가 채보처럼 독립적인 기준은 아니다.
- 머리 매칭은 좌표만으로 한다. 모델 좌표가 거칠면 리듬을 맞게 읽어도 점수가 낮게 나올 수 있어서, 완전한 6개 보표에는 좌표 없이 (마디, 위치, 모양, 붙임줄) 목록을 비교하는 점수를 따로 둔다.
- 요청 한도나 타임아웃으로 결과가 없는 작업은 정답을 놓친 것으로 세고, 모델 정확도 0으로 보고하지는 않는다.
- "이미지를 한 번만 본다", "다른 도구를 쓰지 않는다"는 프롬프트 규칙일 뿐 샌드박스로 막은 것은 아니다.
- 모델만 찾은 머리의 악기는 범례로 따로 확인하기 전까지 내보내지 않는다.
- 오디오 검사는 시간이 맞는지를 확인할 뿐, 음표를 바르게 읽었다는 독립적인 증거는 아니다. 통과한 보표가 없는 것도 정상적인 결과다.
- 테스트용으로 따로 떼어 둔 영상은 개발용 진입점과 채점에 들어가지 못하게 막아 두었다.

자세한 실험 기록과 모델별 비교 결과는 로컬에만 두고 공개하지 않는다.
