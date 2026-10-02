# 원격 GPU에서 학습·추론하기

ADTOF fine-tuning과 곡 전체 추론은 NVIDIA GPU가 있는 별도 머신(Windows의 WSL2 Ubuntu)에서 한다. Mac은 캐시 준비, 전송, 결과 비교를 맡는다. 같은 규모의 학습이 Mac(MPS·CPU)에서는 수십 분 걸리고 GPU에서는 몇 분이면 끝나기 때문이다. 채점은 CPU 작업이라 GPU를 쓰지 않으며, 원격 머신의 CPU에서 돌리거나 예측만 받아 Mac에서 돌린다.

`tools/finetune`의 학습·검출 방식은 그대로 두고, [`tools/gpu`](../tools/gpu)가 실행 위치와 입출력만 나눈다.

## 원격 머신 준비

SSH로 접속할 수 있는 Linux 머신이면 된다. 필요한 것은 다음과 같다.

- 원격 home 아래의 저장소 복사본 `~/$GPU_REMOTE_DIR` (기본 `groovemap`). git checkout이 아니라 `sync.sh push`로 보낸 rsync 복사본이다.
- 그 안의 `.venv`에 CUDA용 PyTorch, `numpy`, `scipy`, `adtof-pytorch`, 채점용 `librosa`, `soundfile`, `pretty_midi`, `mido`, `typer`, `rich`.
- 이 패키지는 `--no-deps`로 editable 설치한다. CUDA venv의 의존성을 다시 풀지 않기 위해서다.

```bash
cd ~/groovemap
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python numpy scipy librosa soundfile pretty_midi mido typer rich \
  'adtof-pytorch @ git+https://github.com/xavriley/ADTOF-pytorch.git'
uv pip install --python .venv/bin/python --no-deps -e .
```

CUDA wheel 버전은 GPU 세대에 맞춘다(최신 세대는 `cu128` 이상). 원격에서는 `uv sync`나 `uv run`을 쓰지 않는다. 준비된 CUDA 환경이 Mac용 lock 기준으로 바뀐다. WSL에서는 `nvidia-smi`가 `/usr/lib/wsl/lib/`에 있고 기본 `PATH`에는 없을 수 있다.

## 접속 대상 지정

도구는 환경 변수 `GPU_HOST`(기본 `gpu-host`)로 접속하고, 원격 저장소 위치는 `GPU_REMOTE_DIR`(기본 `groovemap`, 원격 home 기준 상대 경로)로 정한다. 보통은 `~/.ssh/config`에 별칭을 하나 두면 된다.

```
Host gpu-host
    HostName <원격 머신 주소>
    User <원격 사용자>
```

다른 머신을 쓸 때는 `GPU_HOST=my-box bash tools/gpu/sync.sh push`처럼 지정한다. 두 값에는 공백이나 shell metacharacter를 넣지 않는다. 비대화형 실행이라 `ssh -o BatchMode=yes`로 접속하므로 키 인증이 되어 있어야 한다.

## 주요 명령

```bash
# Mac: 코드와 FT 캐시를 보내고 원격에 gm-status를 설치한다.
bash tools/gpu/sync.sh push
# truth/의 오디오·정답이 필요할 때만 함께 보낸다(영상·이미지는 제외).
bash tools/gpu/sync.sh push --truth

# 원격: 실험 목록(한 줄에 "<tag> <train.py 인자>")을 CUDA로 학습하고 추론한다.
ssh gpu-host 'cd ~/groovemap && setsid nohup .venv/bin/python -u tools/gpu/train_tags.py \
  --jobs 3 .eval-cache/ft/gpu-experiments.txt > logs_run.out 2>&1 < /dev/null &' < /dev/null

# 원격: 진행 상황(학습 log, GPU·CPU 사용, 채점 상태)을 본다.
ssh gpu-host '~/.local/bin/gm-status --once'

# 원격: 예측이 끝난 tag를 CPU로 채점한다.
bash tools/gpu/score_wsl.sh --only CH1,CH2 --queue-log logs_run.out TAG1 TAG2

# Mac: 채점이 끝난 tag의 점수 JSON만 받는다.
bash tools/gpu/sync.sh pull TAG1 TAG2
# Mac에서 채점할 때는 예측과 checkpoint를 받아 score_tags.sh를 쓴다.
bash tools/gpu/sync.sh pull --predictions TAG1
bash tools/gpu/score_tags.sh --only CH1,CH2 --jobs 4 TAG1
```

`train_tags.py`는 tag마다 fold 0, 1, 2를 순서대로 학습하고, 성공하면 held-out 곡과 외부 solo를 추론한다. `--no-predict`로 학습만 하고 나중에 `tools/gpu/predict.py`로 추론할 수도 있다. 로그는 원격의 `logs/gpu/<tag>-fold{k}.log`에 남는다. tag에는 `-`를 넣지 않는다(`[A-Za-z0-9_]+`만 허용).

채보 영상 수집(fetch, 악보 판독, compact)을 원격에서 돌리는 entry point는 [`tools/gpu/harvest_wsl.sh`](../tools/gpu/harvest_wsl.sh)다. 처음에는 `push --init`으로 이관하고, 이후 `fetch`, `read`, `compact`, `status`, `pull` 순서로 쓴다. 각 하위 명령의 인자는 `--help`로 확인한다.

## 주의할 점

- 오래 걸리는 작업은 `setsid nohup ... > log 2>&1 < /dev/null &`로 분리한다. `nohup cmd &`만 쓰면 세 스트림 중 하나가 SSH에 묶여 클라이언트가 끝나지 않는다.
- 동시에 학습하는 tag는 `--jobs 1..3`이다. 원격 채점은 tag당 worker 1..12, 준비 단계는 최대 4개를 쓴다. Mac 채점은 worker 4개 이하로 둔다.
- 채점 도구는 BLAS/OpenMP thread를 1로 고정하고 `FT_*_SCALE` 환경값을 지운 채 비교한다.
- `sync.sh`는 `--delete`를 쓰지 않아 삭제가 전파되지 않고, Mac의 `.venv`와 배포 가중치는 보내지 않는다.
- recipe나 cohort가 다른 실험에는 새 tag를 쓴다. 같은 tag를 재사용하면 옛 fold 예측이 채점에 섞일 수 있다.

장비별 설정과 실측 기록을 담은 상세 runbook은 공개 저장소에 두지 않고 로컬에만 보관한다.
