# 학습 레시피 비교

`tools/gpu/chops_compare.py`가 쓰는 학습 레시피들이다. 같은 GPU, 같은 fold, 같은 step 수, 같은 seed 두 개로 여러 학습 방식을 나란히 돌려서, 빠른 필 점수가 레시피 때문에 바뀌는지 데이터 때문에 바뀌는지 구분하려고 만들었다. 모든 안은 원래 ADTOF 가중치에서 시작하고, 기본값은 1500 step, 배치 12, 8초 크롭, Adam 1e-4, 증류 0.3, BatchNorm 고정이다. 검증 점수는 로그에만 남기고 체크포인트 선택에는 쓰지 않는다.

| 안 | 학습 데이터 | 레시피 |
|---|---|---|
| A | 원래 14개 채널만 | v2 그대로 (`tools/finetune/train.py`) |
| B | 이후 추가한 채널까지 전부 | v2 그대로 |
| C | B와 같음 | `stratified.py`: 빠른 구간 50%, 기존 방식 30%, 느린 구간 20%로 크롭 중심을 고른다 |
| D | B와 같음 | `three_state.py`: 정답 / 검수된 무음 / 모름 세 가지로 나누고, 모름에는 손실을 주지 않는다 |
| E | A와 같은 정답 + 순서만 아는 구간 | `weakorder.py`: v2에 순서 CTC 보조 손실(lambda 0.05)을 더한다 |

`train_v2frozen.py`와 `train_synth.py`는 예전 v5b 실험(합성 빠른 필을 크롭에 붙여 넣는 방식)을 재현하려고 그때 코드를 그대로 옮겨 둔 것이다. 직접 실행하지 않고 `--arms v5b`로 고른다. `train_recipe.py`가 각 레시피를 감싸서, 봉인된 곡을 특징량이나 정답을 읽기 전에 걸러 내고 계획 파일에 고정된 학습/평가 곡 목록을 넘긴다.

## 실행

GPU 머신에서 돌린다. 준비 과정은 [docs/wsl-gpu.md](../../../docs/wsl-gpu.md)에 있다.

```bash
.venv/bin/python -B tools/gpu/chops_compare.py run --name NAME --dry-run      # 채널 목록과 작업 목록만 출력
.venv/bin/python -B tools/gpu/chops_compare.py run --name NAME --arms A E --folds 2 --steps 1500 --gpu 0
.venv/bin/python -B tools/gpu/chops_compare.py score --name NAME
.venv/bin/python -B tools/gpu/chops_compare.py final --name NAME              # 봉인 세트를 한 번만 연다
```

기본은 A–D × seed 2개 × fold 2개로 16개 작업을 순서대로 학습한다. `run`은 실제 학습 전에 채널 구성, fold, 학습 예산, 코드와 정답 파일 해시를 `plan.json`에 고정한다. 코드나 정답이 바뀌면 같은 이름으로 이어 갈 수 없고 새 계획을 만들어야 한다.

## 판정 기준

- E는 외부 드럼 솔로의 빠른 킥·스네어·탐 macro F1이 두 seed 모두에서 A보다 2점 이상 올라야 가능성이 있다고 본다. 다른 지표는 0.5점 넘게 떨어지면 안 된다.
- D는 학습 fold마다 검수된 무음 구간(`kst_audit.json`)이 있어야 후보가 된다. 없으면 정답만 쓰는 스트레스 테스트로 표시하고 후보에서 뺀다.
- D와 C의 차이는 손실 정책 여러 개가 한꺼번에 바뀐 결과라서, 세 가지 라벨만의 효과로 읽으면 안 된다.

## 한계

- E의 순서 학습 데이터(`tools/finetune/weak_set.py`)는 자동 선별 기준이 엄격해서 비어 있을 수 있다. 그러면 학습 전에 멈춘다.
- D에 필요한 검수 무음 구간은 형식만 지원하고, 이 저장소에 실제 검수 데이터는 없다.

자세한 실험 기록은 로컬에만 두고 공개하지 않는다.
