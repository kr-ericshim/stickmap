#!/usr/bin/env python3
"""groovemap WSL 작업 현황 TUI. sync.sh push가 ~/.local/bin/gm-status로 설치한다.

Usage: gm-status [--once] [-n SECONDS] [--all] [--queue-log FILE]
터미널에서는 실시간 화면(q 종료, r 새로고침, a 모든 태그, ↑↓ 스크롤)을 띄우고,
파이프나 `ssh host gm-status`처럼 터미널이 아니면 한 번만 출력한다.
표준 라이브러리만 쓴다(.venv 불필요). 저장소 위치: ~/$GPU_REMOTE_DIR (기본 groovemap).
"""

from __future__ import annotations

import argparse
import locale
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path.home() / os.environ.get("GPU_REMOTE_DIR", "groovemap")
SCORERS = ("prepare_score", "fill_score", "fill_seq", "global_score")
SCORER_COL = {"prepare_score": "prepare", "fill_score": "covers", "fill_seq": "solos", "global_score": "global"}
SPIN = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
LAUNCHER_VALUE_OPTS = {"--jobs", "--parallel", "--only", "--queue-log"}


# ---------------------------------------------------------------- data

@dataclass
class Proc:
    pid: int
    secs: int
    argv: list[str]


@dataclass
class Tag:
    name: str
    order: float = float("inf")
    folds: dict[int, str] = field(default_factory=dict)      # fold -> ok/bad/run
    fold_pct: dict[int, int] = field(default_factory=dict)
    predict: str = "none"
    score: dict[str, str] = field(default_factory=dict)      # prepare/covers/solos/global -> state
    global_done: int = 0
    scoring_secs: int = 0
    scored_at: str = ""
    trained_at: str = ""
    queued_train: bool = False
    queued_score: bool = False

    @property
    def training(self) -> bool:
        return "run" in self.folds.values()

    @property
    def predicting(self) -> bool:
        return self.predict == "run"

    @property
    def scoring(self) -> bool:
        return "run" in self.score.values()

    @property
    def failed(self) -> bool:
        return "bad" in self.folds.values() or self.predict == "bad" or "bad" in self.score.values()

    @property
    def trained(self) -> bool:
        return self.predict == "ok"


@dataclass
class Snapshot:
    taken: float
    gpu: tuple[float, float, float, float] | None
    load: float
    cpus: int
    tags: list[Tag]
    global_total: int
    other: list[tuple[str, str, int, int]]      # script, output, count, max secs
    events: list[tuple[str, str, str]]          # time, tag, kind


def opt(argv: list[str], name: str) -> str | None:
    for i, word in enumerate(argv):
        if word == name and i + 1 < len(argv):
            return argv[i + 1]
        if word.startswith(name + "="):
            return word.split("=", 1)[1]
    return None


def processes() -> list[Proc]:
    try:
        out = subprocess.run(["ps", "-eo", "pid=,etimes=,args="], capture_output=True, text=True,
                             timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    procs = []
    for line in out.splitlines():
        parts = line.split(None, 2)
        if len(parts) == 3 and parts[0].isdigit() and parts[1].isdigit():
            procs.append(Proc(int(parts[0]), int(parts[1]), parts[2].split()))
    return procs


def script_of(argv: list[str]) -> str:
    for word in argv[:3]:
        if word.endswith((".py", ".sh")):
            return word
    return ""


def env_of(pid: int, key: str) -> str | None:
    try:
        for item in Path(f"/proc/{pid}/environ").read_bytes().split(b"\0"):
            if item.startswith(key.encode() + b"="):
                return item.split(b"=", 1)[1].decode()
    except OSError:
        pass
    return None


def gpu_stats() -> tuple[float, float, float, float] | None:
    exe = "/usr/lib/wsl/lib/nvidia-smi"
    if not Path(exe).exists():
        exe = shutil.which("nvidia-smi") or ""
    if not exe:
        return None
    try:
        out = subprocess.run([exe, "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout
        util, used, total, temp = (float(x) for x in out.splitlines()[0].split(","))
        return util, used, total, temp
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def read_text(path: Path) -> str:
    try:
        return path.read_text(errors="replace")
    except OSError:
        return ""


def last_step(text: str) -> int | None:
    steps = re.findall(r'"step"\s*:\s*(\d+)', text)
    return int(steps[-1]) if steps else None


def short_time(stamp: str) -> str:
    match = re.search(r"(\d{2}):(\d{2})(?::(\d{2}))?", stamp)
    return f"{match.group(1)}:{match.group(2)}" if match else ""


def collect(queue_log: str) -> Snapshot:
    procs = processes()
    tags: dict[str, Tag] = {}

    def tag(name: str) -> Tag:
        return tags.setdefault(name, Tag(name))

    # Training / prediction logs (current launcher: logs/gpu, older ad-hoc runs: logs/).
    for logdir in (ROOT / "logs" / "gpu", ROOT / "logs"):
        for log in logdir.glob("*.log") if logdir.is_dir() else ():
            match = re.fullmatch(r"(.+)-(fold(\d)|predict)\.log", log.name)
            if not match or match.group(1) in tags and logdir.name == "logs" and match.group(3) is not None \
                    and int(match.group(3)) in tags[match.group(1)].folds:
                continue
            t = tag(match.group(1))
            mtime = log.stat().st_mtime
            text = read_text(log)
            if match.group(3) is not None:
                t.order = min(t.order, mtime)
                t.folds[int(match.group(3))] = "bad" if "Traceback" in text else "ok"
            else:
                done_folds = len(re.findall(rf"^{re.escape(t.name)}-fold\d \d+", text, re.M))
                t.predict = "bad" if "Traceback" in text else ("ok" if done_folds >= 3 else t.predict)

    # Queue logs: "TAG done [time]" from training, "TAG scored <time>" from scoring.
    events = []
    for line in read_text(ROOT / queue_log).splitlines():
        words = line.split()
        if len(words) >= 2 and words[1] == "done" and words[0] != "ALL":
            t = tag(words[0])
            t.predict = "ok"
            t.trained_at = short_time(line)
            events.append((t.trained_at, t.name, "학습·예측 완료"))
    scores_root = ROOT / "scores"
    for line in read_text(scores_root / "done.txt").splitlines():
        words = line.split()
        if len(words) >= 2 and words[1] == "scored":
            t = tag(words[0])
            t.scored_at = short_time(line)
            events.append((t.scored_at, t.name, "채점 완료"))

    # Running processes.
    other: dict[tuple[str, str], list[int]] = {}
    scorer_procs = []
    for p in procs:
        script = script_of(p.argv)
        name = Path(script).stem
        if script.endswith("finetune/train.py"):
            t = tag(opt(p.argv, "--tag") or "v1")
            fold = opt(p.argv, "--fold") or "0"
            if fold.isdigit():
                f = int(fold)
                steps = int(opt(p.argv, "--steps") or 3000)
                step = 0
                for logdir in (ROOT / "logs" / "gpu", ROOT / "logs"):
                    found = last_step(read_text(logdir / f"{t.name}-fold{f}.log"))
                    if found is not None:
                        step = found
                        break
                t.folds[f] = "run"
                t.fold_pct[f] = min(99, round(100 * step / max(steps, 1)))
                t.order = min(t.order, time.time() - p.secs)
        elif name == "gpu_predict" or script.endswith("gpu/predict.py"):
            for word in p.argv[p.argv.index(script) + 1:]:
                if word.startswith("-"):
                    break
                tag(word).predict = "run"
        elif script.endswith("gpu/train_tags.py"):
            exp = next((w for w in reversed(p.argv) if not w.startswith("-") and w != script), "")
            path = Path(exp) if Path(exp).is_absolute() else ROOT / exp
            for line in read_text(path).splitlines():
                words = line.split("#", 1)[0].split()
                if words:
                    tag(words[0]).queued_train = True
        elif name in ("score_wsl", "wsl_score2", "score_tags"):
            rest = p.argv[p.argv.index(script) + 1:]
            skip = False
            for word in rest:
                if skip:
                    skip = False
                elif word.split("=", 1)[0] in LAUNCHER_VALUE_OPTS and "=" not in word:
                    skip = True
                elif not word.startswith("-"):
                    tag(word).queued_score = True
        elif name in SCORERS and "python" in Path(p.argv[0]).name:
            scorer_procs.append((p, name))

    for p, name in scorer_procs:
        out = opt(p.argv, "--json") or ""
        owner = opt(p.argv, "--tag") or env_of(p.pid, "FT_TAG")
        match = re.search(r"(?:^|/)scores/([^/]+)/", out) or re.search(r"results/gpu/([^/]+)/", out)
        owner = owner or (match.group(1) if match else None)
        if owner and (owner in tags or (scores_root / owner).is_dir()):
            t = tag(owner)
            t.score[SCORER_COL[name]] = "run"
            t.scoring_secs = max(t.scoring_secs, p.secs)
        else:
            shown = re.sub(r"^\.eval-cache/", "", out) or " ".join(p.argv[1:3])
            other.setdefault((name, shown), []).append(p.secs)

    # Score folders: scores/TAG (WSL scripts) or .eval-cache/ft/results/gpu/TAG (score_tags.sh).
    global_total = 0
    for gdir in scores_root.glob("*/global"):
        global_total = max(global_total, len(list(gdir.glob("*.json"))))
    for t in tags.values():
        for base in (scores_root / t.name, ROOT / ".eval-cache" / "ft" / "results" / "gpu" / t.name):
            if not base.is_dir():
                continue
            logs = {k: base / f"{k}.log" for k in ("prepare", "covers", "solos", "global")}
            started = [k for k in ("covers", "solos", "global") if logs[k].exists() or (base / k).is_dir()]
            for col in ("prepare", "covers", "solos", "global"):
                if t.score.get(col) == "run":
                    continue
                if col == "prepare":
                    if started or (base / "covers.json").exists():
                        t.score[col] = "ok"
                    elif logs[col].exists():
                        t.score[col] = "bad"
                    continue
                if (base / f"{col}.json").exists():
                    t.score[col] = "ok"
                elif col == "global" and (base / "global").is_dir() and t.scored_at:
                    t.score[col] = "ok"
                elif logs[col].exists() or (col == "global" and (base / "global").is_dir()):
                    t.score[col] = "bad" if not t.scoring and not t.queued_score else "wait"
            if (base / "global").is_dir():
                t.global_done = len(list((base / "global").glob("*.json")))
            break
        if t.scored_at:
            for col in ("prepare", "covers", "solos", "global"):
                t.score.setdefault(col, "ok")

    for t in tags.values():
        if t.order == float("inf") and (t.queued_train or t.queued_score or t.training or t.scoring):
            t.order = time.time() + 1e6       # queued with no logs yet: bottom
    # Tags launched together share a 15-minute bucket, then sort by name so live rows don't jump.
    ordered = sorted(tags.values(), key=lambda t: (t.order // 900, t.name))

    gpu = gpu_stats()
    try:
        load = float(read_text(Path("/proc/loadavg")).split()[0])
    except (ValueError, IndexError):
        load = os.getloadavg()[0]
    events.sort(key=lambda e: e[0])
    others = sorted(((s, o, len(v), max(v)) for (s, o), v in other.items()), key=lambda x: -x[3])
    return Snapshot(time.time(), gpu, load, os.cpu_count() or 1, ordered, global_total, others, events)


# ---------------------------------------------------------------- layout
# A line is a list of (text, style) segments; styles: title head ok run bad wait dim key.

def width(text: str) -> int:
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 0 if unicodedata.combining(c) else 1
               for c in text)


def pad(text: str, cols: int, right: bool = False) -> str:
    gap = " " * max(0, cols - width(text))
    return gap + text if right else text + gap


def clip(text: str, cols: int) -> str:
    out, used = "", 0
    for c in text:
        w = width(c)
        if used + w > cols:
            break
        out, used = out + c, used + w
    return out


def bar(frac: float, cols: int) -> str:
    filled = round(max(0.0, min(1.0, frac)) * cols)
    return "█" * filled + "░" * (cols - filled)


def level(frac: float) -> str:
    return "bad" if frac >= 0.9 else "run" if frac >= 0.6 else "ok"


def dur(secs: int) -> str:
    if secs < 60:
        return f"{secs}초"
    if secs < 3600:
        return f"{secs // 60}분"
    return f"{secs // 3600}시간 {secs % 3600 // 60}분"


def cell(state: str, frame: int, text: str = "") -> tuple[str, str]:
    if state == "ok":
        return text or "✓", "ok"
    if state == "run":
        return (SPIN[frame % len(SPIN)] + (" " + text if text else "")), "run"
    if state == "bad":
        return "✗", "bad"
    if state == "wait":
        return "…", "wait"
    return "-", "dim"


def tag_status(t: Tag) -> tuple[str, str]:
    if t.scoring:
        return f"채점 중 · {dur(t.scoring_secs)}째", "run"
    if t.training:
        f = max(k for k, v in t.folds.items() if v == "run")
        return f"학습 중 · fold{f} {t.fold_pct.get(f, 0)}%", "run"
    if t.predicting:
        return "예측 중", "run"
    if t.failed:
        return "실패 · 로그 확인", "bad"
    if t.scored_at or (t.score and all(t.score.get(c) == "ok" for c in ("prepare", "covers", "solos", "global"))):
        return "채점 완료" + (f" {t.scored_at}" if t.scored_at else ""), "ok"
    if t.trained and t.queued_score:
        return "채점 대기", "wait"
    if t.trained:
        return "학습 완료" + (f" {t.trained_at}" if t.trained_at else "") + " · 채점 안 함", "dim"
    if t.queued_train:
        return "학습 대기", "wait"
    if t.folds and all(v == "ok" for v in t.folds.values()):
        return "학습 끝 · 예측 안 됨", "dim"
    return "멈춤?", "bad"


def render(s: Snapshot, frame: int, cols: int, show_all: bool, interval: float, live: bool) -> list[list[tuple[str, str]]]:
    lines: list[list[tuple[str, str]]] = []
    host = socket.gethostname()
    stamp = time.strftime("%H:%M:%S", time.localtime(s.taken))
    left, right = " groovemap 작업 현황", f"{host} · {stamp} "
    lines.append([(left + " " * max(1, cols - width(left) - width(right)) + right, "title")])

    active = [t for t in s.tags if t.training or t.predicting or t.scoring]
    waiting = [t for t in s.tags if not (t.training or t.predicting or t.scoring)
               and ((t.queued_score and t.trained and not t.scored_at) or (t.queued_train and not t.folds))]
    now: list[tuple[str, str]] = [(" 지금  ", "head")]
    if active:
        for i, t in enumerate(active):
            text, style = tag_status(t)
            now += [(", " if i else "", "dim"), (t.name, "key"), (" " + text.split(" · ")[0], style)]
    else:
        now.append(("실행 중인 학습·채점 없음", "dim"))
    if waiting:
        now += [("   다음 ", "dim"), (", ".join(t.name for t in waiting), "wait")]
    lines += [[], now, []]

    bw = 20
    if s.gpu:
        util, used, total, temp = s.gpu
        lines.append([(pad(" GPU", 8), "head"), (bar(util / 100, bw), level(util / 100)), (f" {util:3.0f}%", "")])
        lines.append([(pad(" 메모리", 8), "head"), (bar(used / total, bw), level(used / total)),
                      (f" {used / 1024:4.1f} / {total / 1024:.1f} GiB   온도 {temp:.0f}°C", "")])
    else:
        lines.append([(pad(" GPU", 8), "head"), ("정보 없음 (nvidia-smi 실패)", "dim")])
    frac = s.load / s.cpus
    lines.append([(pad(" CPU", 8), "head"), (bar(frac, bw), level(frac)),
                  (f" 부하 {s.load:.1f} / {s.cpus}코어", "")])
    lines.append([])

    cols_spec = [("태그", 14, False), ("fold0", 7, False), ("fold1", 7, False), ("fold2", 7, False),
                 ("예측", 6, False), ("준비", 6, False), ("커버", 6, False), ("솔로", 6, False),
                 ("전역", 9, False), ("상태", 0, False)]
    header = [(" " + "".join(pad(name, w) if w else name for name, w, _ in cols_spec), "head")]
    group = " " + pad("", 14) + pad("── 학습 ──────────", 21) + pad("", 6) + "── 채점 ───────────────────"
    lines += [[(group, "dim")], header]

    shown = s.tags
    if not show_all:
        recent = {t.name for t in s.tags[-8:]}
        keep = {t.name for t in s.tags if t.training or t.predicting or t.scoring or t.queued_train or t.queued_score}
        shown = [t for t in s.tags if t.name in recent or t.name in keep]
    hidden = len(s.tags) - len(shown)
    if not shown:
        lines.append([(" 아직 기록된 태그가 없음", "dim")])
    for t in shown:
        row: list[tuple[str, str]] = [(" " + pad(clip(t.name, 13), 14), "key")]
        for f in range(3):
            state = t.folds.get(f, "wait" if t.queued_train else "none")
            text, style = cell(state, frame, f"{t.fold_pct[f]}%" if state == "run" else "")
            row.append((pad(text, 7), style))
        text, style = cell(t.predict if t.predict != "none" else ("wait" if t.training else "none"), frame)
        row.append((pad(text, 6), style))
        for col in ("prepare", "covers", "solos"):
            state = t.score.get(col, "wait" if t.queued_score else "none")
            text, style = cell(state, frame)
            row.append((pad(text, 6), style))
        gstate = t.score.get("global", "wait" if t.queued_score else "none")
        gtext = ""
        if gstate == "run" or (gstate == "wait" and t.global_done):
            gtext = f"{t.global_done}/{s.global_total}" if s.global_total else f"{t.global_done}개"
        text, style = cell(gstate, frame, gtext)
        if gstate == "wait" and gtext:
            text = "… " + gtext
        row.append((pad(text, 9), style))
        row.append(tag_status(t))
        lines.append(row)
    if hidden:
        lines.append([(f" 이전 태그 {hidden}개 숨김" + (" · a 키로 모두 보기" if live else " · --all로 모두 보기"), "dim")])

    if s.other:
        lines += [[], [(" 다른 작업", "head")]]
        for script, out, count, secs in s.other[:6]:
            lines.append([(f"  {script}", "key"), (f" ×{count}" if count > 1 else "", "dim"),
                          (f"  {out}", ""), (f"  {dur(secs)}째", "run")])

    if s.events:
        lines += [[], [(" 최근 기록", "head")]]
        for at, name, kind in list(reversed(s.events))[:5]:
            lines.append([(f"  {at or '--:--'}  ", "dim"), (name, "key"), (f"  {kind}", "ok" if "채점" in kind else "")])

    if live:
        lines += [[], [(" q", "key"), (" 종료   ", "dim"), ("r", "key"), (" 새로고침   ", "dim"),
                       ("a", "key"), (" 최근만/모든 태그   ", "dim"), ("↑↓", "key"), (" 스크롤   ", "dim"),
                       (f"{interval:g}초마다 갱신", "dim")]]
    return lines


# ---------------------------------------------------------------- output

ANSI = {"title": "\033[1;7m", "head": "\033[1;36m", "ok": "\033[32m", "run": "\033[33m", "bad": "\033[1;31m",
        "wait": "\033[34m", "dim": "\033[2m", "key": "\033[1m", "": ""}


def print_once(s: Snapshot, show_all: bool, color: bool) -> None:
    cols = min(shutil.get_terminal_size((100, 40)).columns, 120)
    for line in render(s, 0, cols, show_all, 0, live=False):
        out = "".join((ANSI[st] + tx + "\033[0m") if color and ANSI[st] else tx for tx, st in line)
        print(out.rstrip())


def run_tui(queue_log: str, interval: float, show_all: bool) -> None:
    import curses

    def main(scr) -> None:
        nonlocal show_all
        curses.curs_set(0)
        curses.use_default_colors()
        palette = {"ok": curses.COLOR_GREEN, "run": curses.COLOR_YELLOW, "bad": curses.COLOR_RED,
                   "head": curses.COLOR_CYAN, "wait": curses.COLOR_BLUE}
        styles = {"": curses.A_NORMAL, "dim": curses.A_DIM, "key": curses.A_BOLD,
                  "title": curses.A_REVERSE | curses.A_BOLD}
        for i, (name, color) in enumerate(palette.items(), 1):
            curses.init_pair(i, color, -1)
            styles[name] = curses.color_pair(i) | (curses.A_BOLD if name in ("head", "bad") else 0)
        scr.timeout(250)
        snap, fetched, frame, offset = None, 0.0, 0, 0
        while True:
            if snap is None or time.time() - fetched >= interval:
                snap, fetched = collect(queue_log), time.time()
            rows, cols = scr.getmaxyx()
            lines = render(snap, frame, cols, show_all, interval, live=True)
            footer = lines[-1]
            body = lines[:-2]
            offset = max(0, min(offset, len(body) - (rows - 2)))
            scr.erase()
            for y, line in enumerate(body[offset:offset + rows - 2]):
                draw(scr, y, line, cols, styles)
            draw(scr, rows - 1, footer, cols, styles)
            scr.refresh()
            frame += 1
            key = scr.getch()
            if key in (ord("q"), ord("Q"), 27):
                return
            if key in (ord("r"), ord("R")):
                snap = None
            elif key in (ord("a"), ord("A")):
                show_all, offset = not show_all, 0
            elif key in (curses.KEY_DOWN, ord("j")):
                offset += 1
            elif key in (curses.KEY_UP, ord("k")):
                offset = max(0, offset - 1)
            elif key == curses.KEY_NPAGE:
                offset += rows - 3
            elif key == curses.KEY_PPAGE:
                offset = max(0, offset - (rows - 3))

    def draw(scr, y: int, line, cols: int, styles) -> None:
        x = 0
        for text, style in line:
            room = cols - x - (1 if y == scr.getmaxyx()[0] - 1 else 0)
            if room <= 0:
                break
            text = clip(text, room)
            try:
                scr.addstr(y, x, text, styles.get(style, 0))
            except Exception:
                pass
            x += width(text)

    curses.wrapper(main)


def use_utf8() -> None:
    for name in ("", "C.UTF-8", "C.utf8", "en_US.UTF-8"):
        try:
            locale.setlocale(locale.LC_ALL, name)
        except locale.Error:
            continue
        if "utf" in (locale.getlocale()[1] or "").lower().replace("-", ""):
            return
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(prog="gm-status", description="groovemap WSL 학습·예측·채점 현황")
    parser.add_argument("--once", action="store_true", help="실시간 화면 대신 한 번만 출력")
    parser.add_argument("-n", "--interval", type=float, default=5.0, help="갱신 간격(초, 기본 5)")
    parser.add_argument("--all", action="store_true", help="이전 태그까지 모두 표시")
    parser.add_argument("--queue-log", default="logs_run.out", help="학습 queue log (기본 logs_run.out)")
    parser.add_argument("--color", choices=("auto", "always", "never"), default="auto",
                        help="--once 출력 색상")
    args = parser.parse_args()
    if not ROOT.is_dir():
        sys.exit(f"저장소를 찾을 수 없음: {ROOT} (GPU_REMOTE_DIR 확인)")
    use_utf8()
    tty = sys.stdout.isatty() and sys.stdin.isatty()
    if args.once or not tty:
        color = args.color == "always" or (args.color == "auto" and sys.stdout.isatty()
                                           and "NO_COLOR" not in os.environ)
        print_once(collect(args.queue_log), args.all, color)
        return
    if not os.environ.get("TERM") or os.environ.get("TERM") == "dumb":
        os.environ["TERM"] = "xterm-256color"
    try:
        run_tui(args.queue_log, max(1.0, args.interval), args.all)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
