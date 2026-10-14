"""Read-only audit of canonical, non-vault strict labels (no detector inference).

    .venv/bin/python -B tools/harvest/label_audit.py --check-known
    .venv/bin/python -B tools/harvest/label_audit.py --spot-checks /tmp/checks.json

Only LABEL_AUDIT.md and label_audit.json are written. Rejected strip systems
provide explicitly labelled neighbour context, never additional strict truth.
Detector disagreement and raster geometry/ties are review cues, not new labels.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "harvest"))
sys.path.insert(0, str(ROOT / "tools" / "truth_video"))
import vault  # noqa: E402
from heads import staffs  # noqa: E402
from ties import mark_continuations  # noqa: E402

OUT = ROOT / "truth" / "harvest"
CORE = ("kick", "snare", "tom")
POLICY = {
    "neighbour_radius_systems": 2,
    "neighbour_min_ks": 6,
    "share_min_heads_each": 8,
    "share_total_variation": 0.45,
    "geometry_spacing_relative_tolerance": 0.08,
    "geometry_shift_spaces": [0.75, 1.25],
    "geometry_residual_spaces": 0.15,
    "detector_fps": 100,
    "detector_window_seconds": 0.03,
    "unison_exclusion_seconds": 0.04,
    "detector_own_below": 0.2,
    "detector_other_above": 0.8,
    "detector_cluster_min": 3,
    "detector_cluster_fraction": 0.3,
}
REASONS = {
    "zero_ks_neighbours": "K/S가 0인데 가까운 시스템에는 K/S가 다수 있음",
    "zero_local_ks_phase": "자체 K/S phase count가 0이며 song-level phase에 의존",
    "instrument_share_jump": "인접 시스템의 악기 비중이 급변",
    "staff_line_shift_neighbour": "인접 시스템 대비 오선 좌표가 약 한 줄 이동",
    "staff_line_span_alternative": "같은 crop의 line-span 후보가 약 한 줄 다른 오선을 선택",
    "tied_continuation_onset": "crop에서 타이 연속음으로 검출된 head가 canonical onset에 존재",
    "detector_disagreement_cluster": "기존 detector 캐시의 다른 악기 고확신 불일치 군집",
    "missing_strip_system": "canonical 이벤트에 대응하는 strip 시스템이 없음",
}


def family(inst):
    if inst.startswith("tom"):
        return "tom"
    if inst in ("ride", "ride_bell", "bell", "crash", "cymbal_unknown"):
        return "cymbal"
    return "snare" if inst == "side_stick" else inst


def counts(notes):
    return dict(sorted(Counter(family(n.get("inst", "unknown")) for n in notes).items()))


def ks(c):
    return c.get("kick", 0) + c.get("snare", 0)


def shift_evidence(a, b):
    """Compare five lines, rejecting zoom/spacing changes and nonrigid shifts."""
    x, y = np.asarray(a.get("lines", [])), np.asarray(b.get("lines", []))
    sa, sb = a.get("spacing", 0), b.get("spacing", 0)
    if x.shape != (5,) or y.shape != (5,) or min(sa, sb) <= 0:
        return None
    if abs(sa / sb - 1) > POLICY["geometry_spacing_relative_tolerance"]:
        return None
    delta = (x - y) / ((sa + sb) / 2)
    offset = float(np.median(delta))
    lo, hi = POLICY["geometry_shift_spaces"]
    if lo <= abs(offset) <= hi and np.max(abs(delta - offset)) <= POLICY["geometry_residual_spaces"]:
        return {"shift_spaces": round(offset, 4), "lines": x.tolist(),
                "reference_lines": y.tolist(), "spacing": sa}
    return None


class Inputs:
    """Hash precisely the inputs opened; detect concurrent changes before writing."""
    def __init__(self):
        self.hashes = {}

    def read(self, path):
        path = path.resolve()
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        key = str(path)
        if key in self.hashes and self.hashes[key] != digest:
            raise RuntimeError(f"input changed during audit: {path}")
        self.hashes[key] = digest
        return raw

    def json(self, path):
        return json.loads(self.read(path))

    def verify(self):
        changed = [p for p, h in self.hashes.items()
                   if not Path(p).exists() or hashlib.sha256(Path(p).read_bytes()).hexdigest() != h]
        if changed:
            raise RuntimeError(f"inputs changed during audit; rerun: {changed}")

    def digest(self, path):
        return self.hashes[str(path.resolve())]


def eligible_rows(seal, songs):
    """Check manifest identity AND folder metadata before any truth/cache read."""
    selected, skipped = [], Counter()
    for row in sorted(seal.rows.values(), key=lambda r: r.get("video", "")):
        if row.get("status") not in ("read", "compacted"):
            skipped["other_status"] += 1
            continue
        if seal.contains_song(row):
            skipped["vault"] += 1
            continue
        slug = row.get("slug", "")
        if not slug or Path(slug).name != slug:
            raise ValueError(f"unsafe/missing slug for {row['video']}")
        folder = songs / slug
        if folder.is_symlink() or not folder.is_dir():
            raise ValueError(f"missing or symlink song folder: {folder}")
        if seal.contains_song(folder):  # metadata only; catches sealed aliases
            skipped["vault"] += 1
            continue
        # Do not follow truth/cache symlinks into another (possibly sealed) song.
        if (folder / "truth").is_symlink():
            raise ValueError(f"symlink truth folder: {folder}")
        selected.append(row)
    return selected, dict(skipped)


def safe_file(path, boundary):
    if path.is_symlink() or not path.resolve().is_relative_to(boundary.resolve()):
        raise ValueError(f"unsafe input path: {path}")
    return path


def detector_candidates(events, path, inputs):
    """Original ADTOF columns K/S/T/H/C, 100 fps; exclude core unisons."""
    import io
    act = np.load(io.BytesIO(inputs.read(path)), allow_pickle=False)
    if act.ndim != 2 or act.shape[1] != 5 or not np.isfinite(act).all():
        raise ValueError(f"invalid detector array: {path}, {act.shape}")
    unique = {(round(e["time"], 5), family(e["inst"])): e
              for e in events if family(e["inst"]) in CORE}
    times = {c: np.array(sorted(t for t, k in unique if k == c)) for c in CORE}
    candidates, inspected, in_range = defaultdict(list), 0, 0
    for (t, c), e in sorted(unique.items()):
        if any(np.any(abs(times[k] - t) <= POLICY["unison_exclusion_seconds"]) for k in CORE if k != c):
            continue
        inspected += 1
        centre = round(t * POLICY["detector_fps"])
        if centre < 0 or centre >= len(act):
            continue
        in_range += 1
        radius = round(POLICY["detector_window_seconds"] * POLICY["detector_fps"])
        peak = act[max(0, centre-radius):min(len(act), centre+radius+1), :3].max(axis=0)
        own = CORE.index(c)
        other = max((i for i in range(3) if i != own), key=lambda i: peak[i])
        if peak[own] < POLICY["detector_own_below"] and peak[other] > POLICY["detector_other_above"]:
            candidates[e["system"]].append({"time": e["time"], "inst": c,
                "pos": e.get("pos"), "shape": e.get("shape"), "other": CORE[other],
                "own_peak": float(peak[own]), "other_peak": float(peak[other])})
    return candidates, {"distinct_core": len(unique), "non_unison": inspected,
                        "in_range": in_range, "candidates": sum(map(len, candidates.values()))}


def audit_song(row, inputs):
    folder = OUT / "songs" / row["slug"]
    truth_dir = folder / "truth"
    coverage = Counter()
    problems = []
    def read_json(name):
        return inputs.json(safe_file(truth_dir / name, truth_dir))
    if not (truth_dir / "truth.json").exists():
        return [], {"video": row["video"], "error": "missing canonical truth.json"}, []
    events = read_json("truth.json")
    strip = read_json("strip.json") if (truth_dir / "strip.json").exists() else []
    alignment = read_json("alignment.json") if (truth_dir / "alignment.json").exists() else {}
    if not strip:
        problems.append("missing/empty strip.json")
    if not alignment:
        problems.append("missing/empty alignment.json")
    systems = {s["system"]: s for s in strip}
    aligned = {s["system"]: s for s in alignment.get("systems", [])}
    grouped = defaultdict(list)
    for event in events:
        grouped[event["system"]].append(event)
    order = sorted(systems)
    context = {i: counts(n for n in s["notes"] if n.get("exportable", True)) for i, s in systems.items()}
    detector, cache_stats, cache_path = {}, {}, None
    # Exact allowlisted song cache paths only; no glob over sealed cached truth.
    for name in ("teacher0.npy", "teacher.npy"):
        path = ROOT / ".eval-cache" / "ft" / row["slug"] / name
        safe_file(path, ROOT / ".eval-cache" / "ft" / row["slug"])
        if path.parent.is_symlink():
            raise ValueError(f"symlink cache folder: {path.parent}")
        if path.exists():
            cache_path = str(path.relative_to(ROOT))
            detector, cache_stats = detector_candidates(events, path, inputs)
            break
    flags = []
    for sid, exported in sorted(grouped.items()):
        c = counts(exported)
        item = {"song": row["slug"], "video": row["video"], "title": row.get("title"),
                "channel": row.get("channel"), "system": sid, "counts": c,
                "event_count": len(exported), "reasons": [], "recommended_action": "reread",
                "canonical_sha256": inputs.digest(truth_dir / "truth.json")}
        def flag(code, evidence):
            item["reasons"].append({"code": code, "reason": REASONS[code], "evidence": evidence})
        if sid not in systems:
            flag("missing_strip_system", {})
            flags.append(item)
            continue
        s = systems[sid]
        staff = s.get("staff", {})
        idx = order.index(sid)
        neighbours = [j for j in order[max(0, idx-2):idx+3] if j != sid and abs(j-sid) <= 2]
        neighbour_info = [{"system": j, "counts": context[j],
                           "strict_usable": bool(aligned.get(j, {}).get("usable")),
                           "count_basis": "strip exportable heads, not timed truth"} for j in neighbours]
        if ks(c) == 0 and any(ks(context[j]) >= POLICY["neighbour_min_ks"] for j in neighbours):
            flag("zero_ks_neighbours", {"neighbours": neighbour_info})
        phase = aligned.get(sid, {}).get("phase") or {}
        item["phase_count"] = phase.get("count")
        if phase.get("count") == 0 and (phase.get("song_level") or {}).get("count", 0) >= 6:
            flag("zero_local_ks_phase", {"count": 0, "song_count": phase["song_level"]["count"]})
        adjacent = [j for j in neighbours if abs(j-sid) == 1]
        jumps, shifts = [], []
        own = context[sid]
        for j in adjacent:
            other = context[j]
            if min(sum(own.values()), sum(other.values())) >= POLICY["share_min_heads_each"]:
                tv = sum(abs(own.get(k, 0)/sum(own.values()) - other.get(k, 0)/sum(other.values()))
                         for k in own.keys() | other.keys()) / 2
                if tv >= POLICY["share_total_variation"]:
                    jumps.append({"system": j, "total_variation": round(tv, 4), "counts": other,
                                  "own_strip_counts": own, "strict_usable": bool(aligned.get(j, {}).get("usable"))})
            # Same crop row only; crop translations can still create false positives.
            if s.get("row", 0) == systems[j].get("row", 0):
                shift = shift_evidence(staff, systems[j].get("staff", {}))
                if shift:
                    shifts.append({"system": j, **shift})
        if jumps:
            flag("instrument_share_jump", {"pairs": jumps, "count_basis": "strip exportable heads"})
        if shifts:
            flag("staff_line_shift_neighbour", {"pairs": shifts, "caution": "crop translation is not a label error"})
        crop = safe_file(truth_dir / f"system_{sid:03d}.png", truth_dir)
        if crop.exists():
            gray = cv2.imdecode(np.frombuffer(inputs.read(crop), np.uint8), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                raise ValueError(f"unreadable crop: {crop}")
            item.update(crop=str(crop.relative_to(ROOT)), crop_sha256=inputs.digest(crop))
            coverage["crops"] += 1
            alternatives = []
            for lines, spacing in staffs(gray, prefer_line_span=True):
                shift = shift_evidence(staff, {"lines": lines, "spacing": spacing})
                if shift:
                    alternatives.append(shift)
            if alternatives:
                flag("staff_line_span_alternative", {"candidates": alternatives})
            if len(staff.get("lines", [])) == 5 and staff.get("spacing", 0) > 0:
                # Preserve saved head coordinates; no re-detection, alignment or file writes.
                notes = copy.deepcopy(s["notes"] + s.get("unvisited_heads", []))
                mark_continuations(gray, notes, staff["lines"], staff["spacing"])
                tied = []
                for e in exported:
                    matches = [n for n in notes if n.get("tied_continuation")
                               and n.get("shape") == e.get("shape")
                               and abs(n["x"]-e["x"]) <= 1 and abs(n["y"]-e["y"]) <= 1]
                    if e.get("tied_continuation") or matches:
                        tied.append({"time": e["time"], "inst": e["inst"], "x": e["x"], "y": e["y"],
                                     "tie_evidence": (matches[0] if matches else e).get("tie_evidence")})
                coverage["tie_checked_systems"] += 1
                if tied:
                    flag("tied_continuation_onset", {"events": tied})
        else:
            problems.append(f"system {sid}: missing crop")
        candidates = detector.get(sid, [])
        clusters = defaultdict(list)
        for e in candidates:
            clusters[(e["inst"], e["other"], e["shape"], e["pos"])].append(e)
        strong = []
        for (inst, other, shape, pos), group in clusters.items():
            total = len({round(e["time"], 5) for e in exported if family(e["inst"]) == inst
                         and e.get("shape") == shape and e.get("pos") == pos})
            if len(group) >= POLICY["detector_cluster_min"] and len(group)/max(1, total) >= POLICY["detector_cluster_fraction"]:
                strong.append({"inst": inst, "other": other, "shape": shape, "pos": pos,
                               "count": len(group), "rule_events": total, "events": group})
        if strong:
            flag("detector_disagreement_cluster", {"cache": cache_path, "clusters": strong})
        if item["reasons"]:
            codes = {r["code"] for r in item["reasons"]}
            if {"staff_line_span_alternative", "zero_local_ks_phase", "detector_disagreement_cluster"} <= codes:
                item["recommended_action"] = "drop system"
            item["action_note"] = ("재판독 완료 전 학습에서 시스템 제외 권고; 자동 삭제/수정 없음"
                                   if item["recommended_action"] == "drop system" else "crop·legend 재판독, 필요시 원음 대조")
            flags.append(item)
    return flags, {"video": row["video"], "song": row["slug"], "events": len(events),
                   "strict_systems": len(grouped), "stored_systems": len(strip),
                   "usable_systems": sum(bool(s.get("usable")) for s in aligned.values()),
                   "cache": cache_path, "detector": cache_stats, "coverage": dict(coverage),
                   "problems": problems}, [dict(song=row["slug"], system=sid, **e)
                                            for sid, items in detector.items() for e in items]


def known_cases(flags):
    index = {(f["video"], f["system"]): f for f in flags}
    geometry = []
    for sid in (29, 32, 35, 36):
        f = index.get(("zAZRKRbWGm0", sid), {})
        codes = [r["code"] for r in f.get("reasons", [])]
        geometry.append({"system": sid, "caught": "staff_line_span_alternative" in codes
                         and "zero_local_ks_phase" in codes, "tom_count": f.get("counts", {}).get("tom", 0)})
    f = index.get(("4VX2domLud8", 10), {})
    ties = [e for r in f.get("reasons", []) if r["code"] == "tied_continuation_onset" for e in r["evidence"]["events"]]
    return {"drumscribe": geometry, "drumscribe_toms": sum(g["tom_count"] for g in geometry),
            "olliewoods_tie": {"system": 10, "caught": any(e["inst"] == "kick" and abs(e["time"]-20.224036) < .001 for e in ties),
                               "events": ties}}


def attach_reviews(flags, reviews):
    index = {(f["video"], f["system"]): f for f in flags}
    seen = set()
    for review in reviews:
        key = (review["video"], review["system"])
        if key in seen:
            raise ValueError(f"duplicate manual review: {key}")
        seen.add(key)
        f = index[(review["video"], review["system"])]
        for key in ("crop_sha256", "canonical_sha256"):
            if f.get(key) != review[key]:
                raise ValueError(f"stale manual review: {review['video']}/{review['system']}/{key}")
        f["manual_review"] = review
        if review["verdict"] == "confirmed_label_error":
            f["recommended_action"] = "drop system"
            f["action_note"] = "crop 대조로 오류 확인; 재판독 완료 전 시스템 제외 권고 (실제 변경 없음)"


def markdown(result):
    s = result["summary"]
    lines = ["# Harvest strict truth 라벨 노이즈 감사", "",
        f"생성: {result['generated_at']} · 정책 v1 · 모든 판정은 검토용이며 truth 수정 없음.", "",
        f"- 대상: read/compacted 비-vault {s['eligible_songs']}곡, 비어 있지 않은 strict {s['nonempty_songs']}곡.",
        f"- strict 이벤트가 있는 {s['strict_systems']}시스템 / {s['events']}이벤트 중 **{s['flagged_systems']}시스템 ({s['system_flag_rate']:.2%})** 경고.",
        f"- 경고 곡 {s['flagged_songs']} / {s['nonempty_songs']} ({s['song_flag_rate']:.2%}); 경고 시스템의 이벤트 {s['flagged_events']} ({s['event_review_fraction']:.2%}). 이 수치는 오라벨 수·오류율이 아니다.",
        f"- crop {s['crops_checked']}개, tie 검사 {s['tie_checked_systems']}시스템; detector 캐시 {s['cache_songs']}곡.",
        f"- detector: distinct core {s['detector_distinct_core']}, unison 제외 {s['detector_non_unison']}, 캐시 범위 내 {s['detector_in_range']}, 개별 불일치 후보 {s['detector_candidates']}개.",
        f"- 제외 집계: `{json.dumps(result['skipped'], ensure_ascii=False)}`. vault truth/캐시는 열지 않는다.", "",
        "## 검사 기준과 한계", "",
        "1. 최신 manifest 병합 및 vault.Vault로 선별한 canonical truth.json만 분모에 넣는다. strict 이벤트가 없는 시스템은 분모에서 제외하며, strip 이웃은 거절된 시스템도 문맥으로만 사용한다.",
        "2. ±2 시스템에 K/S 6개 이상인데 본 시스템 K/S가 0, 또는 자체 phase count 0 + song-level backing을 경고한다.",
        "3. 바로 인접한 시스템의 exportable head가 각각 8개 이상이고 악기 분포 total variation ≥0.45이면 양쪽 strict 시스템을 경고한다. 정상 필인·솔로·withheld 구성 차이도 해당할 수 있다.",
        "4. 같은 row의 인접 오선 spacing 차이 ≤8%, 다섯 줄 이동 0.75–1.25 spacing, 잔차 ≤0.15 spacing을 검사한다. 같은 crop의 prefer_line_span 후보도 비교한다. 좌표 이동만으로는 crop 이동과 라벨 오류를 구별하지 못한다.",
        "5. 저장 head + unvisited head에 현재 ties.mark_continuations를 메모리에서 적용하고 canonical 이벤트와 shape 및 x/y 각각 1px 이내로 대조한다. 잘린/긴 타이, 누락된 head는 놓칠 수 있으며 raster 오검출도 crop 확인이 필요하다.",
        "6. .eval-cache/ft/<허용 slug>/teacher0.npy 우선, 없으면 teacher.npy. 100fps K/S/T/H/C 배열을 사용한다. 다른 core onset ±40ms인 unison은 제외하고 ±30ms max에서 own<0.2, other>0.8을 후보로 저장한다. 같은 시스템·shape·pos·class 전환이 ≥3개이며 해당 rule 이벤트의 ≥30%이면 군집 경고한다. detector 자체의 킷 혼동·시각 오차·캐시 provenance 한계가 남는다.",
        "7. geometry 대안 + phase 0 + detector 군집 동시 발생 또는 crop 수동 오류 확인이면 `drop system`(재판독 전 임시 제외 권고), 나머지는 `reread`. 실제 삭제·재라벨·truth 재생성·캐시 쓰기는 수행하지 않는다.",
        "8. 입력 SHA-256을 기록하고 종료 전 재확인한다. 이 digest는 실행 중 변경 탐지이며 과거 캐시가 어느 모델로 생성됐는지 증명하지 않는다. 파일 누락과 검사 범위는 JSON의 songs/provenance에서 확인한다.", "",
        "## 사유별 시스템 수 (중복 가능)", "", "| 사유 | 시스템 |", "|---|---:|"]
    lines += [f"| {REASONS[k]} (`{k}`) | {v} |" for k, v in s["reason_counts"].items()]
    known = result["known_cases"]
    lines += ["", "## 알려진 사례 회귀 확인", "",
              f"- DRUMSCRIBE zAZRKRbWGm0: `{json.dumps(known['drumscribe'], ensure_ascii=False)}`; tom 합계 {known['drumscribe_toms']}개.",
              f"- Ollie Woods 4VX2domLud8 system 10, 20.224036초 kick: 검출 **{known['olliewoods_tie']['caught']}**.",
              "- 위 ID/시각은 결과 검증에만 사용하며 탐지 조건에 사용하지 않는다.", "", "## 저장 crop 수동 spot check", "",
              "목적 표본이며 무작위·맹검이 아니다. 알려진 사례와 서로 다른 경고 유형을 포함한다. 음향 일치나 전체 오류율 검증이 아니다.", "",
              "| 영상 / 시스템 | 판정 | crop에서 확인한 근거 |", "|---|---|---|"]
    reviews = [f for f in result["flags"] if f.get("manual_review")]
    verdicts = Counter(f["manual_review"]["verdict"] for f in reviews)
    confirmed = [f for f in reviews if f["manual_review"]["verdict"] == "confirmed_label_error"]
    # Manual notes define the scope of the confirmation; don't count all events
    # in a flagged/confirmed system as incorrect positives.
    for f in reviews:
        r = f["manual_review"]
        lines.append(f"| [{f['video']} / {f['system']}]({f['crop'].removeprefix('truth/harvest/')}) | {r['verdict']} | {r['note']} |")
    if not reviews:
        lines.append("| — | 미실시 | --spot-checks로 실제 crop 확인 기록을 첨부할 수 있음 |")
    else:
        lines += ["", f"직접 확인 {len(reviews)}시스템: `{json.dumps(dict(verdicts), ensure_ascii=False)}`.",
                  f"오류 확인 {len(confirmed)}시스템에도 정상 이벤트가 섞일 수 있으므로 전체 이벤트를 오라벨 수로 세지 않는다. `no_label_error_seen`은 해당 경고에 대한 시각적 반례이며 시스템 전체의 정확성 보증이 아니다."]
    lines += ["", "## 전체 경고 목록", "", "자세한 좌표, 이웃 counts, detector 확률, 타이 시각 및 SHA-256은 label_audit.json의 flags에 기록했다.", "",
              "| 영상 / 시스템 | K / S / T / 기타 | 사유 | 권고 |", "|---|---:|---|---|"]
    for f in result["flags"]:
        c = f["counts"]
        link = f"[{f['video']} / {f['system']}]({f['crop'].removeprefix('truth/harvest/')})" if f.get("crop") else f"{f['video']} / {f['system']}"
        count = f"{c.get('kick', 0)} / {c.get('snare', 0)} / {c.get('tom', 0)} / {sum(v for k, v in c.items() if k not in CORE)}"
        lines.append(f"| {link} | {count} | {', '.join(r['code'] for r in f['reasons'])} | {f['recommended_action']} |")
    problems = [song for song in result["songs"] if song.get("problems") or song.get("error")]
    lines += ["", "## 검사 누락/입력 문제", "", f"`{json.dumps(problems, ensure_ascii=False)}`", "",
              "재실행: `.venv/bin/python -B tools/harvest/label_audit.py --check-known`", "",
              "수동 확인 보존: 기존 label_audit.json을 --spot-checks로 넘기면 hash가 일치하는 검토만 재사용한다.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--spot-checks", type=Path, help="manual-review list or a previous label_audit.json")
    parser.add_argument("--check-known", action="store_true", help="fail if the two stale-truth regression cases are not caught")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    for forbidden in (OUT / "songs", ROOT / ".eval-cache", ROOT / "tools" / "truth_video"):
        if output.is_relative_to(forbidden.resolve()):
            parser.error(f"output forbidden in {forbidden}")
    inputs = Inputs()
    # Capture dependency bytes before/after scanning, including concurrently edited helpers.
    for path in (Path(__file__), Path(vault.__file__), ROOT / "tools/harvest/manifest_io.py",
                 ROOT / "tools/truth_video/heads.py", ROOT / "tools/truth_video/ties.py",
                 vault.CONFIG, vault.MANIFEST):
        inputs.read(path)
    seal = vault.Vault()  # invalid/missing config fails closed
    rows, skipped = eligible_rows(seal, OUT / "songs")
    reviews = []
    if args.spot_checks:
        data = json.loads(args.spot_checks.read_text())
        reviews = [f["manual_review"] for f in data["flags"] if f.get("manual_review")] if isinstance(data, dict) else data
    flags, songs, candidates = [], [], []
    for i, row in enumerate(rows, 1):
        meta = OUT / "songs" / row["slug"] / "meta.json"
        if meta.exists():
            inputs.read(meta)
        found, coverage, individual = audit_song(row, inputs)
        flags.extend(found); songs.append(coverage); candidates.extend(individual)
        if i % 10 == 0 or i == len(rows):
            print(f"{i}/{len(rows)} songs; {len(flags)} flagged systems", flush=True)
    attach_reviews(flags, reviews)
    total_systems = sum(s.get("strict_systems", 0) for s in songs)
    total_events = sum(s.get("events", 0) for s in songs)
    nonempty = sum(s.get("events", 0) > 0 for s in songs)
    flagged_songs = len({f["song"] for f in flags})
    flagged_events = sum(f["event_count"] for f in flags)
    summary = {"eligible_songs": len(rows), "nonempty_songs": nonempty,
               "strict_systems": total_systems, "events": total_events,
               "flagged_systems": len(flags), "flagged_songs": flagged_songs,
               "flagged_events": flagged_events, "system_flag_rate": len(flags)/max(1, total_systems),
               "song_flag_rate": flagged_songs/max(1, nonempty),
               "event_review_fraction": flagged_events/max(1, total_events),
               "reason_counts": dict(sorted(Counter(r["code"] for f in flags for r in f["reasons"]).items())),
               "cache_songs": sum(s.get("cache") is not None for s in songs),
               "crops_checked": sum(s.get("coverage", {}).get("crops", 0) for s in songs),
               "tie_checked_systems": sum(s.get("coverage", {}).get("tie_checked_systems", 0) for s in songs),
               "manual_checks": len(reviews)}
    summary.update({f"detector_{key}": sum(s.get("detector", {}).get(key, 0) for s in songs)
                    for key in ("distinct_core", "non_unison", "in_range", "candidates")})
    inputs.verify()
    result = {"schema_version": 1, "generated_at": datetime.now(timezone.utc).isoformat(),
              "policy": POLICY, "summary": summary, "skipped": skipped, "songs": songs,
              "known_cases": known_cases(flags), "flags": flags, "detector_candidates": candidates,
              "provenance": {"input_sha256": {str(Path(p).relative_to(ROOT)): h for p, h in inputs.hashes.items()},
                             "inputs_unchanged_at_end": True,
                             "reader_mode": "saved heads + raster tie check; line-span geometry candidate only"}}
    output.mkdir(parents=True, exist_ok=True)
    for name, content in (("label_audit.json", json.dumps(result, ensure_ascii=False, indent=2) + "\n"),
                          ("LABEL_AUDIT.md", markdown(result))):
        path = output / name
        if path.is_symlink():
            raise ValueError(f"refusing symlink output: {path}")
        path.write_text(content)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.check_known:
        known = result["known_cases"]
        if not all(g["caught"] for g in known["drumscribe"]) or not known["olliewoods_tie"]["caught"]:
            raise SystemExit("known regression case missing; inspect current truth and report")


if __name__ == "__main__":
    main()
