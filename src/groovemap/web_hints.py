"""Best-effort Songsterr metadata for reviewable hints, never score edits."""

from __future__ import annotations

import gzip
import json
import logging
import math
import os
import re
import time
import unicodedata
from collections import Counter
from pathlib import Path
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from .fetch import SongMeta
from .metre import parse_signatures

log = logging.getLogger(__name__)
_VARIANTS = re.compile(r"\b(cover|transcription|backing|live|lesson)\b", re.I)
_NOISE = re.compile(r"\b(official|music video|video|audio|mv|drum cover|lyrics?|live|[48]k|hd|hq)\b", re.I)


def _clean_title(value: str) -> str:
    value = re.sub(r"\([^)]*\)|\[[^]]*\]", lambda m: "" if _NOISE.search(m[0]) else m[0], value)
    value = re.sub(r"\s*[|–—-]\s*(?:official|lyrics?|live|drum cover|[48]k|hd)\b.*$", "", value, flags=re.I)
    value = re.sub(r"\s+(?:(?:official\s+)?(?:music\s+)?(?:video|audio)|lyrics?|[48]k|hd|mv)\s*$", "", value, flags=re.I)
    return value.strip(" \t-–—|")


def song_query(meta: SongMeta, display_title: str) -> tuple[str, str] | None:
    if meta.artist and meta.track:
        artist, title = meta.artist.strip(), _clean_title(meta.track)
        return (artist, title) if artist and title else None
    for value in (display_title, meta.title):
        parts = re.split(r"\s+[-–—]\s+", value, maxsplit=1)
        if len(parts) == 2:
            artist, title = parts[0].strip(), _clean_title(parts[1])
            if artist and title:
                return artist, title
    if meta.artist and (meta.track or display_title):
        title = _clean_title(meta.track or display_title)
        if title:
            return meta.artist.strip(), title
    return None


def _normalized(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    value = re.split(r"\b(?:feat\.?|ft\.?|featuring)\s", value, maxsplit=1)[0]
    return " ".join(token for token in re.findall(r"[^\W_]+", value) if token != "the")


def _matches(result: dict, artist: str, title: str) -> bool:
    candidate = result["title"]
    if set(_VARIANTS.findall(candidate.lower())) - set(_VARIANTS.findall(title.lower())):
        return False
    # Songsterr sometimes appends version information in parentheses.
    base = re.sub(r"(?:\s*\([^)]*\)|\s*\[[^]]*\])+\s*$", "", candidate)
    expected = _normalized(title).replace(" ", "")
    return bool(expected and expected in {_normalized(candidate).replace(" ", ""), _normalized(base).replace(" ", "")}
                and _same_artist(artist, result["artist"]))


def _same_artist(ours: str, theirs: str) -> bool:
    """One side's name words all appear in the other's: "James Fortune" ~ "James Fortune & FIYA", not "John Legend" ~ "John Lennon"."""
    a, b = set(_normalized(ours).split()), set(_normalized(theirs).split())
    return bool(a and b) and (a <= b or b <= a)


def _slug(value: str) -> str:
    return quote("-".join(re.findall(r"[^\W_]+", value.casefold())), safe="-") or "song"


def _get_json(url: str):
    request = Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; Groovemap/0.1)", "Accept-Encoding": "gzip"})
    with urlopen(request, timeout=8) as response:
        body = response.read()
    if body.startswith(b"\x1f\x8b"):
        body = gzip.decompress(body)
    return json.loads(body.decode("utf-8"), strict=False)


def _signature_summary(measures: list[dict]) -> str | None:
    signatures = []
    current = None
    for measure in measures:
        if "signature" in measure:
            current = "/".join(map(str, measure["signature"]))
            parse_signatures(current)
        signatures.append(current)
    counts = Counter(s for s in signatures if s is not None)
    if not counts:
        return None
    common = counts.most_common(2)
    if common[0][1] / len(measures) >= 0.8:
        return common[0][0]
    if len(common) == 2 and sum(n for _, n in common) / len(measures) >= 0.8:
        pair = {s for s, _ in common}
        adjacent = [(a, b) for a, b in zip(signatures, signatures[1:]) if a in pair and b in pair]
        # Allow a few repeated/transition bars without calling separate sections alternating.
        if len(adjacent) >= 2 and sum(a != b for a, b in adjacent) / len(adjacent) >= 0.8:
            ordered = list(dict.fromkeys(s for s in signatures if s in pair))
            if len({s.split("/")[1] for s in ordered}) == 1:
                return "+".join(ordered)
    return None


def _common_signatures(measures: list[dict], limit: int = 3) -> list[str]:
    """The most frequent signatures, so a song too mixed to summarise still reads as mixed."""
    signatures, current = [], None
    for measure in measures:
        if "signature" in measure:
            current = "/".join(map(str, measure["signature"]))
        signatures.append(current)
    return [s for s, _ in Counter(s for s in signatures if s is not None).most_common(limit)]


def _tempo_summary(track: dict, bars: int) -> float | None:
    events = []
    for event in track.get("automations", {}).get("tempo", []):
        position = event.get("position", 0)
        if isinstance(position, list):
            position = position[0] / position[1]
        start = float(event["measure"]) + float(position)
        bpm = float(event["bpm"])
        if math.isfinite(start) and math.isfinite(bpm) and bpm > 0 and 0 <= start < bars:
            events.append((start, bpm))
    events.sort(key=lambda item: item[0])
    weighted = []
    for i, (start, bpm) in enumerate(events):
        end = events[i + 1][0] if i + 1 < len(events) else bars
        if end > start:
            weighted.append((bpm, end - start))
    halfway = sum(weight for _, weight in weighted) / 2
    cumulative = 0.0
    for bpm, weight in sorted(weighted):
        cumulative += weight
        if cumulative >= halfway:
            return round(bpm, 2)
    return None


def lookup(artist: str, title: str) -> dict | None:
    if os.environ.get("GROOVEMAP_WEB_HINTS") == "0":
        return None
    try:
        return _lookup(artist, title)
    except Exception:
        log.warning("Songsterr lookup failed for %s - %s", artist, title, exc_info=True)
        return None


def _lookup(artist: str, title: str) -> dict | None:
    """None means Songsterr has no matching drum tab; network and parse errors raise."""
    results = _get_json("https://www.songsterr.com/api/songs?" + urlencode({"pattern": f"{artist} {title}", "size": 10}))
    match = next((r for r in results if any(t.get("instrumentId") == 1024 for t in r.get("tracks", []))
                  and _matches(r, artist, title)), None)
    if match is None:
        return None
    song_id = match["songId"]
    meta = _get_json(f"https://www.songsterr.com/api/meta/{song_id}")
    tracks = meta["tracks"]
    index = meta.get("popularTrackDrum")
    if not isinstance(index, int) or not 0 <= index < len(tracks) or tracks[index].get("instrumentId") != 1024:
        index = next(i for i, track in enumerate(tracks) if track.get("instrumentId") == 1024)
    track = _get_json(f"https://dqsljvtekg760.cloudfront.net/{song_id}/{meta['revisionId']}/{meta['image']}/{index}.json")
    measures = track["measures"]
    return {"songId": song_id, "artist": match["artist"], "title": match["title"], "bars": len(measures),
            "url": f"https://www.songsterr.com/a/wsa/{_slug(match['artist'] + ' ' + match['title'])}-drum-tab-s{song_id}",
            "signature": _signature_summary(measures), "common": _common_signatures(measures),
            "bpm": _tempo_summary(track, len(measures))}


def source(hint: dict) -> dict:
    return {"name": "Songsterr", **{key: hint[key] for key in ("url", "artist", "title")}}


def suggest_tempo(hint: dict, current_bpm: float) -> dict | None:
    signature = hint.get("signature")
    if signature and any(s.denominator != 4 for s in parse_signatures(signature)):
        return None
    bpm = hint.get("bpm")
    if not bpm or current_bpm <= 0:
        return None
    for scale in (2, 0.5):
        if abs(bpm / current_bpm / scale - 1) <= 0.08:
            return {"tempo_scale": scale, "web_bpm": bpm, "current_bpm": current_bpm, "source": source(hint)}
    return None


def load_or_fetch(song_dir: Path, meta: SongMeta, display_title: str) -> dict | None:
    if os.environ.get("GROOVEMAP_WEB_HINTS") == "0":
        return None
    try:
        query = song_query(meta, display_title)
        if query is None:
            return None
        path = song_dir / "web_hints.json"
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
            if cached.get("query") == list(query) and (cached.get("result") is not None or time.time() - cached["timestamp"] < 7 * 86400):
                return cached["result"]
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            pass  # missing or damaged cache: look it up again
        if os.environ.get("GROOVEMAP_WEB_HINTS") == "0":
            return None
        # A failed request is not cached: only "Songsterr has no such tab" waits a week.
        result = _lookup(*query)
        part = path.with_name(f".part.{path.name}")
        part.write_text(json.dumps({"query": query, "timestamp": time.time(), "result": result}, ensure_ascii=False, indent=1), encoding="utf-8")
        part.replace(path)
        return result
    except Exception:
        log.warning("Songsterr hint cache failed", exc_info=True)
        return None
