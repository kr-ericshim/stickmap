"""Seal future harvests and explicit video IDs from training and routine scoring.

    .venv/bin/python tools/harvest/vault.py init
    .venv/bin/python tools/harvest/vault.py add VIDEO_ID [VIDEO_ID ...]
    .venv/bin/python tools/harvest/vault.py list
    .venv/bin/python tools/harvest/vault.py stats

vault.json contains cutoff (UTC ISO timestamp or Unix seconds), video_ids and
optional train_channels, whose new videos are not sealed unless listed in video_ids.
Membership uses the FIRST committed manifest row, never its latest update.
Missing/invalid configuration fails closed; init persists its cutoff once.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import shlex
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

try:
    from .manifest_io import records
except ImportError:  # direct CLI and the existing tools' flat imports
    from manifest_io import records

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "truth" / "harvest"
CONFIG = OUT / "vault.json"
MANIFEST = OUT / "manifest.jsonl"
LOG = OUT / "vault_log.jsonl"
SONGS = OUT / "songs"


def performance_table_file(path: Path | None = None):
    """Optional JSON groups, with a digest of the parsed bytes. Errors fail closed."""
    path = path if path is not None else OUT / "same_performance.json"
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        if path.is_symlink():
            raise
        return (), None
    data = json.loads(raw)
    groups = data.get("groups") if isinstance(data, dict) else None
    if not isinstance(groups, list) or any(
        not isinstance(g, list) or len(g) < 2 or
        any(not isinstance(v, str) or not v.strip() for v in g) for g in groups
    ):
        raise ValueError(f"invalid same_performance groups: {path}")
    return tuple(tuple(g) for g in groups), hashlib.sha256(raw).hexdigest()


def timestamp(value) -> float:
    if isinstance(value, str):
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            raise ValueError("vault timestamps must include a timezone")
        value = dt.timestamp()
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("vault timestamp must be finite Unix seconds or a timezone-aware ISO timestamp")
    return float(value)


def config() -> dict:
    if not CONFIG.exists():
        raise FileNotFoundError(f"Missing {CONFIG}; initialize the seal with tools/harvest/vault.py init")
    data = json.loads(CONFIG.read_text())
    timestamp(data["cutoff"])
    ids = data["video_ids"]
    if not isinstance(ids, list) or any(not isinstance(v, str) or not v.strip() for v in ids):
        raise ValueError("vault video_ids must be a list of nonempty video IDs")
    return data


def init(cutoff=None) -> dict:
    """Create the seal once; rerunning init never advances an existing cutoff."""
    if CONFIG.exists():
        return config()
    cutoff = cutoff if cutoff is not None else datetime.now(timezone.utc).isoformat()
    timestamp(cutoff)
    data = {"cutoff": cutoff, "video_ids": []}
    CONFIG.parent.mkdir(parents=True, exist_ok=True)
    with CONFIG.open("x") as out:
        out.write(json.dumps(data, indent=2) + "\n")
    return data


class Vault:
    """One fresh config/journal snapshot per song-list operation."""

    def __init__(self):
        data = config()
        self.cutoff = timestamp(data["cutoff"])
        self.explicit = set(data["video_ids"])
        # Channels harvested for training: new videos stay out unless listed explicitly.
        self.train_channels = set(data.get("train_channels", []))
        self.first = {}
        self.channel = {}
        self.rows = {}
        self.slugs = {}
        for row in records(MANIFEST):
            video = row["video"]
            if video not in self.first:
                self.channel[video] = row.get("channel")
                # An undated first row cannot safely be classified as pre-seal.
                self.first[video] = timestamp(row["at"]) if "at" in row else None
            self.rows[video] = {**self.rows.get(video, {}), **row}
            if row.get("slug"):
                self.slugs[row["slug"]] = video

    def is_vault(self, video_id: str) -> bool:
        if video_id in self.explicit:
            return True
        if video_id not in self.first or self.channel.get(video_id) in self.train_channels:
            return False
        first = self.first[video_id]
        return first is None or first > self.cutoff

    def contains_song(self, song) -> bool:
        """Resolve new metadata and legacy cached slugs, including old slug aliases."""
        return any(self.is_vault(v) for v in self.song_ids(song))

    def song_ids(self, song) -> set[str]:
        """Resolve identities without changing seal membership or eligibility."""
        if isinstance(song, (str, Path)):
            folder = Path(song)
            meta = folder / "meta.json"
            song = {**(json.loads(meta.read_text()) if meta.exists() else {}), "slug": folder.name}
        def field(key):
            return song.get(key) if isinstance(song, dict) else getattr(song, key, None)

        slug = field("slug") or ""
        ids = {field("video"), field("video_id"), self.slugs.get(slug)}
        # YouTube IDs may themselves contain hyphens; rsplit('-') loses them.
        if len(slug) >= 12 and slug[-12] == "-":
            ids.add(slug[-11:])
        return {v for v in ids if v}

    def final_exclusions(self) -> dict:
        """Eligibility is separate from sealing: mixed performances never score."""
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gpu"))
        from chops_common import require_sealed_groups
        peers = require_sealed_groups(self)
        return {v: {"video": v, "slug": self.rows.get(v, {}).get("slug"),
                    "train_peers": ids, "reason": "excluded: train peer " + " ".join(ids)}
                for v, ids in peers.items()}


def filter_final(rows, *, seal=None, log=True):
    """Select before truth reads; retain inspectable exclusions for final reports."""
    seal = Vault() if seal is None else seal
    excluded = seal.final_exclusions()
    if log:
        log_exclusions(excluded)
    return [r for r in rows if seal.contains_song(r) and
            not (seal.song_ids(r) & excluded.keys())], excluded


def log_exclusions(excluded) -> None:
    for video, item in sorted(excluded.items()):
        print(f"{video}: {item['reason']}", flush=True)


def is_vault(video_id: str) -> bool:
    return Vault().is_vault(video_id)


def is_vault_song(song) -> bool:
    return Vault().contains_song(song)


def add(video_ids) -> dict:
    video_ids = [video_ids] if isinstance(video_ids, str) else [*video_ids]
    if any(not isinstance(v, str) or not v.strip() for v in video_ids):
        raise ValueError("provide nonempty video IDs")
    init()
    # A stable sidecar lock serializes adders. Readers see a complete config
    # via atomic replacement, including when another process adds IDs.
    with CONFIG.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data = config()
        data["video_ids"] = sorted(set(data["video_ids"]) | set(video_ids))
        tmp = CONFIG.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2) + "\n")
        tmp.replace(CONFIG)
    return data


def list_videos() -> list[str]:
    seal = Vault()
    return sorted(seal.explicit | {v for v in seal.rows if seal.is_vault(v)})


def stats() -> dict:
    """Aggregate notation only: no detector, model, or per-video scores."""
    sys.path.insert(0, str(ROOT / "tools" / "truth_video"))
    from fill_seq import DRUMS, fast_mask, notated

    seal = Vault()
    ids = seal.explicit | {v for v in seal.rows if seal.is_vault(v)}
    excluded = seal.final_exclusions()
    ids -= set(excluded)
    result = {"count": len(ids), "seconds": 0.0, "notated_seconds": 0.0,
              "systems": 0, "fast_moments": {c: 0 for c in DRUMS},
              "missing_notation": 0}
    if excluded:
        result['excluded'] = excluded
    for video in sorted(ids):
        row = seal.rows.get(video, {})
        result["seconds"] += float(row.get("seconds") or 0)
        slug = row.get("slug")
        folder = SONGS / slug / "truth" if slug else None
        if folder is None or not all((folder / f).exists() for f in ("strip.json", "alignment.json")):
            result["missing_notation"] += 1
            continue
        for a, b, ref in notated(folder, 0.75):
            result["systems"] += 1
            result["notated_seconds"] += b - a
            for fast, (_, drums) in zip(fast_mask(ref), ref):
                if fast:
                    for c in drums:
                        result["fast_moments"][c] += 1
    return result


def require_final_writer() -> None:
    if (OUT / ".wsl-mirror").exists():
        raise RuntimeError("WSL owns final scoring and vault_log; run final evaluation on WSL, not the Mac mirror")


def resume_final(opening_id: str, model_tag: str) -> None:
    """A child scorer must verify the durable opening recorded by its runner."""
    require_final_writer()
    config()
    data = LOG.read_bytes()
    if data and not data.endswith(b"\n"):
        raise ValueError("vault log has an unfinished record; refusing to open the vault")
    rows = [json.loads(line) for line in data.splitlines()]
    if not any(row.get("opening_id") == opening_id and row.get("model_tag") == model_tag
               and row.get("pid") == os.getppid() for row in rows):
        raise ValueError("no matching durable vault opening; refusing to open the vault")
    Vault().final_exclusions()  # validate the group table before any sealed read


def open_final(model_tag: str, command: str | None = None) -> str:
    """Record an attempted opening BEFORE any notation/detector evaluation."""
    require_final_writer()
    config()  # fail closed before announcing a usable seal
    Vault().final_exclusions()  # metadata only; malformed/unreadable group tables refuse final
    print("\n" + "!" * 78 + "\n!!! FINAL EVALUATION: OPENING THE SEALED TEST VAULT !!!\n"
          "!!! VAULT ONLY. This invocation is appended to vault_log.jsonl. !!!\n"
          + "!" * 78, flush=True)
    payload = {"timestamp": datetime.now(timezone.utc).isoformat(),
               "command": command or shlex.join([sys.executable, *sys.argv]),
               "model_tag": model_tag, "opening_id": uuid.uuid4().hex, "pid": os.getpid()}
    LOG.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(LOG, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        with LOG.open("rb") as check:
            check.seek(0, os.SEEK_END)
            if check.tell():
                check.seek(-1, os.SEEK_END)
                if check.read(1) != b"\n":
                    raise ValueError("vault log has an unfinished record; refusing to open the vault")
        line = (json.dumps(payload, ensure_ascii=False) + "\n").encode()
        if os.write(fd, line) != len(line):
            raise OSError("short vault log append; refusing to open the vault")
        os.fsync(fd)
    finally:
        os.close(fd)
    return payload["opening_id"]


def model_tag(detector: str, tag: str | None = None) -> str:
    return tag or os.environ.get("FT_TAG") or detector


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("init")
    create.add_argument("--cutoff", help="timezone-aware ISO timestamp (default: now, persisted once)")
    insert = sub.add_parser("add")
    insert.add_argument("video_ids", nargs="+")
    sub.add_parser("list")
    sub.add_parser("stats")
    args = parser.parse_args()
    if args.command == "init":
        print(json.dumps(init(args.cutoff), indent=2))
    elif args.command == "add":
        print(json.dumps(add(args.video_ids), indent=2))
    elif args.command == "list":
        print("\n".join(list_videos()))
    else:
        print(json.dumps(stats(), indent=2))


if __name__ == "__main__":
    main()
