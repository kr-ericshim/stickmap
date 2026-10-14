"""Shared channel/song exclusions for training and fold-local dense quality."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harvest"))
import vault  # noqa: E402

# Keep the original 14 assignments and evaluation cohort fixed for v2 comparisons.
ORIGINAL_FOLDS = (
    ("swartz", "allyeodrum", "drumclass", "mydrumsheet", "taktak"),
    ("drumshin", "drumplace", "nakamura", "chock"),
    ("yeol", "atthedrum", "jilju", "drummate", "drumscribe"),
)
ORIGINAL_CHANNELS = tuple(c for fold in ORIGINAL_FOLDS for c in fold)
# 2026-10-02 manifest: readable song-seconds 10535 / 10730 / 12394.
# Ivannikov has no strict exports yet; assigning it now prevents future leakage.
FOLDS = (
    ORIGINAL_FOLDS[0] + ("olliewoods", "golly", "ivannikov"),
    ORIGINAL_FOLDS[1] + ("moyin",),
    ORIGINAL_FOLDS[2],
)

SAME_SONG = (
    "love-me-or-leave-me", "나의-x", "last-christmas", "벚꽃엔딩",
    # MoYin / Nakamura: Official HIGE DANdism, Mixed Nuts (ミックスナッツ).
    ("SRKKu5mTeAE", "DqSJyUZiHIk"),
)


def same_song(a: str, b: str) -> bool:
    for key in SAME_SONG:
        if isinstance(key, str):
            if key in a and key in b:
                return True
        elif any(a.endswith("-" + vid) for vid in key) and any(b.endswith("-" + vid) for vid in key):
            return True
    return False


def split_songs(songs, fold: str):
    """Accept Song objects or metadata dicts, with identical exclusions in both."""
    seal = vault.Vault()
    songs = [s for s in songs if not seal.contains_song(s)]
    def field(song, key):
        return song[key] if isinstance(song, dict) else getattr(song, key)

    known = {c for group in FOLDS for c in group}
    unknown = {field(s, "channel") for s in songs} - known
    if unknown:
        raise ValueError(f"channels missing from FOLDS: {sorted(unknown)}")
    if fold == "all":
        return songs, []
    if fold not in ("0", "1", "2"):
        raise ValueError(f"invalid fold: {fold}")
    held_channels = FOLDS[int(fold)]
    held = [s for s in songs if field(s, "channel") in held_channels]
    training = [s for s in songs if field(s, "channel") not in held_channels
                and not any(same_song(field(s, "slug"), field(h, "slug")) for h in held)]
    return training, held
