"""Find solo audio outside truth/ on the Mac, or rebase copied song.txt on WSL.

sync.sh uses --external before copying, then --rebase on the remote copy.
Only --rebase writes files; the Mac's song.txt files are left intact.
"""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRUTH = ROOT / "truth"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--external", action="store_true")
    mode.add_argument("--rebase", action="store_true")
    args = parser.parse_args()
    external = set()
    for pointer in sorted(TRUTH.glob("*/song.txt")):
        original = Path(pointer.read_text().strip())
        if args.external:
            song = original if original.is_absolute() else ROOT / original
            if song.is_dir() and not song.resolve().is_relative_to(TRUTH.resolve()):
                if "\n" in str(song):
                    parser.error(f"unsupported newline in audio path: {pointer}")
                external.add(str(song))
            continue
        # Retain the truth-relative layout where possible. Audio copied from
        # outside truth/ is placed in truth/songs/<slug> by sync.sh.
        candidates = []
        if "truth" in original.parts:
            offset = original.parts.index("truth")
            candidates.append(ROOT.joinpath(*original.parts[offset:]))
        candidates += [TRUTH / "songs" / original.name,
                       TRUTH / "harvest" / "songs" / original.name]
        song = next((p for p in candidates if p.is_dir()), None)
        if song is not None:
            pointer.write_text(str(song.resolve()) + "\n")
    for song in sorted(external):
        print(song)


if __name__ == "__main__":
    main()
