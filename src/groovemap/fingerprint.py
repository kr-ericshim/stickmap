"""Fingerprints of the code behind each output, so outputs made by older code can be found and remade.

A fingerprint hashes the syntax tree of a function and of everything in this package it
reaches: functions, classes and module-level values, followed through imports and
`module.name` references. Comments, docstrings and code the function never reaches do not
count, so editing a tom rule leaves the drum-separation fingerprint alone. The fine-tuned
weights shipped in the package count; other model weights and third-party libraries do not.
Read from source, without importing: torch stays unloaded.
"""

from __future__ import annotations

import ast
import hashlib
import re
from functools import cache
from pathlib import Path
from typing import Any

PACKAGE = Path(__file__).parent

# The function that makes each kind of cached model output.
PRODUCERS = {
    "separate": ("models", "separate_drums"),
    "beats": ("models", "track_beats"),
    "onsets": ("models", "drum_activations"),
    "stem_onsets": ("models", "stem_activations"),
    "snare": ("dynamics", "snare_response"),
    "attacks": ("fills", "attack_times"),
    "adtof_input": ("cymbal_veto", "adtof_input"),
}
# Files a producer reads besides its code: the fine-tuned weights shipped in the package.
PRODUCER_FILES = {"stem_onsets": ("adtof-groovemap-v2.pt",)}
# Everything from cached model outputs to score.json, MusicXML and MIDI.
SCORE_ROOT = ("pipeline", "run")


class _Module:
    def __init__(self, name: str) -> None:
        tree = ast.parse((PACKAGE / f"{name}.py").read_text(encoding="utf-8"))
        self.names: dict[str, ast.AST] = {}
        self.imports: dict[str, tuple[str, str | None]] = {}
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                self.names[node.name] = node
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    for name in ast.walk(target):
                        if isinstance(name, ast.Name):
                            self.names[name.id] = node
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                self.names[node.target.id] = node
            elif isinstance(node, ast.ImportFrom):
                self.imports.update(_imported(node))


def _imported(node: ast.ImportFrom) -> dict[str, tuple[str, str | None]]:
    """Local name -> (package module, attribute or None for the module itself)."""
    if node.level != 1:
        return {}
    found: dict[str, tuple[str, str | None]] = {}
    for alias in node.names:
        local = alias.asname or alias.name
        if node.module is None:
            found[local] = (alias.name, None)  # from . import models
        else:
            found[local] = (node.module, alias.name)  # from .models import pick_hits
    return {local: target for local, target in found.items() if (PACKAGE / f"{target[0]}.py").is_file()}


@cache
def _module(name: str) -> _Module:
    return _Module(name)


def _without_docstrings(node: ast.AST) -> ast.AST:
    for part in ast.walk(node):
        body = getattr(part, "body", None)
        if (
            isinstance(part, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and body and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str)
        ):
            part.body = body[1:] or [ast.Pass()]
    return node


def _reached(root: tuple[str, str]) -> dict[tuple[str, str], str]:
    """Every (module, name) the root reaches, with the dump of its syntax tree."""
    seen: dict[tuple[str, str], str] = {}
    todo = [root]
    while todo:
        module_name, name = key = todo.pop()
        if key in seen:
            continue
        module = _module(module_name)
        if name in module.imports and name not in module.names:
            target = module.imports[name]
            if target[1] is not None:
                todo.append((target[0], target[1]))
            continue
        node = module.names.get(name)
        if node is None:
            seen[key] = ""
            continue
        node = _without_docstrings(ast.parse(ast.unparse(node)).body[0])
        seen[key] = ast.dump(node)
        imports = dict(module.imports)
        for part in ast.walk(node):
            if isinstance(part, ast.ImportFrom):  # imports inside a function body
                imports.update(_imported(part))
        for part in ast.walk(node):
            if isinstance(part, ast.Attribute) and isinstance(part.value, ast.Name):
                target = imports.get(part.value.id)
                if target is not None and target[1] is None:
                    todo.append((target[0], part.attr))  # models.pick_hits
            if isinstance(part, ast.Name):
                if part.id in module.names:
                    todo.append((module_name, part.id))
                elif part.id in imports and imports[part.id][1] is not None:
                    todo.append((imports[part.id][0], imports[part.id][1]))
    return seen


def of(root: tuple[str, str], *files: str) -> str:
    digest = hashlib.sha1()
    for (module, name), dump in sorted(_reached(root).items()):
        digest.update(f"{module}.{name}\0{dump}\0".encode())
    for file in files:
        digest.update((PACKAGE / file).read_bytes())
    return digest.hexdigest()[:12]


def _compute() -> dict[str, Any]:
    _module.cache_clear()
    return {
        # The score is remade from the stem activations, so new weights date it even where those caches are gone.
        "score": of(SCORE_ROOT, "engraving.mss", *PRODUCER_FILES["stem_onsets"]),
        "producers": {kind: of(root, *PRODUCER_FILES.get(kind, ())) for kind, root in PRODUCERS.items()},
    }


# The code this process loaded, give or take modules imported later (see `snapshot`).
AT_START = _compute()


def snapshot(check: bool = False) -> dict[str, Any] | None:
    """Fingerprints of the code this process runs.

    With `check`, None when the source on disk has changed since the process started: modules
    imported lazily after that may be newer than the rest, so nothing it makes can be vouched for.
    """
    if check and _compute() != AT_START:
        return None
    return AT_START


def _join(*parts: str | None) -> str | None:
    return None if None in parts else hashlib.sha1("/".join(parts).encode()).hexdigest()[:12]  # type: ignore[arg-type]


def cache_stamp(filename: str, fingerprints: dict[str, Any] | None) -> str | None:
    """The stamp a cache file made by this code carries: its maker and the caches it was made from.

    None for files that are not cached model outputs, or when the code cannot be vouched for.
    """
    if fingerprints is None:
        return None
    made = fingerprints["producers"]
    if match := re.fullmatch(r"drums\.(.+)\.wav", filename):
        return made["separate"]
    if match := re.fullmatch(r"adtof_input\.(.+)\.npy", filename):
        return _join(made["adtof_input"], cache_stamp(f"drums.{match[1]}.wav", fingerprints))
    if filename == "beats.json":
        return made["beats"]
    if filename == "activations.mix.npy":
        return made["onsets"]
    if match := re.fullmatch(r"activations\.drums-(.+)\.npy", filename):
        return _join(made["stem_onsets"], cache_stamp(f"drums.{match[1]}.wav", fingerprints))
    if match := re.fullmatch(r"snare\.(.+)\.npy", filename):
        return _join(made["snare"], cache_stamp(f"activations.drums-{match[1]}.npy", fingerprints))
    if match := re.fullmatch(r"attacks\.(.+)\.npy", filename):
        return _join(made["attacks"], cache_stamp(f"drums.{match[1]}.wav", fingerprints))
    return None
