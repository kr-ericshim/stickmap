"""stickmap CLI: `stickmap transcribe <url | file | search words>`."""

from __future__ import annotations

import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.prompt import IntPrompt
from rich.table import Table

from . import fetch, library
from .fetch import FetchError, SongMeta
from .score import INSTRUMENTS

app = typer.Typer(add_completion=False, no_args_is_help=True, help="Song -> drum sheet music draft.")
console = Console()


def _duration(seconds: float) -> str:
    return f"{int(seconds) // 60}:{int(seconds) % 60:02d}" if seconds else "-"


def _choose(results: list[SongMeta], first: bool) -> SongMeta:
    if first or len(results) == 1 or not sys.stdin.isatty():
        return results[0]
    table = Table(box=None)
    for column in ("#", "title", "channel", "length"):
        table.add_column(column)
    for i, r in enumerate(results, 1):
        table.add_row(str(i), r.title, r.uploader, _duration(r.duration))
    console.print(table)
    pick = IntPrompt.ask("Which one", choices=[str(i) for i in range(1, len(results) + 1)], default=1)
    return results[pick - 1]


def _prepare_song(source: str, out: Path, first: bool) -> tuple[Path, SongMeta]:
    """Choose an interactive search result, then use shared library preparation."""
    local = Path(source).expanduser()
    if fetch.is_url(source) or local.is_file():
        return library.prepare_song(source, out)
    with console.status(f"Searching YouTube for “{source}”"):
        results = fetch.search_youtube(source)
    if not results:
        raise FetchError(f"no YouTube results for “{source}”")
    meta = _choose(results, first)
    return library.prepare_song(meta.source, out, meta=meta)


@app.command()
def transcribe(
    source: Annotated[str, typer.Argument(help="YouTube URL, audio file path, or search words.")],
    grid: Annotated[int, typer.Option(help="Finest note value: 16 (full) or 8 (simplified).")] = 16,
    triplets: Annotated[str, typer.Option(help="Triplet detection: auto or off.")] = "auto",
    meter: Annotated[Optional[int], typer.Option(help="Force N beats per bar instead of detected bars.")] = None,
    time_signature: Annotated[Optional[str], typer.Option(help="Rebuild the pulse for N/4, N/8 or N/16 (N=1..15).")] = None,
    odd_bars: Annotated[
        str, typer.Option(help="none = regular bars, auto = odd bars when detections insist, detected = follow them.")
    ] = "none",
    beat_shift: Annotated[float, typer.Option(help="Move beat 1 later by this many beats (steps of 0.25).")] = 0,
    tempo_scale: Annotated[float, typer.Option(help="2 = count twice as fast as detected, 0.5 = half.")] = 1,
    voices: Annotated[int, typer.Option(help="1 = everything stems-up, 2 = kick in its own voice.")] = 1,
    title: Annotated[Optional[str], typer.Option(help="Title printed on the score.")] = None,
    out: Annotated[Path, typer.Option(help="Directory that holds one folder per song.")] = Path("songs"),
    first: Annotated[bool, typer.Option("--first", help="Take the first search result without asking.")] = False,
    hq: Annotated[bool, typer.Option("--hq", help="Use htdemucs_ft (about 4x slower, cleaner stem).")] = False,
    on_mix: Annotated[bool, typer.Option("--on-mix", help="Detect drums on the full mix, not the stem.")] = False,
    thresholds: Annotated[
        Optional[str], typer.Option(help="Peak thresholds kick,snare,tom,hihat,cymbal (lower = more notes).")
    ] = None,
    sensitivity: Annotated[
        float, typer.Option(help="Detection sensitivity: 2 picks up hits half as strong as the default.")
    ] = 1,
    device: Annotated[str, typer.Option(help="auto, mps, cuda or cpu.")] = "auto",
    pdf: Annotated[bool, typer.Option(help="Engrave a PDF with MuseScore.")] = True,
    preview: Annotated[bool, typer.Option(help="Write preview.wav: song left, transcription right.")] = False,
    open_: Annotated[bool, typer.Option("--open", help="Open the score in MuseScore when done.")] = False,
    force: Annotated[bool, typer.Option(help="Ignore cached model outputs.")] = False,
) -> None:
    """Transcribe the drums of a song into MusicXML, MIDI and PDF."""
    if grid not in (8, 16) or triplets not in ("auto", "off") or voices not in (1, 2):
        raise typer.BadParameter("grid must be 8 or 16, triplets auto or off, voices 1 or 2")
    if odd_bars not in ("none", "auto", "detected") or tempo_scale not in (0.5, 1, 2):
        raise typer.BadParameter("odd-bars must be none, auto or detected; tempo-scale 0.5, 1 or 2")
    if time_signature is not None:
        from .metre import Signature

        try:
            Signature.parse(time_signature)
        except ValueError as error:
            raise typer.BadParameter(str(error)) from error
    parsed_thresholds = None
    if thresholds:
        parsed_thresholds = [float(v) for v in thresholds.split(",")]
        if len(parsed_thresholds) != 5:
            raise typer.BadParameter("thresholds needs 5 comma-separated values")

    try:
        song_dir, meta = _prepare_song(source, out, first)
    except FetchError as err:
        console.print(f"[red]{err}[/red]")
        raise typer.Exit(1)
    if title:
        meta.title = title
    console.print(f"[bold]{meta.title}[/bold]  {_duration(meta.duration)}  → {song_dir}")

    from . import pipeline  # imports torch; keep `--help` fast

    opts = pipeline.Options(
        grid=grid,
        triplets=triplets,
        meter=meter,
        time_signature=time_signature,
        odd_bars=odd_bars,
        beat_shift=beat_shift,
        tempo_scale=tempo_scale,
        voices=voices,
        device=device,
        demucs_model="htdemucs_ft" if hq else "htdemucs",
        onset_input="mix" if on_mix else "drums",
        thresholds=parsed_thresholds,
        sensitivity=sensitivity,
        pdf=pdf,
        preview=preview,
        force=force,
        refresh_stale=True,
    )
    result = pipeline.run(song_dir, meta, opts, log=lambda line: console.print(f"  [dim]{line}[/dim]"))
    library.save_options(song_dir, opts)

    score = result.score
    meters = Counter((bar.beats, bar.denominator) for bar in score.bars)
    triplet_beats = sum(
        any(g in (3, 6) for g in (subdiv if isinstance(subdiv, list) else [subdiv]))
        for bar in score.bars for subdiv in bar.subdivisions
    )
    console.print(
        f"{len(score.bars)} bars, {score.tempo:.0f} quarter BPM, "
        + ", ".join(f"{count}× {beats}/{denominator}" for (beats, denominator), count in meters.most_common())
        + (f", {triplet_beats} triplet beats" if triplet_beats else "")
    )
    console.print(
        "notes: "
        + ", ".join(f"{inst} {result.note_counts[inst]}" for inst in INSTRUMENTS if result.note_counts[inst])
    )
    for kind, path in result.files.items():
        if kind != "json":
            console.print(f"  {kind:9}{path}")
    if open_:
        subprocess.run(["open", str(result.files.get("pdf") or result.files["musicxml"])], check=False)


@app.command("list")
def list_songs(out: Annotated[Path, typer.Option(help="Songs directory.")] = Path("songs")) -> None:
    """List transcribed songs."""
    for meta_path in sorted(out.glob("*/meta.json")):
        meta = SongMeta.load(meta_path)
        done = "✓" if (meta_path.parent / "score.json").exists() else " "
        console.print(f"{done} {meta_path.parent.name}  [dim]{meta.title}[/dim]")


@app.command()
def serve(
    host: Annotated[str, typer.Option(help="Bind address.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="HTTP port.")] = 8765,
    out: Annotated[Path, typer.Option(help="Songs directory.")] = Path("songs"),
    open_: Annotated[bool, typer.Option("--open", help="Open the app in your browser.")] = False,
) -> None:
    """Serve the local song library and transcription API."""
    import threading
    import webbrowser

    import uvicorn

    from .server import ALLOWED_HOSTS, create_app

    static = Path(__file__).resolve().parents[2] / "web" / "dist"
    # Bound to every interface, the names it is reached by are unknown, so only the cross-site check stays.
    hosts = None if host in ("0.0.0.0", "::") else ALLOWED_HOSTS | {host.strip("[]").lower()}
    server = create_app(out, static if static.is_dir() else None, hosts=hosts)
    if open_:
        timer = threading.Timer(1, webbrowser.open, args=[f"http://{host}:{port}"])
        timer.daemon = True
        timer.start()
    uvicorn.run(server, host=host, port=port)


if __name__ == "__main__":
    app()
