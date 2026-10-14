"""Synthetic seals: never open or alter the real test vault during verification."""
import json
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path[:0] = [str(Path(__file__).parent), str(Path(__file__).parents[1] / "truth_video")]
import fill_score
import fill_seq
import global_score
import harvest
import score
import vault
from manifest_io import read

OLD, NEW, LOOSE, EDGE = "old00000001", "new00000001", "loose000001", "edge0000001"
TIMES = [1.0, 1.04, 1.08, 1.4, 1.7, 2.0]
DRUMS = ["kick", "snare", "tom", "kick", "snare", "tom"]


def write_notation(folder, *, rejected=False):
    folder.mkdir(parents=True)
    notes = [dict(x=i * 20., inst=c, exportable=True, ghost=False, pos=3)
             for i, c in enumerate(DRUMS)]
    (folder / "strip.json").write_text(json.dumps([dict(system=0, staff=dict(spacing=10), notes=notes)]))
    report = dict(systems=[dict(system=0, usable=not rejected, times=TIMES,
                  after=dict(within_40ms=.8 if rejected else 1.),
                  reasons=["held-out agreement below 55% / 80%"] if rejected else [])],
                  intervals=[] if rejected else [[TIMES[0], TIMES[-1]]])
    (folder / "alignment.json").write_text(json.dumps(report))
    truth = [{**n, "time": t} for n, t in zip(notes, TIMES)] if not rejected else []
    (folder / "truth.json").write_text(json.dumps(truth))


class InlinePool:
    def __init__(self, jobs):
        assert 1 <= jobs <= 4

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def map(self, func, args):
        return map(func, args)


@pytest.fixture
def seal(tmp_path, monkeypatch):
    monkeypatch.setattr(vault, "OUT", tmp_path)
    monkeypatch.setattr(vault, "CONFIG", tmp_path / "vault.json")
    monkeypatch.setattr(vault, "MANIFEST", tmp_path / "manifest.jsonl")
    monkeypatch.setattr(vault, "LOG", tmp_path / "vault_log.jsonl")
    monkeypatch.setattr(vault, "SONGS", tmp_path / "songs")
    monkeypatch.setattr(harvest, "SONGS", vault.SONGS)
    monkeypatch.setattr(fill_score, "ProcessPoolExecutor", InlinePool)
    monkeypatch.setattr(fill_seq, "ProcessPoolExecutor", InlinePool)
    vault.init(100.)
    rows = [dict(video=v, at=at, slug=f"song-{v}", channel="drumshin", seconds=seconds, status=status)
            for v, at, seconds, status in [(OLD, 90., 20, "read"), (NEW, 110., 30, "read"),
                                          (LOOSE, 120., 40, "unreadable"), (EDGE, 100., 10, "read")]]
    rows.append(dict(video=OLD, at=160., status="compacted"))
    vault.MANIFEST.write_text("".join(json.dumps(r) + "\n" for r in rows))
    for row in rows[:4]:
        write_notation(vault.SONGS / row["slug"] / "truth", rejected=row["video"] == LOOSE)
    seen = []
    detector = types.ModuleType("vault_test_detector")
    def detect(song):
        seen.append(song.name)
        return [types.SimpleNamespace(time=t + .02, inst=c, ghost=False) for t, c in zip(TIMES, DRUMS)]
    detector.detect = detect
    monkeypatch.setitem(sys.modules, detector.__name__, detector)
    return dict(rows=rows, seen=seen, detector=detector)


def test_first_record_not_latest_update_and_strict_cutoff(seal):
    assert read(vault.MANIFEST)[OLD]["at"] == 160.
    assert not vault.is_vault(OLD)
    assert not vault.is_vault(EDGE)
    assert vault.is_vault(NEW)
    assert vault.list_videos() == sorted([NEW, LOOSE])
    # A crashed/in-progress writer's final row is not a committed arrival.
    with vault.MANIFEST.open("a") as out:
        out.write(json.dumps(dict(video="partial0001", at=200.)))
    assert not vault.is_vault("partial0001")


def test_init_once_and_explicit_add_preserves_cutoff(seal):
    assert vault.init(900.)["cutoff"] == 100.
    vault.add([OLD, OLD, "unseen00001"])
    vault.add("another0001")
    assert vault.config()["cutoff"] == 100.
    assert vault.is_vault(OLD) and vault.is_vault("unseen00001")
    assert vault.list_videos() == sorted([OLD, NEW, LOOSE, "unseen00001", "another0001"])
    assert not vault.LOG.exists()


def test_default_init_uses_current_time_once(seal, monkeypatch):
    from datetime import datetime, timezone
    vault.CONFIG.unlink()
    before = datetime.now(timezone.utc).timestamp()
    cutoff = vault.timestamp(vault.init()["cutoff"])
    assert before <= cutoff <= datetime.now(timezone.utc).timestamp()
    assert vault.timestamp(vault.init()["cutoff"]) == cutoff


@pytest.mark.parametrize("cutoff", ["1970-01-01T00:01:40Z", "1970-01-01T09:01:40+09:00", 100.])
def test_cutoff_formats_are_equivalent(seal, cutoff):
    vault.CONFIG.write_text(json.dumps(dict(cutoff=cutoff, video_ids=[])))
    assert not vault.is_vault(EDGE)
    assert vault.is_vault(NEW)


@pytest.mark.parametrize("cutoff", ["2026-10-02T00:00:00", float("nan"), True])
def test_invalid_cutoff_fails_closed(seal, cutoff):
    vault.CONFIG.write_text(json.dumps(dict(cutoff=cutoff, video_ids=[])))
    with pytest.raises(ValueError):
        score.songs(None)


def test_missing_config_fails_closed(seal):
    vault.CONFIG.unlink()
    with pytest.raises(FileNotFoundError, match="initialize the seal"):
        score.songs(None)


def test_undated_first_row_is_sealed(seal):
    with vault.MANIFEST.open("a") as out:
        out.write(json.dumps(dict(video="undated0001")) + "\n")
        out.write(json.dumps(dict(video="undated0001", at=80.)) + "\n")
    assert vault.is_vault("undated0001")


def test_identity_from_alias_slug_metadata_and_hyphenated_id(seal):
    hyphenated = "ab-cd_efghi"
    vault.add(hyphenated)
    membership = vault.Vault()
    assert membership.contains_song(dict(slug=f"old-cache-title-{hyphenated}"))
    assert membership.contains_song(types.SimpleNamespace(video_id=NEW))
    assert membership.contains_song(dict(slug=f"song-{NEW}"))
    # All manifest slug aliases stay associated with the first arrival.
    with vault.MANIFEST.open("a") as out:
        out.write(json.dumps(dict(video=NEW, slug="renamed", at=300.)) + "\n")
    assert vault.Vault().contains_song(dict(slug=f"song-{NEW}"))


def test_selection_defaults_and_final_are_disjoint_including_loose_rows(seal):
    assert {r["video"] for r in score.songs(None)} == {OLD, EDGE}
    assert {r["video"] for r in score.songs(None, final=True)} == {NEW}
    assert {r["video"] for r in score.songs(None, final=True, loose=True)} == {NEW, LOOSE}
    assert not score.songs(["other"], final=True)


def test_score_final_logs_before_detector_and_only_vault(seal, monkeypatch, capsys):
    def detect(song):
        assert vault.LOG.exists()
        assert song.name == f"song-{NEW}"
        return seal["detector"].detect(song)
    monkeypatch.setattr(score, "detect", detect)
    monkeypatch.setattr(sys, "argv", ["score.py", "--final", "--model-tag", "frozen-v5"])
    score.main()
    assert "OPENING THE SEALED TEST VAULT" in capsys.readouterr().out
    log = [json.loads(line) for line in vault.LOG.read_text().splitlines()]
    assert len(log) == 1 and log[0]["model_tag"] == "frozen-v5"
    assert "--final" in log[0]["command"] and vault.timestamp(log[0]["timestamp"]) > 0


def test_score_default_never_reads_vault_truth(seal, monkeypatch):
    # Sealed corrupt exports would fail if the routine scorer touched them.
    (vault.SONGS / f"song-{NEW}" / "truth" / "truth.json").write_text("broken")
    monkeypatch.setattr(score, "detect", seal["detector"].detect)
    monkeypatch.setattr(sys, "argv", ["score.py"])
    score.main()
    assert set(seal["seen"]) == {f"song-{OLD}", f"song-{EDGE}"}
    assert not vault.LOG.exists()


def test_global_wrapper_logs_original_command_once_and_integer_totals(seal, monkeypatch, tmp_path):
    output = tmp_path / "totals.json"
    monkeypatch.setattr(sys, "argv", ["global_score.py", "--detect", "vault_test_detector:detect",
                                      "--final", "--model-tag", "frozen-global", "--json", str(output)])
    # runpy imports evaluate.detect at execution time; restore it after the wrapper.
    monkeypatch.setattr(global_score.evaluate, "detect", global_score.evaluate.detect)
    global_score.main()
    totals = json.loads(output.read_text())
    assert totals["songs"] == 1 and totals["kick"] == [2, 2, 2]
    assert seal["seen"] == [f"song-{NEW}"]
    log = [json.loads(line) for line in vault.LOG.read_text().splitlines()]
    assert len(log) == 1 and log[0]["model_tag"] == "frozen-global"
    assert "global_score.py" in log[0]["command"] and "--detect" in log[0]["command"]


@pytest.mark.parametrize("final", [False, True])
@pytest.mark.parametrize("loose", [False, True])
def test_fill_cli_sealing_and_final_log(seal, monkeypatch, final, loose, tmp_path):
    output = tmp_path / "fill.json"
    argv = ["fill_score.py", "--jobs", "1", "--gap", "0.4", "--detect", "vault_test_detector:detect", "--json", str(output)]
    if final:
        argv += ["--final", "--model-tag", "frozen-fill"]
    if loose:
        argv += ["--loose"]
    monkeypatch.setattr(sys, "argv", argv)
    fill_score.main()
    expected = {NEW, LOOSE} if final and loose else {NEW} if final else {OLD, EDGE}
    assert set(seal["seen"]) == {f"song-{v}" for v in expected}
    if loose:
        totals = json.loads(output.read_text())
        assert totals["systems"] == len(expected)
        assert totals["fast.stroke.ref"] == 3 * len(expected)
        assert totals["fast.stroke.tp"] == 3 * len(expected)
    if final:
        log = [json.loads(line) for line in vault.LOG.read_text().splitlines()]
        assert len(log) == 1 and log[0]["model_tag"] == "frozen-fill"
    else:
        assert not vault.LOG.exists()


def test_loose_reuses_existing_dp_counts_for_timing_rejection(seal):
    row = vault.Vault().rows[LOOSE]
    folder = vault.SONGS / row["slug"]
    assert not score.load(folder / "truth", "strict")[0]
    a, b, ref = fill_seq.notated(folder / "truth", .75)[0]
    hits = seal["detector"].detect(folder)
    expected = fill_seq.tally(ref, fill_seq.detected(hits, a - .05, b + .05), .06)
    _, got = fill_score.one_loose((row, "vault_test_detector:detect", .06, .75))
    assert all(got[k] == value for k, value in expected.items())
    assert got["systems"] == 1 and got["all.stroke.ref"] == 6


@pytest.mark.parametrize("failure", ["phase alias", "page row not visited by cursor", "warp is not strictly monotone",
                                     "unexportable", "staff", "low agreement", "nan agreement", "bad times", "nan", "mapping"])
def test_loose_does_not_rescue_reading_phase_or_invalid_times(seal, failure):
    folder = vault.SONGS / f"song-{LOOSE}" / "truth"
    report = json.loads((folder / "alignment.json").read_text())
    strip = json.loads((folder / "strip.json").read_text())
    r = report["systems"][0]
    if failure == "unexportable":
        strip[0]["notes"][0]["exportable"] = False
    elif failure == "staff":
        strip[0]["staff"]["reasons"] = ["unverified geometry"]
    elif failure == "low agreement":
        r["after"]["within_40ms"] = .74
    elif failure == "nan agreement":
        r["after"]["within_40ms"] = float("nan")
    elif failure == "bad times":
        r["times"][1] = r["times"][0]
    elif failure == "nan":
        r["times"][1] = float("nan")
    elif failure == "mapping":
        r["times"].pop()
    else:
        r["reasons"].append(failure)
    (folder / "alignment.json").write_text(json.dumps(report))
    (folder / "strip.json").write_text(json.dumps(strip))
    assert fill_seq.notated(folder, .75) == []


def test_stats_counts_only_sealed_eligible_notation_without_opening(seal):
    vault.add("unseen00001")
    assert vault.stats() == dict(count=3, seconds=70., notated_seconds=2., systems=2,
                                fast_moments=dict(kick=2, snare=2, tom=2), missing_notation=1)
    assert not seal["seen"] and not vault.LOG.exists()


def test_nearest_gap_and_dp_one_to_one_counts():
    ref = [(0., frozenset({"kick"})), (.04, frozenset({"snare"})), (.09, frozenset({"tom"})),
           (.4, frozenset({"tom"}))]
    assert fill_seq.fast_mask(ref).tolist() == [True, True, True, False]
    assert fill_seq.fast_mask([(0., set()), (.09, set())]).tolist() == [False, False]
    assert not fill_seq.fast_mask([]).size
    tally = fill_seq.tally(ref, [(0.02, frozenset({"kick"}))], .06)
    assert tally["all.stroke.tp"] == 1 and tally["fast.stroke.ref"] == 3
    assert tally["fast.kick.ref"] == tally["fast.snare.ref"] == tally["fast.tom.ref"] == 1


def test_incomplete_log_refuses_open_before_evaluation(seal):
    vault.LOG.write_text('{"timestamp":')
    with pytest.raises(ValueError, match="refusing to open"):
        vault.open_final("frozen")
    assert vault.LOG.read_text() == '{"timestamp":'


def test_repeated_openings_append_and_model_tag_resolution(seal, monkeypatch):
    monkeypatch.setenv("FT_TAG", "frozen-ft")
    assert vault.model_tag("evaluate:detect") == "frozen-ft"
    assert vault.model_tag("evaluate:detect", "explicit") == "explicit"
    vault.open_final("first", "scorer --final")
    vault.open_final("second", "scorer --final --loose")
    rows = [json.loads(line) for line in vault.LOG.read_text().splitlines()]
    assert [r["model_tag"] for r in rows] == ["first", "second"]


@pytest.mark.parametrize('existing', [False, True])
def test_mac_mirror_refuses_final_without_log_changes(seal, existing, monkeypatch):
    if existing:
        vault.open_final('previous')
    before = vault.LOG.read_bytes() if vault.LOG.exists() else None
    (vault.OUT / '.wsl-mirror').touch()
    monkeypatch.setattr(score, 'songs', lambda *a, **kw: pytest.fail('mirror read sealed songs'))
    monkeypatch.setattr(sys, 'argv', ['score.py', '--final'])
    with pytest.raises(RuntimeError, match='WSL owns final scoring'):
        score.main()
    assert (vault.LOG.read_bytes() if vault.LOG.exists() else None) == before


def test_fill_final_reuses_verified_opening_once(seal, monkeypatch):
    opening = vault.open_final('runner', 'chops_compare.py final')
    monkeypatch.setattr(vault.os, 'getppid', vault.os.getpid)
    monkeypatch.setattr(sys, 'argv', ['fill_score.py', '--loose', '--final', '--jobs', '1',
                                    '--model-tag', 'runner', '--opening-id', opening,
                                    '--detect', 'vault_test_detector:detect'])
    fill_score.main()
    rows = [json.loads(line) for line in vault.LOG.read_text().splitlines()]
    assert len(rows) == 1 and rows[0]['opening_id'] == opening
    assert set(seal['seen']) == {f'song-{NEW}', f'song-{LOOSE}'}


@pytest.mark.parametrize('loose', [False, True])
def test_mixed_group_final_excludes_sealed_before_truth_and_logs_reason(seal, monkeypatch, tmp_path, capsys, loose):
    # OLD is routine/training material; transitive links exclude sealed NEW.
    table = {'groups': [[OLD, 'bridge00001'], ['bridge00001', NEW]],
             'source': 'offline audio matching', 'generated': 'fixture'}
    (vault.OUT / 'same_performance.json').write_text(json.dumps(table))
    membership = vault.Vault()
    assert not membership.contains_song({'video': OLD})
    assert membership.contains_song({'video': NEW})
    assert {r['video'] for r in score.songs(None, final=True, loose=True)} == {LOOSE}
    # A corrupt excluded export must never be read, even to check readability.
    for name in ('truth.json', 'strip.json', 'alignment.json'):
        (vault.SONGS / f'song-{NEW}' / 'truth' / name).write_text('excluded truth must not be read')
    output = tmp_path / 'final.json'
    argv = ['fill_score.py', '--final', '--jobs', '1', '--gap', '0.4',
            '--detect', 'vault_test_detector:detect', '--model-tag', 'fixture', '--json', str(output)]
    if loose:
        argv.append('--loose')
    monkeypatch.setattr(sys, 'argv', argv)
    fill_score.main()
    assert set(seal['seen']) == ({f'song-{LOOSE}'} if loose else set())
    result = json.loads(output.read_text())
    excluded = result['excluded'][NEW]
    assert excluded['train_peers'] == sorted([OLD, 'bridge00001'])
    log_line = f"{NEW}: excluded: train peer " + ' '.join(excluded['train_peers'])
    assert log_line in capsys.readouterr().out
    assert log_line in output.with_suffix('.md').read_text()
    if loose:
        assert result['fast.stroke.ref'] == 3 and result['systems'] == 1
    assert len(vault.LOG.read_text().splitlines()) == 1
    # Sealing remains intact, so the excluded row cannot become routine data.
    assert {r['video'] for r in score.songs(None)} == {OLD, EDGE}
    assert vault.Vault().contains_song({'video': NEW})


def test_mixed_group_stats_omit_excluded_notation_and_fast_counts(seal):
    (vault.OUT / 'same_performance.json').write_text(json.dumps({'groups': [[NEW, OLD]]}))
    (vault.SONGS / f'song-{NEW}' / 'truth/strip.json').write_text('excluded, never read')
    result = vault.stats()
    assert result['count'] == 1 and result['seconds'] == 40.
    assert result['systems'] == 1 and result['fast_moments'] == dict(kick=1, snare=1, tom=1)
    assert result['excluded'][NEW]['reason'] == f'excluded: train peer {OLD}'
    assert not seal['seen'] and not vault.LOG.exists()


def test_final_exclusion_resolves_old_slug_aliases_without_unsealing(seal):
    original_slug = f'song-{NEW}'
    with vault.MANIFEST.open('a') as out:
        out.write(json.dumps({'video': NEW, 'slug': 'renamed_song', 'at': 200.}) + '\n')
    (vault.OUT / 'same_performance.json').write_text(json.dumps({'groups': [[NEW, OLD]]}))
    membership = vault.Vault()
    assert membership.contains_song({'slug': original_slug})
    assert membership.contains_song({'slug': 'renamed_song'})
    rows, excluded = vault.filter_final([{'slug': original_slug}, {'slug': 'renamed_song'}], log=False)
    assert rows == [] and excluded[NEW]['train_peers'] == [OLD]


def test_builtin_timbuell_performance_is_auto_sealed_but_not_vault_material(seal):
    train, tim = 'N0PXFlgigjw', '6CLJH8YJtFM'
    config = vault.config()
    config['train_channels'] = ['alexblack']
    vault.CONFIG.write_text(json.dumps(config))
    with vault.MANIFEST.open('a') as out:
        for video, channel in [(train, 'alexblack'), (tim, 'timbuell')]:
            out.write(json.dumps({'video': video, 'channel': channel, 'slug': f'song-{video}',
                                  'at': 200., 'status': 'read'}) + '\n')
    membership = vault.Vault()
    assert not membership.is_vault(train)
    assert membership.is_vault(tim)
    assert train in membership.final_exclusions()[tim]['train_peers']
    assert tim not in {r['video'] for r in score.songs(None, final=True)}
    assert train in {r['video'] for r in score.songs(None)}


@pytest.mark.parametrize('failure', ['json', 'shape', 'unreadable'])
def test_final_table_load_failure_prevents_opening_and_scoring(seal, monkeypatch, failure):
    path = vault.OUT / 'same_performance.json'
    path.write_text('broken' if failure == 'json' else '{"groups": [["one"]]}')
    if failure == 'unreadable':
        original = Path.read_bytes
        def read(self):
            if self == path:
                raise PermissionError('fixture unreadable table')
            return original(self)
        monkeypatch.setattr(Path, 'read_bytes', read)
    monkeypatch.setattr(fill_score, 'songs', lambda *a, **kw: pytest.fail('bad table selected sealed songs'))
    monkeypatch.setattr(sys, 'argv', ['fill_score.py', '--final'])
    with pytest.raises((ValueError, PermissionError)):
        fill_score.main()
    assert not vault.LOG.exists() and not seal['seen']


def test_real_child_can_validate_parent_opening_without_second_log_entry(seal):
    opening = vault.open_final('runner')
    code = ('import sys; from pathlib import Path; '
            'sys.path.insert(0, sys.argv[1]); import vault; '
            'vault.OUT=Path(sys.argv[2]); vault.CONFIG=vault.OUT/"vault.json"; '
            'vault.LOG=vault.OUT/"vault_log.jsonl"; vault.resume_final(sys.argv[3], "runner")')
    result = subprocess.run([sys.executable, '-c', code, str(Path(__file__).parent),
                             str(vault.OUT), opening], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert len(vault.LOG.read_text().splitlines()) == 1


@pytest.mark.parametrize('problem', ['unknown', 'tag', 'unfinished', 'mirror', 'parent'])
def test_opening_reuse_fails_closed_before_selection(seal, monkeypatch, problem):
    opening = vault.open_final('runner')
    monkeypatch.setattr(vault.os, 'getppid', (lambda: -1) if problem == 'parent' else vault.os.getpid)
    if problem == 'unknown':
        opening = 'unknown'
    if problem == 'unfinished':
        with vault.LOG.open('ab') as stream:
            stream.write(b'{"partial":')
    if problem == 'mirror':
        (vault.OUT / '.wsl-mirror').touch()
    before = vault.LOG.read_bytes()
    monkeypatch.setattr(fill_score, 'songs', lambda *a, **kw: pytest.fail('invalid opening read sealed songs'))
    monkeypatch.setattr(sys, 'argv', ['fill_score.py', '--final', '--model-tag',
                                    'wrong' if problem == 'tag' else 'runner', '--opening-id', opening])
    with pytest.raises((ValueError, RuntimeError)):
        fill_score.main()
    assert vault.LOG.read_bytes() == before


@pytest.mark.parametrize("final", [False, True])
def test_solo_pointer_to_harvested_song_cannot_bypass_seal(seal, monkeypatch, tmp_path, final):
    solo_root = tmp_path / "solos"
    monkeypatch.setattr(fill_seq, "TRUTH", solo_root)
    for label, video in [("old-solo", OLD), ("new-solo", NEW)]:
        folder = solo_root / label
        write_notation(folder)
        song = vault.SONGS / f"song-{video}"
        (folder / "song.txt").write_text(str(song))
        np.save(song / "activations.drums-htdemucs.npy", np.zeros((1, 5)))
    argv = ["fill_seq.py", "--videos", "old-solo", "new-solo", "--jobs", "1",
            "--detect", "vault_test_detector:detect"] + (["--final", "--model-tag", "frozen-solo"] if final else [])
    monkeypatch.setattr(sys, "argv", argv)
    fill_seq.main()
    assert seal["seen"] == [f"song-{NEW if final else OLD}"]
    assert vault.LOG.exists() == final
