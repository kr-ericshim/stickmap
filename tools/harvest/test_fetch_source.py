import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fetch_source as fs  # noqa: E402


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    (tmp_path / 'lists').mkdir()
    (tmp_path / 'lists' / 'yt.txt').write_text(''.join(f'vid{i:08d}|NA|title {i}\n' for i in range(6)))
    (tmp_path / 'lists' / 'ig.txt').write_text(''.join(f'reel{i:07d}|NA|\n' for i in range(70)))
    (tmp_path / 'channels.json').write_text(json.dumps({'channels': [
        {'name': 'yt', 'layout': 'x'},
        {'name': 'ig', 'layout': 'y', 'url_template': 'https://www.instagram.com/reel/{id}/'}]}))
    monkeypatch.setattr(fs, 'SOURCES', tmp_path / 'sources')
    monkeypatch.setattr(fs, 'LISTS', tmp_path / 'lists')
    monkeypatch.setattr(fs, 'CHANNELS', tmp_path / 'channels.json')
    return tmp_path


class Clock:
    def __init__(self):
        self.t = 1_800_000_000.0
        self.sleeps = []

    def now(self):
        return self.t

    def sleep(self, s):
        self.sleeps.append(s)
        self.t += s


def fake(results):
    calls = []

    def downloader(url, fmt, target):
        calls.append((url, fmt))
        ok, text = results[len(calls) - 1] if len(calls) <= len(results) else (True, '')
        if ok:
            (target / 'video.mp4').write_bytes(b'x')
        return ok, text
    return downloader, calls


def run(channel, results, **kw):
    clock = Clock()
    downloader, calls = fake(results)
    out = fs.fetch(channel, kw.get('ids'), kw.get('limit'), kw.get('clear', False),
                   sleep=clock.sleep, now=clock.now, rng=lambda: .5, downloader=downloader)
    return out, calls, clock


def test_youtube_gaps_format_and_skip(sandbox):
    out, calls, clock = run('yt', [(True, '')] * 3, limit=3)
    assert out['ok'] == ['vid00000000', 'vid00000001', 'vid00000002'] and out['stopped'] is None
    assert calls[0] == ('https://www.youtube.com/shorts/vid00000000', fs.FORMAT)
    assert clock.sleeps == [11.5, 11.5]  # 8-15 s between requests, none before the first
    out, calls, _ = run('yt', [(True, '')] * 10)
    assert [c[0][-11:] for c in calls] == ['vid00000003', 'vid00000004', 'vid00000005']  # finished ones skipped


def test_youtube_stops_after_two_consecutive_failures(sandbox):
    out, calls, _ = run('yt', [(False, 'HTTP Error 403'), (True, ''), (False, 'x'), (False, 'y')])
    assert len(calls) == 4 and out['stopped'].startswith('2 consecutive')
    assert (fs.SOURCES / 'yt' / 'STOP.json').exists()
    with pytest.raises(SystemExit, match='clear-stop'):
        run('yt', [])
    out, calls, _ = run('yt', [(True, '')] * 5, clear=True)  # YouTube needs no waiting period
    assert out['stopped'] is None


def test_instagram_stops_at_first_failure_and_waits_an_hour(sandbox):
    out, calls, clock = run('ig', [(True, ''), (False, 'HTTP Error 429: Too Many Requests')])
    assert len(calls) == 2 and 'suspicious' in out['stopped']
    assert 45 <= clock.sleeps[0] <= 90
    with pytest.raises(SystemExit, match='min old'):
        run('ig', [], clear=True)
    stop = fs.SOURCES / 'ig' / 'STOP.json'
    data = json.loads(stop.read_text())
    data['unix'] -= 7200
    stop.write_text(json.dumps(data))
    out, calls, _ = run('ig', [(True, '')], clear=True, limit=1)
    assert out['ok'] and not stop.exists()


def test_instagram_rest_and_daily_cap(sandbox):
    out, calls, clock = run('ig', [(True, '')] * 70)
    assert len(calls) == 60 and out['stopped'].startswith('daily cap 60')
    rests = [s for s in clock.sleeps if s >= 600]
    assert len(rests) == 2  # after requests 20 and 40


def test_max_height_uses_watch_url_and_height_cap(sandbox):
    clock = Clock()
    downloader, calls = fake([(True, '')])
    fs.fetch('yt', None, 1, False, sleep=clock.sleep, now=clock.now, rng=lambda: .5, downloader=downloader,
             max_height=1080)
    assert calls == [('https://www.youtube.com/watch?v=vid00000000', fs.height_format(1080))]
    assert '[height<=1080][vcodec^=avc1]' in calls[0][1] and 'width' not in calls[0][1]
    with pytest.raises(SystemExit, match='YouTube'):
        fs.fetch('ig', None, 1, False, sleep=clock.sleep, now=clock.now, rng=lambda: .5,
                 downloader=downloader, max_height=1080)


def test_unknown_ids_rejected(sandbox):
    with pytest.raises(SystemExit, match='not in yt list'):
        run('yt', [], ids=['nope0000000'])


def test_never_touches_harvest_or_detector():
    text = Path(fs.__file__).read_text()
    for forbidden in ('manifest', 'vault', 'harvest-writer'):
        assert forbidden not in text.split('"""', 2)[2].lower(), forbidden
    code = ('import sys; sys.path.insert(0, "tools/harvest"); import fetch_source as f; f.attack_function(); '
            'bad=[m for m in sys.modules if m.startswith(("groovemap", "adtof", "tools.harvest.score"))]; '
            'print(bad); raise SystemExit(bool(bad))')
    run = subprocess.run([sys.executable, '-c', code], cwd=fs.ROOT, capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr


def test_attack_function_matches_fills(tmp_path):
    import numpy as np
    import soundfile as sf
    rng = np.random.default_rng(0)
    audio = rng.normal(0, .01, 44100 * 3).astype('float32')
    for t in (0.5, 1.0, 1.7, 2.4):
        i = int(t * 44100)
        audio[i:i + 2000] += np.hanning(4000)[2000:].astype('float32') * rng.normal(0, 1, 2000).astype('float32')
    path = tmp_path / 'drums.wav'
    sf.write(path, audio, 44100)
    sys.path.insert(0, str(fs.ROOT / 'src'))
    from groovemap.fills import attack_times
    assert np.array_equal(fs.attack_function()(path), attack_times(path))
