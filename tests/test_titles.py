import json

import pytest

from stickmap import library
from stickmap.fetch import SongMeta
from stickmap.library import clean_title


@pytest.mark.parametrize(
    ("raw", "channel", "expected"),
    [
        ("Redline Dash (Drum Cover) #drums #shorts | Ivannikov", "Ivannikov", "Redline Dash"),
        ("Metallica - Enter Sandman [Drum Playthrough]", "", "Metallica - Enter Sandman"),
        ("【드럼커버】 아이유 - 좋은 날 #drumcover", "", "아이유 - 좋은 날"),
        ("NewJeans (뉴진스) 'Super Shy' Drum Cover | 드럼신", "드럼신", "NewJeans (뉴진스) 'Super Shy'"),
        ("드럼 커버 | 잔나비 - 주저하는 연인들을 위해", "", "잔나비 - 주저하는 연인들을 위해"),
        ("Toto - Rosanna (Drum Cam) - Golly", "Golly", "Toto - Rosanna"),
        ("Song Name - Official Video", "", "Song Name"),
        ("[MV] IU(아이유) _ Palette(팔레트) (feat. G-DRAGON)", "", "IU(아이유) _ Palette(팔레트) (feat. G-DRAGON)"),
        # Words that are often real song names stay outside brackets.
        ("Live Wire drum cover", "", "Live Wire"),
        ("Cover Me (Live) - Bruce", "Bruce", "Cover Me"),
        # A trailing name is only cut when it is the known channel.
        ("Groove 17 | Daily Chops", "Ivannikov", "Groove 17 | Daily Chops"),
        ("Rosanna - Golly", "", "Rosanna - Golly"),
    ],
)
def test_clean_title(raw, channel, expected):
    assert clean_title(raw, channel) == expected


def test_all_noise_title_falls_back_to_raw():
    assert clean_title("#drums #shorts") == "#drums #shorts"
    assert clean_title("(Drum Cover) | Ivannikov", "Ivannikov") == "(Drum Cover) | Ivannikov"


def _song(tmp_path, **extra):
    song = tmp_path / "song"
    song.mkdir()
    meta = SongMeta(kind="youtube", source="https://youtu.be/x", title="Redline Dash (Drum Cover) #shorts | Ivannikov",
                    uploader="Ivannikov", video_id="x")
    meta.save(song / "meta.json")
    if extra:
        data = json.loads((song / "meta.json").read_text(encoding="utf-8")) | extra
        (song / "meta.json").write_text(json.dumps(data), encoding="utf-8")
    (song / "audio.wav").write_bytes(b"")
    return song


def test_summary_shows_clean_title_and_keeps_raw(tmp_path):
    summary = library.summarize(_song(tmp_path))
    assert summary["title"] == "Redline Dash"
    assert summary["raw_title"] == "Redline Dash (Drum Cover) #shorts | Ivannikov"


def test_title_override_wins(tmp_path):
    song = _song(tmp_path, title_override="레드라인 대시")
    assert library.summarize(song)["title"] == "레드라인 대시"
    assert library._saved_meta(song) is not None  # an extra field does not break loading


def test_local_file_titles_are_not_cleaned():
    meta = SongMeta(kind="file", source="/x/my live take.wav", title="my live take (cover)")
    assert library.display_title(meta) == "my live take (cover)"
