import json

import numpy as np
import pytest

from groovemap import pipeline
from groovemap.fetch import SongMeta
from groovemap.score import Hit, Score


@pytest.mark.parametrize("times,merged", [([9.54, 9.59], 1), ([9.50, 9.59], 0)])
def test_pipeline_reports_only_nonzero_quantization_merges(tmp_path, monkeypatch, times, merged):
    np.save(tmp_path / "activations.mix.npy", np.zeros((1, 5)))
    (tmp_path / "beats.json").write_text(json.dumps({"beats": [i * 0.6 for i in range(16)], "downbeats": [0]}))
    monkeypatch.setattr(pipeline.models, "pick_device", lambda _: "cpu")
    monkeypatch.setattr(pipeline.models, "pick_hits", lambda *args: [Hit(t, "tom", 1) for t in times])
    logs = []
    result = pipeline.run(tmp_path, SongMeta("file", "test.wav", "Terminal pair"),
                          pipeline.Options(onset_input="mix", pdf=False), log=logs.append)
    assert result.score.merged_strokes == merged
    assert Score.load(result.files["json"]) == result.score
    assert [line for line in logs if line.startswith("quantization:")] == (
        ["quantization: 1 strokes merged into occupied slots"] if merged else []
    )


def test_sound_names_are_validated_and_ignored_without_a_stem(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="sound names"):
        pipeline.Options(sound_names={"kick-1": "cowbell"})
    np.save(tmp_path / "activations.mix.npy", np.zeros((1, 5)))
    (tmp_path / "beats.json").write_text(json.dumps({"beats": [i * 0.6 for i in range(16)], "downbeats": [0]}))
    monkeypatch.setattr(pipeline.models, "pick_device", lambda _: "cpu")
    monkeypatch.setattr(pipeline.models, "pick_hits", lambda *args: [Hit(1.2, "kick", 1)])
    monkeypatch.setattr(pipeline.kit, "load_or_learn", lambda *args: pytest.fail("no sound map on the mix"))
    result = pipeline.run(tmp_path, SongMeta("file", "test.wav", "Mix"),
                          pipeline.Options(onset_input="mix", pdf=False, sound_names={"kick-1": "snare"}), log=lambda line: None)
    assert [n.inst for bar in result.score.bars for n in bar.notes] == ["kick"]
    assert not (tmp_path / "soundmap.json").exists()


def test_sound_names_rename_hits_before_tom_sizing(tmp_path, monkeypatch):
    from groovemap import kit

    for name in ("activations.drums-htdemucs.npy", "activations.mix.npy"):
        np.save(tmp_path / name, np.zeros((1, 5)))
    np.save(tmp_path / "snare.htdemucs.npy", np.zeros((0, 2)))
    np.save(tmp_path / "attacks.htdemucs.npy", np.array([2.4]))
    (tmp_path / "drums.htdemucs.wav").write_bytes(b"")
    (tmp_path / "beats.json").write_text(json.dumps({"beats": [i * 0.6 for i in range(16)], "downbeats": [0]}))
    monkeypatch.setattr(pipeline.models, "pick_device", lambda _: "cpu")
    monkeypatch.setattr(pipeline.models, "pick_hits", lambda *args, **kw: [Hit(1.2, "kick", 1), Hit(1.8, "kick", 1)])
    calls = []
    sound_map = kit.SoundMap([kit.SoundGroup("kick-1", "kick", 2, 0, [1.0], [], {})], {})
    monkeypatch.setattr(pipeline.kit, "load_or_learn", lambda song_dir, *args: (calls.append(song_dir), sound_map)[1])
    monkeypatch.setattr(pipeline.kit, "rename", lambda hits, found, names, *args: [Hit(h.time, names["kick-1"], h.strength) for h in hits] + [Hit(2.4, "snare", 1)])
    monkeypatch.setattr(pipeline, "tom_sizes", lambda drums, hits: (calls.append([h.inst for h in hits]), hits)[1])
    result = pipeline.run(tmp_path, SongMeta("file", "test.wav", "Named"), pipeline.Options(pdf=False, sound_names={"kick-1": "tom"}),
                          log=lambda line: None)
    assert calls == [tmp_path, ["tom", "tom", "snare"]]  # renamed first, then sized
    assert sorted(n.inst for bar in result.score.bars for n in bar.notes) == ["snare", "tom", "tom"]
