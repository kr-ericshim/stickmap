import numpy as np

from stickmap.dynamics import annotate_confidence, ghost_states
from stickmap.quantize import quantize
from stickmap.score import Hit


def test_only_clear_soft_notes_with_confident_identity_get_parentheses():
    hits = [Hit(i * 0.5, "snare", 1, confidence=0.9) for i in range(20)]
    hits += [Hit(10, "snare", 0.2, True, 0.9), Hit(10.5, "snare", 0.2, True, 0.3), Hit(11, "snare", 0.45, True, 0.9)]
    states = ghost_states(hits)
    assert states[10, "snare"] == "ghost"
    assert states[10.5, "snare"] == states[11, "snare"] == "unknown"
    beats = np.arange(25) * 0.5
    score = quantize(hits, beats, beats[::4])
    notes = {n.raw_time: n for b in score.bars for n in b.notes}
    assert notes[10].ghost and not notes[10.5].ghost and not notes[11].ghost
    assert notes[10].velocity == notes[10.5].velocity


def test_relative_dynamics_do_not_force_a_ghost_quota_or_depend_on_absolute_scale():
    for scale in (0.05, 1, 10):
        uniform = [Hit(i * 0.5, "snare", scale * (1 + i / 100), confidence=0.9) for i in range(24)]
        assert set(ghost_states(uniform).values()) == {"normal"}
        uniform += [Hit(12, "snare", scale * 0.15, True, 0.9)]
        assert ghost_states(uniform)[12, "snare"] == "ghost"


def test_tom_overlap_without_snare_share_is_uncertain():
    activations = np.zeros((100, 5))
    activations[50, 1] = 0.9
    response = np.zeros((100, 2))
    response[50] = [0.02, 0.1]
    hits = [Hit(0.5, "snare", 0.02, True), Hit(0.5, "tom", 1)]
    annotated = annotate_confidence(hits, activations, response)
    assert annotated[0].confidence < 0.65
    response[50, 1] = 0.8
    assert annotate_confidence(hits, activations, response)[0].confidence >= 0.65


def test_preview_timing_is_raw_and_legacy_scores_still_use_the_display_map():
    from stickmap.score import Bar, Note, Score

    bar = Bar(0, 1, [0, 0.5], [4], 120, [Note(12, "snare", 90, raw_time=0.29)])
    assert bar.tick_time(12) == 0.25
    assert bar.note_time(bar.notes[0]) == 0.29
    assert Score.from_json(Score("Raw", [bar]).to_json()).bars[0].notes[0].raw_time == 0.29
    assert bar.note_time(Note(12, "snare", 90)) == 0.25
