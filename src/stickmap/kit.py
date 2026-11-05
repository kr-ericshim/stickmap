"""A song's sound map: the detected hits grouped by sound, for a listener to name.

On an unusual kit the detector mislabels whole sounds, and differently in each
recording: a dead snare comes out as kick, a high one as tom. No automatic rule
has told such sounds apart without hurting ordinary kits, so the correction is
opt-in and by ear. Each detected class is split into a few timbre groups, the
attacks the detector left unexplained form groups of their own, and the user
listens to a few clean examples of each group and names it. Names replay from
the saved map without hearing the audio again, and a group's id keeps meaning
the same sound when the map is rebuilt, so names outlive a retranscription.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .library import file_fingerprint
from .models import ADTOF_CLASSES, ADTOF_FPS, GHOST_RATIO
from .score import INSTRUMENTS, Hit, family

NAMES = (*INSTRUMENTS, "ignore")
EXTRA = "extra"  # id prefix of the groups of attacks no detected hit explains
FILE = "soundmap.json"
_BANDS = 32  # mel bands, pooled from the spectrogram's 128: enough to tell drums apart, few enough to cluster
_MAX_GROUPS = 4  # per detected class
_MIN_MEMBERS = 8  # a sound needs this many strokes to be a group of its own
_NEAR = 0.05  # seconds: an attack this close to a hit is that hit, or its flam
_TOGETHER = 0.03  # seconds: hits this close were struck together, so their spectrum is a mixture
_CLEAN = 0.08  # seconds: an example has no other stroke this close
_SPREAD = 0.03  # share of the song between examples, so they come from different sections
_EXAMPLES = 6
_CLIP = (0.6, 0.4)  # seconds before and after an example that a clip should loop
_SPLIT = 1.5  # held-out separation above which the two halves of a group are two sounds (one Gaussian: 1.33)
_SAME = 0.9  # similarity at which a rebuilt group is the same sound as an old one and keeps its id


@dataclass(frozen=True)
class Example:
    time: float  # seconds into the song, the same timeline as the stem, the mix and the video
    start: float  # a clip to loop, from just before the stroke to after its ring
    end: float


@dataclass(frozen=True)
class SoundGroup:
    id: str
    detected: str | None  # the detector's class; None for attacks it left unexplained
    count: int
    mixed: int  # members struck together with another drum, whose spectrum is a mixture
    centre: list[float]  # typical spectrum, square roots of _BANDS unit-sum bands
    examples: list[Example]
    # How close this sound is to each detected class's main sound in this song. Descriptive only: on the
    # JD Beck truth, naming a group after its nearest class was wrong more often than right.
    similar: dict[str, float]


@dataclass(frozen=True)
class SoundMap:
    groups: list[SoundGroup]
    inputs: dict[str, str]  # cache file -> fingerprint of the version the map was learned from
    version: int = 1

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> SoundMap:
        if data.get("version") != 1:
            raise ValueError("unsupported sound map version")
        groups = [
            SoundGroup(**{**row, "examples": [Example(**example) for example in row["examples"]]})
            for row in data["groups"]
        ]
        if len({group.id for group in groups}) != len(groups):
            raise ValueError("sound group ids must be unique")
        return cls(groups, dict(data["inputs"]))

    def save(self, song_dir: Path) -> None:
        path = song_dir / FILE
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self.to_dict(), allow_nan=False), encoding="utf-8")
        temporary.replace(path)

    @classmethod
    def load(cls, song_dir: Path) -> SoundMap | None:
        """The saved map; None when there is none or it is unreadable, which means learning again."""
        try:
            return cls.from_dict(json.loads((song_dir / FILE).read_text(encoding="utf-8")))
        except (OSError, ValueError, KeyError, TypeError):
            return None


def _spectra(mel: np.ndarray, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """What each stroke added, as square roots of unit-sum bands, and how loud it was.

    The peak of the 50 ms after the onset less the level just before it, so what
    was already ringing stays out. Square roots let quiet bands count without
    letting a quiet stroke look like another sound.
    """
    padded = np.pad(mel, ((0, 0), (5, 10)), mode="edge")
    added = []
    for time in times:
        f = int(round(time * ADTOF_FPS)) + 5
        before = padded[:, f - 5:f - 2].mean(axis=1)
        added.append(np.maximum(padded[:, f - 1:f + 4].max(axis=1) - before, 0))
    added = np.array(added, dtype=float).reshape(-1, mel.shape[0])
    per_band = mel.shape[0] // _BANDS
    pooled = added[:, :_BANDS * per_band].reshape(len(added), _BANDS, per_band).sum(axis=2)
    loudness = pooled.sum(axis=1)
    return np.sqrt(pooled / np.maximum(loudness, 1e-9)[:, None]), loudness


def _similarity(features: np.ndarray, centre: np.ndarray) -> np.ndarray:
    """Cosine similarity, 1 for the same spectrum; for these features the Bhattacharyya coefficient."""
    norms = np.linalg.norm(features, axis=-1) * np.linalg.norm(centre)
    return features @ centre / np.maximum(norms, 1e-9)


def _kmeans(x: np.ndarray, k: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """k-means++ seeding and Lloyd updates; seeded, so the same song gives the same groups."""
    rng = np.random.default_rng(seed)
    centres = [x[int(rng.integers(len(x)))]]
    for _ in range(1, k):
        distance = np.min(np.sum((x[:, None] - np.array(centres)) ** 2, axis=2), axis=1)
        if distance.max() < 1e-12:
            break
        centres.append(x[rng.choice(len(x), p=distance / distance.sum())])
    centres = np.array(centres)
    labels = None
    for _ in range(100):
        nearest = _nearest(x, centres)
        if labels is not None and np.array_equal(nearest, labels):
            break
        labels = nearest
        centres = np.array([x[labels == j].mean(axis=0) if np.any(labels == j) else centres[j] for j in range(len(centres))])
    return centres, _nearest(x, centres)


def _nearest(x: np.ndarray, centres: np.ndarray) -> np.ndarray:
    return np.argmin(np.sum((x[:, None] - centres[None]) ** 2, axis=2), axis=1)


def _split(y: np.ndarray) -> tuple[float, np.ndarray]:
    """How clearly a group falls in two along its best two-way split, and which strokes are on the far side.

    The split is found on one half of the strokes and judged on the other, both
    ways round: the gap between the two sides' means along the split axis, over
    their spread. Held out like this, one Gaussian sound reads 1.33 however many
    strokes there are (k-means alone would split it, and ever more confidently
    as they grow), while two sounds read far more. Strokes that are the same to
    rounding are one sound; sides under _MIN_MEMBERS are not sounds.
    """
    # Shuffled halves: by time, a backbeat's kick and its mislabelled snare would alternate into separate halves.
    order = np.random.default_rng(len(y)).permutation(len(y))
    halves = (y[order[0::2]], y[order[1::2]])
    none = 0.0, np.zeros(len(y), dtype=bool)
    scores, sides = [], []
    for fit, judge in ((0, 1), (1, 0)):
        fits = [_kmeans(halves[fit], 2, seed) for seed in range(5)]
        two, _ = min(fits, key=lambda found: np.sum((halves[fit] - found[0][found[1]]) ** 2))
        axis = two[-1] - two[0]
        if len(two) < 2 or np.linalg.norm(axis) < 1e-6:
            return none
        axis /= np.linalg.norm(axis)
        middle = float((two[0] + two[1]) @ axis / 2)
        projected = halves[judge] @ axis
        far = projected > middle
        if min(far.sum(), (~far).sum()) < 3:
            return none
        spread = projected[far].std() + projected[~far].std()
        scores.append((projected[far].mean() - projected[~far].mean()) / max(spread, 1e-12))
        sides.append(y @ axis > middle)
    side = sides[0]
    if min(side.sum(), (~side).sum()) < _MIN_MEMBERS:
        return 0.0, side
    return float(np.mean(scores)), side


def _centres(x: np.ndarray) -> np.ndarray:
    """As many centres as the strokes support, up to _MAX_GROUPS: a group splits while it is clearly two sounds."""
    labels = np.zeros(len(x), dtype=int)
    centres = [x.mean(axis=0)]
    while len(centres) < _MAX_GROUPS:
        candidates = [(*_split(x[labels == j]), j) for j in range(len(centres)) if np.sum(labels == j) >= 2 * _MIN_MEMBERS]
        best = max(candidates, key=lambda candidate: candidate[0], default=None)
        if best is None or best[0] <= _SPLIT:
            break
        _, side, j = best
        members = np.flatnonzero(labels == j)
        labels[members[side]] = len(centres)
        centres[j] = x[members[~side]].mean(axis=0)
        centres.append(x[members[side]].mean(axis=0))
    return np.array(centres)


def _together(times: np.ndarray, hits: list[Hit]) -> np.ndarray:
    """Whether each hit was struck together with a hit of another family."""
    families = np.array([family(hit.inst) for hit in hits])
    together = np.zeros(len(hits), dtype=bool)
    for i, time in enumerate(times):
        near = np.abs(times - time) <= _TOGETHER
        together[i] = bool(np.any(near & (families != families[i])))
    return together


def _gaps(times: np.ndarray, strokes: np.ndarray) -> np.ndarray:
    """Seconds from each time to the nearest other stroke; `strokes` holds the times themselves too."""
    padded = np.concatenate([[-np.inf], np.sort(strokes), [np.inf]])
    at = np.searchsorted(padded, times)  # the time itself, so its neighbours are either side
    return np.minimum(times - padded[at - 1], padded[at + 1] - times)


def _distance(times: np.ndarray, others: np.ndarray) -> np.ndarray:
    """Seconds from each time to the nearest of `others`, which must not be empty."""
    padded = np.concatenate([[-np.inf], np.sort(others), [np.inf]])
    at = np.searchsorted(padded, times)
    return np.minimum(times - padded[at - 1], padded[at] - times)


def _unexplained(attacks: np.ndarray, times: np.ndarray, duration: float) -> np.ndarray:
    """Attacks in the song that no hit is close to."""
    attacks = np.unique(np.asarray(attacks, dtype=float))
    attacks = attacks[np.isfinite(attacks) & (attacks >= 0) & (attacks < duration)]
    if not len(attacks) or not len(times):
        return attacks
    return attacks[_distance(attacks, times) > _NEAR]


def _examples(times: np.ndarray, typicality: np.ndarray, gaps: np.ndarray, duration: float) -> list[Example]:
    """A few strokes good to watch and hear: the most typical clean ones, from different parts of the song.

    When too few are clean, the spread is given up first, then cleanliness,
    the least crowded strokes going first.
    """
    by_typicality = np.argsort(-typicality, kind="stable")
    by_gap = np.argsort(-gaps, kind="stable")
    picked: list[int] = []
    for order, clean, spread in ((by_typicality, _CLEAN, _SPREAD * duration), (by_typicality, _CLEAN, 0), (by_gap, 0, 0)):
        for i in order:
            if len(picked) == _EXAMPLES:
                break
            if i in picked or gaps[i] < clean or any(abs(times[i] - times[j]) < spread for j in picked):
                continue
            picked.append(int(i))
    return [Example(float(t), max(0.0, float(t) - _CLIP[0]), float(t) + _CLIP[1]) for t in sorted(times[picked])]


def _ids(labels: list[str], centres: list[np.ndarray], previous: SoundMap | None) -> list[str]:
    """`<label>-<n>` per group, dark to bright within a label.

    A rebuilt map keeps an old id for the group closest to the old group's
    sound, so a name saved against it still means what the user heard; new
    sounds get numbers no old group had.
    """
    ids: list[str | None] = [None] * len(labels)
    used: set[str] = set()
    if previous is not None:
        pairs = sorted(
            ((float(_similarity(np.array(old.centre), centres[i])), i, old.id)
             for i, label in enumerate(labels) for old in previous.groups
             if (old.detected or EXTRA) == label and len(old.centre) == len(centres[i])),
            reverse=True,
        )
        for similarity, i, old_id in pairs:
            if similarity < _SAME:
                break
            if ids[i] is None and old_id not in used:
                ids[i], used = old_id, used | {old_id}
    for label in dict.fromkeys(labels):
        number = max((int(m[1]) for used_id in used if (m := re.fullmatch(rf"{re.escape(label)}-(\d+)", used_id))), default=0)
        for i in sorted((i for i, other in enumerate(labels) if other == label and ids[i] is None), key=lambda i: _brightness(centres[i])):
            number += 1
            ids[i] = f"{label}-{number}"
    return ids  # type: ignore[return-value]


def _brightness(centre: np.ndarray) -> float:
    return float(np.dot(centre ** 2, np.arange(len(centre))))


def learn(drums_wav: Path, hits: list[Hit], attacks: np.ndarray, inputs: dict[str, str], previous: SoundMap | None = None) -> SoundMap:
    """Group the hits of each detected class by timbre, and the unexplained attacks by theirs.

    Centres are learned from strokes that stand alone; a stroke that lands with
    another drum has a mixed spectrum and only gets assigned to the nearest
    centre of its class, never shown as an example.
    """
    from .dynamics import _spectrogram

    mel = _spectrogram(drums_wav)
    duration = mel.shape[1] / ADTOF_FPS
    hits = sorted((hit for hit in hits if 0 <= hit.time < duration), key=lambda hit: hit.time)  # the same map for any order
    times = np.array([hit.time for hit in hits])
    features, _ = _spectra(mel, times)
    together = _together(times, hits)
    free = _unexplained(attacks, times, duration)
    free_features, _ = _spectra(mel, free)
    strokes = np.concatenate([times, free])
    hit_gaps, free_gaps = _gaps(times, strokes), _gaps(free, strokes)

    proposals: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]] = []
    for label in ADTOF_CLASSES:
        members = np.flatnonzero([family(hit.inst) == label for hit in hits])
        if not len(members):
            continue
        alone = members[~together[members]]
        centres = _centres(features[alone if len(alone) >= _MIN_MEMBERS else members])
        assigned = _nearest(features[members], centres)
        for k, centre in enumerate(centres):
            chosen = members[assigned == k]
            proposals.append((label, centre, times[chosen], features[chosen], hit_gaps[chosen], int(together[chosen].sum())))
    if len(free):
        centres = _centres(free_features)
        assigned = _nearest(free_features, centres)
        for k, centre in enumerate(centres):
            chosen = assigned == k
            proposals.append((EXTRA, centre, free[chosen], free_features[chosen], free_gaps[chosen], 0))

    main = {}
    for label, centre, member_times, *_ in proposals:
        if label != EXTRA and len(member_times) > len(main.get(label, (None, []))[1]):
            main[label] = (centre, member_times)
    ids = _ids([p[0] for p in proposals], [p[1] for p in proposals], previous)
    groups = []
    for identity, (label, centre, member_times, member_features, gaps, mixed) in zip(ids, proposals):
        groups.append(SoundGroup(
            identity, None if label == EXTRA else label, len(member_times), mixed, centre.tolist(),
            _examples(member_times, _similarity(member_features, centre), gaps, duration),
            {other: round(float(_similarity(centre, reference)), 3) for other, (reference, _) in main.items()},
        ))
    order = (*ADTOF_CLASSES, EXTRA)
    return SoundMap(sorted(groups, key=lambda g: (order.index(g.detected or EXTRA), _brightness(np.array(g.centre)))), inputs)


def load_or_learn(song_dir: Path, drums_wav: Path, activations_npy: Path, attacks_npy: Path, hits: list[Hit], attacks: np.ndarray) -> SoundMap:
    """The song's saved map while its caches are the ones it was learned from; otherwise a new one, saved.

    The map lives outside the cache: it is small, and the user's names refer
    to its ids. A new stem, new activations or new attacks rebuild it, inheriting
    the ids of sounds that are still there.
    """
    inputs = {path.name: file_fingerprint(path) for path in (drums_wav, activations_npy, attacks_npy)}
    previous = SoundMap.load(song_dir)
    if previous is not None and previous.inputs == inputs:
        return previous
    learned = learn(drums_wav, hits, attacks, inputs, previous)
    learned.save(song_dir)
    return learned


def rename(hits: list[Hit], sound_map: SoundMap, names: dict[str, str], drums_wav: Path, attacks: np.ndarray) -> list[Hit]:
    """The hits with each named group renamed or dropped, plus the unexplained attacks of named groups.

    Every hit goes to the nearest centre among its class's groups, so a name
    covers strokes detected later with other thresholds too. A renamed hit
    keeps its strength; its ghost flag only survives within the snare family.
    An added attack's strength is its loudness against its group's typical
    stroke, which marks the ghost notes among added snares.
    """
    from .dynamics import _spectrogram

    names = {group.id: names[group.id] for group in sound_map.groups if group.id in names}
    if not names:
        return hits
    mel = _spectrogram(drums_wav)
    duration = mel.shape[1] / ADTOF_FPS
    times = np.array([hit.time for hit in hits])
    features, _ = _spectra(mel, np.clip(times, 0, duration - 1 / ADTOF_FPS))
    new_name: dict[int, str] = {}
    for label in ADTOF_CLASSES:
        groups = [group for group in sound_map.groups if group.detected == label]
        members = np.flatnonzero([family(hit.inst) == label for hit in hits])
        if not len(members) or not groups:
            continue
        assigned = _nearest(features[members], np.array([group.centre for group in groups]))
        new_name.update((int(i), names[groups[k].id]) for i, k in zip(members, assigned) if groups[k].id in names)
    renamed: list[Hit] = []
    for i, hit in enumerate(hits):
        name = new_name.get(i)
        if name is None:
            renamed.append(hit)
        elif name != "ignore":
            renamed.append(Hit(hit.time, name, hit.strength, hit.ghost and family(name) == "snare"))
    extras = [group for group in sound_map.groups if group.detected is None]
    free = _unexplained(attacks, times, duration)
    if extras and len(free):
        free_features, loudness = _spectra(mel, free)
        assigned = _nearest(free_features, np.array([group.centre for group in extras]))
        for k, group in enumerate(extras):
            name = names.get(group.id)
            chosen = np.flatnonzero(assigned == k)
            if name in (None, "ignore") or not len(chosen):
                continue
            typical = max(float(np.median(loudness[chosen])), 1e-9)
            for i in chosen:
                strength = float(loudness[i] / typical)
                renamed.append(Hit(float(free[i]), name, strength, family(name) == "snare" and strength < GHOST_RATIO))
    return sorted(renamed, key=lambda hit: hit.time)
