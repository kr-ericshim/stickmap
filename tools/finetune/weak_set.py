"""Fold-local, fail-closed weak-order sidecars. Never read scoring event exports.

Automatic admission uses reader identities, cursor spans and class-agnostic
drum-stem attacks. Hash-bound human review remains an opt-in legacy path.
"""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'tools/gpu'), str(ROOT / 'tools/harvest')]
from chops_common import fold_policy, load_performance_table, partitions, performance_groups, splits, vault

FPS = 100
TOLERANCE = .65
MIN_MOMENTS = 8
POLICY = 'weak-order-v2-auto8-onset15-dp85-tol065'
ORDER_RADIUS = .65
DENSE_SOLOS = ('jAMDUcKmjD4', 'qLi8dFevOF0', 'n3jYSzj9VX0', 'tK9MGMbPS_w', 'Ly7Pbg4EDug')
FAMILIES = ('kick', 'snare', 'tom')
BITS = dict(zip(FAMILIES, (1, 2, 4)))
REVIEW = 'weak_order_review.json'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def seal_fingerprint():
    # Constructing Vault validates configuration and the committed manifest first.
    vault.Vault()
    return json_hash({'config': digest(vault.CONFIG), 'manifest': digest(vault.MANIFEST)})


def family(name):
    if not isinstance(name, str):
        return 'unknown'
    if name in FAMILIES:
        return name
    if name.startswith('tom'):
        return 'tom'
    if name == 'side_stick':
        return 'snare'
    if name == 'hihat' or name.startswith('hihat_'):
        return 'hihat'
    if name in ('cymbal', 'cymbal_unknown', 'ride', 'ride_bell', 'crash', 'splash', 'china', 'bell'):
        return 'cymbal'
    return 'unknown'


def external_solos(root=ROOT):
    """All truth/<solo>/song.txt sources, including JD folders outside SOLOS.

    Parse source slugs, so Mac/WSL absolute path rebasing is unnecessary here.
    No audio, truth labels, or target directory metadata is opened.
    """
    rows, hashes = [], {}
    for pointer in sorted((Path(root) / 'truth').glob('*/song.txt')):
        slug = Path(pointer.read_text().strip()).name
        if not re.search(r'-[A-Za-z0-9_-]{11}$', slug):
            raise ValueError(f'external solo pointer has no video ID: {pointer}')
        rows.append({'slug': slug, 'video': slug[-11:], 'channel': 'external-solo',
                     'solo': pointer.parent.name})
        hashes[str(pointer.relative_to(root))] = digest(pointer)
    if not rows:
        raise ValueError('no external solo song.txt sources; cannot establish isolation')
    return rows, hashes


def manifest_rows(seal):
    return sorted((dict(r) for r in seal.rows.values() if r.get('slug') and r.get('channel')),
                  key=lambda r: r['slug'])


def external_exclusions(rows, solo, table):
    components = performance_groups(rows + solo, table)
    result = {}
    for row in rows:
        peers = [s for s in solo if components[row['slug']] == components[s['slug']]
                 or splits.same_song(row['slug'], s['slug'])]
        if peers:
            result[row['slug']] = {'video': row['video'], 'reason': 'external-solo/same-performance',
                                   'solo_sources': [s['solo'] for s in peers]}
    return result


def exclusions(plan, fold, rows, solo, table, seal):
    item = plan['folds'][fold]
    all_rows = {r['slug']: r for r in rows + plan.get('songs', []) + plan.get('references', []) + solo}
    groups = performance_groups(all_rows.values(), table)
    held = set(item.get('held', [])) | set(item.get('predict', []))
    held.update(r['slug'] for r in rows if r['channel'] in item['channels'])
    held_groups = {groups[s] for s in held if s in groups}
    solo_groups = {groups[r['slug']] for r in solo}
    result = {}
    for row in rows:
        slug = row['slug']
        if seal.contains_song(row):
            reason = 'vault'
        elif groups[slug] in solo_groups or any(splits.same_song(slug, s['slug']) for s in solo):
            reason = 'external-solo/same-performance'
        elif row['channel'] in item['channels'] or slug in held:
            reason = 'held-out-song/channel'
        elif groups[slug] in held_groups:
            reason = 'held-out-same-performance'
        elif any(splits.same_song(slug, s) for s in held):
            reason = 'held-out-same-song'
        elif row['channel'] not in plan['channels']:
            reason = 'outside-fold-roster'
        else:
            continue
        result[slug] = {'video_id': row['video'], 'channel': row['channel'], 'reason': reason}
        if reason == 'external-solo/same-performance':
            result[slug]['solo_sources'] = [s['solo'] for s in solo if groups[s['slug']] == groups[slug]
                                            or splits.same_song(slug, s['slug'])]
    return result


def reader_reason(system, alignment):
    for key in ('staff', 'clef', 'legend', 'scan', 'reader'):
        metadata = system.get(key, {})
        if isinstance(metadata, dict) and metadata.get('reasons'):
            return 'reader/staff/scan'
    if any(system.get(k) for k in ('scan_reasons', 'reader_reasons', 'unvisited_heads')):
        return 'reader/staff/scan'
    if system.get('staff', {}).get('clef_confirmed') is False:
        return 'reader/staff/scan'
    # Duration/tuplet errors alone do not change the head order target.
    reasons = alignment.get('scoring_reasons', []) + alignment.get('reasons', [])
    forbidden = ('staff', 'clef', 'legend', 'reader', 'scan', 'span',
                 'unknown head', 'unknown note', 'unknown instrument', 'unknown core',
                 'truncated', 'read x', 'missing head', 'duplicat', 'missing note',
                 'extra head', 'extra note', 'unexpanded', 'unresolved identity',
                 'unresolved head', 'unresolved instrument')
    normalized = [r.lower().replace('-', ' ').replace('_', ' ') for r in reasons]
    head_count_fault = any(any(w in r for w in ('head', 'note', 'stroke', 'onset', 'hit')) and
                           any(w in r for w in ('missing', 'extra', 'duplicat', 'doubled', 'incomplete'))
                           for r in normalized)
    if head_count_fault or any(any(word in r for word in forbidden) for r in normalized):
        return 'reader/head/span-reason'
    for head in system.get('notes', []):
        if head.get('tied_continuation'):
            continue
        if family(head.get('inst', 'unknown')) == 'unknown' or not head.get('verified', False):
            return 'unknown/unverified-head'
        if head.get('exportable') is False:
            return 'withheld-head'
        if head.get('roll') or head.get('flam') or head.get('repeat_count', 1) != 1:
            return 'unexpanded-head-multiplicity'
    return None


def candidate_groups(system, alignment):
    _, groups = notation_groups(system)
    return [g for g in groups if any(family(system['notes'][i].get('inst', 'unknown')) in FAMILIES for i in g)]


@lru_cache(maxsize=1)
def reader_api():
    # Read-only reuse; neither build_truth nor any canonical writer is called.
    sys.path.insert(0, str(ROOT / 'tools/truth_video'))
    from align import assignment, moments, prior
    return assignment, moments, prior


def notation_groups(system):
    _, moments, _ = reader_api()
    return moments(system)  # excludes tied_continuation even in stale alignment groups


def rational(value):
    if not isinstance(value, (str, list)) or isinstance(value, list) and (
            len(value) != 2 or any(type(v) is not int for v in value)):
        raise ValueError('rhythm must use rational strings or integer [numerator, denominator]')
    return Fraction(*value) if isinstance(value, list) else Fraction(value)


def reviewed_moments(system, review):
    moments, used, previous = [], set(), None
    for m in review.get('moments', []):
        onset, duration = rational(m['onset']), rational(m['duration'])
        if onset < 0 or duration <= 0 or previous is not None and onset <= previous:
            raise ValueError('rhythm moments must be strictly ordered; merge verified chords')
        previous = onset
        if not isinstance(m['chord_id'], str) or not m['chord_id']:
            raise ValueError('missing chord identity')
        ratio = rational(m['tuplet_ratio'])
        if ratio <= 0 or type(m['repeat_count']) is not int or m['repeat_count'] != 1 or m['grace'] is not False or m['roll'] is not False:
            raise ValueError('unexpanded repeat/grace/roll or invalid tuplet')
        if not isinstance(m['tie_chain'], list):
            raise ValueError('missing tie chain review')
        heads = m['head_indices']
        if not heads or any(type(i) is not int or i < 0 or i >= len(system['notes']) or i in used for i in heads):
            raise ValueError('invalid or reused source head')
        if len(set(heads)) != len(heads):
            raise ValueError('duplicate source head')
        used.update(heads)
        originals = [{key: system['notes'][i].get(key) for key in ('inst', 'pos', 'shape')}
                     for i in heads]
        if any(system['notes'][i].get('tied_continuation') for i in heads):
            raise ValueError('tie continuation cannot be an onset')
        families = sorted({family(h['inst']) for h in originals} & set(FAMILIES))
        if not families:
            raise ValueError('empty-sequence/non-core-moment')
        moments.append({'onset': str(onset), 'duration': str(duration), 'chord_id': m['chord_id'],
                        'tuplet_ratio': str(ratio), 'tie_chain': m['tie_chain'],
                        'grace': False, 'roll': False, 'repeat_count': 1,
                        'head_indices': heads, 'original_heads': originals, 'families': families})
    core = {i for i, h in enumerate(system['notes']) if family(h.get('inst', 'unknown')) in FAMILIES
            and not h.get('tied_continuation')}
    if not moments:
        raise ValueError('empty-sequence')
    if used != core:
        raise ValueError('incomplete core sequence')
    return moments


def span_frames(span, offset=0., origin=0., guard=0):
    a, b = span
    if not all(math.isfinite(x) for x in (a, b, offset, origin)) or a >= b:
        raise ValueError('invalid source span')
    return [math.floor((a + offset - origin) * FPS) - guard,
            math.ceil((b + offset - origin) * FPS) + guard + 1]


def overlaps(a, b):
    return a[0] < b[1] and b[0] < a[1]


def admitted(system, alignment, review, strict, row, fold):
    flags = ('staff_clean', 'clef_verified', 'legend_verified', 'visited_span_verified',
             'reader_complete', 'heads_complete', 'duplicates_removed', 'rhythm_verified',
             'ties_verified', 'rests_verified', 'grace_roll_verified', 'system_location_verified',
             'complete_window_sequence', 'padding_sequence_verified')
    if any(review.get(k) is not True for k in flags):
        raise ValueError('missing reader/rhythm/location/window review')
    if any(type(review.get(k)) is not int or review[k] != 0 for k in ('unresolved_rhythm', 'unresolved_identity')):
        raise ValueError('unresolved rhythm/identity')
    if type(review.get('quality_fold')) is not int or review['quality_fold'] != fold or review.get('reference_scope') != 'training-only':
        raise ValueError('review outside training fold')
    if any(not isinstance(review.get(k), str) or not review[k].strip()
           for k in ('reviewer', 'evidence', 'source_revision')):
        raise ValueError('missing independent review provenance')
    moments = reviewed_moments(system, review)
    if len(moments) < 12:
        raise ValueError('fewer-than-12-verified-moments')
    metrics = [alignment.get('after', {}).get('within_40ms', 0),
               alignment.get('held_out', {}).get('within_40ms', 0),
               alignment.get('held_out_gain_20ms', 0), alignment.get('held_out_gain_40ms', 0)]
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in metrics):
        raise ValueError('nonfinite timing metrics')
    if (metrics[0] < .90 or metrics[1] <
            (.90 if alignment.get('mode') in ('interval', 'rhythm') else .85) or
            max(metrics[2:]) < .15 or
            alignment.get('strictly_monotone') is not True):
        raise ValueError('timing-gates')
    source = review['source_span_seconds']
    if source != system['interval']:
        raise ValueError('source span differs from whole notation system')
    offset, origin = review['audio_offset_seconds'], review['feature_origin_seconds']
    if origin != 0:
        raise ValueError('current full-song feature cache requires origin zero')
    # Source span is already the strip's audio interval; offsets must be reviewed explicitly.
    lo, hi = source[0] + offset - TOLERANCE, source[1] + offset + TOLERANCE
    window = [math.floor((lo - origin) * FPS), math.ceil((hi - origin) * FPS)]
    if lo < 0 or hi > float(row.get('seconds') or 0) or window[1] - window[0] > 800:
        raise ValueError('source-boundary/whole-crop-length')
    tokens = [sum(BITS[c] for c in m['families']) for m in moments]
    if window[1] - window[0] < len(tokens) + sum(a == b for a, b in zip(tokens, tokens[1:])):
        raise ValueError('impossible-ctc-length')
    tested = [span_frames(source, offset, origin), window]
    times = alignment.get('times', [])
    if times:
        tested.append(span_frames([min(times), max(times)]))
    protected = [span_frames(v, guard=1) for v in strict.get('intervals', [])]
    for s in strict.get('systems', []):
        if not s.get('usable'):
            continue
        protected.append(span_frames(s['interval'], guard=1))
        t = s.get('times', [])
        if t:
            protected.append(span_frames([min(t), max(t)], guard=1))
    if any(overlaps(a, b) for a in tested for b in protected):
        raise ValueError('strict-overlap')
    tempo = review.get('diagnostic_tempo_bpm')
    fast = Counter()
    if tempo is not None:
        if not math.isfinite(tempo) or tempo <= 0:
            raise ValueError('invalid diagnostic tempo')
        onsets = [float(Fraction(m['onset'])) * 60 / tempo for m in moments]
        for i, m in enumerate(moments):
            if any(abs(onsets[i] - onsets[j]) < .09 for j in (i - 1, i + 1) if 0 <= j < len(moments)):
                fast.update(m['families'])
    return {'system': system['system'], 'moments': moments, 'tokens': tokens,
            'source_span_seconds': source, 'audio_offset_seconds': offset, 'feature_origin_seconds': origin,
            'tolerance_seconds': TOLERANCE, 'window_start_frame': window[0], 'window_stop_frame': window[1],
            'confidence': 1., 'weight': 1., 'policy_version': POLICY, 'band_check': 'off',
            'strict_overlap_audit': 'disjoint including rounding and +/-1 frame',
            'original_tier': alignment.get('tier'), 'original_reasons': alignment.get('scoring_reasons', []),
            'timing_metric_provenance': 'legacy alignment; may use band-seeded fit',
            'verification': {k: review[k] for k in flags},
            'unresolved_rhythm': 0, 'unresolved_identity': 0,
            'reviewer': review['reviewer'], 'evidence': review['evidence'], 'source_revision': review['source_revision'],
            'diagnostic_fast_kst': {c: fast[c] for c in FAMILIES},
            'diagnostic_fast_known': tempo is not None}


def notated_beats(system):
    """Use explicit quarter beats, or counted measures and their meter.

    Legacy strips contain barlines but no meter field: their fallback is 4/4,
    recorded as an assumption, never a verified duration/rhythm annotation.
    """
    for key in ('notated_beats', 'beats'):
        if key in system:
            beats = float(Fraction(str(system[key])))
            if not math.isfinite(beats) or beats <= 0:
                raise ValueError('invalid-notated-beats')
            return beats, key
    signature = system.get('time_signature', [4, 4])
    if isinstance(signature, str):
        signature = signature.split('/')
    numerator, denominator = map(int, signature)
    if numerator <= 0 or denominator <= 0:
        raise ValueError('invalid-meter')
    notes = [h for h in system['notes'] if not h.get('tied_continuation')]
    # Exclude a leading staff boundary; terminal barlines close whole measures.
    first = min(h['x'] for h in notes)
    bars = sorted(set(float(x) for x in system.get('staff', {}).get('bar_x', []) if x > first))
    measures = system.get('measure_count', len(bars))
    if type(measures) is not int or measures <= 0:
        raise ValueError('missing-notated-beats/barlines')
    return measures * numerator * 4 / denominator, (
        'measure-count/meter' if 'time_signature' in system else 'bar-count; assumed 4/4')


def automatic_window(system, alignment, strict, row):
    source = list(system['interval'])
    span_frames(source)  # finite, positive span
    if alignment.get('interval', source) != source:
        raise ValueError('alignment/strip-span-mismatch')
    beats, beat_source = notated_beats(system)
    tempo = beats * 60 / (source[1] - source[0])
    if not 40 <= tempo <= 300:
        raise ValueError('implausible-span-tempo')
    lo, hi = source[0] - TOLERANCE, source[1] + TOLERANCE
    window = [math.floor(lo * FPS), math.ceil(hi * FPS)]
    if lo < 0 or hi > float(row.get('seconds') or 0) or window[1] - window[0] > 800:
        raise ValueError('source-boundary/whole-crop-length')
    tested = [span_frames(source), window]
    times = alignment.get('times', [])
    if times:
        tested.append(span_frames([min(times), max(times)]))
    protected = [span_frames(v, guard=1) for v in strict.get('intervals', [])]
    for s in strict.get('systems', []):
        if s.get('usable'):
            protected.append(span_frames(s['interval'], guard=1))
            if s.get('times'):
                protected.append(span_frames([min(s['times']), max(s['times'])], guard=1))
    if any(overlaps(a, b) for a in tested for b in protected):
        raise ValueError('strict-overlap')
    return source, window, {'notated_beats': beats, 'beat_source': beat_source, 'tempo_bpm': tempo}


def acoustic_check(predicted, core, attacks, window):
    """Joint DP reserves HH/C-only matches so core heads cannot reuse them.

    We cannot identify an attack's drum from audio alone. Permit at most one
    extra onset per notated HH/C-only moment plus ceil(15% of core moments).
    No neural outputs, frequency-band identity gates or fitted note times.
    """
    import numpy as np
    assignment, _, _ = reader_api()
    attacks = np.asarray(attacks, dtype=float)
    if attacks.ndim != 1 or not np.isfinite(attacks).all() or np.any(attacks < 0):
        raise ValueError('invalid-drum-stem-attacks')
    local = np.unique(attacks[(attacks >= window[0] / FPS) & (attacks < window[1] / FPS)])
    predicted = np.asarray(predicted, dtype=float)
    if (not np.isfinite(predicted).all() or np.any(np.diff(predicted) <= 0) or
            np.any(predicted < window[0] / FPS) or np.any(predicted >= window[1] / FPS)):
        raise ValueError('invalid-cursor-order/window')
    n = sum(core)
    hh = len(core) - n
    excess = math.ceil(.15 * n)
    evidence = {'detector': 'cached attacks.htdemucs.npy; drum-stem class-agnostic attack_times',
                'core_moments': n, 'hh_cymbal_only_moments': hh, 'onsets': len(local),
                'minimum_onsets': n, 'maximum_onsets': n + hh + excess,
                'excess_tolerance': excess, 'order_radius_seconds': ORDER_RADIUS,
                'minimum_core_assignment': .85}
    # This count bound marginalizes HH/C-only attacks instead of labeling them.
    pairs = assignment(predicted, local, ORDER_RADIUS)
    matched = sum(bool(core[i]) for i, _ in pairs)
    evidence.update(matched_core_moments=matched, core_assignment_fraction=matched / max(1, n),
                    matched_hh_cymbal_only_moments=sum(not core[i] for i, _ in pairs))
    reasons = []
    if not n <= len(local) <= n + hh + excess:
        reasons.append('acoustic-onset-count')
    if matched / max(1, n) < .85:
        reasons.append('acoustic-order-assignment')
    return evidence, reasons


def automatic_admitted(system, alignment, strict, row, attacks, evidence):
    import numpy as np
    x, groups = notation_groups(system)
    core = [any(family(system['notes'][i].get('inst')) in FAMILIES for i in g) for g in groups]
    if sum(core) < MIN_MOMENTS:
        raise ValueError('fewer-than-eight-core-moments')
    source, window, span = automatic_window(system, alignment, strict, row)
    evidence['span'] = span
    # Require observed cursor location; an engraving/interval-only prior is insufficient.
    cx, ct = (np.asarray(system.get(k, []), dtype=float) for k in ('cursor_x', 'cursor_t'))
    if (len(cx) < 4 or len(cx) != len(ct) or not np.isfinite(cx).all() or
            not np.isfinite(ct).all() or np.any(np.diff(cx) <= 0) or np.any(np.diff(ct) <= 0) or
            system.get('cursor_coverage', 1) <= .35):
        raise ValueError('missing/invalid-cursor-span')
    _, _, prior = reader_api()
    predicted, mode = prior(system, x)
    if mode not in ('cursor', 'color'):
        raise ValueError('missing-cursor-location')
    acoustic, reasons = acoustic_check(predicted, core, attacks, window)
    evidence.update(acoustic=acoustic, location_source=mode, reasons=reasons)
    if reasons:
        raise ValueError(reasons[0])
    moments = []
    for g, is_core in zip(groups, core):
        if not is_core:
            continue
        originals = [{key: system['notes'][i].get(key) for key in ('inst', 'pos', 'shape')} for i in g]
        families = sorted({family(h['inst']) for h in originals} & set(FAMILIES))
        moments.append({'head_indices': g, 'original_heads': originals, 'families': families})
    tokens = [sum(BITS[c] for c in m['families']) for m in moments]
    if window[1] - window[0] < len(tokens) + sum(a == b for a, b in zip(tokens, tokens[1:])):
        raise ValueError('impossible-ctc-length')
    return {'system': system['system'], 'moments': moments, 'tokens': tokens,
            'source_span_seconds': source, 'audio_offset_seconds': 0., 'feature_origin_seconds': 0.,
            'tolerance_seconds': TOLERANCE, 'window_start_frame': window[0], 'window_stop_frame': window[1],
            'confidence': 1., 'weight': 1., 'policy_version': POLICY, 'band_check': 'off',
            'strict_overlap_audit': 'disjoint including rounding and +/-1 frame',
            'original_tier': alignment.get('tier'), 'original_reasons': alignment.get('scoring_reasons', []),
            'acceptance_evidence': evidence,
            'verification': {'reader_clean': True, 'ties_excluded': True, 'acoustic_completeness': True,
                             'monotone_assignment': True, 'rhythm_review_required': False},
            'diagnostic_fast_kst': dict.fromkeys(FAMILIES, 0), 'diagnostic_fast_known': False}


def build_fold(plan, fold, *, root=ROOT, prepare_feature=None, manual_review=False):
    seal = vault.Vault()  # MUST precede all notation/cache reads.
    fingerprint = seal_fingerprint()
    rows = manifest_rows(seal)
    table, table_sha = load_performance_table(root)
    solo, pointer_hashes = external_solos(root)
    excluded = exclusions(plan, fold, rows, solo, table, seal)
    sources = dict(pointer_hashes)
    channels, skipped, songs = {}, Counter(), {r['slug']: [] for r in rows}
    audits = {}
    for row in rows:
        slug, channel = row['slug'], row['channel']
        counts = channels.setdefault(channel, {'systems': 0, 'moments': 0, 'fast_kst': dict.fromkeys(FAMILIES, 0),
                                               'reader_candidates': 0, 'candidate_moments': 0,
                                               'fast_unknown_systems': 0, 'rejected': {}})
        if slug in excluded:
            audits[slug] = {'video_id': row['video'], 'channel': channel,
                            'status': 'excluded', 'reason': excluded[slug]['reason'], 'systems': {}}
            continue
        fresh = vault.Vault()
        if seal_fingerprint() != fingerprint or fresh.contains_song(row):
            raise ValueError('vault changed before notation/cache read')
        truth = vault.SONGS / slug / 'truth'
        audit = audits[slug] = {'video_id': row['video'], 'channel': channel, 'status': 'read', 'systems': {}}

        def reject(system_id, reason, evidence=None):
            reasons = (evidence or {}).get('reasons') or [reason]
            audit['systems'][str(system_id)] = {'status': 'rejected', 'reasons': reasons,
                                                'evidence': evidence or {}}
            skipped[reason] += 1
            counts['rejected'][reason] = counts['rejected'].get(reason, 0) + 1

        files = [truth / f for f in ('strip.json', 'alignment.json', 'alignment.scoring.json')]
        if any(not p.exists() for p in files):
            skipped['missing-notation/alignment'] += 1
            audit.update(status='missing', missing=[p.name for p in files if not p.exists()])
            continue
        for p in files:
            sources[str(p.relative_to(root))] = digest(p)
        strip, strict, scoring = [json.loads(p.read_text()) for p in files]
        by_id = {s['system']: s for s in strip}
        review_path = truth / REVIEW
        review = {}
        if manual_review and review_path.exists():
            sources[str(review_path.relative_to(root))] = digest(review_path)
            review = json.loads(review_path.read_text())
            if review.get('schema_version') != 1 or review.get('source_hashes') != {p.name: digest(p) for p in files}:
                raise ValueError(f'stale/invalid rhythm review: {review_path}')
        accepted = []
        attacks = None
        for a in scoring['systems']:
            system = by_id.get(a['system'])
            if system is None:
                raise ValueError('scoring system absent from notation')
            if a.get('strict_usable') or any(s['system'] == a['system'] and s.get('usable') for s in strict['systems']):
                reject(a['system'], 'strict-usable')
                continue
            reason = reader_reason(system, a)
            if reason:
                reject(a['system'], reason)
                continue
            groups = candidate_groups(system, a)
            if not groups:
                reject(a['system'], 'empty-sequence')
                continue
            if len(groups) < MIN_MOMENTS:
                reject(a['system'], 'fewer-than-eight-core-moments')
                continue
            counts['reader_candidates'] += 1
            counts['candidate_moments'] += len(groups)
            evidence = {'acceptance_mode': 'manual-review' if manual_review else 'automatic',
                        'reader_clean': True, 'core_moments': len(groups),
                        'tied_continuations_removed': sum(bool(h.get('tied_continuation')) for h in system['notes'])}
            try:
                if manual_review:
                    reviewed = review.get('systems', {}).get(str(a['system']))
                    if reviewed is None:
                        raise ValueError('missing-rhythm-review')
                    entry = admitted(system, a, reviewed, strict, row, fold)
                    entry['acceptance_evidence'] = evidence
                else:
                    # Cheap span/overlap failures precede any acoustic cache read.
                    _, _, span = automatic_window(system, a, strict, row)
                    evidence['span'] = span
                    attack_path = truth.parent / 'attacks.htdemucs.npy'
                    if attacks is None:
                        if not attack_path.exists():
                            raise ValueError('missing-drum-stem-attacks')
                        import numpy as np
                        sources[str(attack_path.relative_to(root))] = digest(attack_path)
                        attacks = np.load(attack_path, allow_pickle=False)
                    entry = automatic_admitted(system, a, strict, row, attacks, evidence)
            except (KeyError, ValueError, TypeError, ZeroDivisionError) as exc:
                reject(a['system'], str(exc), evidence)
                continue
            if any(overlaps([entry['window_start_frame'], entry['window_stop_frame']],
                            [s['window_start_frame'], s['window_stop_frame']]) for s in accepted):
                reject(a['system'], 'weak-window-overlap', evidence)
                continue
            accepted.append(entry)
        if not accepted:
            continue
        if vault.Vault().contains_song(row) or seal_fingerprint() != fingerprint:
            raise ValueError('vault changed before feature read')
        folder = prepare_feature(vault.SONGS / slug) if prepare_feature else Path(root) / '.eval-cache/ft' / slug
        feature = folder / 'feat.npy'
        if not feature.exists():
            for entry in accepted:
                reject(entry['system'], 'missing-features', entry['acceptance_evidence'])
            continue
        import numpy as np
        n = len(np.load(feature, mmap_mode='r', allow_pickle=False))
        for entry in accepted:
            if entry['window_stop_frame'] > n:
                reject(entry['system'], 'feature-boundary', entry['acceptance_evidence'])
                continue
            entry.update(slug=slug, video_id=row['video'], channel=channel, feature_frames=n)
            songs[slug].append(entry)
            audit['systems'][str(entry['system'])] = {'status': 'accepted', 'reasons': [],
                                                    'evidence': entry['acceptance_evidence']}
            counts['systems'] += 1
            counts['moments'] += len(entry['tokens'])
            counts['fast_unknown_systems'] += not entry['diagnostic_fast_known']
            for c in FAMILIES:
                counts['fast_kst'][c] += entry['diagnostic_fast_kst'][c]
    if seal_fingerprint() != fingerprint:
        raise ValueError('vault/manifest changed while building weak set')
    return {'schema_version': 1, 'label_type': 'weak_order', 'fps': FPS, 'quality_fold': fold,
            'reference_scope': 'training-only', 'policy_version': POLICY,
            'acceptance_mode': 'manual-review' if manual_review else 'automatic',
            'fold_policy_hash': json_hash({'channels': plan['channels'], 'folds': plan['folds'],
                                          'same_song': list(splits.SAME_SONG)}),
            'vault_seal_fingerprint': fingerprint, 'same_performance_sha256': table_sha,
            'source_hashes': sources, 'songs': songs, 'exclusions': excluded,
            'counts': channels, 'skipped': dict(skipped), 'system_audits': audits,
            'fast_definition': 'manual only: rational onset with reviewed tempo; automatic fast timing unknown'}


def load_fold(path, plan, fold, *, root=ROOT):
    seal = vault.Vault()  # Check seal BEFORE reading sidecar or any cache.
    data = json.loads(Path(path).read_text())
    if data.get('vault_seal_fingerprint') != seal_fingerprint():
        raise ValueError('weak sidecar vault seal changed')
    # Rebuild selection, including fold, external sources, reviews, overlap and actual feature length.
    if data != build_fold(plan, fold, root=root, manual_review=data.get('acceptance_mode') == 'manual-review'):
        raise ValueError('weak sidecar differs from current fold/source/policy')
    if any(seal.contains_song({'slug': slug}) for slug, windows in data['songs'].items() if windows):
        raise ValueError('sealed weak song')
    return data


def print_counts(data):
    print(f'Weak fold{data["quality_fold"]}: policy={POLICY}; band_check=off')
    for channel, counts in sorted(data['counts'].items()):
        print(f'  {channel}: systems={counts["systems"]} moments={counts["moments"]} '
              f'fast K/S/T={counts["fast_kst"]["kick"]}/{counts["fast_kst"]["snare"]}/{counts["fast_kst"]["tom"]} '
              f'fast-unknown systems={counts["fast_unknown_systems"]} '
              f'reader candidates={counts["reader_candidates"]}/{counts["candidate_moments"]} moments')
        if counts['rejected']:
            print('    rejected: ' + json.dumps(counts['rejected'], sort_keys=True))
    print('  skipped: ' + json.dumps(data['skipped'], sort_keys=True))
    print(f'  external solo source pointers={sum(p.endswith("song.txt") for p in data["source_hashes"])}; '
          f'excluded songs={sum(e["reason"] == "external-solo/same-performance" for e in data["exclusions"].values())}')
    for slug, info in data['exclusions'].items():
        if info['reason'] == 'external-solo/same-performance':
            print(f'  excluded external solo: {slug} -> {", ".join(info["solo_sources"])}')


def print_dense_diagnostics(data):
    print(f'Dense solo diagnostics fold{data["quality_fold"]} (excluded sources remain unopened):')
    for video in DENSE_SOLOS:
        matches = [(slug, audit) for slug, audit in data['system_audits'].items() if audit['video_id'] == video]
        if not matches:
            print(f'  {video}: missing (manifest/source absent); systems=unknown')
            continue
        for slug, audit in matches:
            if audit['status'] != 'read':
                detail = audit.get('reason') or ','.join(audit.get('missing', []))
                print(f'  {video}: {audit["status"]} ({detail}); systems=unknown; slug={slug}')
                continue
            reasons = Counter(r for s in audit['systems'].values() for r in s['reasons'])
            print(f'  {video}: systems={len(audit["systems"])}; rejection counts=' + json.dumps(reasons, sort_keys=True))
            for system_id, result in audit['systems'].items():
                print(f'    system {system_id}: {result["status"]}; ' + ', '.join(result['reasons']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, help='frozen comparison plan, otherwise metadata-only mirror inventory')
    parser.add_argument('--output', type=Path, help='optional sidecar directory; never writes canonical truth')
    parser.add_argument('--prepare-strict', action='store_true', help='run existing strict data.one after metadata exclusions')
    parser.add_argument('--exclude', nargs='*', default=[], help='external-solo harvest peer slugs, before truth/cache access')
    parser.add_argument('--only', nargs='+')
    parser.add_argument('--jobs', type=int, choices=range(1, 5), default=4)
    parser.add_argument('--manual-review', action='store_true', help='opt in to the legacy hash-bound human rhythm/location review')
    parser.add_argument('--diagnose-dense-solos', action='store_true', help='list target video systems and rejection counts; missing/excluded sources are explicit')
    args = parser.parse_args()
    seal = vault.Vault()
    if args.prepare_strict:
        from concurrent.futures import ProcessPoolExecutor
        import data as strict_data
        rows = [r for r in strict_data.songs(args.only) if r['slug'] not in args.exclude
                and not seal.contains_song(r)]
        strict_data.OUT.mkdir(parents=True, exist_ok=True)
        with ProcessPoolExecutor(args.jobs) as pool:
            done = [s for s in pool.map(strict_data.one, rows) if s]
        print(len(done), 'strict songs cached; external-solo peers excluded before annotation reads')
        return
    if args.plan:
        from chops_common import load_plan
        plan = load_plan(args.plan)
    else:
        rows = [r for r in manifest_rows(seal) if not seal.contains_song(r)]
        policy = fold_policy(set(seal.train_channels) | {r['channel'] for r in rows}, 2)
        table, _ = load_performance_table(ROOT)
        plan = {'channels': [c for group in policy for c in group],
                'folds': partitions([], rows, policy, table), 'songs': [], 'references': rows}
    for fold in range(len(plan['folds'])):
        data = build_fold(plan, fold, manual_review=args.manual_review)
        print_counts(data)
        if args.diagnose_dense_solos:
            print_dense_diagnostics(data)
        if args.output:
            args.output.mkdir(parents=True, exist_ok=True)
            (args.output / f'weak-order-fold{fold}.json').write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
