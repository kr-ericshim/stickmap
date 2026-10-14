"""Vault-only canonical CV reader, invoked as vault_absolute's direct child.

No development override is exposed. resume_final is verified before request,
source metadata, images or audio are read. The unchanged channel_read_core
pipeline writes only vault_truth/olliewoods_shorts/<id>/{work,truth,...}.
common.py and the development entry points retain their confinement/seal checks.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
import sys
from unittest.mock import patch

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / 'gpu')]
import vault_absolute as absolute
from vault_absolute_io import guard, hashes, load, safe, tree, write

# These exact scanner rejections mean there is no admissible native notation
# candidate. Keep them UNKNOWN, with no truth events or fabricated negatives.
# Decode failures, new diagnostics and all downstream errors still abort.
NO_ADMISSIBLE_SCAN = frozenset({'no sustained bottom white Shorts page',
                              'Shorts page needs at least two observed native staffs',
                              'unstable native Shorts page boundary'})


def require_video(job, video):
    absolute.video_id(video)
    selected = {s['video'] for s in job['songs'] if s['cohort'] == 'ollie'}
    seal = absolute.vault.Vault()
    split = load(absolute.SPLIT)
    manual = absolute.manual_exclusions()
    mixed = seal.final_exclusions()
    if (video not in selected or video not in split['vault'] or video not in seal.explicit or
            not seal.is_vault(video) or video in manual or video in mixed or video in seal.rows):
        raise ValueError('video is not an eligible vault-only canonical input: ' + video)


@contextmanager
def canonical_paths(folder, authorize):
    """Process-local narrowing, never a generic allow-vault/common.DATA switch.

    Imported dump aliases still call common.confined at runtime. Its temporary
    implementation REJECTS all development output paths. hybrid's imported
    confined alias needs the same narrow binding for OpenCV annotations.
    The CV-only unsealed callback becomes a fresh eligible-vault check for this
    one authorized video. It is restored before returning to any other caller.
    """
    import common
    import hybrid
    import channel_pilot

    def confined(path, base=None):
        path = safe(path, absolute.ROOT)
        if not path.is_relative_to(folder):
            raise PermissionError('canonical CV output escapes this vault video')
        return path

    with patch.object(common, 'confined', confined), patch.object(hybrid, 'confined', confined), \
            patch.object(channel_pilot, 'confined', confined), patch.object(hybrid, 'unsealed', authorize):
        yield


def source(item):
    from media import video_path, VIDEO_NAMES
    song = tree(absolute.ROOT / item['source'], absolute.ROOT)
    expected = absolute.ROOT / 'sources/olliewoods_shorts' / item['video']
    if song != expected:
        raise ValueError('invalid vault source path')
    meta = load(song / 'meta.json')
    if (meta.get('video_id') != item['video'] or meta.get('channel') != 'olliewoods_shorts' or
            meta.get('video') not in (None, item['video'], *VIDEO_NAMES)):
        raise ValueError('vault source identity/channel mismatch')
    video_path(song)
    return song


def process(job, item):
    video = item['video']
    require_video(job, video)
    song = source(item)
    folder = tree(absolute.ROOT / item['song'], absolute.ROOT)
    if folder != absolute.VAULT_TRUTH / video or item['cache'] != item['song']:
        raise ValueError('invalid vault canonical destination')
    if folder.exists():
        raise ValueError('canonical output exists; refuse stale/partial vault truth')
    folder.mkdir(parents=True)
    before = hashes([p for p in song.rglob('*') if p.is_file()], absolute.ROOT)
    import channel_read_core as core

    def authorize(value):
        if value != video:
            raise ValueError('CV callback attempted a different video')
        require_video(job, value)

    def system_id(value, index):
        authorize(value)
        if type(index) is not int or index < 0:
            raise ValueError('invalid native system index')
        return f'{value}_src_{index:04d}'

    legend = absolute.ROOT / 'tools/truth_video/legends/olliewoods_shorts.json'
    # No subprocess/network access is needed for this CV path.
    with guard(writes=[folder, absolute.NUMBA_CACHE], subprocesses=False), canonical_paths(folder, authorize), \
            (folder / 'read.log').open('w') as log, redirect_stdout(log), redirect_stderr(log):
        work = folder / 'work'
        scan, read = work / 'notation', work / 'truth-reviewed'
        authorize(video)
        try:
            core.notation(song, scan, legend)
        except ValueError as error:
            if str(error) not in NO_ADMISSIBLE_SCAN:
                raise
            after = hashes([p for p in song.rglob('*') if p.is_file()], absolute.ROOT)
            if before != after:
                raise ValueError('source changed during rejected canonical scan')
            rejected = {'purpose': 'canonical_truth', 'vault': True, 'video': video,
                        'opening_id': job['opening_id'], 'model_tag': job['model_tag'],
                        'status': 'reader_rejected', 'reason': str(error), 'source_hashes': before,
                        'source_readonly_verified': True, 'truth_status': 'unknown; no admissible native scan',
                        'strict_notes': 0, 'scoring_notes': 0, 'gpt_invocations': 0}
            write(folder / 'truth/reader_rejection.json', rejected)
            write(folder / 'provenance.json', rejected)
            return  # no candidate strip, no gate input, no metric references
        authorize(video)
        core.read_cv(song, read, legend, scan)
        cv_only = core.counts(read)
        authorize(video)
        inventory = core.extract(song, 'olliewoods_shorts', work / 'native', read, scan, system_id)
        inventory.update(vault=True, purpose='canonical_truth', opening_id=job['opening_id'], model_tag=job['model_tag'])
        write(work / 'native/inventory.json', inventory)
        authorize(video)
        core.prepare(inventory, work)
        core.stage(inventory, work, folder / 'truth')
        authorize(video)
        summary = core.gate(song, folder / 'truth')
        after = hashes([p for p in song.rglob('*') if p.is_file()], absolute.ROOT)
        if before != after:
            raise ValueError('source changed during canonical vault read')
        write(folder / 'provenance.json', {'purpose': 'canonical_truth', 'vault': True, 'video': video,
              'status': 'completed',
              'opening_id': job['opening_id'], 'model_tag': job['model_tag'], 'source_hashes': before,
              'source_readonly_verified': True, 'reader': core.VERSION, 'cv_only': cv_only,
              'representative_cv': summary, 'gpt_invocations': 0,
              'gate_hashes': hashes([absolute.ROOT / 'tools/truth_video' / name
                                     for name in ('align.py', 'bands.py', 'tiers.py')], absolute.ROOT)})
    print(f'{video}: canonical vault CV completed', flush=True)


def execute(request, opening_id, model_tag):
    job = absolute.child_job(request, opening_id, model_tag)  # BEFORE any sealed read
    import os
    before = Path.cwd()
    try:
        os.chdir(absolute.ROOT)
        for item in job['songs']:
            if item['cohort'] == 'ollie':
                process(job, item)
        absolute.verify_metadata(job)
    finally:
        os.chdir(before)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--opening-id', required=True)
    parser.add_argument('--model-tag', required=True)
    args = parser.parse_args(argv)
    execute(args.request, args.opening_id, args.model_tag)


if __name__ == '__main__':
    main()
