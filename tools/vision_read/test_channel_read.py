"""Offline safety/portability contracts; fixtures live only in vision scratch."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import channel_read as reader
import channel_read_core as core
from common import ROOT, dump, load, sha


class ChannelReadTest(unittest.TestCase):
    def setUp(self):
        reader.output_path(reader.BASE).mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix='_test-', dir=reader.BASE)
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.video = 'MPoZ-2nOESc'
        self.allowed = frozenset([self.video])

    def allow_file(self, value):
        path = self.folder / 'allow.json'
        dump(path, value)
        return path

    def fixture(self):
        song = self.folder / 'sources/olliewoods_shorts' / self.video
        song.mkdir(parents=True)
        for name in reader.INPUT_NAMES:
            (song / name).write_bytes(b'fixture')
        (song / 'meta.json').write_text(json.dumps(dict(video_id=self.video,
             channel='olliewoods_shorts', video='video.webm')))
        (song / 'video.webm').write_bytes(b'not opened by preflight')
        return song

    def test_cli_refuses_no_allow_before_media(self):
        run = subprocess.run([sys.executable, '-B', str(ROOT / 'tools/vision_read/channel_read.py'),
                              '--channel', 'olliewoods_shorts'], capture_output=True, text=True)
        self.assertEqual(run.returncode, 2)
        self.assertIn('--allow', run.stderr)

    def test_allow_validation_and_unlisted_refusal_before_vault(self):
        for value in ({}, {'train': 'oops'}, {'train': ['../bad']}, {'train': [self.video, self.video]}):
            with self.assertRaises(ValueError):
                reader.read_allow(self.allow_file(value))
        with self.assertRaisesRegex(ValueError, '--allow'):
            reader.read_allow(None)
        from tools.harvest import vault
        with patch.object(vault, 'Vault', side_effect=AssertionError('unlisted input reached seal')):
            with self.assertRaisesRegex(ValueError, 'not in --allow'):
                reader.require_allowed(self.video, frozenset())

    def test_vault_explicit_independent_of_membership(self):
        from tools.harvest import vault
        with patch.object(vault, 'config', return_value={'video_ids': [self.video]}), \
                patch.object(vault.Vault, '__init__', return_value=None), \
                patch.object(vault.Vault, 'is_vault', return_value=False):
            with self.assertRaisesRegex(ValueError, 'vault input forbidden'):
                reader.source(self.folder / 'missing', 'olliewoods_shorts', self.video, self.allowed)

    def test_vault_cutoff_refusal_before_source_access(self):
        from tools.harvest import vault
        with patch.object(vault, 'config', return_value={'video_ids': []}), \
                patch.object(vault.Vault, '__init__', return_value=None), \
                patch.object(vault.Vault, 'is_vault', return_value=True):
            with self.assertRaisesRegex(ValueError, 'vault input forbidden'):
                reader.source(self.folder / 'missing', 'olliewoods_shorts', self.video, self.allowed)

    def test_bad_vault_configuration_fails_closed(self):
        from tools.harvest import vault
        with patch.object(vault, 'config', side_effect=ValueError('invalid seal')):
            with self.assertRaisesRegex(ValueError, 'invalid seal'):
                reader.require_allowed(self.video, self.allowed)

    def test_whole_cohort_vault_preflight_precedes_workers(self):
        allow = self.allow_file({'train': [self.video, '8iCRbuU-JnE']})
        with patch.object(reader, 'source', side_effect=AssertionError('source opened before cohort check')), \
                patch.object(reader.concurrent.futures, 'ProcessPoolExecutor',
                             side_effect=AssertionError('worker started before cohort check')):
            with self.assertRaises(SystemExit) as error:
                reader.main(['--channel', 'olliewoods_shorts', '--allow', str(allow),
                             '--sources-root', str(self.folder)])
            self.assertEqual(error.exception.code, 2)

    def test_sources_root_layouts_and_metadata_identity(self):
        song = self.fixture()
        before = reader.snapshot(song)
        with patch.object(reader, 'require_allowed'):
            for root in (self.folder / 'sources', song.parent):
                self.assertEqual(reader.source(root, 'olliewoods_shorts', self.video, self.allowed), song)
            meta = load(song / 'meta.json'); meta['video_id'] = '8iCRbuU-JnE'
            dump(song / 'meta.json', meta)
            with self.assertRaisesRegex(ValueError, 'identity/channel mismatch'):
                reader.source(song.parent, 'olliewoods_shorts', self.video, self.allowed)
        dump(song / 'meta.json', dict(video_id=self.video, channel='olliewoods_shorts', video='video.webm'))
        self.assertEqual((song / 'video.webm').read_bytes(), b'not opened by preflight')
        self.assertEqual(set(before), set(reader.snapshot(song)))

    def test_sources_readonly_even_when_band_cache_invalid(self):
        import numpy as np
        import soundfile as sf
        import align
        import batch
        import bands
        song = self.fixture()
        sf.write(song / 'drums.htdemucs.wav', np.zeros(8192, np.float32), 44100)
        np.save(song / bands.CACHE, np.array([1], np.float32))  # forces compute fallback
        before = reader.snapshot(song)
        out = self.folder / 'work'; out.mkdir()
        original_load = bands.load
        seen = []

        def fake_worker(source, target, profile_path, notation_scan):
            align.build_truth(source, target)  # exercise worker's automatic path

        def observe_load(source, cache=True):
            seen.append(cache)
            return original_load(source, cache=cache)

        with patch.object(batch, 'worker', side_effect=fake_worker), \
                patch.object(align, 'build_truth', return_value=([], {})) as build, \
                patch.object(bands, 'load', side_effect=observe_load):
            core.read_cv(song, out, self.folder / 'legend.json', self.folder / 'notation')
            self.assertIsInstance(build.call_args.kwargs['bands'], dict)
        self.assertEqual(seen, [False])
        self.assertEqual(before, reader.snapshot(song))

    def test_write_guard_and_offline_guard(self):
        song = self.fixture()
        out = self.folder / 'work'; out.mkdir()
        before = reader.snapshot(song)
        with reader.write_guard(out):
            (out / 'ok.txt').write_text('scratch')
            with self.assertRaises(PermissionError):
                (song / 'forbidden.txt').write_text('input mutation')
            with self.assertRaises(PermissionError):
                (song / 'video.webm').unlink()
            with self.assertRaises(PermissionError):
                (out / 'ok.txt').replace(song / 'video.webm')
            with self.assertRaises(PermissionError):
                subprocess.run([sys.executable, '-B', '-c', 'pass'])
        self.assertEqual(before, reader.snapshot(song))

    def test_symlink_inputs_and_outputs_refused(self):
        song = self.fixture()
        alias = self.folder / 'alias'; alias.symlink_to(song, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'symlink'):
            reader.no_symlinks(alias)
        out = self.folder / 'output'; out.mkdir()
        (out / 'system_000.png').symlink_to(song / 'video.webm')
        with self.assertRaisesRegex(ValueError, 'symlink'):
            reader.output_path(out)
        (song / 'alias.webm').symlink_to(song / 'video.webm')
        with patch.object(reader, 'require_allowed'):
            with self.assertRaisesRegex(ValueError, 'symlink'):
                reader.source(song.parent, 'olliewoods_shorts', self.video, self.allowed)

    def test_id_format_and_no_pilot_collision(self):
        ids = [reader.system_id(self.video, n) for n in (0, 1, 9999, 10000)]
        self.assertEqual(ids[0], 'MPoZ-2nOESc_src_0000')
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(all('_chan_' not in sid for sid in ids))
        for value in ('../MPoZ-2nOESc', 'short', 'MPoZ-2nOESc/path'):
            with self.assertRaises(ValueError):
                reader.system_id(value, 0)
        for value in (-1, True, '0'):
            with self.assertRaises(ValueError):
                reader.system_id(self.video, value)

    def test_prepare_isolates_shared_hybrid_state_and_paths(self):
        import hybrid
        work = self.folder / 'work'; native = work / 'native'
        sid = reader.system_id(self.video, 0)
        strip = work / 'truth-reviewed/strip.json'; dump(strip, [])
        geometry = dict(offset_xy=[0, 0], matches=[])
        dump(native / 'manifest.json', {'systems': [dict(id=sid, channel_geometry=geometry)]})
        inventory = dict(native=reader.relative(native), systems=[sid],
                         source_strip=reader.relative(strip), inputs={'strip': sha(strip)})
        old_hybrid = hybrid.HYBRID

        def fake_prepare(systems, bench):
            job = hybrid.HYBRID / 'jobs' / sid
            dump(job / 'job.json', dict(system_id=sid, job_dir=str(job)))
            dump(job / 'cv.json', {})
            dump(hybrid.HYBRID / 'prepared.json', {'systems': systems})

        with patch.object(hybrid, 'prepare', side_effect=fake_prepare), \
                patch.object(hybrid, 'instructions', side_effect=lambda meta, cv: meta['job_dir']):
            core.prepare(inventory, work)
        self.assertEqual(hybrid.HYBRID, old_hybrid)
        meta = load(work / 'hybrid/jobs' / sid / 'job.json')
        self.assertFalse(Path(meta['job_dir']).is_absolute())
        self.assertFalse(Path(meta['bench']).is_absolute())
        self.assertEqual(meta['source_strip_sha256'], sha(strip))
        self.assertTrue((work / 'hybrid/prepared.json').is_file())

    def test_portable_normalizes_nested_provenance(self):
        path = self.folder / 'scan.json'
        value = dict(video=str(ROOT / 'sources/channel/video/video.mp4'), nested=[{'song': str(self.folder)}])
        dump(path, value); core.portable(self.folder)
        normalized = load(path)
        self.assertEqual(normalized['video'], 'sources/channel/video/video.mp4')
        self.assertFalse(Path(normalized['nested'][0]['song']).is_absolute())


if __name__ == '__main__':
    unittest.main(verbosity=2)
