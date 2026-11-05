"""Journal recovery and compaction contracts using tiny disposable fixtures."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent));sys.path.insert(0,str(HERE.parents[1]/'harvest'))
import batch
import harvest
from manifest_io import read,append


SCRATCH=Path(tempfile.gettempdir())/'stickmap-truth-video-tests';SCRATCH.mkdir(parents=True,exist_ok=True)

class BatchSafety(unittest.TestCase):
    def test_watch_retries_a_busy_song_without_consuming_limit(self):
        with tempfile.TemporaryDirectory(dir=SCRATCH) as name:
            root=Path(name);append(root/'manifest.jsonl',video='id',channel='atthedrum',slug='fixture',status='fetched')
            calls=[]
            def process(*args):
                calls.append(1)
                if len(calls)==1:return None
                append(root/'manifest.jsonl',video='id',status='read')
                return {'complete':True}
            with patch.object(sys,'argv',['batch.py','--root',str(root),'--watch','--limit','1','--min-free-gb','0']),patch('batch.process',side_effect=process),patch('batch.report'),patch('batch.time.sleep'):
                batch.main()
            self.assertEqual(len(calls),2)

    def test_timeout_publishes_empty_diagnostic_and_preserves_sources(self):
        import subprocess
        with tempfile.TemporaryDirectory(dir=SCRATCH) as name:
            root=Path(name);song=root/'songs/fixture';song.mkdir(parents=True)
            (song/'video.mp4').write_bytes(b'video');(song/'attacks.htdemucs.npy').write_bytes(b'attacks')
            append(root/'manifest.jsonl',video='id',channel='atthedrum',slug='fixture',status='fetched')
            profile,path=batch.channel_profile('atthedrum')
            with patch('batch.subprocess.run',side_effect=subprocess.TimeoutExpired('fixture',1)):
                r=batch.process(root,read(root/'manifest.jsonl')['id'],profile,path,1)
            self.assertFalse(r['complete']);self.assertEqual(read(root/'manifest.jsonl')['id']['status'],'unreadable')
            self.assertEqual(json.loads((song/'truth/truth.json').read_text()),[])
            self.assertTrue((song/'truth/QUALITY.md').exists());self.assertEqual((song/'video.mp4').read_bytes(),b'video')

    def test_song_lock_is_exclusive_and_released(self):
        with tempfile.TemporaryDirectory(dir=SCRATCH) as name:
            p=Path(name)/'lock'
            with batch.lock(p) as first:
                self.assertTrue(first)
                with batch.lock(p) as second:self.assertFalse(second)
            with batch.lock(p) as third:self.assertTrue(third)

    def test_recover_completed_export_without_running_reader(self):
        with tempfile.TemporaryDirectory(dir=SCRATCH) as name:
            root=Path(name);song=root/'songs'/'fixture';out=song/'truth';out.mkdir(parents=True)
            (song/'video.mp4').write_bytes(b'fixture');(song/'attacks.htdemucs.npy').write_bytes(b'fixture')
            profile,path=batch.channel_profile('atthedrum')
            result=dict(complete=True,signature=batch.signature(song,profile),scoring_systems=1,scoring_notes=2)
            batch.dump(out/'result.json',result)
            append(root/'manifest.jsonl',video='id',channel='atthedrum',slug='fixture',status='fetched')
            with patch('batch.subprocess.run') as run:
                recovered=batch.process(root,read(root/'manifest.jsonl')['id'],profile,path,1)
                run.assert_not_called()
            self.assertEqual(recovered,result);self.assertEqual(read(root/'manifest.jsonl')['id']['status'],'read')

    def test_compaction_retains_truth_overlays_caches_and_live_song(self):
        with tempfile.TemporaryDirectory(dir=SCRATCH) as name:
            root=Path(name);songs=root/'songs';song=songs/'finished';truth=song/'truth';truth.mkdir(parents=True)
            (truth/'truth.json').write_text('[]');(truth/'overlay.png').write_bytes(b'png')
            for f in ('video.mp4','audio.wav','drums.htdemucs.wav','attacks.htdemucs.npy','meta.json'):(song/f).write_bytes(b'fixture')
            active=songs/'active';active.mkdir();(active/'video.mp4').write_bytes(b'active')
            manifest=root/'manifest.jsonl';append(manifest,video='a',slug='finished',status='read');append(manifest,video='b',slug='active',status='fetched')
            def encode(args,**kwargs):Path(args[-1]).write_bytes(b'opus')
            with patch.multiple(harvest,OUT=root,SONGS=songs,MANIFEST=manifest),patch('harvest.subprocess.run',side_effect=encode):
                harvest.compact(False)
            self.assertTrue((truth/'truth.json').exists());self.assertTrue((truth/'overlay.png').exists())
            self.assertTrue((song/'attacks.htdemucs.npy').exists());self.assertTrue((song/'drums.htdemucs.opus').exists())
            self.assertFalse((song/'video.mp4').exists());self.assertFalse((song/'audio.wav').exists());self.assertFalse((song/'drums.htdemucs.wav').exists())
            self.assertTrue((active/'video.mp4').exists());self.assertEqual(read(manifest)['a']['status'],'compacted')


if __name__=='__main__':unittest.main()
