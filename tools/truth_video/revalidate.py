"""Re-read retained sources into a scratch cohort without publishing harvest state.

Inputs are linked, not copied. New caches and all reader output stay under OUT;
the canonical song directories and append-only manifest are never written.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys

from batch import ROOT, HERE, manifest_read, song_path, dump
from layouts import channel_profile


def run(root, out, channels, jobs=3, videos=None, reuse_scan=False):
    if not 1 <= jobs <= 4:
        raise ValueError('jobs must be between one and four')
    out=out.resolve()
    if not out.is_relative_to(ROOT/'.eval-cache'):
        raise ValueError('scratch output must be inside .eval-cache')
    out.mkdir(parents=True,exist_ok=True)
    rows=[r for r in manifest_read(root/'manifest.jsonl').values()
          if r.get('channel') in channels and (not videos or r['video'] in videos)]

    def attempt(row):
        source=song_path(root,row);stage=out/'inputs'/row['video'];stage.mkdir(parents=True,exist_ok=True)
        folder=out/'reads'/row['video']
        if folder.exists():
            raise FileExistsError(f'preserve existing attempt: {folder}')
        folder.mkdir(parents=True)
        before=json.loads((source/'truth/result.json').read_text()) if (source/'truth/result.json').exists() else None
        # Only input files; a missing cache will be computed in this scratch song.
        for name in ('video.mp4','meta.json','attacks.htdemucs.npy','drums.htdemucs.wav',
                     'drums.htdemucs.opus','onsets.bands.htdemucs.npy'):
            p=source/name
            if p.exists() and not (stage/name).exists():
                (stage/name).symlink_to(p.resolve())
        _,profile=channel_profile(row['channel'])
        cmd=[sys.executable,str(HERE/'batch.py'),'--worker',str(stage),str(folder),str(profile)]
        if reuse_scan:cmd+=['--reuse',str(source/'truth')]
        env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1',
             'MKL_NUM_THREADS':'1','VECLIB_MAXIMUM_THREADS':'1'}
        with (folder/'run.log').open('w') as log:
            try:
                p=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,timeout=1800)
                code=p.returncode
            except subprocess.TimeoutExpired:
                code='timeout'
        result=json.loads((folder/'result.json').read_text()) if (folder/'result.json').exists() else None
        item=dict(video=row['video'],channel=row['channel'],title=row.get('title'),
                  source=str(source),output=str(folder),returncode=code,before=before,after=result,
                  previous_status=row.get('status'))
        dump(out/(row['video']+'.json'),item)
        print(row['channel'],row['video'],code,
              None if result is None else {k:result[k] for k in ('systems','strict_systems','scoring_systems','strict_notes','scoring_notes')},flush=True)
        return item

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        results=list(pool.map(attempt,rows))
    dump(out/'results.json',results)
    return results


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT/'truth/harvest')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--only',nargs='+',required=True)
    p.add_argument('--videos',nargs='+')
    p.add_argument('--jobs',type=int,default=3)
    p.add_argument('--reuse-scan',action='store_true')
    a=p.parse_args();run(a.root,a.out,a.only,a.jobs,a.videos,a.reuse_scan)
