"""Bounded native notation diagnostics. Never imports audio/detector/scorer code."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

import cv2

from harvest_scan import write
from layouts import channel_profile, detect_region
from media import video_path
from strip import read


def run(song, channel, out, start, seconds):
    if start<0 or not 0<seconds<=20:
        raise ValueError('reader-only window requires start >=0 and 0 < seconds <=20')
    out=Path(out)
    if out.exists():raise FileExistsError('use a fresh diagnostic output folder')
    profile,path=channel_profile(channel)
    if not path or not profile.get('score_region'):
        raise ValueError('bounded strip check requires an inspected channel score region')
    video=video_path(Path(song))
    cap=cv2.VideoCapture(str(video))
    try:fps=cap.get(cv2.CAP_PROP_FPS)
    finally:cap.release()
    started=time.monotonic()
    geometry=detect_region(video,profile)
    scan=write(video,out,geometry=geometry,start_frame=round(start*fps),
               end_frame=round((start+seconds)*fps),
               reject_overlays=profile.get('reject_chromatic_overlays',False),
               native_fallback=profile.get('native_frame_fallback',False))
    systems=read(out,path)
    clean=[s for s in systems if not s['staff']['reasons']]
    counts=Counter(q for s in systems for q in s['staff']['reasons'])
    result=dict(channel=channel,source=str(video.resolve()),
        requested_window=[start,start+seconds],
        frame_window=[scan['scan_start_frame'],scan['frames']],fps=fps,
        found=len(systems),reader_clean=len(clean),
        heads_in_clean=sum(len(s['notes']) for s in clean),
        verified_position_heads=sum(n.get('exportable',False) for s in clean for n in s['notes']),
        printed_clef=sum(s['staff'].get('clef_confirmed',False) for s in systems),
        rejections=dict(counts),
        variants=dict(Counter(r.get('foreground') for r in scan['records'])),
        elapsed_seconds=round(time.monotonic()-started,3),
        scope='notation only; no audio, detector, scorer, strict/scoring alignment or export')
    (out/'window-result.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('song',type=Path)
    p.add_argument('--channel',required=True)
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--start',required=True,type=float)
    p.add_argument('--seconds',type=float,default=16)
    a=p.parse_args()
    print(json.dumps(run(a.song,a.channel,a.out,a.start,a.seconds),indent=2))


if __name__=='__main__':main()
