"""Build inspectable ground truth without requiring detector inference."""
import argparse
import subprocess
import sys
from pathlib import Path
import scan
import strip
from align import build_truth


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('song',type=Path);p.add_argument('out',type=Path)
    p.add_argument('--legend');p.add_argument('--reuse-scan',action='store_true')
    p.add_argument('--score',action='store_true')
    p.add_argument('--crop',nargs=4,type=int,metavar=('LEFT','TOP','RIGHT','BOTTOM'))
    p.add_argument('--no-cursor',action='store_true')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    (a.out/'song.txt').write_text(str(a.song.resolve()))
    video=a.song/('video.hi.mp4' if (a.song/'video.hi.mp4').exists() else 'video.mp4')
    if not a.reuse_scan:
        roi={} if not a.crop else dict(zip(('left','top','right','bottom'),a.crop))
        scan.write(video,a.out,no_cursor=a.no_cursor,**roi)
    strip.read(a.out,a.legend)
    build_truth(a.song,a.out)
    if a.score:
        subprocess.run([sys.executable,str(Path(__file__).with_name('evaluate.py')),str(a.song),str(a.out/'strip.json'),'--reuse-truth'],check=True)


if __name__=='__main__':main()
