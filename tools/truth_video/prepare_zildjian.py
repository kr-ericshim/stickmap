"""Re-extract the single verified bottom staff from a saved full-frame scan.

The generic scan can mistake beam rows for additional staves. This publication
has exactly one staff at native y=932..988 within a 1080p frame.
"""
import json
from pathlib import Path
import cv2
import numpy as np
from heads import staffs


def run(folder):
    p=Path(folder);original=p/'scan.initial.json'
    if not original.exists():original.write_bytes((p/'scan.json').read_bytes())
    d=json.loads(original.read_text());groups={r['start']:r['end'] for r in d['records']}
    cap=cv2.VideoCapture(d['video']);records=[]
    for start,end in groups.items():
        images=[]
        for frame in np.linspace(start+2,end-2,5).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES,int(frame));ok,im=cap.read()
            if ok:images.append(im[820:1080].max(axis=2))
        gray=np.median(images,axis=0).astype('uint8')
        found=[(ls,sp) for ls,sp in staffs(gray,min_space=8) if 115<(ls[0]+ls[-1])/2<180]
        if not found:continue
        ls,sp=found[0];sid=len(records);cv2.imwrite(str(p/f'system_{sid:03d}.png'),gray)
        records.append(dict(system=sid,start=start,end=end,row=0,page_rows=1,crop=[0,260],native_space=sp,cursor_frames=[],cursor_x=[]))
    cap.release();d['records']=records;d['systems']=[r['start'] for r in records]
    d['layout_evidence']='Single bottom percussion staff verified; beam rows above it excluded.'
    (p/'scan.json').write_text(json.dumps(d))


if __name__=='__main__':
    import sys
    run(sys.argv[1])
