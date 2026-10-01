"""Image-derived geometry; identity/notation keys are never guessed from geometry."""
import json
from pathlib import Path
import cv2
import numpy as np
from heads import staffs

HERE = Path(__file__).parent
LAYOUTS = {'kr_strip', 'drumscribe', 'page', 'dailychops', 'olliewoods'}


def channel_profile(channel=None):
    # Manifest strings must never become arbitrary file paths.
    paths = {p.stem: p for p in (HERE/'legends').glob('*.json')}
    p = paths.get(channel)
    if p:
        data = json.loads(p.read_text())
        if data.get('harvest_profile'):
            return data, p
    return dict(name='unverified', enabled=False, rules={}, min_x_fraction=.04,
                harvest_profile=True, reason='unknown channel; no verified legend'), None


def ink(frame):
    """Preserve dark colored played heads, whiten pale colored cursor backgrounds."""
    lo, hi = frame.min(2), frame.max(2)
    return np.where((hi > 170) & (lo > 105), hi, lo).astype(np.uint8)


def cursor(frame, space=11):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Yellow bar shading is deliberately not a playhead. Blue, green, pink,
    # lavender and orange vertical cursors are accepted on geometric evidence.
    mask = ((hsv[:, :, 1] > 27) & (hsv[:, :, 2] > 120) &
            ~((hsv[:, :, 0] >= 22) & (hsv[:, :, 0] <= 38))).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((max(5, round(space)), 1), np.uint8))
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    candidates = [s for s in stats[1:] if s[3] >= max(18, space*2.5)
                  and 1 <= s[2] <= min(90, space*7) and s[4] >= s[3]*.6]
    if not candidates:
        return None
    x,y,w,h,_ = max(candidates, key=lambda s: s[3]/max(1, s[2])**.2)
    return [float(x+(w-1)/2), int(y), int(y+h), int(w)]


def detect(video, requested=None, y_min_fraction=0):
    cap = cv2.VideoCapture(str(video))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    observations = []; paper_tops=[]
    h=w=0
    for fraction in (.1, .25, .5, .75):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(count*fraction))
        ok, frame = cap.read()
        if not ok:
            continue
        h,w=frame.shape[:2]
        gray = ink(frame)
        for lines, sp in staffs(gray, min_space=6):
            if lines[0] < h*y_min_fraction:
                continue
            y0=max(0,round(lines[0]-sp*4));y1=min(h,round(lines[-1]+sp*3))
            # A light field is required; inverted score/video overlays need a
            # separately validated foreground extractor, not global inversion.
            if (gray[y0:y1] > 180).mean() > .55:
                observations.append((lines,sp))
                desired=max(0,int(lines[0]-sp*7.25))
                end=max(desired,int(lines[0]-sp*3.7))
                window=max(7,round(sp))
                light=np.convolve((gray>180).mean(1),np.ones(window)/window,mode='same')
                bad=np.flatnonzero(light[desired:end]<.6)+desired
                paper_tops.append(max(desired,int(bad[-1])+1) if len(bad) else desired)
    cap.release()
    if not observations:
        raise ValueError('no stable light five-line staff; count-in/inverted/occluded notation or missing score')
    low=min(paper_tops)
    high=max(ls[-1]+sp*3 for ls,sp in observations)
    mids=np.array([(ls[0]+ls[-1])/2 for ls,sp in observations])
    inferred='page' if np.ptp(mids)>h*.3 else 'dailychops' if np.median(mids)<h*.5 else 'kr_strip'
    layout=requested if requested in LAYOUTS else inferred
    # Manifest is a hint; broad page/top searches remain driven by observed ink.
    return dict(layout=layout, image_layout=inferred, left=0, right=w,
                top=max(0,int(low)), bottom=min(h,int(high)+1),
                native_spacing=[round(sp,3) for _,sp in observations],
                identity_source='manifest' if requested in LAYOUTS else 'image geometry only')
