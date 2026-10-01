"""Keep score changes separate from cursor motion, including cursors that move down a page."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np


def find_strip(video, samples=24):
    cap = cv2.VideoCapture(str(video))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    white = []
    for k in range(samples):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(count * (k + .5) / samples))
        ok, frame = cap.read()
        if ok:
            white.append((cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) > 225).mean(axis=1))
    cap.release()
    rows = np.where(np.percentile(white, 70, axis=0) > .4)[0]
    bands = np.split(rows, np.where(np.diff(rows) > 6)[0] + 1)
    band = max(bands, key=len)
    if len(band) < 60:
        raise ValueError("no notation strip found")
    return int(band[0]), int(band[-1]) + 1


def paper(frame):
    """Colored highlighting leaves black ink intact; whitening whole columns deletes heads."""
    return frame.max(axis=2)


def cursor_box(frame):
    b, g, r = (frame[:, :, i].astype(np.int16) for i in range(3))
    mask = (((b > 130) & (b-r > 25) & (b-g > 5)) |
            ((r > 80) & (r-b > 20) & (r-g > 15))).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15,1), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    candidates = [s for s in stats[1:] if s[3] >= 18 and s[2] <= 90 and s[4] >= 20]
    if not candidates:
        return None
    x, y, w, h, _ = max(candidates, key=lambda s: s[3])
    return [float(x+(w-1)/2), int(y), int(y+h), int(w)]


def write(video, out, top=None, bottom=None, left=0, right=None, no_cursor=False):
    cv2.setNumThreads(1)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if top is None:
        top, bottom = find_strip(video)
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cursor, boxes, segments = [], [], []
    ref = None
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        crop = frame[top:bottom, left:right]
        gray = paper(crop)
        box = None if no_cursor else cursor_box(crop)
        cursor.append(box[0] if box else -1)
        boxes.append(box)
        small = cv2.resize(gray, (320, max(40, gray.shape[0]//3)), interpolation=cv2.INTER_AREA)
        # The clipped difference ignores compression and faint colored highlighting.
        diff = 999 if ref is None else np.maximum(np.abs(small.astype(float)-ref)-25, 0).mean()
        if diff > 4:
            segments.append(dict(start=n, samples=[]))
            ref = small.astype(float)
        if n % 6 == 0 and len(segments[-1]['samples']) < 40:
            segments[-1]['samples'].append(gray)
        n += 1
    cap.release()
    records = []
    from heads import staffs
    for i, seg in enumerate(segments):
        end = segments[i+1]['start'] if i+1 < len(segments) else n
        if end-seg['start'] < max(5, fps*.25) or not seg['samples']:
            continue
        gray = np.median(np.stack(seg['samples']), axis=0).astype(np.uint8)
        found = staffs(gray, min_space=3)
        if not found:
            continue
        centers = [(lines[0]+lines[-1])/2 for lines, _ in found]
        for row, (lines, space) in enumerate(found):
            y0 = 0 if row == 0 else int((centers[row-1]+centers[row])/2)
            y1 = len(gray) if row+1 == len(found) else int((centers[row]+centers[row+1])/2)
            sid = len(records)
            cv2.imwrite(str(out/f'system_{sid:03d}.png'), gray[y0:y1])
            on = [(j, boxes[j]) for j in range(seg['start'], end) if boxes[j] and
                  (len(found)==1 or boxes[j][1] <= centers[row] <= boxes[j][2])]
            records.append(dict(system=sid, start=seg['start'], end=end, row=row, page_rows=len(found),
                                crop=[y0,y1], native_space=space,
                                cursor_frames=[j for j,_ in on], cursor_x=[b[0] for _,b in on]))
    # Images no longer referenced by this scan are stale derived output.
    keep = {f"system_{s['system']:03d}.png" for s in records}
    for p in out.glob('system_*.png'):
        if p.name not in keep:
            p.unlink()
    data = dict(version=2, video=str(Path(video).resolve()), left=left, right=right,
                fps=fps, top=top, bottom=bottom, cursor=cursor,
                systems=[s['start'] for s in records], records=records, frames=n)
    (out/'scan.json').write_text(json.dumps(data))
    print(f'{out.name}: {n/fps:.1f}s, {len(records)} systems, cursor {sum(x>=0 for x in cursor)/fps:.1f}s', flush=True)
    return data


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('video'); p.add_argument('args', nargs='+')
    a = p.parse_args()
    if len(a.args) == 3:
        write(a.video, a.args[2], int(a.args[0]), int(a.args[1]))
    else:
        write(a.video, a.args[0])
