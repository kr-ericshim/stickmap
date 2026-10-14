"""Native geometry selection for known score ROIs; protected checks stay intact.

The original detector can mistake long beams for another staff. The scan's
image-only geometry identifies which staff is the drum row. Function bindings
reuse the exact existing head/label implementations without mutating modules.
"""
import types
import numpy as np


def measured_lines(gray, expected):
    expected = np.asarray(expected, dtype=float)
    space = float(np.median(np.diff(expected)))
    if len(expected) != 5 or space < 6:
        raise ValueError('invalid native staff prior')
    # Follow local vertical motion, but never synthesize missing horizontal ink.
    profile = np.mean(gray < 230, axis=1)
    candidates = []
    for delta in range(-round(space / 2), round(space / 2) + 1):
        rows = []
        for y in expected + delta:
            lo, hi = max(0, round(y)-2), min(len(gray), round(y)+3)
            if hi <= lo:
                break
            peak = lo + int(np.argmax(profile[lo:hi]))
            yy = np.arange(max(0, peak-1), min(len(gray), peak+2))
            weight = np.maximum(profile[yy] - .15, 0)
            rows.append(float(np.average(yy, weights=weight)) if weight.sum() else float(peak))
        if len(rows) != 5:
            continue
        sp = (rows[-1]-rows[0])/4
        if sp < 6 or max(abs(np.diff(rows)-sp)) > 1.1:
            continue
        support = [float(profile[round(y)]) for y in rows]
        if min(support) < .45:
            continue
        score = min(support) + .05*np.mean(support) - .002*abs(delta)
        candidates.append((score, rows, sp))
    if not candidates:
        # Score scene changes can move the whole row (Steve moves 18 px).
        # A unique complete native staff may replace the stale location prior;
        # its scale, line support and spacing residual must still agree.
        import heads as h
        found=h.staffs(gray,prefer_line_span=True)
        if len(found)==1:
            ls,sp=found[0]
            support=[float((gray[round(y)]<230).mean()) for y in ls]
            if abs(sp-space)<=1.1 and min(support)>=.45 and max(abs(np.diff(ls)-sp))<=1.1:
                return ls,sp
        raise ValueError('native five-line geometry unsupported in scan-selected drum row')
    _, lines, spacing = max(candidates)
    return lines, spacing


def bound_readers(expected):
    import heads as h
    import strip as s
    def select(gray, prefer_line_span=False):
        # Prefer the detector's own subpixel staff when it agrees with the ROI.
        found = h.staffs(gray, prefer_line_span=True)
        space = float(np.median(np.diff(expected)))
        local = [(ls, sp) for ls, sp in found
                 if max(abs(np.asarray(ls)-expected)) <= space*.5]
        if len(local) == 1:
            return local[0]
        return measured_lines(gray, expected)
    heads = types.FunctionType(h.heads.__code__, {**h.heads.__globals__, 'staff_lines': select},
                               h.heads.__name__, h.heads.__defaults__, h.heads.__closure__)
    label = types.FunctionType(s.label_heads.__code__, {**s.label_heads.__globals__, 'heads': heads},
                               s.label_heads.__name__, s.label_heads.__defaults__, s.label_heads.__closure__)
    return heads, label
