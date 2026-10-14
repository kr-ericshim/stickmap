"""Optional, channel-calibrated composite glyphs for explicit printed marks."""
from functools import lru_cache
from pathlib import Path
import cv2
import numpy as np

HERE=Path(__file__).parent


@lru_cache(maxsize=16)
def template(path):
    im=cv2.imread(str(HERE/path),0)
    if im is None:raise ValueError(f'missing calibrated mark template: {path}')
    return im


def ghost_score(gray,n,space,config):
    if (n['shape']!='filled' or n['inst'] not in config.get('instruments',['snare'])
            or n['pos'] not in config.get('positions',[5])):return 0.
    scale=space/config['reference_spacing']
    t=cv2.resize(template(config['path']),None,fx=scale,fy=scale)
    h,w=t.shape;x=round(n['x']-config['anchor'][0]*scale);y=round(n['y']-config['anchor'][1]*scale)
    patch=gray[max(0,y-2):y+h+3,max(0,x-2):x+w+3]
    if patch.shape[0]<h or patch.shape[1]<w:return 0.
    response=cv2.matchTemplate(patch,t,cv2.TM_CCOEFF_NORMED)
    if not config.get('side_columns'):
        return float(response.max())
    # A composite head dominates correlation even when both parentheses are
    # absent. Validate each side independently, removing horizontal staff ink
    # by centring each row. A nearby stem or beam must not supply a ghost mark.
    best=0.
    for yy,xx in np.argwhere(response>=config['threshold']):
        live=patch[yy:yy+h,xx:xx+w];scores=[]
        for a,b in config['side_columns']:
            a,b=round(a*scale),round(b*scale)
            reference=t[:,a:b].astype(float);side=live[:,a:b].astype(float)
            reference-=reference.mean(axis=1,keepdims=True)
            side-=side.mean(axis=1,keepdims=True)
            scores.append(float(np.sum(reference*side)/max(1,np.linalg.norm(reference)*np.linalg.norm(side))))
        if min(scores)>=config['side_min_score']:
            best=max(best,float(response[yy,xx]))
    return best


def clef_score(gray, lines, space, notes, config):
    """An opt-in native grey paired-bar clef; absent/partial glyphs reject.

    Search only before the first head and inside the observed staff's middle.
    Both calibrated strokes must match after removing horizontal row intensity.
    Correlation recognizes the grey source glyph without changing staff support
    or globally raising the darkness threshold used by the ordinary reader.
    """
    scale=space/config['reference_spacing']
    t=cv2.resize(template(config['path']),None,fx=scale,fy=scale)
    h,w=t.shape
    first=min((n['x']-max(space,n.get('w',0)/2) for n in notes),default=gray.shape[1])
    end=min(round(gray.shape[1]*config['max_x_fraction']),int(first))
    y=round(lines[-1]-config['bottom_to_top_spaces']*space)
    patch=gray[max(0,y-2):y+h+3,:end]
    if patch.shape[0]<h or patch.shape[1]<w:return 0.
    response=cv2.matchTemplate(patch,t,cv2.TM_CCOEFF_NORMED);best=0.
    for yy,xx in np.argwhere(response>=config['threshold']):
        live=patch[yy:yy+h,xx:xx+w];scores=[]
        for a,b in config['side_columns']:
            a,b=round(a*scale),round(b*scale)
            reference=t[:,a:b].astype(float);side=live[:,a:b].astype(float)
            reference-=reference.mean(axis=1,keepdims=True)
            side-=side.mean(axis=1,keepdims=True)
            scores.append(float(np.sum(reference*side)/max(1,np.linalg.norm(reference)*np.linalg.norm(side))))
        if min(scores)>=config['side_min_score']:best=max(best,float(response[yy,xx]))
    return best
