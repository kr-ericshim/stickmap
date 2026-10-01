"""Optional, channel-calibrated composite glyphs for explicit printed marks."""
from functools import lru_cache
from pathlib import Path
import cv2

HERE=Path(__file__).parent


@lru_cache(maxsize=16)
def template(path):
    im=cv2.imread(str(HERE/path),0)
    if im is None:raise ValueError(f'missing calibrated mark template: {path}')
    return im


def ghost_score(gray,n,space,config):
    if n['inst']!='snare' or n['shape']!='filled' or n['pos']!=5:return 0.
    scale=space/config['reference_spacing']
    t=cv2.resize(template(config['path']),None,fx=scale,fy=scale)
    h,w=t.shape;x=round(n['x']-config['anchor'][0]*scale);y=round(n['y']-config['anchor'][1]*scale)
    patch=gray[max(0,y-2):y+h+3,max(0,x-2):x+w+3]
    if patch.shape[0]<h or patch.shape[1]<w:return 0.
    return float(cv2.matchTemplate(patch,t,cv2.TM_CCOEFF_NORMED).max())
