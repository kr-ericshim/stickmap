"""Read-only same-performance evidence from local audio; never transfers truth.

Independent 6-second waveform-envelope queries search the whole retained
reference. High correlations at consistent lags suggest shared recorded audio;
titles or matching performers alone never establish performance identity.
"""
import json
from pathlib import Path
import subprocess

import numpy as np
from scipy.signal import correlate
from pilot_new import PILOTS, ROOT


def envelope(folder, stem=False):
    for name in (('drums.htdemucs.wav',) if stem else ('audio.m4a','audio.wav','drums.htdemucs.wav')):
        path=folder/name
        if path.exists():break
    else:raise FileNotFoundError(folder)
    data=subprocess.run(['ffmpeg','-loglevel','error','-threads','1','-i',str(path),
        '-f','f32le','-ar','4000','-ac','1','pipe:1'],check=True,capture_output=True).stdout
    x=np.frombuffer(data,dtype='<f4');x=x[:len(x)//20*20]
    return np.abs(x).reshape(-1,20).mean(1).astype(float)


def compare(query, reference, stem=False):
    x,y=envelope(query,stem),envelope(reference,stem);rows=[];width=1200
    # References often retain only a short excerpt of a full performance.
    # Cover every query passage instead of relying on four whole-video probes.
    for start in range(0,max(0,len(x)-width+1),600):
        if start<0 or len(y)<width:continue
        window=x[start:start+width];window=window-window.mean()
        if np.linalg.norm(window)<1e-7:continue
        cc=correlate(y,window,mode='valid',method='fft')
        cs=np.r_[0,np.cumsum(y)];sq=np.r_[0,np.cumsum(y*y)]
        variance=np.maximum(sq[width:]-sq[:-width]-(cs[width:]-cs[:-width])**2/width,0)
        scores=cc/np.maximum(np.sqrt(variance)*np.linalg.norm(window),1e-9)
        k=int(scores.argmax())
        rows.append(dict(query_seconds=start/200,reference_seconds=k/200,
                         lag_seconds=(k-start)/200,correlation=float(scores[k])))
    high=[r for r in rows if r['correlation']>=.8]
    # Repeated lesson phrases can match one window; multiple independent
    # windows at a stable offset are necessary for a confirmed shared passage.
    clusters=[[r for r in high if abs(r['lag_seconds']-seed['lag_seconds'])<=.05]
              for seed in high]
    best=max(clusters,key=len,default=[])
    consistent=len(best)>=3
    return dict(query=str(query.relative_to(ROOT)),reference=str(reference.relative_to(ROOT)),
                audio_kind='cached Demucs drum stem' if stem else 'retained full-mix audio',
                windows=rows,consistent_high_windows=best,shared_audio_confirmed=bool(consistent),
                limitation='A failed match does not prove different recordings; edited clips, narration, speed changes or mix differences may defeat this narrow check.')


def find(pattern):
    return next(ROOT.glob(pattern))


def main():
    ivan=find('truth/harvest/songs/*-ZsMtIfjkubo')
    alfio=find('truth/samples/alfio/*-K_AtZoNIoKo')
    jrod=find('songs/*-bd17mxbLaNE')
    eric=find('truth/harvest/songs/*-9juoKgimeR4')
    larnell=find('truth/songs/*-XUdFq81ZSRQ')
    daily_larnell=ROOT/'.eval-cache/pilot1080/i1nnfqkv5WY'
    alex=PILOTS/'alexblack/N0PXFlgigjw'
    pairs=[(alex,ivan),(alex,alfio),(PILOTS/'alexblack/2AzwR38fHKU',jrod),
           (PILOTS/'nyenty/koBhkNbncAw',eric),
           (PILOTS/'timbuell/JG2xMCBwL1Y',eric),
           (PILOTS/'timbuell/eik7Wk03o-8',larnell),
           (PILOTS/'timbuell/eik7Wk03o-8',daily_larnell),
           (alex,larnell)]
    results=[compare(*pair) for pair in pairs]
    results.extend(compare(a,b,stem=True) for a,b in ((alex,ivan),(alex,alfio),(alfio,ivan)))
    (PILOTS/'duplicates.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()
