"""Transfer verified notation between two audio versions using waveform envelopes."""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import correlate
from tiers import write as write_tiers


def envelope(path):
    x,sr=sf.read(path,dtype='float32',always_2d=True);hop=round(sr/200)
    x=x[:len(x)//hop*hop]
    return np.abs(x).reshape(-1,hop,x.shape[1]).mean(axis=(1,2))


def run(source,target_song,out):
    out.mkdir(parents=True,exist_ok=True)
    source_song=Path((source/'song.txt').read_text().strip())
    x=envelope(source_song/'audio.wav');y=envelope(target_song/'drums.htdemucs.wav')
    rows=[]
    for center in np.arange(20,min(len(x),len(y))/200-15,30):
        k=int(center*200);w=x[k-2000:k+2000];ss=y[k-2400:k+2400]
        w=w-w.mean();ss=ss-ss.mean()
        cc=correlate(ss,w,mode='valid',method='fft');best=int(cc.argmax());seg=ss[best:best+len(w)]
        rows.append(dict(center_seconds=float(center),offset_seconds=(best-400)/200,
                         correlation=float(np.dot(w,seg)/(np.linalg.norm(w)*np.linalg.norm(seg)))))
    centers=np.array([r['center_seconds'] for r in rows]);offsets=np.array([r['offset_seconds'] for r in rows])
    slope,intercept=np.polyfit(centers,offsets,1)
    held=[]
    for i in range(len(rows)):
        keep=np.arange(len(rows))!=i;a,b=np.polyfit(centers[keep],offsets[keep],1)
        held.append(float(offsets[i]-(a*centers[i]+b)))
    if min(r['correlation'] for r in rows)<.8 or max(abs(np.array(held)))>.02:
        raise ValueError('audio transfer does not pass correlation / held-out lag consistency')
    transform=lambda t:float(t*(1+slope)+intercept)
    evidence=dict(method='200 Hz mean-absolute waveform envelope; 20 s windows, +/-2 s search',
                  source=str(source),windows=rows,slope=float(slope),intercept_seconds=float(intercept),
                  held_out_offset_errors_seconds=held,independent_truth=False)
    (out/'audio-transfer.json').write_text(json.dumps(evidence,indent=2))
    (out/'song.txt').write_text(str(target_song.resolve()))
    for suffix in ['', '.scoring']:
        truth=json.loads((source/f'truth{suffix}.json').read_text())
        for n in truth:n['time']=transform(n['time']);n['transferred_from']=str(source)
        r=json.loads((source/f'alignment{suffix}.json').read_text())
        r['song']=str(target_song.resolve());r['transferred_from']=str(source);r['transfer_evidence']='audio-transfer.json'
        for s in r['systems']:
            for k in ['times','initial_times','held_out_times']:
                if k in s:s[k]=[transform(t) for t in s[k]]
            s['interval']=[transform(t) for t in s['interval']]
        for k in ['intervals','strict_intervals']:
            if k in r:r[k]=[[transform(t) for t in interval] for interval in r[k]]
        r['trustworthy_seconds']=sum(b-a for a,b in r['intervals'])
        (out/f'truth{suffix}.json').write_text(json.dumps(truth,indent=2));(out/f'alignment{suffix}.json').write_text(json.dumps(r,indent=2))
    # The optical reading is the same reference; link to original overlays in QA.
    (out/'strip.json').write_text((source/'strip.json').read_text())
    (out/'truth.additional.json').write_text(json.dumps([n for n in json.loads((out/'truth.scoring.json').read_text()) if n.get('tier')=='scoring'],indent=2))
    print('transferred',len(json.loads((out/'truth.scoring.json').read_text())),'notes;',evidence,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source',type=Path);p.add_argument('song',type=Path);p.add_argument('out',type=Path)
    a=p.parse_args();run(a.source,a.song,a.out)
