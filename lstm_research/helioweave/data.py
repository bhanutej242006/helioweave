"""Kaggle ingestion; timestamps remain source-local (assumed Asia/Kolkata)."""
import csv, json, hashlib
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
DT=.25
HISTORY=24
HORIZON=16


def stamp(s):
    for f in ('%d-%m-%Y %H:%M','%Y-%m-%d %H:%M:%S','%Y-%m-%d %H:%M'):
        try: return datetime.strptime(s,f)
        except ValueError: pass
    raise ValueError('Unrecognized DATE_TIME: '+s)


def prepare(plant=1):
    raw=ROOT/'data/raw'; gen=raw/f'Plant_{plant}_Generation_Data.csv'; weather=raw/f'Plant_{plant}_Weather_Sensor_Data.csv'
    if not gen.exists() or not weather.exists():
        raise FileNotFoundError('Run python download_dataset.py or put the four Kaggle CSVs in data/raw.')
    groups=defaultdict(dict); keys=set(); duplicates=0
    with gen.open(newline='',encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            t=stamp(row['DATE_TIME']); key=row['SOURCE_KEY']; value=float(row['AC_POWER'])
            if key in groups[t]: duplicates+=1
            if np.isfinite(value) and value>=0: groups[t][key]=value; keys.add(key)
    sensors={}
    with weather.open(newline='',encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            vals=[float(r['IRRADIATION']),float(r['AMBIENT_TEMPERATURE'])]
            if np.all(np.isfinite(vals)): sensors[stamp(r['DATE_TIME'])]=vals
    start=min(groups); end=max(groups); times=[]; power=[]; irr=[]; temp=[]; valid=[]; coverage=[]
    t=start; prev=[0.,0.,25.]
    while t<=end:
        g=groups.get(t,{}); cov=len(g)/len(keys)
        ok=cov>=.9 and t in sensors
        if ok: prev=[float(np.mean(list(g.values()))),*sensors[t]]
        times.append(t.isoformat()); power.append(prev[0]); irr.append(prev[1]); temp.append(prev[2]); valid.append(ok); coverage.append(cov)
        t+=timedelta(minutes=15)
    n=len(times); train_end=int(n*.6); val_end=int(n*.8); early_end=(train_end+val_end)//2
    # All normalization constants are training-only, including the scenario scale.
    scale=max(float(np.max(np.array(power)[:train_end][np.array(valid)[:train_end]])),1.)
    pv=np.clip(np.array(power)/scale,0,1.2)  # normalized measured mean inverter AC, not claimed nameplate output
    minutes=np.array([datetime.fromisoformat(t).hour*60+datetime.fromisoformat(t).minute for t in times])
    features=np.column_stack([pv,np.array(irr),np.array(temp)/50,
                              np.sin(2*np.pi*minutes/1440),np.cos(2*np.pi*minutes/1440),np.array(valid,dtype=float)])
    np.savez(ROOT/'data/processed.npz',features=features,pv=pv,valid=valid,times=times,
             split=[train_end,early_end,val_end],scale=scale)
    with (ROOT/'data/processed_solar.csv').open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['timestamp_source_local','normalized_ac_power','irradiation','ambient_c','valid_target','inverter_coverage'])
        w.writerows(zip(times,pv,irr,temp,valid,coverage))
    report={'source':'Kaggle / anikannal/solar-power-generation-data','plant':plant,'interval_minutes':15,
            'rows':n,'valid_rows':int(sum(valid)),'inverters':len(keys),'duplicates':duplicates,
            'training_scale_mean_inverter_ac':scale,'start':times[0],'end':times[-1],
            'split_indices':[train_end,early_end,val_end],
            'split_dates':[times[x] for x in [train_end,early_end,val_end]],
            'treatment':'Mean AC across available inverters; require >=90% inverter coverage and weather. Invalid intervals forward-filled only for history; invalid forecast targets excluded. Scenario PV = normalized AC x chosen capacity.',
            'sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in raw.glob('*.csv')}}
    (ROOT/'reports/data_quality.json').write_text(json.dumps(report,indent=2))
    return report


def load():
    with np.load(ROOT/'data/processed.npz',allow_pickle=False) as d: return {k:d[k].copy() for k in d.files}


def prior_day(d,t,h=HORIZON):
    ids=np.arange(t,t+h)-96
    return d['pv'][np.clip(ids,0,t-1)]


def samples(d,start,end,stride=1):
    ids=[t for t in range(max(96,start),end-HORIZON+1,stride) if np.all(d['valid'][t:t+HORIZON])]
    X=np.array([d['features'][t-HISTORY:t] for t in ids]); base=np.array([prior_day(d,t) for t in ids])
    Y=np.array([d['pv'][t:t+HORIZON] for t in ids])
    return X,Y,base,np.array(ids)


def forecast(net,d,t):
    base=prior_day(d,t)
    return np.clip(base+net.forward(d['features'][t-HISTORY:t][None])[0],0,1.2)
