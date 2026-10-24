"""Causal historical replay; no hardware or live weather connection is implied."""
import json
from functools import lru_cache
import numpy as np
from .data import ROOT,load,forecast
from .lstm import LSTM
from .scheduler import scenario,simulate


@lru_cache(maxsize=1)
def resources():
    d=load(); solar=LSTM.load(ROOT/'models/solar_lstm.npz'); policy=LSTM.load(ROOT/'models/scheduler_lstm.npz')
    with np.load(ROOT/'models/calibration.npz') as a: radius=a['radius'].copy()
    return d,solar,policy,radius


def days():
    d,_,_,_=resources(); c=int(d['split'][2])
    return [{'index':i,'date':str(d['times'][i])[:10], 'observed_pct':round(float(np.mean(d['valid'][i:i+96]))*100,1)}
            for i in range(((c+95)//96)*96,len(d['pv'])-95,96) if np.mean(d['valid'][i:i+96])>=.9]


def get_forecasts(start,capacity=60.):
    d,net,_,_=resources(); forecasts=[]
    for t in range(96):
        issue=start+t; horizon=96-t
        f=d['pv'][issue-96:start]*capacity  # past day only, extended to today's end
        short=forecast(net,d,issue)*capacity
        f[:min(16,horizon)]=short[:min(16,horizon)]
        f[0]=d['pv'][issue]*capacity  # current measured PV available to controller
        forecasts.append(f)
    return forecasts


@lru_cache(maxsize=24)
def run(day,capacity=60.,grid=25.,risk=0.,seed=42,stress=1.):
    if day not in [a['index'] for a in days()]: raise ValueError('Choose an available held-out date')
    if not 10<=capacity<=120 or not 15<=grid<=80 or not 0<=risk<=1 or not 1<=stress<=2: raise ValueError('Settings out of range')
    d,net,policy,radius=resources(); actual=d['pv'][day:day+96]*capacity
    fs=get_forecasts(day,capacity); ev,load_kw,tariff=scenario(seed,stress=stress)
    results={m:simulate(actual,fs,ev,load_kw,tariff,m,policy,grid,risk,float(np.mean(radius))*capacity)
             for m in ['immediate','learned','rolling','oracle']}
    ref=results['immediate']['summary']['cost_inr']
    for r in results.values():
        r['summary']['savings_pct']=100*(ref-r['summary']['cost_inr'])/ref if ref>0 else None
        r['summary']['cost_gap_to_oracle_inr']=r['summary']['cost_inr']-results['oracle']['summary']['cost_inr']
    noon=48; predicted=forecast(net,d,day+noon)*capacity
    return {'mode':'Historical replay + simulated EV/building/tariff scenario','date':str(d['times'][day])[:10],
            'settings':{'capacity_kw':capacity,'grid_limit_kw':grid,'risk':risk,'seed':seed,'stress':stress},
            'vehicles':ev,'results':results,'observed':d['valid'][day:day+96].tolist(),
            'forecast':{'issue_hour':12,'actual':actual[noon:noon+16].tolist(),'predicted':predicted.tolist(),
                        'lower':np.maximum(0,predicted-radius*capacity).tolist(),'upper':(predicted+radius*capacity).tolist()}}


def meta():
    return {'days':days(),'training':json.loads((ROOT/'reports/training.json').read_text()),
            'data_source':'https://www.kaggle.com/datasets/anikannal/solar-power-generation-data'}
