"""Chronological training: solar LSTM + optimization-distilled per-EV LSTM."""
import argparse,json,platform
import numpy as np
import scipy
from helioweave.data import ROOT,prepare,load,samples,forecast,prior_day
from helioweave.lstm import LSTM
from helioweave.scheduler import scenario,optimize,policy_features


def metrics(y,p):
    e=y-p; daylight=y>.01
    return {'mae_normalized':float(np.mean(abs(e))),'rmse_normalized':float(np.sqrt(np.mean(e*e))),
            'daylight_rmse_normalized':float(np.sqrt(np.mean(e[daylight]**2))),
            'rmse_by_horizon':[float(x) for x in np.sqrt(np.mean(e*e,axis=0))]}


def main(epochs=25,plant=1):
    quality=prepare(plant); d=load(); a,b,c=map(int,d['split'])
    X,Y,B,_=samples(d,96,a,2); VX,VY,VB,_=samples(d,a,b)
    if min(len(X),len(VX))<20: raise ValueError('Insufficient valid chronological samples')
    net=LSTM(6,20,16); history=net.fit(X,Y-B,VX,VY-VB,epochs=epochs)
    net.save(ROOT/'models/solar_lstm.npz')
    CX,CY,CB,_=samples(d,b,c); cp=np.clip(CB+net.forward(CX),0,1.2)
    # Separate calibration period; marginal empirical interval per horizon (dependent series).
    residual=np.abs(CY-cp); radius=np.quantile(residual,.9,axis=0,method='higher')
    TX,TY,TB,ids=samples(d,c,len(d['pv'])); pred=np.clip(TB+net.forward(TX),0,1.2)
    result={'python_tested':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,
            'source':quality,'samples':{'train':len(X),'early_stop_validation':len(VX),'calibration':len(CX),'test':len(TX)},
            'lstm':metrics(TY,pred),'previous_day':metrics(TY,TB),
            'coverage_90_by_horizon':np.mean((TY>=np.maximum(0,pred-radius))&(TY<=pred+radius),axis=0).tolist(),
            'interval_radius_normalized':radius.tolist(),'training_history':history,
            'note':'Overlapping test windows and short 34-day source; uncertainty is empirical, not a coverage guarantee.'}
    np.savez(ROOT/'models/calibration.npz',radius=radius)
    # Teacher trajectories from TRAIN days only. Inputs use causal previous-day forecasts,
    # never fitted solar predictions or future actual PV. Each day remains in one split.
    sequences=[]; targets=[]; dayids=[]
    for day in range(2,a//96):
        start=day*96; actual=d['pv'][start:start+96]*60
        if np.mean(d['valid'][start:start+96])<.9: continue
        ev,load_kw,tariff=scenario(100+day)
        rem=np.array([e['need'] for e in ev]); actions,_=optimize(actual,load_kw,tariff,ev,rem)
        hist=[]
        for t in range(96):
            f=d['pv'][start+t-96:start+96-96]*60
            hist.append(policy_features(t,ev,rem,actual[t],f[:16],load_kw[t],tariff[t],25))
            seq=hist[-8:]; seq=[seq[0]]*(8-len(seq))+seq
            sequences.extend(np.array(seq).transpose(1,0,2)); targets.extend((actions[t]/7.2)[:,None]); dayids.extend([day]*len(ev))
            rem=np.maximum(0,rem-actions[t]*.25)
    SX=np.array(sequences); SY=np.array(targets); days=np.array(dayids)
    if not len(SX): raise ValueError('No suitable training days for scheduler')
    cut=sorted(set(dayids))[max(1,int(len(set(dayids))*.8))-1]; mask=days<=cut
    if np.all(mask): raise ValueError('Need multiple teacher days')
    scheduler=LSTM(9,16,1,seed=43)
    sh=scheduler.fit(SX[mask],SY[mask],SX[~mask],SY[~mask],epochs=epochs,batch=128)
    scheduler.save(ROOT/'models/scheduler_lstm.npz')
    result['scheduler']={'teacher':'lexicographic perfect-information HiGHS LP','training_rows':int(mask.sum()),
                         'validation_rows':int((~mask).sum()),'teacher_days':sorted(set(dayids)),
                         'history':sh,'caveat':'Training inputs use previous-day forecasts. Deployment uses solar LSTM forecasts; distribution shift and teacher-forcing error are evaluated by closed-loop replay.'}
    (ROOT/'reports/training.json').write_text(json.dumps(result,indent=2))
    with (ROOT/'reports/forecast_test.csv').open('w') as f:
        f.write('issue_time,lead_minutes,actual_normalized,lstm_normalized,previous_day_normalized\n')
        for j,t in enumerate(ids):
            for h in range(16): f.write(f"{d['times'][t]},{(h+1)*15},{TY[j,h]},{pred[j,h]},{TB[j,h]}\n")
    print(json.dumps({k:result[k] for k in ['python_tested','samples','lstm','previous_day']},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--epochs',type=int,default=25);p.add_argument('--plant',type=int,choices=[1,2],default=1)
    args=p.parse_args();main(args.epochs,args.plant)
