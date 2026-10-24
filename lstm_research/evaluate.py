"""Paired test-day experiments; controlled solar-error study is explicitly oracle-derived."""
import argparse,csv,json
import numpy as np
from helioweave.data import ROOT
from helioweave.engine import days,run,resources
from helioweave.scheduler import scenario,simulate


def main(sensitivity=False):
    rows=[]
    for day in days():
        result=run(day['index'])
        (ROOT/f"reports/replay_{day['date']}.json").write_text(json.dumps(result,indent=2))
        for r in result['results'].values():rows.append({'date':day['date'],**r['summary']})
        print(day['date'],[(r['summary']['method'],round(r['summary']['unmet_kwh'],2)) for r in result['results'].values()],flush=True)
    with (ROOT/'reports/controller_comparison.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    if sensitivity:
        d,_,policy,radius=resources();out=[]
        for day in days():
            actual=d['pv'][day['index']:day['index']+96]*60;ev,load,tariff=scenario()
            for seed in range(5):
                noise=np.random.default_rng(seed).normal(size=96)
                for level in [0,.05,.1,.2]:
                    raw=np.maximum(0,actual+noise*60*level);raw[actual<.01]=0
                    fs=[raw[t:].copy() for t in range(96)]
                    for t in range(96):fs[t][0]=actual[t]
                    r=simulate(actual,fs,ev,load,tariff,'learned',policy)
                    out.append({'date':day['date'],'seed':seed,'target_nrmse':level,
                                'achieved_nrmse':float(np.sqrt(np.mean((raw-actual)**2))/60),**r['summary']})
        with (ROOT/'reports/controlled_error_study.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=out[0].keys());w.writeheader();w.writerows(out)
    print('Saved measured results under reports/')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--sensitivity',action='store_true');main(p.parse_args().sensitivity)
