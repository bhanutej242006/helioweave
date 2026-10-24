"""Charging-only lexicographic LP, recurrent policy, and measured-state projection."""
import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix, vstack, csr_matrix
from .data import DT


def scenario(seed=42,ports=10,stress=1.):
    rng=np.random.default_rng(seed)
    ev=[]
    for i in range(ports):
        arrival=int(rng.integers(26,43)); departure=int(rng.integers(60,85))
        ev.append({'id':f'EV-{i+1:02d}','arrival':arrival,'departure':departure,
                   'need':float(rng.uniform(9,23)*stress),'max_kw':7.2})
    hours=np.arange(96)/4
    load=7+5*((hours>=8)&(hours<19))+1.2*np.sin(hours*np.pi/12)**2
    tariff=np.where((hours>=17)&(hours<22),10.,np.where((hours>=6)&(hours<17),6.,4.))
    return ev,load,tariff


def optimize(pv,load,tariff,ev,remaining,now=0,grid_cap=25.):
    """Minimize unmet kWh first, then import INR. No export, V2G, or demand charges.
    Continuous charge powers make this an LP; MILP binaries are unnecessary here.
    """
    H=len(pv); N=len(ev); size=N*H+H+N; grid=N*H; slack=grid+H
    bounds=[]
    for e in ev:
        bounds += [(0,e['max_kw'] if e['arrival']<=now+t<e['departure'] else 0) for t in range(H)]
    bounds += [(0,grid_cap)]*H+[(0,float(max(0,r))) for r in remaining]
    eq=lil_matrix((N,size)); rhs=np.array(remaining,dtype=float)
    for i in range(N): eq[i,i*H:(i+1)*H]=DT; eq[i,slack+i]=1
    ub=lil_matrix((H,size))
    for t in range(H):
        for i in range(N): ub[t,i*H+t]=1
        ub[t,grid+t]=-1
    ub=ub.tocsr(); b=np.array(pv)-np.array(load); eq=eq.tocsr()
    service=np.zeros(size); service[slack:]=1
    first=linprog(service,A_ub=ub,b_ub=b,A_eq=eq,b_eq=rhs,bounds=bounds,method='highs')
    if not first.success: raise RuntimeError('LP infeasible or failed: '+first.message)
    cost=np.zeros(size); cost[grid:slack]=np.array(tariff)*DT
    lock=csr_matrix(service.reshape(1,-1))
    second=linprog(cost,A_ub=vstack([ub,lock]),b_ub=np.r_[b,first.fun+1e-7],A_eq=eq,b_eq=rhs,bounds=bounds,method='highs')
    if not second.success: raise RuntimeError('Cost LP failed: '+second.message)
    return second.x[:grid].reshape(N,H).T, {'status':second.message,'unmet_kwh':float(second.x[slack:].sum()),'objective_inr':float(second.fun),'solver':'HiGHS LP (optimal status)'}


def policy_features(t,ev,remaining,pv,forecast,load,tariff,grid_cap):
    return np.array([[t/96, max(0,e['departure']-t)/96, remaining[i]/40,
                      float(e['arrival']<=t<e['departure']),pv/60,
                      float(np.mean(forecast))/60,load/30,tariff/10,grid_cap/50] for i,e in enumerate(ev)])


def project(proposed,t,ev,remaining,pv,load,grid_cap):
    """Physical cap + earliest-departure urgency. Does not promise joint future feasibility."""
    cap=max(0,grid_cap+pv-load); p=np.zeros(len(ev)); upper=np.zeros(len(ev)); urgent=np.zeros(len(ev))
    order=sorted(range(len(ev)),key=lambda i:ev[i]['departure'])
    for i,e in enumerate(ev):
        if e['arrival']<=t<e['departure']:
            upper[i]=min(e['max_kw'],max(0,remaining[i])/DT)
            future=max(0,e['departure']-t-1)*DT*e['max_kw']
            urgent[i]=min(upper[i],max(0,remaining[i]-future)/DT)
    for i in order: p[i]=min(urgent[i],cap); cap-=p[i]
    for i in order:
        extra=min(max(0,float(proposed[i])-p[i]),upper[i]-p[i],cap); p[i]+=extra; cap-=extra
    return p, float(np.abs(p-proposed).sum()),max(0,load-pv-grid_cap)


def simulate(actual,forecasts,ev,load,tariff,method='learned',policy=None,grid_cap=25.,risk=0.,radius=0.):
    remaining=np.array([e['need'] for e in ev]); rows=[]; powers=[]; hist=[]; elapsed=[]; oracle_status=None
    if method=='oracle': fixed,oracle_status=optimize(actual,load,tariff,ev,remaining,grid_cap=grid_cap)
    for t in range(96):
        started=time.perf_counter(); f=np.maximum(0,np.array(forecasts[t])-risk*radius)
        if method=='oracle': proposed=fixed[t]
        elif method=='immediate': proposed=np.array([e['max_kw'] if e['arrival']<=t<e['departure'] else 0 for e in ev])
        elif method=='rolling':
            try: proposed=optimize(f,load[t:],tariff[t:],ev,remaining,t,grid_cap)[0][0]
            except RuntimeError: proposed=np.array([e['max_kw'] for e in ev])
        elif method=='learned':
            features=policy_features(t,ev,remaining,actual[t],f[:16],load[t],tariff[t],grid_cap)
            hist.append(features); seq=hist[-8:]; seq=[seq[0]]*(8-len(seq))+seq
            proposed=np.clip(policy.forward(np.array(seq).transpose(1,0,2))[:,0],0,1)*7.2
        else: raise ValueError('Unknown policy')
        p,correction,overload=project(proposed,t,ev,remaining,actual[t],load[t],grid_cap)
        remaining=np.maximum(0,remaining-p*DT); demand=load[t]+p.sum()
        grid=max(0,demand-actual[t]); used=min(actual[t],demand)
        rows.append({'slot':t,'hour':t/4,'solar_kw':float(actual[t]),'building_kw':float(load[t]),
                     'ev_kw':float(p.sum()),'proposed_ev_kw':float(np.sum(proposed)),'grid_kw':float(grid),'solar_used_kw':float(used),'tariff':float(tariff[t]),
                     'cost_inr':float(grid*DT*tariff[t]),'correction_kw':correction,'building_overload_kw':overload,
                     'reason':'deadline/capacity/connection/energy projection' if correction>1e-5 else 'proposal accepted',
                     **{f'{e["id"]}_proposed_kw':float(proposed[i]) for i,e in enumerate(ev)},
                     **{f'{e["id"]}_executed_kw':float(p[i]) for i,e in enumerate(ev)}})
        powers.append(p.tolist()); elapsed.append((time.perf_counter()-started)*1000)
    summary={'method':method,'cost_inr':sum(r['cost_inr'] for r in rows),'unmet_kwh':float(remaining.sum()),
             'served_sessions':int(np.sum(remaining<.05)),'peak_grid_kw':max(r['grid_kw'] for r in rows),
             'solar_utilization_pct':100*sum(r['solar_used_kw'] for r in rows)/max(sum(actual),1e-9),
             'interventions':sum(r['correction_kw']>1e-5 for r in rows),
             'remaining_grid_violations':sum(r['grid_kw']>grid_cap+1e-6 for r in rows),
             'median_decision_ms':float(np.median(elapsed)),'p95_decision_ms':float(np.quantile(elapsed,.95))}
    return {'summary':summary,'rows':rows,'power':powers,'remaining_kwh':remaining.tolist(),'solver':oracle_status}
