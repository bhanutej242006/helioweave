"""Hourly scenario planner with battery-side efficiency and service-first optimization."""
import csv,io,math
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix,vstack,csr_matrix
from .schema import FEATURES


def finite(value,label,lo,hi):
    try:v=float(value)
    except (ValueError,TypeError):raise ValueError(label+' must be numeric')
    if not math.isfinite(v) or not lo<=v<=hi:raise ValueError(f'{label} must be between {lo} and {hi}')
    return v


def requests(raw):
    if not isinstance(raw,list) or not 1<=len(raw)<=12:raise ValueError('Provide between 1 and 12 EVs')
    out=[]
    for i,e in enumerate(raw):
        if not isinstance(e,dict):raise ValueError('Each EV must be an object')
        cap=finite(e.get('battery_kwh'),'battery_kwh',5,200);initial=finite(e.get('initial_soc'),'initial_soc',0,100);target=finite(e.get('target_soc'),'target_soc',0,100)
        if target<initial:raise ValueError('Target SOC must be at least initial SOC')
        a=finite(e.get('arrival'),'arrival',0,23);d=finite(e.get('departure'),'departure',1,24)
        if int(a)!=a or int(d)!=d or d<=a:raise ValueError('Use whole-hour arrival/departure; departure must be later, within the same day')
        out.append({'id':f'EV-{i+1:02d}','battery_kwh':cap,'initial_soc':initial,'target_soc':target,'arrival':int(a),'departure':int(d),'max_kw':finite(e.get('charger_kw'),'charger_kw',1,50),'need':cap*(target-initial)/100})
    return out


def profile(preset,stats,cloud=30.):
    """Explicit illustrative weather profile, not observed or issued future weather."""
    out=[];peak=max(s['max'] for s in stats if s['name']=='shortwave_radiation_backwards_sfc')
    for h in range(24):
        x=dict(preset);daylight=max(0,math.sin(math.pi*(h-6)/12))
        x['shortwave_radiation_backwards_sfc']=peak*.85*daylight*(1-.006*cloud)
        x['zenith']=min(170,90-65*daylight) if 6<h<18 else 105
        x['angle_of_incidence']=min(90,x['zenith']*.8)
        x['azimuth']=(90+15*(h-6))%360
        x['total_cloud_cover_sfc']=cloud
        for key in ['high_cloud_cover_high_cld_lay','medium_cloud_cover_mid_cld_lay','low_cloud_cover_low_cld_lay']:x[key]=cloud/2
        x['temperature_2_m_above_gnd']=preset['temperature_2_m_above_gnd']+4*daylight
        out.append([x[k] for k in FEATURES])
    return np.array(out)


def plan(solar,ev,grid_limit=30,building=8,efficiency=.92):
    solar=np.array(solar,dtype=float);H=24;N=len(ev);grid=N*H;slack=grid+H;size=slack+N
    tariff=np.array([4 if h<6 or h>=22 else 10 if h>=17 else 6 for h in range(H)],dtype=float)
    if np.any(solar+grid_limit<building-1e-9):raise ValueError('Grid limit cannot serve the building during low-solar hours; increase grid limit or reduce building load')
    bounds=[]
    for e in ev:bounds.extend((0,e['max_kw'] if e['arrival']<=t<e['departure'] else 0) for t in range(H))
    bounds.extend([(0,grid_limit)]*H);bounds.extend((0,e['need']) for e in ev)
    eq=lil_matrix((N,size));b_eq=np.array([e['need'] for e in ev])
    for i in range(N):eq[i,i*H:(i+1)*H]=efficiency;eq[i,slack+i]=1
    ub=lil_matrix((H,size))
    for t in range(H):
        for i in range(N):ub[t,i*H+t]=1
        ub[t,grid+t]=-1
    service=np.zeros(size);service[slack:]=1
    args=dict(A_eq=eq.tocsr(),b_eq=b_eq,bounds=bounds,method='highs')
    first=linprog(service,A_ub=ub.tocsr(),b_ub=solar-building,**args)
    if not first.success:raise ValueError('Planner failed: '+first.message)
    objective=np.zeros(size);objective[grid:slack]=tariff
    second=linprog(objective,A_ub=vstack([ub.tocsr(),csr_matrix(service.reshape(1,-1))]),b_ub=np.r_[solar-building,first.fun+1e-7],**args)
    if not second.success:raise ValueError('Cost planner failed: '+second.message)
    power=second.x[:grid].reshape(N,H).T
    baseline=np.zeros_like(power);remaining=b_eq.copy()
    for t in range(H):
        available=max(0,grid_limit+solar[t]-building)
        for i,e in enumerate(ev):
            if e['arrival']<=t<e['departure']:
                p=min(e['max_kw'],remaining[i]/efficiency,available);baseline[t,i]=p;remaining[i]-=p*efficiency;available-=p
    def summarize(p):
        g=np.maximum(0,building+p.sum(axis=1)-solar);needs=np.maximum(0,b_eq-p.sum(axis=0)*efficiency)
        return {'estimated_cost_inr':float(g@tariff),'peak_grid_kw':float(g.max()),'unmet_kwh':float(needs.sum()),'served_evs':int(np.sum(needs<.01)),'solar_used_kwh':float(np.minimum(solar,building+p.sum(axis=1)).sum()),'delivered_kwh':float(p.sum()*efficiency),'grid_kwh':float(g.sum()),'remaining':needs.tolist()}
    optimized=summarize(power);immediate=summarize(baseline)
    rows=[]
    for t in range(H):
        rows.append({'hour':t,'solar_kw':float(solar[t]),'building_kw':building,'ev_kw':float(power[t].sum()),'grid_kw':float(max(0,building+power[t].sum()-solar[t])),'tariff_inr_kwh':float(tariff[t]),**{e['id']+'_kw':float(power[t,i]) for i,e in enumerate(ev)}})
    buf=io.StringIO();w=csv.DictWriter(buf,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    for i,e in enumerate(ev):e['delivered_kwh']=float(power[:,i].sum()*efficiency);e['final_soc']=e['initial_soc']+e['delivered_kwh']/e['battery_kwh']*100
    return {'optimized':optimized,'immediate':immediate,'rows':rows,'vehicles':ev,'power':power.tolist(),'csv':buf.getvalue(),'solver_status':second.message,'mode':'Illustrative hourly weather + user EV requests; estimated planned outcomes, not live operation','efficiency':efficiency}
