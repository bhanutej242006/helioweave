import unittest
import numpy as np
from helioweave.lstm import LSTM
from helioweave.scheduler import optimize,project,scenario,simulate
from helioweave.data import load,samples,forecast


class CoreTests(unittest.TestCase):
    def test_lstm_gradient(self):
        net=LSTM(2,3,1); rng=np.random.default_rng(2); X=rng.normal(size=(2,3,2));Y=rng.normal(size=(2,1))
        _,g=net.gradients(X,Y)
        for key,index in [('W',(0,0)),('W',(3,8)),('b',(4,)),('V',(1,0)),('a',(0,))]:
            v=net.p[key][index]; eps=1e-5
            net.p[key][index]=v+eps; plus=np.mean((net.forward(X)-Y)**2)
            net.p[key][index]=v-eps;minus=np.mean((net.forward(X)-Y)**2);net.p[key][index]=v
            self.assertAlmostEqual((plus-minus)/(2*eps),g[key][index],places=6)

    def test_projection_invariants(self):
        rng=np.random.default_rng(8);ev,_,_=scenario()
        for t in range(96):
            rem=rng.uniform(0,20,len(ev)); pv=float(rng.uniform(0,50)); load=12;cap=25
            p,_,_=project(rng.normal(5,12,len(ev)),t,ev,rem,pv,load,cap)
            self.assertTrue(np.all(p>=0)); self.assertTrue(np.all(p<=7.2+1e-9));self.assertTrue(np.all(p*.25<=rem+1e-9))
            self.assertLessEqual(load+p.sum()-pv,cap+1e-9)
            for i,e in enumerate(ev):
                if not e['arrival']<=t<e['departure']:self.assertEqual(p[i],0)

    def test_optimizer_known_solution(self):
        ev=[{'arrival':0,'departure':2,'max_kw':4,'need':1}]
        p,status=optimize(np.array([0,0]),np.array([0,0]),np.array([10,2]),ev,[1],grid_cap=4)
        self.assertAlmostEqual(p[1,0],4,places=5);self.assertAlmostEqual(status['objective_inr'],2,places=5)

    def test_infeasible_request_reported(self):
        ev=[{'arrival':0,'departure':1,'max_kw':4,'need':20}]
        p,status=optimize([0],[0],[5],ev,[20],grid_cap=4)
        self.assertAlmostEqual(status['unmet_kwh'],19,places=5)

    def test_forecast_causal(self):
        d=load();net=LSTM(6,20,16);t=2200;before=forecast(net,d,t);d['features'][t:]=999;d['pv'][t:]=999
        np.testing.assert_allclose(before,forecast(net,d,t))

    def test_partition_targets(self):
        d=load();a,b,c=map(int,d['split'])
        for start,end in [(96,a),(a,b),(b,c),(c,len(d['pv']))]:
            X,Y,B,ids=samples(d,start,end)
            self.assertTrue(np.all(ids>=start));self.assertTrue(np.all(ids+16<=end))
            self.assertTrue(all(np.all(d['valid'][t:t+16]) for t in ids))

if __name__=='__main__':unittest.main()
