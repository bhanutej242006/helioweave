import io,json,unittest
import numpy as np
import pandas as pd
from app import app,resources
from core.schema import FEATURES,ROOT
from core.ev import plan,requests

class ProjectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client=app.test_client();cls.bundle,cls.metrics,cls.presets=resources()
    def test_all_requested_models_trained(self):
        self.assertEqual(set(self.metrics['test']),{'LightGBM','XGBoost','CatBoost','KNN','Stacking Model'})
        self.assertEqual(len(self.bundle['models']),4);self.assertEqual(self.bundle['meta'].n_features_in_,4)
    def test_exact_schema(self):
        self.assertEqual(list(pd.read_csv(ROOT/'data/spg.csv').columns),FEATURES+['generated_power_kw'])
    def test_prediction_finite(self):
        r=self.client.post('/api/predict',json={'features':self.presets[1]['features']});self.assertEqual(r.status_code,200)
        self.assertTrue(all(np.isfinite(v) and v>=0 for v in r.json['predictions'].values()))
    def test_target_not_used_as_feature(self):
        f=dict(self.presets[1]['features']);r1=self.client.post('/api/predict',json={'features':f}).json
        f['generated_power_kw']=999999999;r2=self.client.post('/api/predict',json={'features':f}).json
        self.assertEqual(r1['predictions'],r2['predictions'])
    def test_bad_payloads(self):
        self.assertEqual(self.client.post('/api/predict',json=[]).status_code,400)
        self.assertEqual(self.client.post('/api/predict',json={'features':{}}).status_code,400)
        f=dict(self.presets[1]['features']);f[FEATURES[0]]='nan'
        self.assertEqual(self.client.post('/api/predict',json={'features':f}).status_code,400)
    def test_stacking_split_isolation(self):
        for f in self.metrics['stacking_folds']:
            self.assertLess(f['train_last'],f['held_out_first']);self.assertLess(f['held_out_last'],self.metrics['split']['train'][1])
        self.assertLessEqual(self.metrics['split']['train'][1],self.metrics['split']['validation'][0]);self.assertLessEqual(self.metrics['split']['validation'][1],self.metrics['split']['test'][0])
    def test_batch_predictions(self):
        csv=pd.DataFrame([self.presets[1]['features']]).to_csv(index=False)
        r=self.client.post('/api/batch',data={'file':(io.BytesIO(csv.encode()),'weather.csv')});self.assertEqual(r.status_code,200)
        self.assertEqual(len(pd.read_csv(io.StringIO(r.data.decode())).columns),25)
    def test_bad_batch(self):
        r=self.client.post('/api/batch',data={'file':(io.BytesIO(b'x,y\n1,2\n'),'wrong.csv')});self.assertEqual(r.status_code,400)
    def test_energy_and_power_constraints(self):
        ev=requests([{'battery_kwh':50,'initial_soc':20,'target_soc':80,'arrival':8,'departure':17,'charger_kw':7.2},{'battery_kwh':70,'initial_soc':30,'target_soc':90,'arrival':10,'departure':20,'charger_kw':11}])
        pv=np.maximum(0,np.sin(np.arange(24)*np.pi/12-np.pi/2))*30;r=plan(pv,ev,20,8);power=np.array(r['power'])
        for i,e in enumerate(ev):
            self.assertTrue(np.all(power[:e['arrival'],i]<1e-8));self.assertTrue(np.all(power[e['departure']:,i]<1e-8))
            self.assertLessEqual(float(power[:,i].max()),e['max_kw']+1e-7);self.assertLessEqual(float(power[:,i].sum()*.92),e['need']+1e-7)
        self.assertTrue(all(row['grid_kw']<=20+1e-7 for row in r['rows']));self.assertAlmostEqual(r['optimized']['delivered_kwh'],float(power.sum()*.92))
    def test_impossible_request_has_shortfall(self):
        ev=requests([{'battery_kwh':100,'initial_soc':0,'target_soc':100,'arrival':8,'departure':9,'charger_kw':7.2}]);r=plan(np.zeros(24),ev,20,8)
        self.assertAlmostEqual(r['optimized']['unmet_kwh'],100-7.2*.92,places=5)
    def test_building_infeasibility_rejected(self):
        ev=requests([{'battery_kwh':50,'initial_soc':20,'target_soc':80,'arrival':8,'departure':17,'charger_kw':7.2}])
        with self.assertRaises(ValueError):plan(np.zeros(24),ev,3,8)
    def test_lstm_retained(self):
        r=self.client.get('/api/lstm/metrics');self.assertEqual(r.status_code,200);self.assertIn('scheduler',r.json)
    def test_assets_local(self):
        for p in ['/','/static/app.js','/static/scene.js','/static/vendor/three.module.js','/api/download/dataset']:
            r=self.client.get(p);self.assertEqual(r.status_code,200);r.close()

if __name__=='__main__':unittest.main()
