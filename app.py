"""Flask entry point; binds locally, serves bundled HTML/CSS/JS and model APIs."""
import argparse,io,json,sys
from functools import lru_cache
import numpy as np
import pandas as pd
import joblib
from flask import Flask,request,jsonify,render_template,Response,send_file
from core.schema import ROOT,FEATURES,TARGET,validate_features
from core.modeling import predict_all
from core.ev import requests as ev_requests,finite,profile,plan
sys.path.insert(0,str(ROOT/'lstm_research'))
from helioweave.engine import meta as lstm_meta,run as lstm_run,days as lstm_days

app=Flask(__name__);app.config['MAX_CONTENT_LENGTH']=3*1024*1024

@lru_cache(maxsize=1)
def resources():
    return (joblib.load(ROOT/'models/ensemble.joblib'),json.loads((ROOT/'reports/metrics.json').read_text()),json.loads((ROOT/'data/presets.json').read_text()))

@app.errorhandler(ValueError)
def invalid(e):return jsonify(error=str(e)),400
@app.errorhandler(413)
def too_large(e):return jsonify(error='CSV upload exceeds 3 MB limit'),413

def json_object():
    body=request.get_json(silent=True)
    if not isinstance(body,dict):raise ValueError('Send a JSON object')
    return body

@app.get('/')
def home():return render_template('index.html')

@app.get('/api/bootstrap')
def bootstrap():
    bundle,metrics,presets=resources()
    return jsonify(metrics=metrics,presets=presets,models=list(metrics['test']),lstm_days=lstm_days(),dataset_source='https://www.kaggle.com/datasets/stucom/solar-energy-power-generation-dataset')

@app.post('/api/predict')
def predict():
    body=json_object();values=validate_features(body.get('features'));bundle,metrics,_=resources()
    predictions={k:float(v[0]) for k,v in predict_all(bundle,np.array([values])).items()}
    warnings=[f"{f['name']} is outside the training range" for f,v in zip(metrics['features'],values) if not f['min']<=v<=f['max']]
    return jsonify(predictions=predictions,selected=bundle['selected'],warnings=warnings,mode='Weather-conditioned power estimate; not a timestamped future forecast')

@app.post('/api/batch')
def batch():
    file=request.files.get('file')
    if not file:raise ValueError('Select a CSV file')
    try:frame=pd.read_csv(file)
    except Exception:raise ValueError('Cannot read CSV; use UTF-8 comma-separated data')
    if not 1<=len(frame)<=10000:raise ValueError('Use 1–10,000 rows')
    missing=[f for f in FEATURES if f not in frame]
    if missing:raise ValueError('Missing columns: '+', '.join(missing))
    X=frame[FEATURES].apply(pd.to_numeric,errors='coerce').to_numpy()
    if not np.all(np.isfinite(X)) or np.max(abs(X))>1e8:raise ValueError('All 20 input columns must contain finite numeric values')
    # Return only numeric schema columns, avoiding formula injection from arbitrary uploaded fields.
    out=frame[FEATURES].copy()
    for name,pred in predict_all(resources()[0],X).items():out[name+'_prediction_kw']=pred
    return Response(out.to_csv(index=False),mimetype='text/csv',headers={'Content-Disposition':'attachment; filename=solar_predictions.csv'})

@app.post('/api/schedule')
def schedule():
    body=json_object();bundle,metrics,presets=resources();ev=ev_requests(body.get('vehicles'))
    capacity=finite(body.get('capacity',60),'PV scenario scale',5,500);grid=finite(body.get('grid',30),'grid limit',1,500);building=finite(body.get('building',8),'building load',0,100)
    cloud=finite(body.get('cloud',30),'cloud',0,100);model=body.get('model',bundle['selected'])
    if not isinstance(model,str) or model not in metrics['test']:raise ValueError('Unknown model')
    features=profile(presets[1]['features'],metrics['features'],cloud)
    raw=predict_all(bundle,features)[model];solar=np.clip(raw/bundle['scale']*capacity,0,capacity)
    solar[:7]=0;solar[18:]=0  # declared illustrative sunrise/sunset, not inferred site astronomy
    result=plan(solar,ev,grid,building);result['model']=model;result['scenario_scale_kw']=capacity
    return jsonify(result)

@app.get('/api/lstm')
def lstm():
    day=request.args.get('day',type=int,default=lstm_days()[0]['index'])
    return jsonify(lstm_run(day))

@app.get('/api/lstm/metrics')
def lstm_metrics():return jsonify(lstm_meta()['training'])

@app.get('/api/dataset')
def dataset():
    frame=pd.read_csv(ROOT/'data/spg.csv');offset=max(0,request.args.get('offset',default=0,type=int));q=frame.iloc[offset:offset+15]
    return jsonify(columns=list(frame.columns),rows=json.loads(q.to_json(orient='records')),total=len(frame),offset=offset)

@app.get('/api/download/<name>')
def download(name):
    known={'dataset':ROOT/'data/spg.csv','metrics':ROOT/'reports/metrics.json','test_predictions':ROOT/'reports/test_predictions.csv'}
    if name not in known:return jsonify(error='Unknown download'),404
    return send_file(known[name],as_attachment=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=5000);args=p.parse_args()
    try:resources()
    except FileNotFoundError:raise SystemExit('Run python train.py to create the models first.')
    app.run(host='127.0.0.1',port=args.port,debug=False)
