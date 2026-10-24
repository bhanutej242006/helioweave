"""Five requested regressors; blocked out-of-fold stacking without target leakage."""
import argparse,hashlib,json,platform,importlib.metadata
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from core.schema import ROOT,FEATURES,TARGET,LABELS,UNITS
from core.modeling import learners,predict_all


def score(y,p):return {'mae_kw':float(mean_absolute_error(y,p)),'rmse_kw':float(np.sqrt(mean_squared_error(y,p))),'r2':float(r2_score(y,p))}


def main(path):
    frame=pd.read_csv(path)
    missing=[c for c in FEATURES+[TARGET] if c not in frame]
    if missing:raise ValueError('Dataset missing required columns: '+', '.join(missing))
    frame=frame[FEATURES+[TARGET]].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan)
    before=len(frame);frame=frame.dropna(subset=[TARGET]);frame=frame[frame[TARGET]>=0].drop_duplicates().reset_index(drop=True)
    if len(frame)<300:raise ValueError('Need at least 300 valid rows')
    n=len(frame);a=int(n*.6);b=int(n*.8);X=frame[FEATURES].to_numpy();y=frame[TARGET].to_numpy()
    if np.isnan(X[:a]).all(axis=0).any():raise ValueError('A feature is completely missing in training block')
    scale=max(float(y[:a].max()),1);names=list(learners());oof=np.full((a,len(names)),np.nan);folds=[]
    for fold,(tr,va) in enumerate(TimeSeriesSplit(n_splits=4).split(X[:a])):
        fold_models=learners()
        for j,name in enumerate(names):
            fold_models[name].fit(X[tr],y[tr]);oof[va,j]=np.maximum(0,fold_models[name].predict(X[va]))
        folds.append({'train_first':int(tr[0]),'train_last':int(tr[-1]),'held_out_first':int(va[0]),'held_out_last':int(va[-1])})
        print('OOF fold',fold+1,'complete',flush=True)
    valid=np.all(np.isfinite(oof),axis=1);meta=Ridge(alpha=1.).fit(oof[valid]/scale,y[:a][valid]/scale)
    models=learners()
    for name,m in models.items():m.fit(X[:a],y[:a]);print('Trained',name,flush=True)
    bundle={'models':models,'meta':meta,'base_names':names,'scale':scale,'features':FEATURES}
    val=predict_all(bundle,X[a:b]);test=predict_all(bundle,X[b:]);validation={k:score(y[a:b],v) for k,v in val.items()}
    selected=min(validation,key=lambda k:validation[k]['rmse_kw']);bundle['selected']=selected
    report={'dataset':'stucom/solar-energy-power-generation-dataset','python':platform.python_version(),
            'versions':{k:importlib.metadata.version(k) for k in ['numpy','scipy','scikit-learn','lightgbm','xgboost','catboost','flask','pandas','joblib']},
            'sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest(),'raw_rows':before,'clean_rows':n,'removed_rows':before-n,
            'split':{'train':[0,a],'validation':[a,b],'test':[b,n]},'split_note':'Contiguous source-row blocks, 60/20/20. Source CSV has no timestamp; chronological forecasting validation cannot be claimed.',
            'stacking_folds':folds,'stacking_oof_rows':int(valid.sum()),'selected_on_validation':selected,'validation':validation,
            'test':{k:score(y[b:],v) for k,v in test.items()},'scale_kw':scale,
            'target_stats':{'min':float(y.min()),'max':float(y.max()),'mean':float(y.mean())},
            'features':[{'name':f,'label':LABELS[j],'unit':UNITS[j],'min':float(np.nanmin(X[:a,j])),'max':float(np.nanmax(X[:a,j])),'median':float(np.nanmedian(X[:a,j])),'missing':int(np.isnan(X[:,j]).sum())} for j,f in enumerate(FEATURES)]}
    importance=permutation_importance(models['LightGBM'],X[a:b],y[a:b],n_repeats=3,random_state=42,scoring='neg_mean_absolute_error')
    report['importance']=[{'feature':FEATURES[i],'increase_mae_kw':float(importance.importances_mean[i])} for i in np.argsort(importance.importances_mean)[::-1]]
    joblib.dump(bundle,ROOT/'models/ensemble.joblib',compress=3)
    (ROOT/'reports/metrics.json').write_text(json.dumps(report,indent=2))
    testframe=frame.iloc[b:].copy();testframe.insert(0,'source_row',np.arange(b,n))
    for name,p in test.items():testframe[name+'_prediction_kw']=p
    testframe.to_csv(ROOT/'reports/test_predictions.csv',index=False)
    presets=[];order=np.argsort(y[b:])
    for label,q in [('Low generation',.15),('Medium generation',.5),('High generation',.85)]:
        i=b+int(order[int((len(order)-1)*q)])
        vals=np.where(np.isnan(X[i]),np.nanmedian(X[:a],axis=0),X[i])
        presets.append({'name':label,'source_row':i,'features':dict(zip(FEATURES,map(float,vals))),'actual_kw':float(y[i])})
    (ROOT/'data/presets.json').write_text(json.dumps(presets,indent=2))
    print('Validation-selected:',selected,'| Test:',report['test'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--csv',default=str(ROOT/'data/spg.csv'));main(p.parse_args().csv)
