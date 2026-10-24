import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor


def learners():
    return {
      'LightGBM':make_pipeline(SimpleImputer(strategy='median'),LGBMRegressor(n_estimators=240,max_depth=6,num_leaves=31,learning_rate=.045,verbosity=-1,random_state=42,n_jobs=2)),
      'XGBoost':make_pipeline(SimpleImputer(strategy='median'),XGBRegressor(n_estimators=240,max_depth=5,learning_rate=.045,subsample=.9,colsample_bytree=.9,objective='reg:squarederror',random_state=42,n_jobs=2,tree_method='hist')),
      'CatBoost':make_pipeline(SimpleImputer(strategy='median'),CatBoostRegressor(iterations=240,depth=6,learning_rate=.045,loss_function='RMSE',verbose=False,random_seed=42,thread_count=2,allow_writing_files=False)),
      'KNN':make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),KNeighborsRegressor(n_neighbors=9,weights='distance',n_jobs=2))}


def predict_all(bundle,X):
    preds={name:np.maximum(0,m.predict(X)) for name,m in bundle['models'].items()}
    stack=np.column_stack([preds[n] for n in bundle['base_names']])/bundle['scale']
    preds['Stacking Model']=np.maximum(0,bundle['meta'].predict(stack)*bundle['scale'])
    return preds
