import os

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
from matplotlib import pyplot

from xgboost import plot_importance
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from hyperopt import fmin, hp, tpe, Trials, space_eval, STATUS_OK
from hyperopt.pyll import scope as ho_scope
from hyperopt.pyll.stochastic import sample as ho_sample
from functools import partial
from sklearn.model_selection import train_test_split, GridSearchCV, ShuffleSplit, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_log_error, mean_squared_error
from src.functions import loading, get_var, setting_X, setting_y, objetivo  #importing loading functions
import json

file_path = "/home/slimbook/git-repos/eolo-project/data/.raw/GFS_data"
csv_path = ("/home/slimbook/git-repos/eolo-project/data/processed/power_data.csv")
list_var = ["Vel100m"]

print("loading raw data and processing it...")

data_dictionary = loading(file_path) # All files and variables loaded as dictionary
var_to_test = get_var(data_dictionary, list_var, nz=5)

print("Setting X and y data...")

meteo = setting_X(var_to_test)
power = setting_y(csv_path)

train = pd.concat([power, meteo], axis=1, join="inner")
train.sort_index(ascending=True, inplace=True)

X = train[[x for x in train.columns if x != 'Production']]
y = pd.DataFrame(train["Production"])

##############################################
scaler= StandardScaler()
X = scaler.fit_transform(X)
y = scaler.fit_transform(y)
print("Yay! Now let's train!")

###########################################

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


space ={'n_estimators': hp.quniform('n_estimators', 10, 1000, 25),
        'learning_rate': hp.uniform('learning_rate', 0.0001, 1.0),
        'max_depth': hp.quniform('x_max_depth', 4, 16, 1),
        'min_child_weight': hp.quniform ('x_min_child', 1, 10, 1),
        'subsample': hp.uniform ('x_subsample', 0.7, 1),
        'gamma' : hp.uniform ('x_gamma', 0.1,0.5),
        'reg_lambda' : hp.uniform ('x_reg_lambda', 0,1)
    }
def objetivo(space):
    clf = xgb.XGBRegressor(n_estimators =int(space['n_estimators']),
                           learning_rate = space['learning_rate'],
                           max_depth = int(space['max_depth']),
                           min_child_weight = space['min_child_weight'],
                           subsample = space['subsample'],
                           gamma = space['gamma'],
                           reg_lambda = space['reg_lambda'],
                           objective='reg:squarederror')

    eval_set=[(X_train, y_train), (X_test, y_test)]

    clf.fit(X_train, y_train,
            eval_set=eval_set, eval_metric="rmse", verbose=False)

    y_pred = clf.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred)**(0.5)

    return {'loss':rmse, 'status': STATUS_OK }

trials_reg = Trials()
best = fmin(fn=objetivo,
            space=space,
            algo=tpe.suggest,
            max_evals=1000,
            trials=trials_reg)
print (best)

modelo=xgb.XGBRegressor(n_estimators=int(best['n_estimators']),
                        x_gamma=best['x_gamma'],
                        learning_rate=best['learning_rate'],
                        x_max_depth= best['x_max_depth'],
                        x_min_child= best['x_min_child'],
                        x_reg_lambda=best['x_reg_lambda'],
                        x_subsample= best['x_subsample'],
                        objective='reg:squarederror')
bst = modelo.fit(X_train, y_train)
y_pred = modelo.predict(X_test)
 #ntree_limit=xgb1.best_iteration
print ('RMSE: {}'.format(mean_squared_error(y_test, y_pred)**(0.5)))


x = bst.get_booster().get_score(importance_type='gain')
w = bst.get_weigth

def xgb_feature_importance(model_xgb, fnames=None):
    fs = model_xgb.get_fscore()
    all_features = [fs.get(f, 0.) for f in b.feature_names]

    all_features = np.array(all_features, dtype=np.float32)

    all_features_imp = all_features / all_features.sum()
    if fnames is not None:
        return pd.DataFrame({'X':fnames, 'IMP': all_features_imp})
    else:
        return all_features_imp

z = xgb_feature_importance(bst)#, train.columns)
print(z)

print(bst.best_score)
print(bst.best_iteration)
print(bst.best_ntree_limit)

from xgboost import plot_importance
plot_importance(bst, max_num_features=10) # top 10 most important features
plt.show()
