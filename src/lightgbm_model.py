import os

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_curve, auc, accuracy_score)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
#from bayes_opt import BayesianOptimization
from sklearn.metrics import mean_squared_error, mean_squared_log_error
from src.functions import loading, get_var, setting_X, setting_y  #importing loading functions
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

print("Yay! Now let's train!")

###########################################3

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

hyper_params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': ['l2', 'auc'],
    'learning_rate': 0.005,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.7,
    'bagging_freq': 10,
    'verbose': 0,
    "max_depth": 8,
    "num_leaves": 128,
    "max_bin": 512,
    "num_iterations": 100000,
    "n_estimators": 1000
}
gbm = lgb.LGBMRegressor(**hyper_params)
gbm.fit(X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric='l1',
        early_stopping_rounds=1000)

gbm.fit(X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric='l1',
        early_stopping_rounds=1000)

y_pred = gbm.predict(X_train, num_iteration=gbm.best_iteration_)
# Basic RMSE
print('The rmse of prediction is:', round(mean_squared_error(y_pred, y_train) ** 0.5, 5))
