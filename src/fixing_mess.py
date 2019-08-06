import os

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from src.functions import loading, get_var  #importing loading functions
import json


# Importing data from path

file_path = "/home/slimbook/git-repos/eolo-project/data/.raw/GFS_data"

data_dictionary = loading(file_path) # All files and variables loaded as dictionary

list_var = ["Vel100m"]

var_to_test = get_var(data_dictionary, list_var, nz=5) # Selected variables (all, this time) and Nz from main dictionary

meteo_df = pd.DataFrame(var_to_test).T
power_df = pd.read_csv("/home/slimbook/git-repos/eolo-project/data/processed/power_data.csv")
print(meteo_df)

meteo_df = meteo_df.copy().iloc[1:]
meteo_df.reset_index(level=0, inplace=True)
meteo_df["date"]=pd.to_datetime(meteo_df['index'], format='%d/%m/%Y %H:%M')
meteo_df=meteo_df[[x for x in meteo_df.columns if x != 'index']]


power_df=power_df.sort_values(by='date',ascending=True)
meteo_df=meteo_df.sort_values(by='date',ascending=True)

power_df['date']=pd.to_datetime(power_df['date'], format='%Y%m%d %H:%M:%S')

meteo_df=meteo_df.set_index("date").sort_index().loc[:'31/12/2016 00:00']
power_df=power_df.set_index("date").sort_index().loc[:'31/12/2016 00:00']

train = pd.concat([power_df, meteo_df], axis=1, join="inner")
train.sort_index(ascending=True, inplace=True)

X = train[[x for x in X.columns if x != 'Production']]
y = pd.DataFrame(train["Production"])

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

y_pred = gbm.predict(X_train, num_iteration=gbm.best_iteration_)
# Basic RMSE
print('The rmse of prediction is:', round(mean_squared_error(y_pred, y_train) ** 0.5, 5))

