import os
from os import listdir
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import folium
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from src.functions import loading, get_var, setting_X, setting_y
from sklearn.ensemble import RandomForestRegressor
from src.functions import *


base_dir = '/home/slimbook/git-repos/eolo-project/data/.raw/GFS_data'
csv_path = ("/home/slimbook/git-repos/eolo-project/data/processed/power_data.csv")


df_data = get_vvel(base_dir)
dates = get_date(base_dir)
df_data.index = dates

meteo = get_X(df_data)
power = setting_y(csv_path)

train = pd.concat([power, meteo], axis=1, join="inner")
train.sort_index(ascending=True, inplace=True)

X = train[[x for x in train.columns if x != 'Production']]
y = pd.DataFrame(train["Production"])

##########################################

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

model = RandomForestRegressor(n_estimators=1000)
trained = model.fit(X_train.values, y_train.values[:,0])
prediction =model.predict(X_test)

#model_saved = model.save("rf_model.h5")

feten = model.feature_importances_

############################################
print(feten)

#plotting_feature_importance(feten, model)

#estimate_coord(feten) # Plotting where is max relationship between features
                      # As we are relating wind speed and power generated, we try to estimate where is the park


