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

base_dir = 'data/.raw/GFS_data'

df_data = get_vvel(base_dir)

csv_path = ("/home/slimbook/git-repos/eolo-project/data/processed/power_data.csv")

def get_X(dataframe):
    meteo = dataframe
    meteo.reset_index(level=0, inplace=True)
    meteo["date"] = pd.to_datetime(meteo['index'], format='%d/%m/%Y %H:%M')
    meteo = meteo.sort_values(by='date',ascending=True)
    meteo = meteo.set_index("date").sort_index().loc[:'31/12/2016 00:00']
    meteo = meteo[[x for x in meteo.columns if x != 'index']]

    return meteo


def setting_y(csv_path):
    power_df = pd.read_csv(csv_path)
    power_df['date'] = pd.to_datetime(power_df['date'], format='%d/%m/%Y %H:%M')
    power_df = power_df.sort_values(by='date',ascending=True)
    power_df=power_df.set_index("date").sort_index().loc[:'31/12/2016 00:00']

    return power_df

meteo = get_X(df_data)
power = setting_y(csv_path)

train = pd.concat([power, meteo], axis=1, join="inner")
train.sort_index(ascending=True, inplace=True)

X = train[[x for x in train.columns if x != 'Production']]
y = pd.DataFrame(train["Production"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

model = RandomForestRegressor(n_estimators=1000)
trained = model.fit(X_train.values, y_train.values[:,0])
prediction =model.predict(X_test)

feten = model.feature_importances_


def plotting_feature_importance(importance):
    """Plot the feature importances of the forest"""
    std = np.std([modelo.feature_importances_ for modelo in model.estimators_],
                 axis=0)
    index = np.argsort(feten)
    plt.figure(figsize=(15, 15))
    plt.title("Feature importances")
    plt.barh(range(X_train.values.shape[1]), feten[index],
           color="r", xerr=std[index], align="center")

    plt.yticks(range(X_train.values.shape[1]), index)
    plt.ylim([-1, X_train.values.shape[1]])
    plt.show()


plotting_feature_importance(feten)

def estimate_coord_plot(feten):
    lon_res = 13
    lat_res = 9
    nz = 26

    lat_step = 0.5
    lon_step = 0.5

    lat_start = 44
    lat_end = lat_start + lat_step  * (lat_res - 1) # calculas lat final
    lon_start = -123
    lon_end = lon_start + lon_step * (lon_res -1)  # calculas lon final - con esto puedes construir mesh

    lat = np.linspace(start=lat_start, stop=lat_end, endpoint=lat_end, num=lat_res)
    lon = np.linspace(start=lon_start, stop=lon_end, endpoint=lon_end, num=lon_res) #
    lon, lat = np.meshgrid(lon, lat)
    Z = feten.reshape(lat_res, lon_res)
    ptos = np.hstack((lat.reshape((lat.size,1)), lon.reshape((lon.size,1))))
    fig = plt.figure(figsize=(12, 10))
    im = plt.pcolormesh(lat, lon, Z) # Asignas valores a su posición en el mapa
    return plt.colorbar(mappable=im)

estimate_coord(feten) # Plotting where is max relationship between features
                      # As we are relating wind speed and power generated, we try to estimate where is the park


def get_location():
    lon_res = 13
    lat_res = 9
    nz = 26

    lat_step = 0.5
    lon_step = 0.5

    lat_start = 44
    lat_end = lat_start + lat_step  * (lat_res - 1) # calculas lat final
    lon_start = -123
    lon_end = lon_start + lon_step * (lon_res -1)  # calculas lon final - con esto puedes construir mesh

    lat = np.linspace(start=lat_start, stop=lat_end, endpoint=lat_end, num=lat_res)
    lon = np.linspace(start=lon_start, stop=lon_end, endpoint=lon_end, num=lon_res)
    lon, lat = np.meshgrid(lon, lat)
    Z = feten.reshape(lat_res, lon_res)
    point = Z.argmax()

    ptos = np.hstack((lat.reshape((lat.size,1)), lon.reshape((lon.size,1))))
    max_z_position = Z.argmax()
    coordinates = list(ptos[point])
    return coordinates
