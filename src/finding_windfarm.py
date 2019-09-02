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
import pickle
from src.pickle_save_load import to_pickle
import webbrowser

# Data loading and pre-processing
print("Reading the data...")
base_dir = '/home/slimbook/git-repos/eolo-project/data/.raw/GFS_data'
csv_path = ("/home/slimbook/git-repos/eolo-project/data/processed/power_data.csv")

df_data = get_vvel(base_dir)
dates = get_date(base_dir)
df_data.index = dates

print("We are almost ready to train!")

# Making X and y for machine learning training funnel

meteo = get_X(df_data) # Data obtained from previous pre-processing
power = setting_y(csv_path) # Data from wind farm power station

train = pd.concat([power, meteo], axis=1, join="inner")
train.sort_index(ascending=True, inplace=True) # We must sort index in order to later filter by date

print("Let's find that wind farm")
X = train[[x for x in train.columns if x != 'Production']]
y = pd.DataFrame(train["Production"])

print("Split and train...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

model_rf = RandomForestRegressor(n_estimators=1000)
trained = model_rf.fit(X_train.values, y_train.values[:,0])
print("Prediction and feature importance")

prediction = model_rf.predict(X_test)
feten = model_rf.feature_importances_

# Now that we have the data for the section where this feature is stronger with the power, we apply what we know
# about the location.
# The meteorological data is obtained from a section of Washington state (USA), therefore, we stablish the area in
# where we expect to find the power plant.

print("Let's see where it is...")
lon_res = 13
lat_res = 9
nz = 26

lat_step = 0.5
lon_step = 0.5
lat_start = 44    # The aprox. latitude from where the meteorological data were taken
lon_start = -123  # The aprox. longitude from where the meteorological data were taken

lat_end = lat_start + lat_step * (lat_res - 1)
lon_end = lon_start + lon_step * (lon_res -1)

lat = np.linspace(start=lat_start, stop=lat_end, endpoint=lat_end, num=lat_res)
lon = np.linspace(start=lon_start, stop=lon_end, endpoint=lon_end, num=lon_res)
lon, lat = np.meshgrid(lon, lat)
Z = feten.reshape(lat_res, lon_res)
point = Z.argmax() # Where the importance is higher

ptos = np.hstack((lat.reshape((lat.size,1)), lon.reshape((lon.size,1))))

coordinates = list(ptos[point])

# Map creation with folium

m = folium.Map(
        location=[(lat_start + lat_end) / 2, (lon_start + lon_end) / 2, ],
        zoom_start=10,
        tiles='Stamen Terrain'
    )

tooltip = 'I am here!'
#folium.CircleMarker(location = [45.58163, -120.15285], radius = 100, popup = ' FRI ').add_to(m)
#    folium.PolyLine(locations = [(result_point), (45.18163, -120.15285)], line_opacity = 0.5).add_to(m)
folium.Marker([45.18163, -120.15285], popup='<b>Condon WindFarm</b>', tooltip='Condon WindFarm').add_to(m)
folium.Marker(coordinates, popup='<i>Result</i>', tooltip=tooltip).add_to(m)


filepath = '/home/slimbook/git-repos/eolo-project/data/map.html'

m.save(filepath)

webbrowser.open('file://' + filepath)
