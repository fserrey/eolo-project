
import os
from os import listdir
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import xgboost as xgb
import matplotlib as plt
import folium

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
#from power_forecast.pickle_save_load import to_pickle
import webbrowser


def get_date(base_dir):
    new_time = []
    for file in listdir(base_dir):
        file_path = f'{base_dir}/{file}'
        match=file.split("_")[1]
        date = pd.to_datetime(match, format = "%Y%m%d%H").strftime('%d/%m/%Y')
        time = (datetime.strptime(match, "%Y%m%d%H") + timedelta(hours=6)).strftime('%H:%M')
        new_time.append(date + " " + time)
    return new_time

def get_variables(base_dir, var_list, diccionario, nz=26):
  d3_var = ["HGTprs", "CLWMRprs", "RHprs","Velprs","UGRDprs","VGRDprs","TMPprs"]
  d2_var = ["HGTsfc", "MSLETmsl", "PWATclm", "RH2m", "Vel100m", "UGRD100m", "VGRD100m",
            "Vel80m", "UGRD80m", "VGRD80m", "Vel10m", "UGRD10m", "VGRD10m", "GUSTsfc",
            "TMPsfc", "TMP2m", "no4LFTXsfc", "CAPEsfc", "SPFH2m", "SPFH80m"]

  lst = []
  
  for file in listdir(base_dir):
    file_path = f'{base_dir}/{file}'
    e_file = []
    for key, value in diccionario.items():
    
      if key in set(var_list).intersection(d3_var): #d3_var:
        corte = value[0] + int(((value[1])/26)*nz)
        e_file.append(np.fromfile(file_path, dtype=np.float32)[value[0]:corte])

      elif key in set(var_list).intersection(d2_var):#d2_var:
        e_file.append(np.fromfile(file_path, dtype=np.float32)[value[0]:value[1]])
    lst.append(e_file)
  
  return lst


def setup_x(dataframe):
  """Flat variables values for model training"""
  dataframe.reset_index(level=0, inplace=True)
  row_list =[] 
  for index, rows in dataframe.iterrows(): 
      my_list = [rows.RHprs, rows.Velprs, rows.TMPprs, rows.Vel100m, rows.Vel80m,rows.TMPsfc, rows.SPFH80m]
      row_list.append(my_list) 

  a = [np.concatenate(row_list[i]) for i in range(len(row_list))]
  train_ = pd.DataFrame(a, index=dataframe["index"])
  return train_



def get_var(main_dic, list_var, nz=26):
    """This function provides the selected variables in a nested dictionary with the given array
    and level (consider that each level is around 50m heigth). Output is given as dictionary
    :rtype: object
    """
    dict_final = {}
    size_3d = 13*9*nz
    print("Now, we get the variables we want")
    for datetime_key in main_dic: # iteración sobre las keys de 1º nivel
        res = []
        for var in list_var:  # compruebo que la variable que saco está en mi lista
            if var in main_dic.get(datetime_key).get("var_3d").keys():
                 # compruebo que esa variable está en las de 2º nivel
                array_3d = main_dic[datetime_key]["var_3d"][var]["data"]
                # Asigno el array del value de 4º nivel a una variable
                arr_3d_nz = []
                for j in range(0,len(array_3d), size_3d):
                    res.extend(array_3d[j: j+size_3d])

        for var in list_var:
            if var in main_dic.get(datetime_key).get("var_2d").keys():
                array_2d = main_dic[datetime_key]["var_2d"][var]["data"]
                res.extend(array_2d)

        #for i in range(len(main_dic.keys())):
        dict_final.update({datetime_key:res})

    return dict_final

def get_X(dataframe):

    meteo = dataframe
    meteo.reset_index(level=0, inplace=True)
    meteo["date"] = pd.to_datetime(meteo['index'], format='%d/%m/%Y %H:%M')
    meteo = meteo.sort_values(by='date',ascending=True)
    meteo = meteo.set_index("date").sort_index().loc[:'31/12/2016 00:00']
    meteo = meteo[[x for x in meteo.columns if x != 'index']]

    return meteo



def setting_X(dictionary):
    meteo_df = pd.DataFrame(dictionary).T
    meteo_df.reset_index(level=0, inplace=True)
    meteo_df["date"]=pd.to_datetime(meteo_df['index'], format='%d/%m/%Y %H:%M')
    meteo_df=meteo_df.sort_values(by='date',ascending=True)
    meteo_df=meteo_df.set_index("date").sort_index().loc[:'31/12/2016 00:00']
    meteo_df=meteo_df[[x for x in meteo_df.columns if x != 'index']]

    return meteo_df


def setting_y(csv_file):
    power_df = pd.read_csv(csv_file)
    power_df['date'] = pd.to_datetime(power_df['date'], format='%d/%m/%Y %H:%M')
    power_df = power_df.sort_values(by='date',ascending=True)
    power_df=power_df.set_index("date").sort_index().loc[:'31/12/2016 00:00']

    return power_df



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


def get_vvel(base_dir):
    """This function gives you the values of all Velocity at 100m height as pandas data frame
    """
    content = []
    filenames = []
    filenames.append(get_date(base_dir))
    for file in os.listdir(base_dir):
        file_path = f'{base_dir}/{file}'
        filenames.append(file)
        content.append(np.fromfile(file_path, dtype=np.float32)[21762:21879])

    return pd.DataFrame(data=content)


def plotting_feature_importance(importance, model):
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
    return plt.show()


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

def get_location(feten):
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

def drawing_map(result_point, radio=False, distance=False):
    m = folium.Map(
        location=[(lat_start + lat_end) / 2, (lon_start + lon_end) / 2, ],
        zoom_start=7,
        tiles='Stamen Terrain'
    )

    tooltip = 'I am here!'
    if radio == True | distance == True:
        folium.CircleMarker(location = [45.58163, -120.15285], radius = 100, popup = ' FRI ').add_to(m)
        folium.PolyLine(locations = [(result_point), (45.18163, -120.15285)], line_opacity = 0.5).add_to(m)
        folium.Marker([45.18163, -120.15285], popup='<b>Condon WindFarm</b>', tooltip=tooltip).add_to(m)
    folium.Marker(result_point, popup='<i>Result</i>', tooltip=tooltip).add_to(m)

    return m

