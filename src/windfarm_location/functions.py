
import os
from os import listdir
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import xgboost as xgb
import matplotlib as plt
import folium

def loading(file_path):
    """
    Given GFS data in several .gra files, this function iterate over a given folder path importing and
    organising its content in a dictionary format.
    """
    total={}
    file_data = []
    new_time = []
    print("Loading files and classifing")
    for root, subdirs, files in os.walk(file_path):
        for file in files:
            array = np.fromfile(file_path +"/"+ file, dtype=np.float32)

    for root, subdirs, files in os.walk(file_path):
        for file in files:
            match=file.split("_")[1]
            date = pd.to_datetime(match, format = "%Y%m%d%H").strftime('%d/%m/%Y')
            time = (datetime.strptime(match, "%Y%m%d%H") + timedelta(hours=6)).strftime('%H:%M')
            new_time.append(date + " " + time)


    for file in os.listdir(file_path):
        start = 0
        step_3d = 13*9*26
        end = step_3d
        features_3d = {
                  "HGTprs": {'dimesiones': [13, 9, 26], 'data': None},
                  "CLWMRprs": {'dimesiones': [13, 9, 26], 'data': None},
                  "RHprs": {'dimesiones': [13, 9, 26], 'data': None},
                  "Velprs": {'dimesiones': [13, 9, 26], 'data': None},
                  "UGRDprs": {'dimesiones': [13, 9, 26], 'data': None},
                  "VGRDprs": {'dimesiones': [13, 9, 26], 'data': None},
                  "TMPprs": {'dimesiones': [13, 9, 26], 'data': None}
               }

        end = end - step_3d
        step_2d = 13*9
        end = end +step_2d
        features_2d = {
                   "HGTsfc": {'dimesiones': [13, 9, 1], 'data': None},
                   "MSLETmsl": {'dimesiones': [13, 9, 1], 'data': None},
                   "PWATclm": {'dimesiones': [13, 9, 1], 'data': None},
                   "RH2m": {'dimesiones': [13, 9, 1], 'data': None},
                   "Vel100m": {'dimesiones': [13, 9, 1], 'data': None},
                   "UGRD100m": {'dimesiones': [13, 9, 1], 'data': None},
                   "VGRD100m": {'dimesiones': [13, 9, 1], 'data': None},
                   "Vel80m": {'dimesiones': [13, 9, 1], 'data': None},
                   "UGRD80m": {'dimesiones': [13, 9, 1], 'data': None},
                   "VGRD80m": {'dimesiones': [13, 9, 1], 'data': None},
                   "Vel10m":{'dimesiones': [13, 9, 1], 'data': None},
                   "UGRD10m": {'dimesiones': [13, 9, 1], 'data': None},
                   "VGRD10m": {'dimesiones': [13, 9, 1], 'data': None},
                   "GUSTsfc": {'dimesiones': [13, 9, 1], 'data': None},
                   "TMPsfc": {'dimesiones': [13, 9, 1], 'data': None},
                   "TMP2m": {'dimesiones': [13, 9, 1], 'data': None},
                   "no4LFTXsfc":{'dimesiones': [13, 9, 1], 'data': None},
                   "CAPEsfc": {'dimesiones': [13, 9, 1], 'data': None},
                   "SPFH2m": {'dimesiones': [13, 9, 1], 'data': None},
                   "SPFH80m": {'dimesiones': [13, 9, 1], 'data': None},
               }

        size_3d = 13*9*26
        array_3d = array[:size_3d*7]
        for variable, length in zip(features_3d.keys(), range(len(features_3d))):
            features_3d[variable]["data"] = array_3d[length*size_3d:(length +1)*size_3d]

        size_2d = 13*9
        array_2d = array[size_3d*7:]
        for variable, length in zip(features_2d.keys(), range(len(features_2d))):
            features_2d[variable]["data"] = array_2d[length*size_2d:(length +1)*size_2d]



        file_data.append( {
            "file_name": file,
            #"datetime": new_time,
            "var_3d": features_3d,
            "var_2d":features_2d,
        })

    for i in range(len(new_time)):
        total.update({new_time[i]:file_data[i]})

    print("It's done!")

    return  total


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


def get_date(base_dir):
    new_time = []
    for file in listdir(base_dir):
        file_path = f'{base_dir}/{file}'
        match=file.split("_")[1]
        date = pd.to_datetime(match, format = "%Y%m%d%H").strftime('%d/%m/%Y')
        time = (datetime.strptime(match, "%Y%m%d%H") + timedelta(hours=6)).strftime('%H:%M')
        new_time.append(date + " " + time)
    return new_time



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

