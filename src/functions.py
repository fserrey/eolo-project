
import os

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json



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
            features_3d[variable]["data"] = array_3d[length*size_3d:(length +1)*size_3d]#.reshape((len(features_3d.keys()), corte))

        size_2d = 13*9
        array_2d = array[size_3d*7:]
        for variable, length in zip(features_2d.keys(), range(len(features_2d))):
            features_2d[variable]["data"] = array_2d[length*size_2d:(length +1)*size_2d]#.reshape((len(features_2d.keys()), corte))



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
    """
    dict_final = {}
    size_3d = 13*9*nz
    print("Now, we get the variables we want")
    for datetime_key in main_dic: #Quiero iterar sobre las keys de 1º nivel
        res = []
        for var in list_var:  # compruebo que la variable que voy a sacar está en mi lista
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


#names = ['date','data']
#formats = ['f8','f8']
#dtype = dict(names = names, formats=formats)
#array = np.array(list(var_to_test.items()), dtype=dtype)

#print(repr(array))



def setting_X(dictionary):
    meteo_df = pd.DataFrame(var_to_test).T
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
