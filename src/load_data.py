import os

#numerical

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
#import klepto



# others

import json


# Importing data

file_path = "/home/slimbook/git-repos/eolo-project/data/.raw/GFS_data"





data_dictionary = loading(file_path)






list_var = ["HGTprs", "CLWMRprs", "RHprs", "Velprs", "UGRDprs", "VGRDprs","TMPprs","HGTsfc","MSLETmsl","PWATclm","RH2m","Vel100m","UGRD100m","VGRD100m","Vel80m",
"UGRD80m","VGRD80m","Vel10m","UGRD10m","VGRD10m","GUSTsfc","TMPsfc","TMP2m","no4LFTXsfc","CAPEsfc","SPFH2m""SPFH80m"]


var_to_test = get_var(data_dictionary, list_var, nz=5)


print("Save it as file")

import pickle

def to_pickle(input_file):
    print("Enter desired name:")
    name = input()
    pickle_out = open(name + '.pickle','wb')
    pickle.dump(input_file, pickle_out)
    pickle_out.close()


#to_pickle(var_to_test)

#with open('meteo_var.json', 'w') as fp:
#        json.dump(var_to_test, fp)

#meteo_var = dir_archive('meteo_var', var_to_test, serialized=True)
#meteo_var.dump()
print("Apparently, you got a new file!")
