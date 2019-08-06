import os

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from src.functions import loading, get_var, setting_X, setting_y  #importing loading functions
import json

file_path = "/home/slimbook/git-repos/eolo-project/data/.raw/GFS_data"

data_dictionary = loading(file_path) # All files and variables loaded as dictionary

list_var = ["Vel100m"]

var_to_test = get_var(data_dictionary, list_var, nz=5)

print("Setting X...")
print(type(var_to_test))

meteo = setting_X(var_to_test)


csv_path = ("/home/slimbook/git-repos/eolo-project/data/processed/power_data.csv")




print("Setting y...")
power = setting_y(csv_path)


train = pd.concat([power, meteo], axis=1, join="inner")
train.sort_index(ascending=True, inplace=True)

X = train[[x for x in train.columns if x != 'Production']]
y = pd.DataFrame(train["Production"])
print("Yay! Your mode input is ready!")
