import pandas as pd

from sklearn.model_selection import train_test_split
from src.functions import loading, get_var, setting_X, setting_y

from sklearn.ensemble import RandomForestRegressor

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


###########################################

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

model = RandomForestRegressor(n_estimators=100)
model.fit(X_train.values, y_train.values)

print(model.feature_importances_)
