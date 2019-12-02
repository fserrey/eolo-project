from functions import *

# Modeling
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor

# Evaluation of the model
from sklearn.model_selection import train_test_split

# Data loading 
print("Reading the data...")
base_dir = '/home/slimbook/git-repos/eolo-project/data/.raw/GFS_data'
power_csv = "/home/slimbook/git-repos/eolo-project/data/processed/power_data.csv"
gfs_dict_path = '/home/slimbook/git-repos/eolo-project/data/processed/gfs_info.json'

with open(gfs_dict_path, 'r', encoding='utf-8') as data_file:    
    gfs_data_dict = json.load(data_file)

# Selection of variables and pre-processing
list_var = ["RHprs", "Velprs", "TMPprs", "Vel100m","Vel80m", "TMPsfc", "SPFH80m"]

lista_dates = get_date(base_dir)
variables_ready = get_variables(base_dir, list_var, gfs_data_dict, nz=5)

df_gfs = pd.DataFrame(data=variables_ready, index=lista_dates, columns=list_var)
df_power = pd.read_csv(power_csv)
df_power['date'] =  pd.to_datetime(df_power['date'], format='%d/%m/%Y %H:%M')
df_power = df_power.set_index("date")

df_gfs.sort_index(axis=0, level=None, ascending=True, inplace=True)
df_gfs.loc[:'31/12/2016 00:00']
df_power.sort_index(axis=0, level=None, ascending=True, inplace=True)
df_power.loc[:'31/12/2016 00:00']

# Data preparation for model train
trained = df_power.merge(df_gfs, left_index=True, right_index=True) # df intersection based on dates

df_X = trained[[x for x in trained.columns if x != 'Production']]

X = setup_x(df_X)
y = pd.DataFrame(trained["Production"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


# specify your configurations as a dict
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_logloss',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}


#XGBRegressor
boost_params = {'eval_metric': 'rmse'}
xgb0 = xgb.XGBRegressor(
    max_depth=8,
    learning_rate=0.1,
    n_estimators=1000,
    objective='reg:linear',
    gamma=0,
    min_child_weight=1,
    subsample=1,
    colsample_bytree=1,
    scale_pos_weight=1,
    seed=27,
    **boost_params)

#LGBMRegressor
gbm0 = lgb.LGBMRegressor(
    objective='regression',
    num_leaves=60,
    learning_rate=0.1,
    n_estimators=1000)


print("Fitting XGBRegressor model...")
xboost_fit = xgb0.fit(X_train, y_train)
print("Finished fitting XGBRegressor model")

print("Fitting LGBMRegressor model...")
gbm_fit = gbm0.fit(X_train, y_train, eval_metric='rmse')
print("Finished fitting LGBMRegressor model")

# Prediction
predict_lightGBM = gbm0.predict(X_test)
predict_XGB = xgb0.predict(X_test)

# W.I.P