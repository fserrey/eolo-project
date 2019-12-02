from power_forecast.functions import *

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

X = trained[[x for x in train.columns if x != 'Production']]
y = pd.DataFrame(trained["Production"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)











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
