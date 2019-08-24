# Eolo project

### Data analytics and Machine Learning techniques applied to meteorological and wind farm power data

## Overview

This project aims to detect the relation between several types of data features in order to locate where, ideally, would this wind farm be placed. The code in this repository takes meteorological data (GFS) and the power registered for 2 years in a wind farm. We will analyse data meteorological predictions for 6 hours horizon in more than 36 hectares space.

The current project it is under final assesment framwork of the Ironhack academic program for the Data Analytics Bootcamp.

## Data
**Meteorological predictions**
The Global Forecast System (GFS) is a weather forecast model produced by the [National Centers for Environmental Prediction (NCEP)](ftp://nomads.ncdc.noaa.gov/). Dozens of atmospheric and land-soil variables are available through this dataset, from temperatures, winds, and precipitation to soil moisture and atmospheric ozone concentration. The entire globe is covered by the GFS at a base horizontal resolution of 18 miles (28 kilometers) between grid points

The data were structured in more than 4K arrays-format-files (.gra) holding information related to these meteorological variables:

| Var       | Nz | Description                                                                    |
|-----------|----|--------------------------------------------------------------------------------|
| HGTpr     | 26 | (1000 975 950 925 900.. 7 5 3 2 1) Geopotential Height [gpm]                   |
| CLWMRprs  | 26 | (1000 975 950 925 900.. 300 250 200 150 100) Cloud Mixing Ratio [kg/kg]        |
| RHprs     | 26 | (1000 975 950 925 900.. 7 5 3 2 1) Relative Humidity [%]                       |
| Velprs    | 26 | (1000 975 950 925 900.. 7 5 3 2 1) Vel [m/s]                                   |
| UGRDprs   | 26 | (1000 975 950 925 900.. 7 5 3 2 1) U-Component of Wind [m/s]                   |
| VGRDprs   | 26 | (1000 975 950 925 900.. 7 5 3 2 1) V-Component of Wind [m/s]                   |
| TMPprs    | 26 | (1000 975 950 925 900.. 7 5 3 2 1) Temperature [K]                             |
| HGTsfc    | 1 | surface Geopotential Height [gpm]                                               |
| MSLETmsl  | 1 | 	mean sea level MSLP (Eta model reduction) [Pa]                                |
| PWATclm   | 1 | entire atmosphere (considered as a single layer) Precipitable Water [kg/m^2]    |
| RH2m      | 1 | 2 m above ground Relative Humidity [%]                                          |
| Vel100m   | 1 | 100 m above ground Vel [m/s]                                                    |
| UGRD100m  | 1 | 100 m above ground U-Component of Wind [m/s]                                    |
| VGRD100m  | 1 | 100 m above ground V-Component of Wind [m/s]                                    |
| Vel80m    | 1 | 80 m above ground Vel [m/s]                                                     |
| UGRD80m   | 1 | 80 m above ground U-Component of Wind [m/s]                                     |
| VGRD80m   | 1 | 80 m above ground V-Component of Wind [m/s]                                     |
| Vel10m    | 1 | 10 m above ground Vel [m/s]                                                     |
| UGRD10m   | 1 | 10 m above ground U-Component of Wind [m/s]                                     |
| VGRD10m   | 1 | 10 m above ground V-Component of Wind [m/s]                                     |
| GUSTsfc   | 1 | surface Wind Speed (Gust) [m/s]                                                 |
| TMPsfc    | 1 | surface Temperature [K]                                                         |
| TMP2m     | 1 | 2 m above ground Temperature [K]                                                |
| no4LFTXsfc| 1 | surface Best (4 layer) Lifted Index [K]                                         |
| CAPEsfc   | 1 | surface Convective Available Potential Energy [J/kg]                            |
| SPFH2m    | 1 | 2 m above ground Specific Humidity [kg/kg]                                      |
| SPFH80m   | 1 | 80 m above ground Specific Humidity [kg/kg]                                     |

<INSERT IMAGE OF VARIABLE REPRESENTATION>

All the variables are included on each file. Therefore, an extraction and posterior organisation of the data were needed before employing the model. 


**Power data**
The wind farm power records were taken from a energy company. The company provided information related to date and power (kilowatts) that we need to match with the GFS predictions.
 	

## Folders
This repository have several components in order to make everyone easier the task to understand of what is going on:

### src
The folder that contains two of the main files, one that keeps all the functions used for this project and the one that actually locates and represent it on a map.

<INSERT MAP CAPTION>

### notebooks
This folder contains the data cleaning of the power dataset and some other experiments made previous to build the code at src.

### images
Ouputs results of the current project

### data 
This folder store all the data employed for this project.

## To Do
As this project is made out of the purpose of predicting the power data of a wind farm out of meteorological predictions, the next step is to actually detect the variables that influence the most and test several predictions models.