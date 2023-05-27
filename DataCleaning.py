import math
import random
import pandas as pd
import numpy as np
import os
import datetime
import sys

t = datetime.datetime.now()
print(datetime.datetime.now()-t, "Starting import...", sep=': ')

# plane capacities
planevalues = pd.read_csv("capacity.csv", sep=';')
planevalues = planevalues[planevalues['Capacity merged'] > 0]
planevalues = {plane['ICAO Code']: plane['Capacity merged'] for _, plane in planevalues.iterrows()}

print(datetime.datetime.now()-t, "Plane capacities imported.", sep=': ')
print(datetime.datetime.now()-t, "Importing files...", sep=': ')

loc = "E:\BME\Önlab1"
files = os.listdir(loc)
files = [file for file in files if file[-7:] == ".csv.gz"]


# Data import and cleaning
filecount = len(files)
i = 0
data = pd.DataFrame(columns=['flight_number', 'capacity', 'scheduled_departure_time_utc', 'timestamp', 'x', 'y', 'z'])

for file in files:
    i += 1
    try:
        tr = (datetime.datetime.now()-t)*(filecount/(i-1)-1)
        if tr > datetime.timedelta(hours=1):
            tr = f'{int(tr.total_seconds() // 3600)} h {int(tr.total_seconds() // 60 % 60)} min'
        elif tr > datetime.timedelta(minutes=5):
            tr = f'{int(tr.total_seconds() // 60 % 60)} min'
        else:
            tr = f'{int(tr.total_seconds())} seconds'
    except ZeroDivisionError:
        tr = "N/A"
    print(datetime.datetime.now()-t, f"Importing data file {i}/{filecount}. Estimated time remaining: {tr}. Memory use: {int(sys.getsizeof(data)/1024/1024)} MB", sep=': ')
    path = loc+'\\'+file
    df = pd.read_csv(path, compression="gzip", low_memory=False)
    cols = ['timestamp', 'latitude', 'longitude', 'altitude_baro', 'on_ground', 'icao_actype', 'flight_number', 'scheduled_departure_time_utc']
    # dropping unnecessary columns
    df = df[cols]
    # removing rows with gaps
    df = df.dropna()
    # removing planes on ground
    df = df[df['on_ground'] == False]
    df = df.drop('on_ground', axis=1)
    # removing small planes
    df = df[df['icao_actype'].isin(planevalues)]
    # formatting timestamps
    df['timestamp'] = df['timestamp'].replace(to_replace='\.\d*', value='', regex=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['scheduled_departure_time_utc'] = df['scheduled_departure_time_utc'].replace(to_replace='\.\d*', value='', regex=True)
    df['scheduled_departure_time_utc'] = pd.to_datetime(df['scheduled_departure_time_utc'])
    # changing spherical coordinates to Cartesian
    lat = np.radians(df['latitude'])
    lon = np.radians(df['longitude'])
    alt = df['altitude_baro'].to_numpy()
    df['x'] = np.cos(lat)*np.cos(lon)*(6371+alt*0.0003048)
    df['y'] = np.cos(lat)*np.sin(lon)*(6371+alt*0.0003048)
    df['z'] = np.sin(lat)*(6371+alt*0.0003048)
    df = df.drop(['latitude', 'longitude', 'altitude_baro'], axis=1)
    # aggregating by minute
    df = df.groupby(['flight_number', 'scheduled_departure_time_utc', 'icao_actype', pd.Grouper(key='timestamp', freq="1Min")])
    df = df.agg({'x': ['mean'], 'y': ['mean'], 'z': ['mean']}).reset_index().sort_index(axis=1)
    df.columns = ['flight_number', 'icao_actype', 'scheduled_departure_time_utc', 'timestamp', 'x', 'y', 'z']
    # replacing aircraft type by capacity
    df['icao_actype'] = df['icao_actype'].replace(planevalues)
    df = df.rename(columns={'icao_actype': 'capacity'})
    # adding new data to dataset
    data = pd.concat([data, df], ignore_index=True)

print(datetime.datetime.now()-t, "All files imported. Exporting cleaned data...", sep=': ')

data.to_csv('clean_data.csv.gz', compression="gzip")

print(datetime.datetime.now()-t, "Data export completed.", sep=': ')
