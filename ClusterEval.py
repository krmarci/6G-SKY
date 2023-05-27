import math
import random
import pandas as pd
import numpy as np
import datetime
import folium

def gcdist(p1, p2):
    """
    Calculates the great circle distance of two points p1 and p2 (lat-lon tuples) on Earth.
    """
    d = (math.radians(p2[0]-p1[0]), math.radians(p2[1]-p1[1]))
    a = math.sin(d[0]/2) ** 2 + math.cos(math.radians(p1[0])) * math.cos(math.radians(p2[0])) * math.sin(d[1]/2) ** 2
    c = 6371 * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return c

def cartdist(p1, p2):
    """
    Calculates the cartesian distance of two points p1 and p2 (x-y-z numpy arrays) on Earth.
    """
    return np.linalg.norm(p2-p1)

def cartesianToSpherical(p):
    """
    Converts an x-y-z tuple to a lat-lon-alt tuple.
    """
    alt = np.linalg.norm(p)
    lat = math.asin(p[2] / alt)
    lon = signum(p[1]) * math.acos(p[0] / (alt * math.cos(lat)))
    alt = (alt - 6371) * 3280.8399
    lat = math.degrees(lat)
    lon = math.degrees(lon)
    return (lat, lon, alt)

def signum(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0

# Model values
nTest = 1000             # number of planes sampled for modelling

t = datetime.datetime.now()
print(datetime.datetime.now()-t, "Starting import...", sep=': ')

data = pd.read_csv('clean_data.csv.gz', index_col=0, compression='gzip')
clusters = pd.read_csv('clusters.csv', index_col=0)

print(datetime.datetime.now()-t, "Import completed. Starting evaluation...", sep=': ')

planes = data.sample(n=nTest)

df = pd.DataFrame([{cidx: cartdist(clusters[['x', 'y', 'z']].loc[cidx], planes[['x', 'y', 'z']].loc[pidx]) for cidx in clusters.index} for pidx in planes.index], columns=clusters.index)
result = df.apply(lambda row: (row < clusters['radius'].transpose()).any(), axis=1)
result.index = planes.index
print(datetime.datetime.now()-t, f"Evaluation complete. Coverage: {result.value_counts().loc[True]}/{nTest}", sep=': ')
print(datetime.datetime.now()-t, "Starting map export...", sep=': ')

m = folium.Map()
for cidx, c in clusters.iterrows():
    r = c['radius']*1000
    lat, lon, alt = cartesianToSpherical(c[['x', 'y', 'z']].tolist())
    folium.Circle(radius=r, location=[lat, lon], popup=f"{cidx}\n{lat:.4f}, {lon:.4f}").add_to(m)

m.save("clusters.html")
    
for pidx, p in planes.iterrows():
    lat, lon, alt = cartesianToSpherical(p[['x', 'y', 'z']].tolist())
    if result.loc[pidx] == True:
        folium.Marker(location=[lat, lon], popup=f"{cidx}\n{lat:.4f}, {lon:.4f}", icon=folium.Icon(color="green", icon="glyphicon-star")).add_to(m)
    else:
        folium.Marker(location=[lat, lon], popup=f"{pidx}\n{lat:.4f}, {lon:.4f}", icon=folium.Icon(color="red", icon="glyphicon-star")).add_to(m)

m.save("eval.html")

print(datetime.datetime.now()-t, "Map export completed.", sep=': ')
