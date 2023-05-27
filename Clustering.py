import math
import random
from OSMPythonTools.overpass import Overpass
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import datetime
import warnings
import logging
import folium
logging.getLogger("OSMPythonTools").setLevel(logging.ERROR)
warnings.filterwarnings("error")
overpass = Overpass()

def gcdist(p1, p2):
    """
    Calculates the great circle distance of two points p1 and p2 (lat-lon tuples) on Earth, where p1 and p2 are lat-lon tuples.
    """
    d = (math.radians(p2[0]-p1[0]), math.radians(p2[1]-p1[1]))
    a = math.sin(d[0]/2) ** 2 + math.cos(math.radians(p1[0])) * math.cos(math.radians(p2[0])) * math.sin(d[1]/2) ** 2
    c = 6371 * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return c

def sphericalToCartesian(p):
    """
    Converts a lat-lon-alt tuple to an x-y-z tuple.
    """
    p_rad = (math.radians(p[0]), math.radians(p[1]))
    x = math.cos(p_rad[0])*math.cos(p_rad[1])*(6371+p[2]*0.0003048)
    y = math.cos(p_rad[0])*math.sin(p_rad[1])*(6371+p[2]*0.0003048)
    z = math.sin(p_rad[0])*(6371+p[2]*0.0003048)
    return (x,y,z)

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

def cartdist(p1, p2):
    """
    Calculates the Cartesian distance of two points.
    """
    return np.linalg.norm(p2 - p1)

def midpoint(*points):
    """
    Calculates the midpoint of any number of Cartesian points.
    """
    return np.sum(points, axis=0) / np.shape(points)[0]

def radius(df):
    """
    Calculates radius and midpoint of points in dataframe.
    """
    arr = df[['x', 'y', 'z']].to_numpy()
    x2 = np.sum(arr**2, axis=1)
    xy = np.matmul(arr, arr.T)
    y2 = x2.reshape(-1, 1)
    dists = np.sqrt(np.abs(x2 - 2*xy + y2))
    argmax = np.unravel_index(np.argmax(dists), dists.shape)
    mp = midpoint(arr[argmax[0]], arr[argmax[1]])
    dists2 = [cartdist(p, mp) for p in arr]
    if max(dists2) > cartdist(mp, arr[argmax[0]]):
        argmax2 = np.argmax(dists2)
        return circumscribedCircle(arr[argmax[0]], arr[argmax[1]], arr[argmax2])
    else:
        return cartdist(mp, arr[argmax[0]]), *mp

def circumscribedCircle(p1, p2, p3):
    """
    Returns the radius of the circumscribed circle, and the coordinates of its midpoint.
    """
    try:
        denom = np.linalg.norm(np.cross((p1 - p2), (p2 - p3)))
        r = np.linalg.norm(p1 - p2) * np.linalg.norm(p2 - p3) * np.linalg.norm(p3 - p1) / (2 * denom)
        a = np.linalg.norm(p2 - p3)**2 * np.dot((p1 - p2), (p1 - p3)) / (2 * denom**2)
        b = np.linalg.norm(p1 - p3)**2 * np.dot((p2 - p1), (p2 - p3)) / (2 * denom**2)
        c = np.linalg.norm(p1 - p2)**2 * np.dot((p3 - p1), (p3 - p2)) / (2 * denom**2)
        mp = a * p1 + b * p2 + c * p3
        return r, *mp.tolist()
    except RuntimeWarning:
        return cartdist(p1, p2)/2, *midpoint(p1, p2)

def onSea(p):
    """
    For a tuple of spherical coordinates, it returns a boolean value indicating whether the point is located in a body of water.
    """
    elements = overpass.query('is_in(' + f"{p[0]:.6f}" + ', ' + f"{p[1]:.6f}" + '); out body;').elements()
    elements_alods = [item.tags() for item in elements]       # alods: As List Of DictionarieS
    country = None
    water = False
    boundary = None
    for idx, item in enumerate(elements_alods):
        if item is not None:
            if 'admin_level' in item and item['admin_level'] == '2' and country is None:
                if 'name:en' in item:
                    country = item['name:en']
                elif 'name' in item:
                    country = item['name']
            if 'natural' in item:
                if item['natural'] == 'bay':
                    water = True
                elif item['natural'] == 'reef':
                    water = True
                elif item['natural'] == 'shoal':
                    water = True
                elif item['natural'] == 'strait':
                    water = True
                elif item['natural'] == 'water':
                    water = True
                    boundary = elements[idx]
            if 'place' in item:
                if item['place'] == 'sea':
                    water = True
                if item['place'] == 'ocean':
                    water = True
            if 'boundary' in item and item['boundary'] == 'maritime':
                water = True
    
    if country is None:
        water = True
        # known errors: unclaimed territory (e.g. Gornja Siga, Marie Byrd Land) is classified as water
        
    return water, boundary

# Model values
towers = {50: 300, 100: 300}                       # number of towers to be placed
nTrain = 50000                            # number of planes sampled for modelling
seaAllowed = True                         # are towers allowed to be placed on sea

t = datetime.datetime.now()
print(datetime.datetime.now()-t, "Starting import...", sep=': ')

data = pd.read_csv('clean_data.csv.gz', compression='gzip')

print(datetime.datetime.now()-t, f"Import completed. Number of data points: {data.shape[0]}", sep=': ')

data = data.sample(n=nTrain)

# Clustering
clusters = pd.DataFrame(columns=['score','planes','radius','x','y','z'])
planes_clustered = pd.DataFrame(columns=[*data.columns, 'cluster'])
cl_count = 0

for r,n in sorted(towers.items(), reverse=True):
    # Determining clusters
    X = data[['x', 'y', 'z']].to_numpy()
    print(datetime.datetime.now()-t, f"Starting clustering algorithm with p={data.shape[0]}, n={n} and r={r}...", sep=': ')
    # Note to distance threshold: In some circles, the largest distance between two points might be smaller than the diameter of the circumscribed circle. An 8% compensation for this effect (by trial and error) is included in the clustering.
    comp = 0.08
    clustering = AgglomerativeClustering(distance_threshold=r*2*(1-comp), n_clusters=None, metric="euclidean", compute_full_tree=True, linkage="complete").fit(X)
    data['cluster'] = np.array(clustering.labels_) + cl_count
    cl_count = data['cluster'].max() + 1
    print(datetime.datetime.now()-t, "Clustering completed.", sep=': ')

    # Scoring clusters, removing covered planes from data
    clusters_buff = data[['cluster', 'capacity']].groupby('cluster').agg({'capacity': ['sum', 'count']})
    clusters_buff.columns = ['score', 'planes']
    clusters_buff['radius'] = [radius(data[data['cluster'] == idx]) for idx, _ in clusters_buff.iterrows()]
    clusters_buff[['radius', 'x', 'y', 'z']] = pd.DataFrame(clusters_buff['radius'].tolist(), index=clusters_buff.index)
    d = (clusters_buff['x']**2 + clusters_buff['y']**2 + clusters_buff['z']**2)**0.5
    clusters_buff['alt'] = (d - 6371) * 3280.8399
    clusters_buff['lat'] = np.arcsin(clusters_buff['z'] / d)
    clusters_buff['lon'] = clusters_buff['y'].apply(signum) * np.arccos(clusters_buff['x'] / (d * np.cos(clusters_buff['lat'])))
    clusters_buff['lat'] = np.degrees(clusters_buff['lat'])
    clusters_buff['lon'] = np.degrees(clusters_buff['lon'])
    clusters_buff = clusters_buff.sort_values('score', ascending=False)
    # Are clusters with sea midpoints allowed?
    if not seaAllowed:
        print(datetime.datetime.now()-t, "Removing clusters on sea...", sep=': ')
        idx = 0
        while idx < n and idx < clusters_buff.shape[0]:
            cl_idx = clusters_buff.index[idx]
            mp = clusters_buff.loc[cl_idx, ['lat', 'lon']].to_numpy()
            room = r - clusters_buff.loc[cl_idx, 'radius']
            clusters_buff.loc[cl_idx, 'onSea'], boundary = onSea(mp)
            if clusters_buff.loc[cl_idx, 'onSea'] == True:
                if room > 0:
                    if boundary is None:
                        nodes = overpass.query(f'way(around:{1000*room:.6f}, {clusters_buff.loc[cl_idx, "lat"]:.6f}, {clusters_buff.loc[cl_idx, "lon"]:.6f})["natural"="coastline"]; (._;>>;); out;').elements()
                    elif boundary.type() == 'way':
                        nodes = overpass.query(f'way({boundary.id()}); (._;>>;); out;').elements()
                    elif boundary.type() == 'area':
                        nodes = overpass.query(f'rel({boundary.id()}); (._;>>;); out;').elements()

                    if len(nodes) > 0:
                        nodes = random.sample(nodes, k=min(30, len(nodes)))       # Querying all nodes increases computing time excessively
                        nodes = np.array([(node.lat(), node.lon()) for node in nodes if node.lat() is not None and node.lon() is not None])
                        dists = [gcdist(node, mp) for node in nodes]
                        if min(dists) < room:
                            nearest = nodes[dists.index(min(dists))]
                            clusters_buff.loc[cl_idx, ['lat', 'lon']] = nearest
                            clusters_buff.loc[cl_idx, ['x', 'y', 'z']] = sphericalToCartesian(clusters_buff.loc[cl_idx, ['lat', 'lon', 'alt']].to_numpy())
                        else:
                            data.loc[data['cluster'] == cl_idx, 'cluster'] = -1
                            clusters_buff = clusters_buff.drop(cl_idx)
                            idx -= 1
                    else:
                        data.loc[data['cluster'] == cl_idx, 'cluster'] = -1
                        clusters_buff = clusters_buff.drop(cl_idx)
                        idx -= 1
                else:
                    data.loc[data['cluster'] == cl_idx, 'cluster'] = -1
                    clusters_buff = clusters_buff.drop(cl_idx)
                    idx -= 1
            idx += 1
        print(datetime.datetime.now()-t, "Cluster removal completed.", sep=': ')

    clusters_buff = clusters_buff.head(n)
    clusters_buff['radius'] = r
                
    # Adding clusters to collection
    planes_clustered_buff = data[data['cluster'].isin(list(clusters_buff.index))]
    data = data[~data['cluster'].isin(list(clusters_buff.index))]
    clusters = pd.concat([clusters, clusters_buff])
    planes_clustered = pd.concat([planes_clustered, planes_clustered_buff])
    if data.shape[0] == 0:
        print("Break")
        break

print(datetime.datetime.now()-t, "All tower locations determined. Exporting data...", sep=': ')

clusters.to_csv('clusters.csv')

print(datetime.datetime.now()-t, "Data export completed.", sep=': ')

print(datetime.datetime.now()-t, "Exporting a random cluster to map...", sep=': ')
m = folium.Map()
cluster = clusters.sample(n=1)
r = cluster['radius'].iloc[0]*1000
lat, lon, alt = cartesianToSpherical(cluster[['x', 'y', 'z']].to_numpy()[0])
folium.Circle(radius=r, location=[lat, lon], popup=f"{cluster.index[0]}\n{lat:.4f}, {lon:.4f}").add_to(m)
folium.Marker(location=[lat, lon], popup=f"{cluster.index[0]}\n{lat:.4f}, {lon:.4f}", icon=folium.Icon(color="orange", icon="glyphicon-star")).add_to(m)

for pidx, p in planes_clustered[planes_clustered['cluster'] == cluster.index[0]].iterrows():
    lat, lon, alt = cartesianToSpherical(p[['x', 'y', 'z']].to_numpy())
    folium.Marker(location=[lat, lon], popup=f"{cluster.index[0]}\n{lat:.4f}, {lon:.4f}", icon=folium.Icon(icon="glyphicon-plane")).add_to(m)

for cidx, c in clusters.drop(cluster.index[0]).iterrows():
    if cartdist(c[['x', 'y', 'z']].to_numpy(), cluster[['x', 'y', 'z']].to_numpy()[0]) < c['radius'] + cluster['radius'].iloc[0]:
        r = c['radius']*1000
        lat, lon, alt = cartesianToSpherical(c[['x', 'y', 'z']].to_numpy())
        folium.Circle(radius=r, location=[lat, lon], popup=f"{cidx}\n{lat:.4f}, {lon:.4f}", color="red").add_to(m)

m.save("cluster.html")
print(datetime.datetime.now()-t, "Map export completed.", sep=': ')
