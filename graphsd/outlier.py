import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import distance, Voronoi, voronoi_plot_2d
import math
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt
from shapely.geometry import MultiPoint, Point, Polygon

from graphsd.utils import *


def voronoi_finite_polygons_2d(vor, radius=None): ## NOT MY ORIGINAL CODE
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        #print(p1, region)
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)

def getLOF(df, k = 5, contamination = .1, print_mode = False):
    clf = LocalOutlierFactor(n_neighbors=k, contamination=.1)
    y_pred = clf.fit_predict(df)
    #print(y_pred)
    LOF_Scores = clf.negative_outlier_factor_
    
    return [-x for x in LOF_Scores]


def getSeveralLOF(movement, socialData, start_time, end_time, nseconds = 1, k = 5, contamination = .1):
    #nseconds = 1
    start_window = pd.Timestamp(start_time)
    areas_df = pd.DataFrame()
    ids = movement.id.unique()

    
    while start_window <= pd.Timestamp(end_time):
        temp_df = socialData[socialData['id'].isin(ids)].copy()
        position = movement[str(start_window)]
        pids = [row.id for index, row in position.iterrows()]
        #print(getAreas(positions))
        #try:
        areasp = getLOF(position[['x','y']], k = k, contamination = contamination)
        #print(areasp)
        #return areasp
    
        tempareas = []
        temptimes = []
        
        try:
            i = 0
            for pid in ids:
                temptimes += [start_window]
                if pid in pids:
                    tempareas += [areasp[i]]
                    i += 1
                else:
                    tempareas += [np.nan]


            temp_df['lof'] = tempareas
            temp_df['timestamp'] = temptimes

        except:
            print("An exception occurred at: ", start_window)

        areas_df = pd.concat([areas_df, temp_df[np.isfinite(temp_df['lof'])]])
        start_window = start_window + pd.Timedelta(seconds = nseconds)
        
    #maxW = max(list(counter.values()))
    
    #counter = counter/count
    return areas_df

def getAreas(points, pids, print_mode = False):
    vor = Voronoi(points)
    #voronoi_plot_2d(vor)
    #plt.show()
    new_vertices = []

    regions, vertices = voronoi_finite_polygons_2d(vor)

    pts = MultiPoint([Point(i) for i in points])
    #mask = MultiPoint(pts.bounds).convex_hull
    minx, miny, maxx, maxy = pts.bounds
    
    factor = 0.1
    margin = minx*factor
    #print("minx: ", minx - margin, ", miny: ", miny - margin, ", maxx: ", maxx + margin, ", maxy: ", maxy + margin)
    mask = MultiPoint([Point(i) for i in [[minx - margin, miny - margin],[minx - margin, maxy + margin], [maxx + margin, miny - margin], [maxx + margin, maxy + margin]]]).convex_hull
    
    areas = []
    for region in regions:
        polygon = vertices[region]
        shape = list(polygon.shape)
        shape[0] += 1
        p = Polygon(np.append(polygon, polygon[0]).reshape(*shape)).intersection(mask)
        areas += [p.area]
        #print(p.area)
        poly = np.array(list(zip(p.boundary.coords.xy[0][:-1], p.boundary.coords.xy[1][:-1])))
        new_vertices.append(poly)
        if print_mode:
            plt.fill(*zip(*poly), alpha=0.4)
    #if print_mode:
        #print(points)
        #print(areas)
        #'2', '3', '4', '5', '6', '8', '9', '10', '11', '12', '13', '14','15', '16', '17', '18', '19', '21'
        #pidi = 0
        #for [x,y] in points:
        #    if pids[pidi] == "17":
        #        plt.plot(x, y, 'ko', color = 'red')
        #    else:
        #        plt.plot(x, y, 'ko', color = 'blue')
        #    pidi += 1
        #plt.title("Clipped Voronois")
        #plt.show()
    return areas 

def getSeveralAreas(movement, socialData, start_time, end_time, nseconds = 1):
    #nseconds = 1
    start_window = pd.Timestamp(start_time)
    areas_df = pd.DataFrame()
    counter = 0
    ids = movement.id.unique()

    
    while start_window <= pd.Timestamp(end_time):
        #print(start_window)
        temp_df = socialData[socialData['id'].isin(ids)].copy()
        position = movement[str(start_window)].query("x != 'NaN'")
        positions = [[row.x,row.y] for index, row in position.iterrows()]
        pids = [row.id for index, row in position.iterrows()]
        #print(getAreas(positions))
        #areasp = getAreas(positions, pids, print_mode = True)
        try:
            areasp = getAreas(positions, pids, print_mode = False)
            tempareas = []
            temptimes = []

            i = 0
            for pid in ids:
                temptimes += [start_window]
                if pid in pids:
                    tempareas += [areasp[i]]
                    i += 1
                else:
                    tempareas += [np.nan]

            temp_df['area'] = tempareas
            temp_df['timestamp'] = temptimes

        except:
           print("An exception occurred at: ", start_window)
           start_window = start_window + pd.Timedelta(seconds = nseconds)
           continue

        areas_df = pd.concat([areas_df, temp_df[np.isfinite(temp_df['area'])]])
        start_window = start_window + pd.Timedelta(seconds = nseconds)
        counter += 1
        
    #maxW = max(list(counter.values()))
    
    #counter = counter/count
    return areas_df

def setAtts(dataset, atts):
    ids = dataset.id.unique()
    transactions = []
    tr = []
    for nid in ids:
        tr = []
        eattr = {}
        for att in atts:
            res = list(dataset[dataset.id == nid][att])[0]
            tr.append(NominalSelector(att, res))
            
        transactions.append(tr)
                
    return transactions