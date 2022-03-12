import numpy as np
from scipy.spatial import distance
import networkx as nx

from graphsd.utils import *


def count_interactions(dataframe, proximity=1, time_step=10):
    """

    get interactions for simple graphs

    Parameters
    ----------
    dataframe
    proximity
    time_step

    Returns
    -------

    """
    counter = {}
    start = min(dataframe.time)
    end = start + time_step
    while start <= max(dataframe.time):
        position = dataframe.query("@start <= time <= @end").groupby(['id']).mean()
        dists = distance.cdist(position[['x', 'y']], position[['x', 'y']], 'euclidean')

        # distances < proximity -> add 1 to that relationship
        np.fill_diagonal(dists, np.inf)
        xs, ys = np.where(dists <= proximity)
        for i in range(len(xs)):
            if xs[i] >= ys[i]:
                continue
            id_x = position.index[xs[i]]
            id_y = position.index[ys[i]]

            if (id_x, id_y) in counter:
                counter[(id_x, id_y)] += 1
            else:
                counter[(id_x, id_y)] = 1

        start = end
        end = end + time_step

    return [(list(key)[0], list(key)[1], value) for key, value in counter.items()]


def count_interactions_digraph(dataframe, proximity=1, time_step=10):
    """

    getDInteractions

    Parameters
    ----------
    dataframe
    proximity
    time_step

    Returns
    -------

    """
    counter = {}
    start = min(dataframe.time)
    end = start + time_step
    while start <= max(dataframe.time):
        position = dataframe.query("@start <= time <= @end").groupby(['id']).mean()
        dists = distance.cdist(position[['x', 'y']], position[['x', 'y']], 'euclidean')

        # distances < proximity -> add 1 to that relationship
        np.fill_diagonal(dists, np.inf)
        xs, ys = np.where(dists <= proximity)
        for i in range(len(xs)):
            if xs[i] == ys[i]:
                continue
            id_x = position.index[xs[i]]
            id_y = position.index[ys[i]]
            vel1X = float(position.loc[id_x].velX)
            vel1Y = float(position.loc[id_x].velY)

            vx = float(position.loc[id_y].x) - float(position.loc[id_x].x)
            vy = float(position.loc[id_y].y) - float(position.loc[id_x].y)

            cosine = 0

            if (vel1X * vx + vel1Y * vy) != 0:
                cosine = (vel1X * vx + vel1Y * vy) / (math.sqrt(vel1X ** 2 + vel1Y ** 2) * math.sqrt(vx ** 2 + vy ** 2))

            if cosine >= 0:
                if (id_x, id_y) in counter:
                    counter[(id_x, id_y)] += 1
                else:
                    counter[(id_x, id_y)] = 1

        start = end
        end = end + time_step

    return {key: value for key, value in counter.items()}


def count_interactions_digraph_all(dataframe, proximity=1, time_step=10):
    ids = dataframe.id.unique()

    counter = {}
    for id1 in ids:
        for id2 in ids:
            if id1 != id2:
                counter[(id1, id2)] = 0

    start = min(dataframe.time)
    end = start + time_step
    while start <= max(dataframe.time):
        position = dataframe.query("@start <= time <= @end").groupby(['id']).mean()
        dists = distance.cdist(position[['x', 'y']], position[['x', 'y']], 'euclidean')

        # distances < proximity -> add 1 to that relationship
        np.fill_diagonal(dists, np.inf)
        xs, ys = np.where(dists <= proximity)
        for i in range(len(xs)):
            if xs[i] == ys[i]:
                continue
            id_x = position.index[xs[i]]
            id_y = position.index[ys[i]]
            vel1X = float(position.loc[id_x].velX)
            vel1Y = float(position.loc[id_x].velY)

            vx = float(position.loc[id_y].x) - float(position.loc[id_x].x)
            vy = float(position.loc[id_y].y) - float(position.loc[id_x].y)

            cosine = 0

            if (vel1X * vx + vel1Y * vy) != 0:
                cosine = (vel1X * vx + vel1Y * vy) / (math.sqrt(vel1X ** 2 + vel1Y ** 2) * math.sqrt(vx ** 2 + vy ** 2))

            if cosine >= 0:
                counter[(id_x, id_y)] += 1

        start = end
        end = end + time_step

    return {key: value for key, value in counter.items()}


def count_interactions_multi_digraph(dataframe, proximity, time_step=10):
    ids = dataframe.id.unique()
    nids = len(ids)
    oldInter = np.zeros((nids, nids))
    counter = []
    start = min(dataframe.time)
    end = start + time_step
    while start <= max(dataframe.time):
        position = dataframe.query("@start <= time <= @end").groupby(['id']).mean()
        dists = distance.cdist(position[['x', 'y']], position[['x', 'y']], 'euclidean')

        # distances < proximity -> add 1 to that relationship
        np.fill_diagonal(dists, np.inf)
        xs, ys = np.where(dists <= proximity)
        for i in range(len(xs)):
            if xs[i] == ys[i]:
                continue
            id_x = position.index[xs[i]]
            id_y = position.index[ys[i]]
            vel1X = float(position.loc[id_x].velX)
            vel1Y = float(position.loc[id_x].velY)

            vx = float(position.loc[id_y].x) - float(position.loc[id_x].x)
            vy = float(position.loc[id_y].y) - float(position.loc[id_x].y)

            cosine = 0

            if (vel1X * vx + vel1Y * vy) != 0:
                cosine = (vel1X * vx + vel1Y * vy) / (math.sqrt(vel1X ** 2 + vel1Y ** 2) * math.sqrt(vx ** 2 + vy ** 2))

            old_id_x = np.where(ids == id_x)[0][0]
            old_id_y = np.where(ids == id_y)[0][0]
            if cosine >= 0:  # following
                oldInter[old_id_x][old_id_y] += 1
            else:
                if oldInter[xs[i]][ys[i]] > 0:
                    counter += [(id_x, id_y, oldInter[old_id_x][old_id_y])]
                    oldInter[old_id_x][old_id_y] = 0

        start = end
        end = end + time_step

    # add last edges (the ones that never stop existing)
    xs, ys = np.where(oldInter > 0)
    for i in range(len(xs)):
        counter += [(ids[xs[i]], ids[ys[i]], oldInter[xs[i]][ys[i]])]

    # counter = counter/count
    # maxW = max([w for x, y, w in counter])

    return [(x, y, w) for x, y, w in counter]


def count_interactions_multi_digraph2(dataframe, proximity, time_step=10):
    ids = dataframe.id.unique()
    nids = len(ids)
    oldInter = np.zeros((nids, nids))
    counter = []
    start = min(dataframe.time)
    end = start + time_step
    while start <= max(dataframe.time):
        position = dataframe.query("@start <= time <= @end").groupby(['id']).mean().reindex(ids).reset_index()
        dists = distance.cdist(position[['x', 'y']], position[['x', 'y']], 'euclidean')

        # distances < proximity -> add 1 to that relationship
        dists = (np.array(dists) <= proximity) + 0
        xs, ys = np.where(dists > 0)
        for i in range(len(xs)):
            if xs[i] == ys[i]:
                continue
            vel1X = float(position.loc[position['id'] == ids[xs[i]]].velX)
            vel1Y = float(position.loc[position['id'] == ids[xs[i]]].velY)

            vx = float(position.loc[position['id'] == ids[ys[i]]].x) - float(
                position.loc[position['id'] == ids[xs[i]]].x)
            vy = float(position.loc[position['id'] == ids[ys[i]]].y) - float(
                position.loc[position['id'] == ids[xs[i]]].y)

            cosine = 0

            if (vel1X * vx + vel1Y * vy) != 0:
                cosine = (vel1X * vx + vel1Y * vy) / (math.sqrt(vel1X ** 2 + vel1Y ** 2) * math.sqrt(vx ** 2 + vy ** 2))

            if cosine >= 0:  # following
                oldInter[xs[i]][ys[i]] += 1
            else:
                if oldInter[xs[i]][ys[i]] > 0:
                    counter += [(ids[xs[i]], ids[ys[i]], oldInter[xs[i]][ys[i]])]
                    oldInter[xs[i]][ys[i]] = 0

        start = end
        end = end + time_step

    # add last edges (the ones that never stop existing)
    xs, ys = np.where(oldInter > 0)
    for i in range(len(xs)):
        counter += [(ids[xs[i]], ids[ys[i]], oldInter[xs[i]][ys[i]])]

    # counter = counter/count
    # maxW = max([w for x, y, w in counter])

    return [(x, y, w) for x, y, w in counter]


def getMultiDInteractions_all(dataframe, proximity, time_step=10):
    ids = dataframe.id.unique()
    nids = len(ids)
    oldInter = np.zeros((nids, nids))
    inter = np.zeros((nids, nids))

    counter = []
    start = min(dataframe.time)
    end = start + time_step
    while start <= max(dataframe.time):
        position = dataframe.query("@start <= time <= @end").groupby(['id']).mean().reindex(ids).reset_index()
        dists = distance.cdist(position[['x', 'y']], position[['x', 'y']], 'euclidean')

        # distances < proximity -> add 1 to that relationship
        dists = (np.array(dists) <= proximity) + 0
        xs, ys = np.where(dists > 0)
        for i in range(len(xs)):
            if xs[i] == ys[i]:
                continue
            idx_x = position['id'] == ids[xs[i]]
            idx_y = position['id'] == ids[ys[i]]

            vel1X = float(position.loc[idx_x].velX)
            vel1Y = float(position.loc[idx_x].velY)

            vx = float(position.loc[idx_y].x) - float(position.loc[idx_x].x)
            vy = float(position.loc[idx_y].y) - float(position.loc[idx_x].y)

            cosine = 0

            if (vel1X * vx + vel1Y * vy) != 0:
                cosine = (vel1X * vx + vel1Y * vy) / (math.sqrt(vel1X ** 2 + vel1Y ** 2) * math.sqrt(vx ** 2 + vy ** 2))

            if cosine >= 0:  # following
                oldInter[xs[i]][ys[i]] += 1
            else:
                if oldInter[xs[i]][ys[i]] > 0:
                    counter += [(ids[xs[i]], ids[ys[i]], oldInter[xs[i]][ys[i]])]
                    inter[xs[i]][ys[i]] = 1
                    oldInter[xs[i]][ys[i]] = 0

        start = end
        end = end + time_step

    # add last edges (the ones that never stop existing)
    xs, ys = np.where(oldInter > 0)
    for i in range(len(xs)):
        counter += [(ids[xs[i]], ids[ys[i]], oldInter[xs[i]][ys[i]])]
        inter[xs[i]][ys[i]] = 1

    xs, ys = np.where(inter == 0)
    for i in range(len(xs)):
        if ids[xs[i]] != ids[ys[i]]:
            # print((ids[xs[i]], ids[ys[i]], 0))
            counter += [(ids[xs[i]], ids[ys[i]], 0)]

    # counter = counter/count
    # maxW = max([w for x, y, w in counter])

    return [(x, y, w) for x, y, w in counter]


def getDInteractions_between(dataframe, start_time, end_time, proximity):
    ids = dataframe.id.unique()

    counter = {}
    timestamps = {}
    interacting = {}
    nseconds = 1
    start_window = pd.Timestamp(start_time)
    while start_window <= pd.Timestamp(end_time):
        position = dataframe[str(start_window)].set_index("id").reindex(ids).reset_index()
        dists = distance.cdist(position[['x', 'y']], position[['x', 'y']], 'euclidean')

        # distances < proximity -> add 1 to that relationship
        dists = (np.array(dists) <= proximity) + 0
        xs, ys = np.where(dists > 0)
        for i in range(len(xs)):
            if xs[i] == ys[i]:
                continue
            key = (ids[xs[i]], ids[ys[i]])

            vel1X = float(position.loc[position['id'] == ids[xs[i]]].velX)
            vel1Y = float(position.loc[position['id'] == ids[xs[i]]].velY)

            vx = float(position.loc[position['id'] == ids[ys[i]]].x) - float(
                position.loc[position['id'] == ids[xs[i]]].x)
            vy = float(position.loc[position['id'] == ids[ys[i]]].y) - float(
                position.loc[position['id'] == ids[xs[i]]].y)

            cosine = 0

            if (vel1X * vx + vel1Y * vy) != 0:
                cosine = (vel1X * vx + vel1Y * vy) / (math.sqrt(vel1X ** 2 + vel1Y ** 2) * math.sqrt(vx ** 2 + vy ** 2))

            if cosine >= 0:
                if key in interacting and interacting[key] is False:
                    if key in counter:
                        # print(start_window, timestamps[(ids[xs[i]], ids[ys[i]])])
                        counter[key] += (start_window - timestamps[key]).seconds - 1
                        # print(counter[(ids[xs[i]], ids[ys[i]])])
                    else:
                        # counter[key] = (start_window - timestamps[key]).seconds - 1
                        counter[key] = 0
                interacting[key] = True
                timestamps[key] = start_window
            else:
                if key in interacting:
                    interacting[key] = False

        start_window = start_window + pd.Timedelta(seconds=nseconds)

    # maxW = max(list(counter.values()))

    # counter = counter/count
    return {key: value for key, value in counter.items()}


def getMultiDInteractions_between(dataframe, start_time, end_time, proximity, nseconds=1):
    ids = dataframe.id.unique()
    nids = len(ids)
    oldInter = np.zeros((nids, nids)) - np.ones((nids, nids))
    counter = []
    start_window = pd.Timestamp(start_time)
    timestamps = {}

    while start_window <= pd.Timestamp(end_time):
        position = dataframe[str(start_window)].set_index("id").reindex(ids).reset_index()
        dists = distance.cdist(position[['x', 'y']], position[['x', 'y']], 'euclidean')

        # distances < proximity -> add 1 to that relationship
        dists = (np.array(dists) <= proximity) + 0
        xs, ys = np.where(dists > 0)
        for i in range(len(xs)):
            if xs[i] == ys[i]:
                continue
            vel1E = float(position.loc[position['id'] == ids[xs[i]]].velE)
            vel1N = float(position.loc[position['id'] == ids[xs[i]]].velN)

            vx = float(position.loc[position['id'] == ids[ys[i]]].x) - float(
                position.loc[position['id'] == ids[xs[i]]].x)
            vy = float(position.loc[position['id'] == ids[ys[i]]].y) - float(
                position.loc[position['id'] == ids[xs[i]]].y)

            cosine = 0

            if (vel1E * vx + vel1N * vy) != 0:
                cosine = (vel1E * vx + vel1N * vy) / (math.sqrt(vel1E ** 2 + vel1N ** 2) * math.sqrt(vx ** 2 + vy ** 2))

            if cosine >= 0:  # following
                if oldInter[xs[i]][ys[i]] == -1:
                    if (ids[xs[i]], ids[ys[i]]) in timestamps:
                        oldInter[xs[i]][ys[i]] = (start_window - timestamps[(ids[xs[i]], ids[ys[i]])]).seconds
                        print(start_window, timestamps[(ids[xs[i]], ids[ys[i]])], oldInter[xs[i]][ys[i]])
                    else:
                        oldInter[xs[i]][ys[i]] = 0

            else:
                if oldInter[xs[i]][ys[i]] >= 0:
                    counter += [(ids[xs[i]], ids[ys[i]], oldInter[xs[i]][ys[i]])]
                    oldInter[xs[i]][ys[i]] = -1
                    timestamps[(ids[xs[i]], ids[ys[i]])] = start_window

        start_window = start_window + pd.Timedelta(seconds=nseconds)

    # add last edges (the ones that never stop existing)
    xs, ys = np.where(oldInter > 0)
    for i in range(len(xs)):
        counter += [(ids[xs[i]], ids[ys[i]], oldInter[xs[i]][ys[i]])]

    # counter = counter/count
    # maxW = max([w for x, y, w in counter])

    return [(x, y, w) for x, y, w in counter]


def getWEdges(counter):
    gedges = []
    for key in counter:
        x, y = key
        w = counter[key]
        gedges += [(x, y, w)]

    return gedges

def edgesInPDescription(G, P):
    edges = []
    nodes = set()
    for e in list(G.edges(data=True)):
        eInP = True
        for sel in P:
            if e[2][sel.attribute] != sel.value:
                eInP = False
                break

        if eInP:
            edges.append(e)
            nodes = nodes | {e[0], e[1]}

    return edges, nodes


def edgesInP(G, P):
    edges = []
    nodes = set()
    wsum = 0

    for e in list(G.edges(data=True)):
        eInP = True
        for sel in P:
            if e[2][sel.attribute] != sel.value:
                eInP = False
                break

        if eInP:
            edges.append(e)
            nodes = nodes | {e[0], e[1]}
            wsum += e[2]['weight']

    nEp = len(nodes)  # number of nodes covered by a pattern P
    nE = nEp * (nEp - 1)  # number of all possible edges

    if nE == 0:
        w = 0
    else:
        w = wsum / nE

    Gpattern = nx.DiGraph()
    Gpattern.add_nodes_from(list(nodes))
    Gpattern.add_edges_from(edges)

    pat = Pattern(P, Gpattern, w)

    return pat


def infoPats_nodes(listOfPatterns, dataset):
    col_names = ['Pattern', 'N', 'ids', 'Mean Weight', 'Score']
    my_df = pd.DataFrame(columns=col_names)
    for p in listOfPatterns:
        n = len(p.ids)
        tempids = len(dataset.iloc[p.ids, 0].unique())
        my_df = my_df.append({'Pattern': p.name, 'N': n, 'ids': tempids, 'Mean Weight': round(p.weight, 1),
                              'Score': round(p.quality, 1)}, ignore_index=True)

    return my_df
