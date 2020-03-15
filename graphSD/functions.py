import numpy as np
import pandas as pd
from scipy.spatial import distance
import math
import networkx as nx
from orangecontrib.associate.fpgrowth import *
import dicts
from plotly.offline import init_notebook_mode, iplot
from IPython.display import display, HTML
import matplotlib.pyplot as plt
import utils


class NominalSelector:
    def __init__(self, attribute, value):
        self.attribute = attribute
        self.value = value

    def __repr__(self):
        return "("+self.attribute+", " + str(self.value)+")"

    def __str__(self):
        return "("+self.attribute+", " + str(self.value)+")"

    def __eq__(self, other):
        return self.attribute == other.attribute and self.value == other.value

    def __lt__(self, other):
        if self.attribute != other.attribute:
            return self.attribute < other.attribute
        else:
            return self.value < other.value

    def __hash__(self):
        return hash(str(self))


class Pattern:
    def __init__(self, name, graph, weight):  # name has to be of type list of NominalSelector
        self.name = name
        self.graph = graph
        self.weight = weight
        self.quality = 0

    def __repr__(self):
        return str(self.name)

    def __str__(self):
        return str(self.name)

    def __eq__(self, other):
        set(self.name) == set(other.name)

    def __lt__(self, other):
        return self.quality < other.quality


def index_by_date(dataset, time_unit = "ms"):
    dataset['date'] = pd.to_datetime(dataset.utc, unit=time_unit)
    dataset.set_index('date', inplace=True)
    dataset = dataset.groupby('id').resample("S").mean().reset_index()
    dataset.set_index('date', inplace=True)

    return dataset


# dataset should have a column x, y, id and indexed by date
# (in datetime, grouped by 1 second).
# If you wish to index it and group it use indexByDate(dataset).
def get_dir_interactions(dataset, ids, start_time, end_time, proximity, n_seconds=1):
    nids = len(ids)
    counter = {}
    start_window = pd.Timestamp(start_time)

    while start_window <= pd.Timestamp(end_time):
        position = dataset[str(start_window)].set_index("id").reindex(ids).reset_index()
        dists = distance.cdist(position[['x', 'y']], position[['x', 'y']], 'euclidean')
        # print(position)

        # distances < proximity -> add 1 to that relationship
        dists = (np.array(dists) <= proximity) + 0
        xs, ys = np.where(dists > 0)
        for i in range(len(xs)):
            if xs[i] == ys[i]:
                continue
            vel1E = float(position.loc[position['id'] == ids[xs[i]]].velX)
            vel1N = float(position.loc[position['id'] == ids[xs[i]]].velY)

            vx = float(position.loc[position['id'] == ids[ys[i]]].x) - float(
                position.loc[position['id'] == ids[xs[i]]].x)
            vy = float(position.loc[position['id'] == ids[ys[i]]].y) - float(
                position.loc[position['id'] == ids[xs[i]]].y)

            cosine = 0

            if (vel1E * vx + vel1N * vy) != 0:
                cosine = (vel1E * vx + vel1N * vy) / (math.sqrt(vel1E ** 2 + vel1N ** 2) * math.sqrt(vx ** 2 + vy ** 2))

            if cosine >= 0:
                if (ids[xs[i]], ids[ys[i]]) in counter:
                    counter[(ids[xs[i]], ids[ys[i]])] += 1
                else:
                    counter[(ids[xs[i]], ids[ys[i]])] = 1

        start_window = start_window + pd.Timedelta(seconds=n_seconds)

    maxW = max(list(counter.values()))

    # counter = counter/count
    return {key: value / maxW for key, value in counter.items()}


def get_w_edges(counter):
    g_edges = []
    for key in counter:
        x, y = key
        w = counter[key]
        g_edges += [(x, y, w)]
    
    return g_edges


# dataset should have a column x, y, id and indexed by date
# (in datetime, grouped by 1 second).
# If you wish to index it and group it use indexByDate(dataset).
# This dataset should have a utc column

def create_directed_graph(dataset, start_time, end_time, proximity):
    ids = dataset.id.unique()
    G = nx.DiGraph()
    G.add_nodes_from(ids)
    counter = get_dir_interactions(dataset, start_time, end_time, proximity)
    G.add_weighted_edges_from(get_w_edges(counter))

    return G


# attributes is a list of cloumn names that should be considered for comparison
# ex: ['Gender', 'Age']
def set_comp_att_edges(G, dataset, nominalatts, numericatts):
    attr = {}
    transactions = []
    tr = []
    for e in list(G.edges()):
        tr = []
        nid1, nid2 = e
        eattr = {}
        for att in nominalatts:
            eattr[att] = str((dataset[dataset.id == nid1][att].item(), dataset[dataset.id == nid2][att].item()))
            tr.append(NominalSelector(att, eattr[att]))

        for att in numericatts:
            if dataset[dataset.id == nid1][att].item() == dataset[dataset.id == nid2][att].item():
                eattr[att] = "EQ"
            elif dataset[dataset.id == nid1][att].item() > dataset[dataset.id == nid2][att].item():
                eattr[att] = ">"
            else:
                eattr[att] = "<"
            tr.append(NominalSelector(att, eattr[att]))

        attr[e] = eattr
        transactions.append(tr)

    nx.set_edge_attributes(G, attr)
    return transactions


def set_from_att_edges(G, dataset, attributes):
    attr = {}
    transactions = []
    tr = []
    for e in list(G.edges()):
        tr = []
        nid1, nid2 = e
        eattr = {}
        for att in attributes:
            eattr[att] = dataset[dataset.id == nid1][att].item()
            tr.append(NominalSelector(att, eattr[att]))

        attr[e] = eattr
        transactions.append(tr)

    nx.set_edge_attributes(G, attr)
    return transactions


def set_to_att_edges(G, dataset, attributes):
    attr = {}
    transactions = []
    tr = []
    for e in list(G.edges()):
        tr = []
        nid1, nid2 = e
        eattr = {}
        for att in attributes:
            eattr[att] = dataset[dataset.id == nid2][att].item()
            tr.append(NominalSelector(att, eattr[att]))

        attr[e] = eattr
        transactions.append(tr)

    nx.set_edge_attributes(G, attr)
    return transactions


def get_z(val, mean, std):
    return (val - mean)/std


def get_freq_itemsets(transactions, minfreq = 1):
    intTransactionsDict = {}
    lastFrom = 1
    intT = []
    newTransactions = {}

    for transaction in transactions:
        temp = []
        for att in trans:
            if transaction in intTransactionsDict:
                return intTransactionsDict[transaction]
            intTransactionsDict[transaction] = lastFrom
            lastFrom += 1
            temp += [intTransactionsDict[transaction]]
        intT += [temp]

    inv_intTransactionsDict = {v: k for k, v in intTransactionsDict.items()}

    itemsets = list(frequent_itemsets(intT, minfreq))

    for fset, count in itemsets:
        first = True
        for n in fset:
            if first:
                temp = (inv_Dict[n],)
                first = False
            else:
                temp += (inv_Dict[n],)

        newTransactions[temp] = count

    return newTransactions


def edges_in_p(G, P):
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


# Measure quality of subgroups
def qs1aux(edges):
    nodes = set()
    wsum = 0
    for e in edges:
        nodes = nodes | {e[0], e[1]}
        wsum += e[2]['weight']
    nEp = len(nodes)  # number of nodes covered by a pattern P
    nE = nEp * (nEp - 1)  # number of all possible edges

    return wsum / nE, round(nE), round(nEp), nodes


def qs1(G, P, freq, nsamples):
    pat = edgesInP(G, P)
    # print('edges ', len(edges))

    totalE = round((len(list(G.nodes())) * (len(list(G.nodes())) - 1)))
    # print(P)
    # print(qres)
    # print(nodes)

    sample = []

    for r in range(nsamples):
        indxs = np.random.choice(range(totalE), len(list(pat.graph.edges())), replace=False)
        # print(indxs)
        randomE = [list(G.edges(data=True))[i] for i in indxs if i < len(list(G.edges()))]
        # print(len(randomE))
        tempres, tempnE, tempnEp, nodestemp = qs1aux(randomE)
        # print(tempres)
        sample = np.append(sample, [tempres])

    mean = np.mean(sample)
    # print('mean ',mean)
    std = np.std(sample)
    # print('std ',std)

    pat.quality = (pat.weight - mean) / std

    return pat


def treeQuality(G, nodes, qsfunc):
    qs = []
    for k, val in nodes.items():
        try:
            qs += [qsfunc(G, k, val, 100)]
        except ZeroDivisionError:
            continue

    return qs


def print_weighted_graph(g):
    ecolors = list(nx.get_edge_attributes(g, 'weight').values())
    pos = nx.circular_layout(g)
    nx.draw(g, pos, edge_color=ecolors,
            width=4, edge_cmap=plt.cm.Blues, with_labels=True, cmap=plt.cm.Reds)

    plt.show()


def info_pats(list_of_patterns):
    col_names = ['Pattern', 'Nodes', 'Edges', 'Mean Weight', 'Score']
    my_df = pd.DataFrame(columns=col_names)
    for p in list_of_patterns:
        nnodes = len(list(p.graph.nodes()))
        nedges = len(list(p.graph.edges()))
        my_df = my_df.append({'Pattern': p.name, 'Nodes': nnodes, 'Edges': nedges, 'Mean Weight': round(p.weight, 1),
                              'Score': round(p.quality, 1)}, ignore_index=True)

    return my_df


def getMultiDInteractions(dataset, start_time, end_time, proximity, nseconds=1):
    nids = len(ids)
    oldInter = np.zeros((nids, nids))
    counter = []
    start_window = pd.Timestamp(start_time)

    while start_window <= pd.Timestamp(end_time):
        position = dataset[str(start_window)].set_index("id").reindex(ids).reset_index()
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

        start_window = start_window + pd.Timedelta(seconds=nseconds)

    # add last edges (the ones that never stop existing)
    xs, ys = np.where(oldInter > 0)
    for i in range(len(xs)):
        counter += [(ids[xs[i]], ids[ys[i]], oldInter[xs[i]][ys[i]])]

    # counter = counter/count
    # maxW = max([w for x, y, w in counter])

    return [(x, y, w) for x, y, w in counter]


def add_network_metrics(g, social_data):
    hubs, auths = nx.hits(g)
    degC = nx.centrality.degree_centrality(g)
    inDeg = nx.centrality.in_degree_centrality(g)
    outDeg = nx.centrality.out_degree_centrality(g)
    eigC = nx.centrality.eigenvector_centrality(g)
    closeness = nx.centrality.closeness_centrality(g)
    betweeness = nx.centrality.betweenness_centrality(g)
    pagerank = nx.pagerank(g)

    social_data['hubs'] = utils.get_bins(3, list(hubs.values()))
    social_data['auths'] = utils.get_bins(3, list(auths.values()))
    social_data['degC'] = utils.getBins(3, list(degC.values()))
    social_data['outDeg'] = utils.get_bins(3, list(outDeg.values()))
    social_data['inDeg'] = utils.get_bins(3, list(inDeg.values()))
    social_data['eigC'] = utils.get_bins(3, list(eigC.values()))
    social_data['closeness'] = utils.get_bins(3, list(closeness.values()))
    social_data['betweeness'] = utils.get_bins(3, list(betweeness.values()))
    social_data['pagerank'] = utils.get_bins(3, list(pagerank.values()))
