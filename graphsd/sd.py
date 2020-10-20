import numpy as np
import pandas as pd
import networkx as nx
import multiprocessing as mp
from multiprocessing.dummy import Pool as ThreadPool

from graphsd.graph import *


def edgesOfP(G, P):
    edges = []

    for e in list(G.edges(data=True)):
        eInP = True
        for sel in P:
            if e[2][sel.attribute] != sel.value:
                eInP = False
                break

        if eInP:
            edges.append(e)

    return edges


def m(G, nodes):
    count = 0
    for n1 in nodes:
        for n2 in nodes:
            if n1 != n2:
                count += (G.number_of_edges(n1, n2) - 1)
    return count


def qSaux(G, edges, multi, metric='mean'):
    nodes = set()
    weights = []
    for e in edges:
        nodes = nodes | {e[0], e[1]}
        weights += [e[2]['weight']]
    nEp = (len(nodes) * 1.0)  # number of nodes covered by a pattern P
    nE = nEp * (nEp - 1)  # number of all possible edges

    if nE == 0:
        w = 0
    else:
        if multi is True:
            nE += m(G, nodes)
        mean = sum(weights) / nE
        if metric == 'mean':
            w = mean
        elif metric == 'var':
            var = sum(abs(np.array(weights) - mean) ** 2) / nE
            w = var
    return w


def qS(G, P, nsamples, metric='mean'):

    multi = False
    totalE = G.number_of_nodes() * (G.number_of_nodes() - 1)

    if type(G) == nx.Graph:
        Gpattern = nx.Graph()
    elif type(G) == nx.DiGraph:
        Gpattern = nx.DiGraph()
    elif type(G) == nx.MultiGraph:
        Gpattern = nx.MultiGraph()
        multi = True
        totalE += m(G, G.nodes)
    elif type(G) == nx.MultiDiGraph:
        Gpattern = nx.MultiDiGraph()
        multi = True
        totalE += m(G, G.nodes)

    edges = edgesOfP(G, P)
    Gpattern.add_edges_from(edges)

    w = qSaux(G, edges, multi, metric)

    pat = Pattern(P, Gpattern, w)

    sample = []

    pool = ThreadPool(2)

    for r in range(nsamples):
        indxs = np.random.choice(range(totalE), len(edges), replace=False)
        randomE = [list(G.edges(data=True))[i] for i in indxs if i < len(list(G.edges()))]
        pool.apply_async(qSaux, args=(G, randomE, multi, metric), callback=sample.append)

    pool.close()
    pool.join()

    mean = np.mean(sample)
    std = np.std(sample)

    pat.quality = (pat.weight - mean) / std

    return pat


def qPaux(edges, metric='mean'):
    weights = [e[2]['weight'] for e in edges]

    if metric == 'mean':
        w = np.mean(weights)
    elif metric == 'var':
        w = np.var(weights)
    return w


def qP(G, P, nsamples, metric='mean'):
    edges = edgesOfP(G, P)
    multi = False

    if type(G) == nx.Graph:
        Gpattern = nx.Graph()
    elif type(G) == nx.DiGraph:
        Gpattern = nx.DiGraph()
    elif type(G) == nx.MultiGraph:
        Gpattern = nx.MultiGraph()
        multi = True
    elif type(G) == nx.MultiDiGraph:
        Gpattern = nx.MultiDiGraph()
        multi = True

    Gpattern.add_edges_from(edges)

    w = qPaux(edges, metric)
    pat = Pattern(P, Gpattern, w)

    sample = []

    pool = ThreadPool(2)

    for r in range(nsamples):
        print("sample:", r)
        indxs = np.random.choice(range(len(list(G.edges()))), len(list(pat.graph.edges())), replace=False)
        randomE = [list(G.edges(data=True))[i] for i in indxs]
        pool.apply_async(qPaux, args=(randomE, metric), callback=sample.append)

    pool.close()
    pool.join()

    mean = np.mean(sample)
    std = np.std(sample)

    pat.quality = (pat.weight - mean) / std

    return pat


##############################
##### Outlier Detection ######
##############################


def idsInP(dataset, P, target):
    pids = []
    wsum = 0

    query = ""
    first = True

    for sel in P:
        if first:
            first = False
        else:
            query += " & "
        if type(sel.value) == str:
            query += sel.attribute + " == '" + sel.value + "'"
        else:
            query += sel.attribute + " == " + str(sel.value)

    print(query)

    for index, row in dataset.query(query).iterrows():
        pids += [index]
        wsum += row[target]

    nEp = len(pids)  # number of nodes covered by a pattern P
    nE = len(dataset)  # number of all possible edges

    if nE == 0:
        w = 0
    else:
        w = wsum / nEp

    print(w)
    pat = NoGPattern(P, pids, w)

    return pat


# Measure quality of subgroups
def qs_nodes_aux(dataframe, eids, target):
    wsum = 0

    for index, row in dataframe.iloc[eids, :].iterrows():
        wsum += row[target]

    return wsum / len(eids)


def qs_nodes(dataset, P, freq, nsamples, target):
    pat = idsInP(dataset, P, target)
    # print('edges ', len(edges))

    # totalE = round((len(list(G.nodes())) * (len(list(G.nodes())) - 1)))
    # print(P)
    # print(qres)
    # print(nodes)

    temp_ids = dataset.index.values.tolist()

    sample = []

    # print('P ', P)
    # print('n: ', pat.ids)

    for r in range(nsamples):
        indxs = np.random.choice(temp_ids, len(pat.ids), replace=False)
        # print(indxs)
        # print(len(randomE))
        tempres = qs_nodes_aux(dataset, indxs, target)
        # print(tempres)
        sample = np.append(sample, [tempres])

    mean = np.mean(sample)
    # print('mean ',mean)
    std = np.std(sample)
    # print('std ',std)
    # print('weight ', pat.weight)

    pat.quality = (pat.weight - mean) / std
    # print(pat)
    return pat


def treeQuality_nodes(dataset, nodes, target):
    dataset2 = dataset.copy()
    dataset2 = dataset2.reset_index()
    qs = []
    for k, val in nodes.items():
        try:
            pat = qs_nodes(dataset2, k, val, 100, target)
            qs += [pat]
        except ZeroDivisionError:
            continue

    return qs
