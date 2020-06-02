import numpy as np
import pandas as pd
import networkx as nx
import multiprocessing as mp
from multiprocessing.dummy import Pool as ThreadPool


from graphSD.graph import *

def edgesOfP(G, P):
    edges = []

    for e in list(G.edges(data = True)):
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


def qSaux(G, edges, multi, metric = 'mean'):
    nodes = set()
    weights = []
    for e in edges:
        nodes = nodes | {e[0],e[1]}
        weights += [e[2]['weight']]
    nEp = (len(nodes) * 1.0) # number of nodes covered by a pattern P
    nE = nEp*(nEp - 1) # number of all possible edges
    
    if nE == 0:
        w = 0
    else:
        if multi == True:
            nE += m(G, nodes)
        mean = sum(weights)/nE
        if metric == 'mean':
            w = mean
        elif metric == 'var':
            var = sum(abs(np.array(weights) - mean)**2)/nE
            w = var
    return w


def qS(G, P, nsamples, metric = 'mean'):    
    edges = edgesOfP(G, P)
    multi = False
    totalE = round((len(list(G.nodes())) * (len(list(G.nodes())) - 1)))

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

    Gpattern.add_edges_from(edges)

    w = qSaux(G, edges, multi, metric)

    pat = Pattern(P, Gpattern, w)
    
    sample = []

    pool = ThreadPool(2)
    
    for r in range(nsamples):
        indxs = np.random.choice(range(totalE), len(edges), replace = False) 
        randomE = [list(G.edges(data = True))[i] for i in indxs if i < len(list(G.edges()))]
        #tempres = qSaux(G, randomE, multi, metric)
        #sample = np.append(sample, [tempres])
        pool.apply_async(qSaux, args=(G, randomE, multi, metric), callback=sample.append)

    pool.close()
    pool.join()
        
    mean = np.mean(sample)
    #print('mean ',mean)
    std = np.std(sample)
    #print('std ',std)
    
    #print(sample, mean, std)
    pat.quality = (pat.weight - mean)/std

    return pat


def qPaux(edges, metric = 'mean'):
    weights = [e[2]['weight'] for e in edges]

    if metric == 'mean':
        w = np.mean(weights)
    elif metric == 'var':
        w = np.var(weights)
    return w


def qP(G, P, nsamples, metric = 'mean'):    
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

    totalE = len(list(G.edges()))

    sample = []
    
    #lst = [edict['weight'] for n1,n2,edict in list(G.edges(data=True))]
    #mean = reduce(lambda a, b: a + b, lst) / len(lst) 
    
    # For only positive Z value
    #if mean > pat.weight:
    #    return 0

    pool = ThreadPool(2)

    for r in range(nsamples):
        indxs = np.random.choice(range(len(list(G.edges()))), len(list(pat.graph.edges())), replace = False) 
        randomE = [list(G.edges(data = True))[i] for i in indxs]
        #tempres = qPaux(randomE, metric)
        #sample = np.append(sample, [tempres])
        pool.apply_async(qPaux, args=(randomE, metric), callback=sample.append)

    pool.close()
    pool.join()
        
    mean = np.mean(sample)
    #print('mean ',mean)
    std = np.std(sample)
    #print('std ',std)
    
    pat.quality = (pat.weight - mean)/std
    
    return pat    


def treeQuality(G, nodes, q, metric = 'mean', multiprocess = True, samples = 1000):
    if multiprocess:
        pool = mp.Pool(mp.cpu_count())
        qs = []
        for k,_ in nodes.items():
            pool.apply_async(q, args=(G, k, samples, metric), callback=qs.append)
        pool.close()
        pool.join()
    else:
        qs = [q(G, k, 1000, metric) for k,_ in nodes.items()]

    return qs





