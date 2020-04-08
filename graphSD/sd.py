import numpy as np
import pandas as pd
import networkx as nx

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


def qSD(G, P, nsamples, metric = 'mean'):    
    edges = edgesOfP(G, P)
    Gpattern = nx.DiGraph()
    Gpattern.add_edges_from(edges)

    w = qSaux(G, edges, False, metric)

    pat = Pattern(P, Gpattern, w)
    
    totalE = round((len(list(G.nodes())) * (len(list(G.nodes())) - 1)))

    
    sample = []
    
    for r in range(nsamples):
        indxs = np.random.choice(range(totalE), len(list(pat.graph.edges())), replace = False) 
        randomE = [list(G.edges(data = True))[i] for i in indxs if i < len(list(G.edges()))]
        tempres, tempnE, tempnEp, nodestemp = qSDaux(randomE)
        sample = np.append(sample, [tempres])
        
    mean = np.mean(sample)
    #print('mean ',mean)
    std = np.std(sample)
    #print('std ',std)
    
    pat.quality = (pat.weight - mean)/std
    return pat


def qPaux(edges, metric = 'mean'):
    weights = [e[2]['weight'] for e in edges]

    if metric == 'mean':
        w = np.mean(weights)
    elif metric == 'var':
        w = np.var(weights)
    return w


def qPD(G, P, nsamples, metric = 'mean'):    
    edges = edgesOfP(G, P)
    Gpattern = nx.DiGraph()
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
    
    for r in range(nsamples):
        indxs = np.random.choice(range(len(list(G.edges()))), len(list(pat.graph.edges())), replace = False) 
        randomE = [list(G.edges(data = True))[i] for i in indxs]
        tempres = qPaux(randomE)
        sample = np.append(sample, [tempres])
        
    mean = np.mean(sample)
    #print('mean ',mean)
    std = np.std(sample)
    #print('std ',std)
    
    pat.quality = (pat.weight - mean)/std
    
    return pat    


def qSM(G, P, nsamples, metric = 'mean'):
    edges = edgesOfP(G, P)
    Gpattern = nx.MultiDiGraph()
    Gpattern.add_edges_from(edges)
    #print(edges)

    w = qSaux(G = G, edges = edges, multi = True, metric = metric)

    pat = Pattern(P, Gpattern, w)        
    totalE = round((len(list(G.nodes())) * (len(list(G.nodes())) - 1)))
    totalmE = m(G, G.nodes)

    sample = []
    
    for r in range(nsamples):
        indxs = np.random.choice(range(totalE + totalmE), len(edges), replace = False) 
        randomE = [list(G.edges(data = True))[i] for i in indxs if i < len(list(G.edges()))]
        sample = np.append(sample, [qSaux(G, randomE, True, metric)])
        
    mean = np.mean(sample)
    #print('mean ',mean)
    std = np.std(sample)
    #print('std ',std)
    
    quality = (qres - mean)/std
    
    pat.quality = quality
    
    return pat


def qPM(G, P, nsamples, metric = 'mean'):    
    edges = edgesOfP(G, P)
    Gpattern = nx.MultiDiGraph()
    Gpattern.add_edges_from(edges)
    #print(edges)

    w = qPaux(edges, metric)
    pat = Pattern(P, Gpattern, w)
    
    totalE = len(list(G.edges()))
    #print(P)
    #print(qres)
    #print(nodes)
    
    sample = []
    
    #lst = [edict['weight'] for n1,n2,edict in list(G.edges(data=True))]
    #mean = reduce(lambda a, b: a + b, lst) / len(lst) 
    
    #if mean > pat.weight:
    #    return 0
    
    for r in range(nsamples):
        indxs = np.random.choice(range(len(list(G.edges()))), len(list(pat.graph.edges())), replace = False) 
        #print(indxs)
        randomE = [list(G.edges(data = True))[i] for i in indxs]
        #print(len(randomE))
        tempres = qPaux(randomE)
        #print(tempres)
        sample = np.append(sample, [tempres])
        
    mean = np.mean(sample)
    #print('mean ',mean)
    std = np.std(sample)
    #print('std ',std)
    
    pat.quality = (pat.weight - mean)/std
    
    return pat


def treeQuality(G, nodes,q, metric = 'mean'):
    qs = []
    for k, val in nodes.items(): 
        patqs = q(G, k, 1000, metric)
        if type(patqs) != int:
            qs += [q(G, k, 1000)]
    return qs





