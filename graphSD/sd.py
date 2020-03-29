import numpy as np
import pandas as pd
import networkx as nx

from graphSD.graph import *

def edgesInP(G, P):
    edges = []
    nodes_in = set()
    nodes_out = set()
    wsum = 0
    nE = 0

    for e in list(G.edges(data = True)):
        eInP = True
        for sel in P:
            if e[2][sel.attribute] != sel.value:
                eInP = False
                break
                
        if eInP:
            nE += 1
            edges.append(e)
            wsum += e[2]['weight']
    
    w = wsum/nE
        
    Gpattern = nx.DiGraph()
    Gpattern.add_edges_from(edges)
    
    pat = Pattern(P, Gpattern, w)
    
    return pat

def qSDaux(edges):
    nodes = set()
    wsum = 0
    for e in edges:
        nodes = nodes | {e[0],e[1]}
        wsum += e[2]['weight']
    nEp = len(nodes) # number of nodes covered by a pattern P
    nE = nEp*(nEp - 1) # number of all possible edges
    
    if nE == 0:
        w = 0
    else:
        w = wsum/nE
    
    return w, round(nE), round(nEp), nodes

def qSD(G, P, nsamples):    
    pat = edgesInP(G, P)
    #print('edges ', len(edges))
    
    totalE = round((len(list(G.nodes())) * (len(list(G.nodes())) - 1)))
    #print(P)
    #print(qres)
    #print(nodes)
    
    sample = []
    
    for r in range(nsamples):
        indxs = np.random.choice(range(totalE), len(list(pat.graph.edges())), replace = False) 
        #print(indxs)
        randomE = [list(G.edges(data = True))[i] for i in indxs if i < len(list(G.edges()))]
        #print(len(randomE))
        tempres, tempnE, tempnEp, nodestemp = qSDaux(randomE)
        #print(tempres)
        sample = np.append(sample, [tempres])
        
    mean = np.mean(sample)
    #print('mean ',mean)
    std = np.std(sample)
    #print('std ',std)
    
    pat.quality = (pat.weight - mean)/std
    return pat

def qPDaux(edges):
    wsum = 0
    for e in edges:
        wsum += e[2]['weight']

    w = wsum/len(edges)
    
    return w

def qPD(G, P, nsamples):    
    pat = edgesInP(G, P)
    #print('edges ', len(edges))
    
    totalE = len(list(G.edges()))
    #print(P)
    #print(qres)
    #print(nodes)
    
    sample = []
    
    lst = [edict['weight'] for n1,n2,edict in list(G.edges(data=True))]
    mean = reduce(lambda a, b: a + b, lst) / len(lst) 
    
    # For only positive Z value
    #if mean > pat.weight:
    #    return 0
    
    for r in range(nsamples):
        indxs = np.random.choice(range(len(list(G.edges()))), len(list(pat.graph.edges())), replace = False) 
        #print(indxs)
        randomE = [list(G.edges(data = True))[i] for i in indxs]
        #print(len(randomE))
        tempres = qPDaux(randomE)
        #print(tempres)
        sample = np.append(sample, [tempres])
        
    mean = np.mean(sample)
    #print('mean ',mean)
    std = np.std(sample)
    #print('std ',std)
    
    pat.quality = (pat.weight - mean)/std
    
    return pat    

#def treeQuality_(G, nodes, q):
#    qs = []
#    for k, val in nodes.items(): 
#        try:
#            qs += [q(G, k, 1000)]
#        except ZeroDivisionError:
#            continue
#   
#    return qs

def treeQuality(G, nodes,q):
    qs = []
    for k, val in nodes.items(): 
        patqs = q(G, k, 1000)
        if type(patqs) != int:
            qs += [q(G, k, 1000)]
    return qs





