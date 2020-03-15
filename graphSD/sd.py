import numpy as np
import pandas as pd
import networkx as nx

from graphSD.graph import *


# Measure quality of subgroups
def qs1aux(edges):
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

def qs1(G, P, nsamples):    
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
        tempres, tempnE, tempnEp, nodestemp = qs1aux(randomE)
        #print(tempres)
        sample = np.append(sample, [tempres])
        
    mean = np.mean(sample)
    #print('mean ',mean)
    std = np.std(sample)
    #print('std ',std)
    
    pat.quality = (pat.weight - mean)/std
    return pat

def treeQuality(G, nodes):
    qs = []
    for k, val in nodes.items(): 
        try:
            qs += [qs1(G, k, 1000)]
        except ZeroDivisionError:
            continue
   
    return qs





