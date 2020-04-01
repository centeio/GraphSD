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

def multiedgesInP(G, P):
    edges = []
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
        
    Gpattern = nx.MultiDiGraph()
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


def m(G, nodes):
    count = 0
    for n1 in nodes:
        for n2 in nodes:
            if n1 != n2:
                count += (G.number_of_edges(n1, n2) - 1)
    return count


# Measure quality of subgroups
def qSMaux(G, P):
    edges = []
    wsum = 0
    nodes = set()
    
    for e in list(G.edges(data = True)):
        eInP = True
        for sel in P:
            if e[2][sel.attribute] != sel.value:
                eInP = False
                break
                
        if eInP:
            edges.append(e)
            wsum += e[2]['weight']
            nodes = nodes | {e[0],e[1]}
        
    Gpattern = nx.MultiDiGraph()
    Gpattern.add_nodes_from(list(nodes))
    Gpattern.add_edges_from(edges)

    
    pat = Pattern(P, Gpattern, wsum)
    return pat

def qSM(G, P, nsamples):        
    totalE = round((len(list(G.nodes())) * (len(list(G.nodes())) - 1)))
    totalmE = m(G, G.nodes)

    pat = qSMaux(G, P)
    nodes = list(pat.graph.nodes())
    edges = list(pat.graph.edges())
    nEp = len(nodes) # number of nodes covered by a pattern P
    nE = nEp*(nEp - 1) # number of all possible edges in pattern P
    mE = m(G, nodes)
    
    qres = pat.weight/(nE+mE)
    pat.weight = qres
    #print(qres)
    #print(nodes)
    sample = []
    
    for r in range(nsamples):
        samplewsum = 0
        samplenodes = set()
        indxs = np.random.choice(range(totalE + totalmE), len(edges), replace = False) 
        #print(indxs)
        randomE = [list(G.edges(data = True))[i] for i in indxs if i < len(list(G.edges()))]
        #print(len(randomE))
        for e in randomE:
            samplewsum += e[2]['weight']
            samplenodes = samplenodes | {e[0],e[1]}
        #print(tempres)
        samplenEp = len(samplenodes)
        samplenE = samplenEp*(samplenEp - 1)
        
        samplemE = m(G, samplenodes)
        
        sample = np.append(sample, [samplewsum/(samplenE + samplemE)])
        
    mean = np.mean(sample)
    #print('mean ',mean)
    std = np.std(sample)
    #print('std ',std)
    
    quality = (qres - mean)/std
    
    pat.quality = quality
    
    return pat

def qPMaux(edges):
    wsum = 0
    for e in edges:
        wsum += e[2]['weight']

    w = wsum/len(edges)
    
    return w

def qPM(G, P, nsamples):    
    pat = multiedgesInP(G, P)
    #print('edges ', len(edges))
    
    totalE = len(list(G.edges()))
    #print(P)
    #print(qres)
    #print(nodes)
    
    sample = []
    
    lst = [edict['weight'] for n1,n2,edict in list(G.edges(data=True))]
    mean = reduce(lambda a, b: a + b, lst) / len(lst) 
    
    if mean > pat.weight:
        return 0
    
    np.random.seed(1234)
    for r in range(nsamples):
        indxs = np.random.choice(range(len(list(G.edges()))), len(list(pat.graph.edges())), replace = False) 
        #print(indxs)
        randomE = [list(G.edges(data = True))[i] for i in indxs]
        #print(len(randomE))
        tempres = qPMaux(randomE)
        #print(tempres)
        sample = np.append(sample, [tempres])
        
    mean = np.mean(sample)
    #print('mean ',mean)
    std = np.std(sample)
    #print('std ',std)
    
    pat.quality = (pat.weight - mean)/std
    
    return pat

def treeQuality(G, nodes,q):
    qs = []
    for k, val in nodes.items(): 
        patqs = q(G, k, 1000)
        if type(patqs) != int:
            qs += [q(G, k, 1000)]
    return qs





