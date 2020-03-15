import numpy as np
import pandas as pd
from scipy.spatial import distance
import math
import networkx as nx

from graphSD.utils import *

class NominalSelector:
    def __init__(self, attribute, value):
        self.attribute = attribute
        self.value = value
    def __repr__(self):
        return "("+self.attribute+", "+ str(self.value)+")"
    def __str__(self):
        return "("+self.attribute+", "+ str(self.value)+")"
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
    def __init__(self, name, graph, weight): #name has to be of type list of NominalSelector
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

def getDInteractions(dataframe, start_time, end_time, proximity):
    ids = dataframe.id.unique()
    nids = len(ids)
    counter = {}
    nseconds = 1
    start_window = pd.Timestamp(start_time)
    while start_window <= pd.Timestamp(end_time):
        position = dataframe[str(start_window)].set_index("id").reindex(ids).reset_index()
        dists = distance.cdist(position[['x','y']], position[['x','y']], 'euclidean')
        
        #distances < proximity -> add 1 to that relationship
        dists = (np.array(dists) <= proximity) + 0
        xs, ys = np.where(dists > 0)
        for i in range(len(xs)):
            if xs[i] == ys[i]:
                continue
            vel1E = float(position.loc[position['id'] == ids[xs[i]]].velE)
            vel1N = float(position.loc[position['id'] == ids[xs[i]]].velN)
            
            vx = float(position.loc[position['id'] == ids[ys[i]]].x) - float(position.loc[position['id'] == ids[xs[i]]].x)
            vy = float(position.loc[position['id'] == ids[ys[i]]].y) - float(position.loc[position['id'] == ids[xs[i]]].y)
            
            cosine = 0
            
            if (vel1E * vx + vel1N * vy) != 0:
                cosine = (vel1E * vx + vel1N * vy)/(math.sqrt(vel1E**2 + vel1N**2) * math.sqrt(vx**2 + vy**2))
            
            if cosine >= 0:
                if (ids[xs[i]], ids[ys[i]]) in counter:
                    counter[(ids[xs[i]], ids[ys[i]])] += 1
                else:
                    counter[(ids[xs[i]], ids[ys[i]])] = 1
                
        start_window = start_window + pd.Timedelta(seconds = nseconds)
        
    maxW = max(list(counter.values()))
    
    #counter = counter/count
    return {key: value for key, value in counter.items()}

def getWEdges(counter):
    gedges = []
    for key in counter:
        x, y = key
        w = counter[key]
        gedges += [(x,y,w)]
    
    return gedges

def createDiGraph(counter, ids):
    graph = nx.DiGraph()
    graph.add_nodes_from(ids)
    graph.add_weighted_edges_from(getWEdges(counter))

    return graph

def filterEdges(graph,n_bins):
    subedges = []
    weights_bins = getBins(n_bins, [edict['weight'] for e1,e2,edict in list(graph.edges(data=True))])
    i = 0
    for eid in list(graph.edges()):
        if weights_bins[i] == (n_bins - 1):
            subedges += [eid]
        i += 1
    print(subedges)
    return subedges

def setCompAttEdges(graph, dataframe, attributes):
    attr = {}
    transactions = []
    tr = []
    for e in list(graph.edges()):
        tr = []
        nid1, nid2 = e
        eattr = {}
        for att in attributes:
            if att == "Gender":
                eattr[att] = str((dataframe[dataframe.id == nid1][att].item(), dataframe[dataframe.id == nid2][att].item()))
            elif dataframe[dataframe.id == nid1][att].item() == dataframe[dataframe.id == nid2][att].item():
                eattr[att] = "EQ"
            elif dataframe[dataframe.id == nid1][att].item() > dataframe[dataframe.id == nid2][att].item():
                eattr[att] = ">"
            else:
                eattr[att] = "<"
            tr.append(NominalSelector(att, eattr[att]))
            
        attr[e] = eattr
        transactions.append(tr)
                
    nx.set_edge_attributes(graph, attr)
    return transactions

def setFromAttEdges(graph, dataframe, attributes):
    attr = {}
    transactions = []
    tr = []
    for e in list(graph.edges()):
        tr = []
        nid1, nid2 = e
        eattr = {}
        for att in attributes:
            eattr[att] = dataframe[dataframe.id == nid1][att].item()
            tr.append(NominalSelector(att, eattr[att]))
            
        attr[e] = eattr
        transactions.append(tr)
                
    nx.set_edge_attributes(graph, attr)
    return transactions

def setToAttEdges(graph, dataframe, attributes):
    attr = {}
    transactions = []
    tr = []
    for e in list(graph.edges()):
        tr = []
        nid1, nid2 = e
        eattr = {}
        for att in attributes:
            eattr[att] = dataframe[dataframe.id == nid2][att].item()
            tr.append(NominalSelector(att, eattr[att]))
            
        attr[e] = eattr
        transactions.append(tr)
                
    nx.set_edge_attributes(graph, attr)
    return transactions

def edgesInPDescription(G, P):
    edges = []
    nodes = set()
    for e in list(G.edges(data = True)):
        eInP = True
        for sel in P:
            if e[2][sel.attribute] != sel.value:
                eInP = False
                break
                
        if eInP:
            edges.append(e)
            nodes = nodes | {e[0],e[1]}
    
    return edges, nodes   

def edgesInP(G, P):
    edges = []
    nodes = set()
    wsum = 0

    for e in list(G.edges(data = True)):
        eInP = True
        for sel in P:
            if e[2][sel.attribute] != sel.value:
                eInP = False
                break
                
        if eInP:
            edges.append(e)
            nodes = nodes | {e[0],e[1]}
            wsum += e[2]['weight']
    
    nEp = len(nodes) # number of nodes covered by a pattern P
    nE = nEp*(nEp - 1) # number of all possible edges
    
    if nE == 0:
        w = 0
    else:
        w = wsum/nE
        
    Gpattern = nx.DiGraph()
    Gpattern.add_nodes_from(list(nodes))
    Gpattern.add_edges_from(edges)

    
    pat = Pattern(P, Gpattern, w)
    
    return pat

def infoPats(listOfPatterns):
    col_names =  ['Pattern', 'Nodes', 'in', 'out','Edges', 'Mean Weight', 'Score']
    my_df  = pd.DataFrame(columns = col_names)
    for p in listOfPatterns:
        nnodes = len(list(p.graph.nodes()))
        nedges = len(list(p.graph.edges()))
        in_nodes = len([y for (x,y)  in list(p.graph.in_degree()) if y > 0])
        out_nodes = len([y for (x,y)  in list(p.graph.out_degree()) if y > 0])
        my_df = my_df.append({'Pattern': p.name, 'Nodes': nnodes, 'in': in_nodes, 'out': out_nodes, 'Edges': nedges, 'Mean Weight': round(p.weight,1),'Score': round(p.quality,1)}, ignore_index=True)
            
    return my_df