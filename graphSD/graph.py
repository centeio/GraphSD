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
            vel1X = float(position.loc[position['id'] == ids[xs[i]]].velX)
            vel1Y = float(position.loc[position['id'] == ids[xs[i]]].velY)
            
            vx = float(position.loc[position['id'] == ids[ys[i]]].x) - float(position.loc[position['id'] == ids[xs[i]]].x)
            vy = float(position.loc[position['id'] == ids[ys[i]]].y) - float(position.loc[position['id'] == ids[xs[i]]].y)
            
            cosine = 0
            
            if (vel1X * vx + vel1Y * vy) != 0:
                cosine = (vel1X * vx + vel1Y * vy)/(math.sqrt(vel1X**2 + vel1Y**2) * math.sqrt(vx**2 + vy**2))
            
            if cosine >= 0:
                if (ids[xs[i]], ids[ys[i]]) in counter:
                    counter[(ids[xs[i]], ids[ys[i]])] += 1
                else:
                    counter[(ids[xs[i]], ids[ys[i]])] = 1
                
        start_window = start_window + pd.Timedelta(seconds = nseconds)
        
    #maxW = max(list(counter.values()))
    
    #counter = counter/count
    return {key: value for key, value in counter.items()}

def getDInteractions_all(dataframe, start_time, end_time, proximity):
    ids = dataframe.id.unique()
    nids = len(ids)
    counter = {}
    nseconds = 1
    start_window = pd.Timestamp(start_time)
    
    for id1 in ids:
        for id2 in ids:
            if id1 != id2:
                counter[(id1,id2)] = 0

    while start_window <= pd.Timestamp(end_time):
        position = dataframe[str(start_window)].set_index("id").reindex(ids).reset_index()
        dists = distance.cdist(position[['x','y']], position[['x','y']], 'euclidean')
        
        #distances < proximity -> add 1 to that relationship
        dists = (np.array(dists) <= proximity) + 0
        xs, ys = np.where(dists > 0)
        for i in range(len(xs)):
            if xs[i] == ys[i]:
                continue
            vel1X = float(position.loc[position['id'] == ids[xs[i]]].velX)
            vel1Y = float(position.loc[position['id'] == ids[xs[i]]].velY)
            
            vx = float(position.loc[position['id'] == ids[ys[i]]].x) - float(position.loc[position['id'] == ids[xs[i]]].x)
            vy = float(position.loc[position['id'] == ids[ys[i]]].y) - float(position.loc[position['id'] == ids[xs[i]]].y)
            
            cosine = 0
            
            if (vel1X * vx + vel1Y * vy) != 0:
                cosine = (vel1X * vx + vel1Y * vy)/(math.sqrt(vel1X**2 + vel1Y**2) * math.sqrt(vx**2 + vy**2))
            
            if cosine >= 0:
                counter[(ids[xs[i]], ids[ys[i]])] += 1

                
        start_window = start_window + pd.Timedelta(seconds = nseconds)
        
    #maxW = max(list(counter.values()))
    
    #counter = counter/count
    
    return {key: value for key, value in counter.items()}

def getMultiDInteractions(dataframe, start_time, end_time, proximity, nseconds = 1):
    ids = dataframe.id.unique()
    nids = len(ids)
    oldInter = np.zeros((nids, nids))
    counter = []
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
            vel1X = float(position.loc[position['id'] == ids[xs[i]]].velX)
            vel1Y = float(position.loc[position['id'] == ids[xs[i]]].velY)
            
            vx = float(position.loc[position['id'] == ids[ys[i]]].x) - float(position.loc[position['id'] == ids[xs[i]]].x)
            vy = float(position.loc[position['id'] == ids[ys[i]]].y) - float(position.loc[position['id'] == ids[xs[i]]].y)
            
            cosine = 0
            
            if (vel1X * vx + vel1Y * vy) != 0:
                cosine = (vel1X * vx + vel1Y * vy)/(math.sqrt(vel1X**2 + vel1Y**2) * math.sqrt(vx**2 + vy**2))
            
            if cosine >= 0: # following
                oldInter[xs[i]][ys[i]] += 1
            else:
                if oldInter[xs[i]][ys[i]] > 0:
                    counter += [(ids[xs[i]], ids[ys[i]], oldInter[xs[i]][ys[i]])]
                    oldInter[xs[i]][ys[i]] = 0
            
        start_window = start_window + pd.Timedelta(seconds = nseconds)
        
    #add last edges (the ones that never stop existing)  
    xs, ys = np.where(oldInter > 0)
    for i in range(len(xs)):
        counter += [(ids[xs[i]], ids[ys[i]], oldInter[xs[i]][ys[i]])]
    
    #counter = counter/count
    #maxW = max([w for x, y, w in counter])
    
    return [(x,y,w) for x, y, w in counter]

def getMultiDInteractions_all(dataframe, start_time, end_time, proximity, nseconds = 1):
    ids = dataframe.id.unique()
    nids = len(ids)
    oldInter = np.zeros((nids, nids))
    inter = np.zeros((nids, nids))

    counter = []
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
            vel1X = float(position.loc[position['id'] == ids[xs[i]]].velX)
            vel1Y = float(position.loc[position['id'] == ids[xs[i]]].velY)
            
            vx = float(position.loc[position['id'] == ids[ys[i]]].x) - float(position.loc[position['id'] == ids[xs[i]]].x)
            vy = float(position.loc[position['id'] == ids[ys[i]]].y) - float(position.loc[position['id'] == ids[xs[i]]].y)
            
            cosine = 0
            
            if (vel1X * vx + vel1Y * vy) != 0:
                cosine = (vel1X * vx + vel1Y * vy)/(math.sqrt(vel1X**2 + vel1Y**2) * math.sqrt(vx**2 + vy**2))
            
            if cosine >= 0: # following
                oldInter[xs[i]][ys[i]] += 1
            else:
                if oldInter[xs[i]][ys[i]] > 0:
                    counter += [(ids[xs[i]], ids[ys[i]], oldInter[xs[i]][ys[i]])]
                    inter[xs[i]][ys[i]] = 1
                    oldInter[xs[i]][ys[i]] = 0
            
        start_window = start_window + pd.Timedelta(seconds = nseconds)
        
    #add last edges (the ones that never stop existing)  
    xs, ys = np.where(oldInter > 0)
    for i in range(len(xs)):
        counter += [(ids[xs[i]], ids[ys[i]], oldInter[xs[i]][ys[i]])]
        inter[xs[i]][ys[i]] = 1
        
    xs, ys = np.where(inter == 0)
    for i in range(len(xs)):
        if ids[xs[i]] != ids[ys[i]]:
            #print((ids[xs[i]], ids[ys[i]], 0))
            counter += [(ids[xs[i]], ids[ys[i]], 0)]

    #counter = counter/count
    maxW = max([w for x, y, w in counter])
    
    return [(x,y,w) for x, y, w in counter]

def getWEdges(counter):
    gedges = []
    for key in counter:
        x, y = key
        w = counter[key]
        gedges += [(x,y,w)]
    
    return gedges

def createGraph(counter, ids):
    # counter must be of type tuple (e1, e2, w)

    graph = nx.Graph()
    graph.add_nodes_from(ids)
    graph.add_weighted_edges_from(counter)

    return graph

def createDiGraph(counter, ids):
    graph = nx.DiGraph()
    graph.add_nodes_from(ids)
    graph.add_weighted_edges_from(getWEdges(counter))

    return graph

def createMultiDiGraph(counter, ids):
    graph = nx.MultiDiGraph()
    graph.add_nodes_from(ids)
    graph.add_weighted_edges_from(counter)

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

def setCompAttEdges(graph, dataframe, attributes, GL = None):
    attr = {}
    transactions = []
    tr = []
    for e in list(graph.edges()):
        tr = []
        nid1, nid2 = e
        eattr = {}
        for att in attributes:
            if dataframe[dataframe.id == nid1][att].item() == dataframe[dataframe.id == nid2][att].item():
                eattr[att] = "EQ"
            else:
                eattr[att] = "NEQ"
            tr.append(NominalSelector(att, eattr[att]))
        
        if GL != None:
            if GL.has_edge(nid1,nid2):
                eattr['weight'] = 1
            else:
                eattr['weight'] = 0
            
        attr[e] = eattr
        transactions.append(tr)
                
    nx.set_edge_attributes(graph, attr)
    return transactions


def setCompAttDiEdges(graph, dataframe, attributes, GL = None):
    attr = {}
    transactions = []
    tr = []
    for e in list(graph.edges()):
        tr = []
        nid1, nid2 = e
        eattr = {}
        for att in attributes:
            print(att, type(dataframe[dataframe.id == nid1][att].item()))
            if type(dataframe[dataframe.id == nid1][att].item()) == type('str'):
                print('here')
                eattr[att] = str((dataframe[dataframe.id == nid1][att].item(), dataframe[dataframe.id == nid2][att].item()))
            elif dataframe[dataframe.id == nid1][att].item() == dataframe[dataframe.id == nid2][att].item():
                eattr[att] = "EQ"
            elif dataframe[dataframe.id == nid1][att].item() > dataframe[dataframe.id == nid2][att].item():
                eattr[att] = ">"
            else:
                eattr[att] = "<"
            tr.append(NominalSelector(att, eattr[att]))
        
        if GL != None:
            if GL.has_edge(nid1,nid2):
                eattr['weight'] = 1
            else:
                eattr['weight'] = 0
            
        attr[e] = eattr
        transactions.append(tr)
                
    nx.set_edge_attributes(graph, attr)

    return transactions

def setFromAttDiEdges(graph, dataframe, attributes, GL = None):

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
                
        if GL != None:
            if GL.has_edge(nid1,nid2):
                eattr['weight'] = 1
            else:
                eattr['weight'] = 0

    nx.set_edge_attributes(graph, attr)

    return transactions

def setToAttDiEdges(graph, dataframe, attributes, GL = None):
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

        if GL != None:
            if GL.has_edge(nid1,nid2):
                eattr['weight'] = 1
            else:
                eattr['weight'] = 0    

    nx.set_edge_attributes(graph, attr)
    return transactions

def setMultiCompAttDiEdges(G, demogdata, attributes, GL = None):
    attr = {}
    transactions = []
    tr = []
    i = 0
    for e in list(G.edges(keys = True, data = True)):
        tr = []
        nid1, nid2, ekey, edict = e
        #eattr = {}
        for att in attributes:
            if att == "Gender":
                #eattr[att] = str((demogdata[demogdata.id == nid1][att].item(), demogdata[demogdata.id == nid2][att].item()))
                edict[att] = str((demogdata[demogdata.id == nid1][att].item(), demogdata[demogdata.id == nid2][att].item()))
            elif demogdata[demogdata.id == nid1][att].item() == demogdata[demogdata.id == nid2][att].item():
                #eattr[att] = "EQ"
                edict[att] = "EQ"
            elif demogdata[demogdata.id == nid1][att].item() > demogdata[demogdata.id == nid2][att].item():
                #eattr[att] = ">"
                edict[att] = ">"
            else:
                #eattr[att] = "<"
                edict[att] = "<"
            tr.append(NominalSelector(att, edict[att]))

        if GL != None:
            if GL.has_edge(nid1,nid2):
                edict['weight'] = 1
            else:
                edict['weight'] = 0
            
        #attr[e] = eattr
        transactions.append(tr)
        i += 1
                
    #nx.set_edge_attributes(G, attr)
    return transactions

def setMultiFromAttDiEdges(G, demogdata, attributes, GL = None):
    attr = {}
    transactions = []
    tr = []
    i = 0
    for e in list(G.edges(keys = True, data = True)):
        tr = []
        nid1, nid2, ekey, edict = e
        #eattr = {}
        for att in attributes:
            #eattr[att] = demogdata[demogdata.id == nid1][att].item()
            edict[att] = demogdata[demogdata.id == nid1][att].item()
            
            tr.append(NominalSelector(att, edict[att]))
            
        #attr[e] = eattr
        transactions.append(tr)
        i += 1
                
        if GL != None:
            if GL.has_edge(nid1,nid2):
                edict['weight'] = 1
            else:
                edict['weight'] = 0

    #nx.set_edge_attributes(G, attr)
    return transactions

def setMultiToAttDiEdges(G, demogdata, attributes, GL = None):
    attr = {}
    transactions = []
    tr = []
    i = 0
    for e in list(G.edges(keys = True, data = True)):
        tr = []
        nid1, nid2, ekey, edict = e
        #eattr = {}
        for att in attributes:
            #eattr[att] = demogdata[demogdata.id == nid2][att].item()
            edict[att] = demogdata[demogdata.id == nid2][att].item()
                
            tr.append(NominalSelector(att, edict[att]))
            
        #attr[e] = eattr
        transactions.append(tr)
        i += 1

        if GL != None:
            if GL.has_edge(nid1,nid2):
                edict['weight'] = 1
            else:
                edict['weight'] = 0

    #nx.set_edge_attributes(G, attr)
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
        if type(p) == Pattern :
            nnodes = len(list(p.graph.nodes()))
            nedges = len(list(p.graph.edges()))
            in_nodes = len([y for (x,y)  in list(p.graph.in_degree()) if y > 0])
            out_nodes = len([y for (x,y)  in list(p.graph.out_degree()) if y > 0])
            my_df = my_df.append({'Pattern': p.name, 'Nodes': nnodes, 'in': in_nodes, 'out': out_nodes, 'Edges': nedges, 'Mean Weight': round(p.weight,1),'Score': round(p.quality,1)}, ignore_index=True)
                
    return my_df