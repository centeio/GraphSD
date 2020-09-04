import numpy as np
import pandas as pd
from scipy.spatial import distance
import math
import networkx as nx

from graphSD.utils import *

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
    #maxW = max([w for x, y, w in counter])
    
    return [(x,y,w) for x, y, w in counter]


def getDInteractions_between(dataframe, start_time, end_time, proximity):
    ids = dataframe.id.unique()
    nids = len(ids)
    counter = {}
    timestamps = {}
    interacting = {}
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
            key = (ids[xs[i]], ids[ys[i]]) 
            
            vel1X = float(position.loc[position['id'] == ids[xs[i]]].velX)
            vel1Y = float(position.loc[position['id'] == ids[xs[i]]].velY)
            
            vx = float(position.loc[position['id'] == ids[ys[i]]].x) - float(position.loc[position['id'] == ids[xs[i]]].x)
            vy = float(position.loc[position['id'] == ids[ys[i]]].y) - float(position.loc[position['id'] == ids[xs[i]]].y)
            
            cosine = 0
            
            if (vel1X * vx + vel1Y * vy) != 0:
                cosine = (vel1X * vx + vel1Y * vy)/(math.sqrt(vel1X**2 + vel1Y**2) * math.sqrt(vx**2 + vy**2))
            
            if cosine >= 0:
                if key in interacting and interacting[key] == False:
                    if key in counter:
                        #print(start_window, timestamps[(ids[xs[i]], ids[ys[i]])])
                        counter[key] += (start_window - timestamps[key]).seconds - 1
                        #print(counter[(ids[xs[i]], ids[ys[i]])])
                    else:
                        #counter[key] = (start_window - timestamps[key]).seconds - 1
                        counter[key] = 0
                interacting[key] = True
                timestamps[key] = start_window
            else:
                if key in interacting:
                    interacting[key] = False
                
        start_window = start_window + pd.Timedelta(seconds = nseconds)
        
    #maxW = max(list(counter.values()))
    
    #counter = counter/count
    return {key: value for key, value in counter.items()}

def getMultiDInteractions_between(dataframe, start_time, end_time, proximity, nseconds = 1):
    ids = dataframe.id.unique()
    nids = len(ids)
    oldInter = np.zeros((nids, nids)) - np.ones((nids, nids))
    counter = []
    start_window = pd.Timestamp(start_time)
    timestamps = {}
    
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
            
            if cosine >= 0: # following
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
            
        start_window = start_window + pd.Timedelta(seconds = nseconds)
        
    #add last edges (the ones that never stop existing)
    xs, ys = np.where(oldInter > 0)
    for i in range(len(xs)):
        counter += [(ids[xs[i]], ids[ys[i]], oldInter[xs[i]][ys[i]])]
    
    #counter = counter/count
    #maxW = max([w for x, y, w in counter])
    
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
            if not GL.has_edge(nid1,nid2):
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
            if type(dataframe[dataframe.id == nid1][att].item()) == type('str'):
                eattr[att] = str((dataframe[dataframe.id == nid1][att].item(), dataframe[dataframe.id == nid2][att].item()))
            elif dataframe[dataframe.id == nid1][att].item() == dataframe[dataframe.id == nid2][att].item():
                eattr[att] = "EQ"
            elif dataframe[dataframe.id == nid1][att].item() > dataframe[dataframe.id == nid2][att].item():
                eattr[att] = ">"
            else:
                eattr[att] = "<"
            tr.append(NominalSelector(att, eattr[att]))
        
        if GL != None:
            if not GL.has_edge(nid1,nid2):
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
            if not GL.has_edge(nid1,nid2):
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
            if not GL.has_edge(nid1,nid2):
                eattr['weight'] = 0

    nx.set_edge_attributes(graph, attr)
    return transactions

def setMultiCompAttDiEdges(G, dataframe, attributes, GL = None):
    attr = {}
    transactions = []
    tr = []
    i = 0
    for e in list(G.edges(keys = True, data = True)):
        tr = []
        nid1, nid2, ekey, edict = e
        #eattr = {}
        for att in attributes:
            if type(dataframe[dataframe.id == nid1][att].item()) == type('str'):
                #eattr[att] = str((dataframe[dataframe.id == nid1][att].item(), dataframe[dataframe.id == nid2][att].item()))
                edict[att] = str((dataframe[dataframe.id == nid1][att].item(), dataframe[dataframe.id == nid2][att].item()))
            elif dataframe[dataframe.id == nid1][att].item() == dataframe[dataframe.id == nid2][att].item():
                #eattr[att] = "EQ"
                edict[att] = "EQ"
            elif dataframe[dataframe.id == nid1][att].item() > dataframe[dataframe.id == nid2][att].item():
                #eattr[att] = ">"
                edict[att] = ">"
            else:
                #eattr[att] = "<"
                edict[att] = "<"
            tr.append(NominalSelector(att, edict[att]))

        if GL != None:
            if not GL.has_edge(nid1,nid2):
                edict['weight'] = 0
            
        #attr[e] = eattr
        transactions.append(tr)
        i += 1
                
    #nx.set_edge_attributes(G, attr)
    return transactions

def setMultiFromAttDiEdges(G, dataframe, attributes, GL = None):
    attr = {}
    transactions = []
    tr = []
    i = 0
    for e in list(G.edges(keys = True, data = True)):
        tr = []
        nid1, nid2, ekey, edict = e
        #eattr = {}
        for att in attributes:
            #eattr[att] = dataframe[dataframe.id == nid1][att].item()
            edict[att] = dataframe[dataframe.id == nid1][att].item()
            
            tr.append(NominalSelector(att, edict[att]))
            
        #attr[e] = eattr
        transactions.append(tr)
        i += 1
                
        if GL != None:
            if not GL.has_edge(nid1,nid2):
                edict['weight'] = 0

    #nx.set_edge_attributes(G, attr)
    return transactions

def setMultiToAttDiEdges(G, dataframe, attributes, GL = None):
    attr = {}
    transactions = []
    tr = []
    i = 0
    for e in list(G.edges(keys = True, data = True)):
        tr = []
        nid1, nid2, ekey, edict = e
        #eattr = {}
        for att in attributes:
            #eattr[att] = dataframe[dataframe.id == nid2][att].item()
            edict[att] = dataframe[dataframe.id == nid2][att].item()
                
            tr.append(NominalSelector(att, edict[att]))
            
        #attr[e] = eattr
        transactions.append(tr)
        i += 1

        if GL != None:
            if not GL.has_edge(nid1,nid2):
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

def infoPats_nodes(listOfPatterns, dataset):
    col_names =  ['Pattern', 'N', 'ids', 'Mean Weight', 'Score']
    my_df  = pd.DataFrame(columns = col_names)
    for p in listOfPatterns:
        n = len(p.ids)
        tempids = len(dataset.iloc[p.ids,0].unique())
        my_df = my_df.append({'Pattern': p.name, 'N': n, 'ids': tempids, 'Mean Weight': round(p.weight,1),'Score': round(p.quality,1)}, ignore_index=True)
            
    return my_df