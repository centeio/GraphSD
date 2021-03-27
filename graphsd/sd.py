from multiprocessing.dummy import Pool as ThreadPool
from networkx import Graph, DiGraph, MultiGraph, MultiDiGraph

from graphsd.graph import *


# def edges_in_pattern(graph, pattern):
#     edges = []
#
#     for edge in list(graph.edges(data=True)):
#         edge_in_pattern = True
#         for sel in pattern:
#             if edge[2][sel.attribute] != sel.value:
#                 edge_in_pattern = False
#                 break
#
#         if edge_in_pattern:
#             edges.append(e)
#
#     return edges
#
#
# def count_edges(graph, nodes):
#     """
#
#     This used to be called function 'm'
#
#     """
#
#     count = 0
#     for node1 in nodes:
#         for node2 in nodes:
#             if node1 != node2:
#                 count += (graph.number_of_edges(node1, node2) - 1)
#     return count


# def qSaux(graph, edges, multi, metric='mean'):
#     nodes = set()
#     weights = []
#     for e in edges:
#         nodes = nodes | {e[0], e[1]}
#         weights += [e[2]['weight']]
#     num_edges_in_pattern = (len(nodes) * 1.0)  # number of nodes covered by a pattern P
#     max_num_edges = num_edges_in_pattern * (num_edges_in_pattern - 1)  # number of all possible edges
#
#     if max_num_edges == 0:
#         quality = 0
#     else:
#         if multi is True:
#             max_num_edges += count_edges(graph, nodes)
#         mean = sum(weights) / max_num_edges
#         if metric == 'mean':
#             quality = mean
#         elif metric == 'var':
#             var = sum((np.array(weights) - mean) ** 2) / max_num_edges
#             quality = var
#     return quality
#
#
# def qS(graph, pattern, n_samples, metric='mean'):
#
#     multi = False
#     totalE = graph.number_of_nodes() * (graph.number_of_nodes() - 1)
#
#     if isinstance(graph, Graph):
#         graph_pattern = Graph()
#     elif type(graph) == DiGraph:
#         graph_pattern = DiGraph()
#     elif type(graph) == MultiGraph:
#         graph_pattern = MultiGraph()
#         multi = True
#         totalE += count_edges(graph, graph.nodes)
#     elif type(graph) == MultiDiGraph:
#         graph_pattern = MultiDiGraph()
#         multi = True
#         totalE += count_edges(graph, graph.nodes)
#
#     edges = edges_in_pattern(graph, pattern)
#     graph_pattern.add_edges_from(edges)
#
#     weight = qSaux(graph, edges, multi, metric)
#
#     pat = Pattern(pattern, graph_pattern, weight)
#
#     sample = []
#
#     pool = ThreadPool(2)
#
#     for r in range(n_samples):
#         indexes = np.random.choice(range(totalE), len(edges), replace=False)
#         random_edges = [list(graph.edges(data=True))[i] for i in indexes if i < len(list(graph.edges()))]
#         pool.apply_async(qSaux, args=(graph, random_edges, multi, metric), callback=sample.append)
#
#     pool.close()
#     pool.join()
#
#     mean = np.mean(sample)
#     std = np.std(sample)
#
#     pat.quality = (pat.weight - mean) / std
#
#     return pat


# def qPaux(edges, metric='mean'):
#     weights = [e[2]['weight'] for e in edges]
#
#     if metric == 'mean':
#         quality = np.mean(weights)
#     elif metric == 'var':
#         quality = np.var(weights)
#     return quality


# def qP(graph, pattern, nsamples, metric='mean'):
#     edges = edges_in_pattern(graph, pattern)
#     multi = False
#
#     if type(graph) == Graph:
#         Gpattern = Graph()
#     elif type(graph) == DiGraph:
#         Gpattern = DiGraph()
#     elif type(graph) == MultiGraph:
#         Gpattern = MultiGraph()
#         multi = True
#     elif type(graph) == MultiDiGraph:
#         Gpattern = MultiDiGraph()
#         multi = True
#
#     Gpattern.add_edges_from(edges)
#
#     w = qPaux(edges, metric)
#     pat = Pattern(pattern, Gpattern, w)
#
#     sample = []
#
#     pool = ThreadPool(2)
#
#     for r in range(nsamples):
#         print("sample:", r)
#         indxs = np.random.choice(range(len(list(graph.edges()))), len(list(pat.graph.edges())), replace=False)
#         randomE = [list(graph.edges(data=True))[i] for i in indxs]
#         pool.apply_async(qPaux, args=(randomE, metric), callback=sample.append)
#
#     pool.close()
#     pool.join()
#
#     mean = np.mean(sample)
#     std = np.std(sample)
#
#     pat.quality = (pat.weight - mean) / std
#
#     return pat


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
