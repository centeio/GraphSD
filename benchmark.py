import numpy as np
import pandas as pd
from scipy.spatial import distance
import math

from graphSD.graph import *
from graphSD.viz import *
from graphSD.utils import *
from graphSD.sd import *

from graphSD.benchmark_graph import digraph1

np.random.seed(1234)

if __name__ ==  '__main__':

    GComp = nx.DiGraph()
    GTo = nx.DiGraph()
    GFrom = nx.DiGraph()

    graphs = [GComp,GTo,GFrom]

    for graph in graphs:
        graph.add_nodes_from(digraph1['nodes'])
        graph.add_weighted_edges_from(digraph1['edges'])

    attributes = ['gender', 'age']

    transactionsComp = setCompAttDiEdges(GComp, digraph1['social_data'], attributes)
    transactionsFrom = setFromAttDiEdges(GFrom, digraph1['social_data'], attributes)
    transactionsTo = setToAttDiEdges(GTo, digraph1['social_data'], attributes)

     #### Subgroup Discovery

    compTQ = treeQuality(GComp,freqItemsets(transactionsComp, 1), qS)
    compTQ.sort(reverse=True)
    infoPats(compTQ).to_csv('output/bench_Comp_qSD_mean.csv', index=True)

    compFrom = treeQuality(GFrom,freqItemsets(transactionsFrom, 1), qS)
    compFrom.sort(reverse=True)
    infoPats(compFrom).to_csv('output/bench_From_qSD_mean.csv', index=True)

    compTo = treeQuality(GTo,freqItemsets(transactionsTo, 1), qS)
    compTo.sort(reverse=True)
    infoPats(compTo).to_csv('output/bench_To_qSD_mean.csv', index=True)

    # Using variance as metric

    compTQ = treeQuality(GComp,freqItemsets(transactionsComp, 1), qS, metric = 'var')
    compTQ.sort(reverse=True)
    infoPats(compTQ).to_csv('output/bench_Comp_qSD_var.csv', index=True)

    compFrom = treeQuality(GFrom,freqItemsets(transactionsFrom, 1), qS, metric = 'var')
    compFrom.sort(reverse=True)
    infoPats(compFrom).to_csv('output/bench_From_qSD_var.csv', index=True)

    compTo = treeQuality(GTo,freqItemsets(transactionsTo, 1), qS, metric = 'var')
    compTo.sort(reverse=True)
    infoPats(compTo).to_csv('output/bench_To_qSD_var.csv', index=True)