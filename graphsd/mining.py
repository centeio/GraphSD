from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool

import numpy as np
import pandas as pd
from networkx import Graph, DiGraph, MultiGraph, MultiDiGraph
import networkx as nx
from orangecontrib.associate.fpgrowth import frequent_itemsets

from graphsd.graph import count_interactions_digraph, count_interactions_multi_digraph, count_interactions, getWEdges
from graphsd.utils import Pattern, addVelXY, NominalSelector



class GraphSDMiningBase(object):

    def __init__(self,
                 n_bins=3,
                 n_samples=100,
                 metric='mean',
                 mode="comparison",
                 random_state=None,
                 n_jobs=1
                 ):
        self.n_bins = n_bins
        self.n_samples = n_samples
        self.metric = metric
        self.mode = mode
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.graph = None
        self.social_data = None

        self.graph_type = None
        self.multi = False

    @staticmethod
    def get_frequent_items(transactions, min_support):
        transactions_dict = {}
        last = 1
        transactions_as_int = []

        for trans in transactions:
            temp = []
            for att in trans:
                if att not in transactions_dict:
                    transactions_dict[att] = last
                    last += 1
                temp += [transactions_dict[att]]
            transactions_as_int += [temp]

        transactions_dict = {v: k for k, v in transactions_dict.items()}

        itemsets = list(frequent_itemsets(transactions_as_int, min_support=min_support))

        frequent_items = {}
        for itemset, support in itemsets:
            first = True
            for n in itemset:
                if first:
                    temp = (transactions_dict[n],)
                    first = False
                else:
                    temp += (transactions_dict[n],)

            frequent_items[temp] = support

        return frequent_items

    def _get_transactions(self, mode):
        pass

    def subgroup_discovery(self, mode="comparison", min_support=0.10, metric='mean', quality_measure='qS'):

        # TODO: random_state is not working
        np.random.seed(self.random_state)
        transactions = self._get_transactions(mode)
        frequent_itemset = self.get_frequent_items(transactions, min_support)

        if quality_measure in ['qP', 'qS']:
            quality_function = self.quality_measure_base
        # elif quality_measure == 'new_qm']::
            # Refer to alternative quality measures here
            # self.measure_score = self.measure_score
        else:
            msg = "Unknown quality function. Current ones are: ['qP','qS']"
            ValueError(msg)

        if self.n_jobs > 1:
            pool = Pool(self.n_jobs)
            subgroups = []
            for frequent_items, _ in frequent_itemset.items():
                pool.apply_async(quality_function, args=(frequent_items, metric, quality_measure),
                                 callback=subgroups.append)
            pool.close()
            pool.join()
        else:
            subgroups = [quality_function(frequent_items, metric=metric, measure_type=quality_measure)
                         for frequent_items, _ in frequent_itemset.items()]

        subgroups.sort(reverse=True)

        return subgroups

    def quality_measure_base(self, pattern, metric, measure_type='qS'):

        graph_of_pattern = self.get_pattern_graph(pattern)

        # The difference between the two quality measures is how they consider the maximum number of edges
        if measure_type == 'qS':
            max_n_edges = self.graph.max_edges
            n_nodes = len(graph_of_pattern)
            max_pattern_edges = float(n_nodes * (n_nodes - 1))  # number of all possible edges
            if self.multi:
                max_pattern_edges += self.count_edges(graph_of_pattern)
        elif measure_type == 'qP':
            max_n_edges = self.graph.size()
            max_pattern_edges = None

        score = self.measure_score(graph_of_pattern, metric, max_pattern_edges)
        subgroup = Pattern(pattern, graph_of_pattern, score)

        mean, std = self.statistical_validation(self.n_samples,
                                                max_n_edges=max_n_edges,
                                                pattern_size=graph_of_pattern.number_of_edges(),
                                                metric=metric,
                                                max_pattern_edges=max_pattern_edges)

        subgroup.quality = (subgroup.weight - mean) / std

        return subgroup

    def statistical_validation(self, n_samples, max_n_edges, pattern_size, metric, max_pattern_edges):
        """

        This function randomly generates graphs and measures their score. Then outputs
        the mean and standard deviation of these scores.

        Parameters
        ----------
        n_samples
        max_n_edges
        pattern_size
        metric
        max_pattern_edges

        Returns
        -------
        mean
        std
        """
        sample = []

        pool = ThreadPool(2)
        list_of_edges = list(self.graph.edges)
        graph_size = self.graph.size()

        for _ in range(n_samples):
            # indexes = np.random.choice(range(interval), pattern_size, replace=False)
            # random_edges = [list_of_edges[i] for i in indexes]
            # TODO: Needs to be confirmed if commented version is the correct or not!
            indexes = np.random.choice(range(max_n_edges), pattern_size, replace=False)
            random_edges = [list_of_edges[i] for i in indexes if i < graph_size]
            random_subgraph = self.graph.edge_subgraph(random_edges)
            pool.apply_async(self.measure_score, args=(random_subgraph,
                                                       metric,
                                                       max_pattern_edges),
                             callback=sample.append)

        pool.close()
        pool.join()

        mean = np.mean(sample)
        std = np.std(sample)

        return mean, std

    @staticmethod
    def measure_score(graph_of_pattern, metric, max_pattern_edges=None):

        if max_pattern_edges is None:
            max_pattern_edges = graph_of_pattern.size()

        if max_pattern_edges == 0:
            quality = 0
        else:
            mean = graph_of_pattern.size(weight="weight") / max_pattern_edges
            if metric == 'mean':

                quality = mean
            elif metric == 'var':

                weights = [e[2]['weight'] for e in graph_of_pattern.edges(data=True)]
                var = sum((np.array(weights) - mean) ** 2) / max_pattern_edges
                quality = var
        return quality

    def get_pattern_graph(self, pattern):

        if type(self.graph) == Graph:
            graph_of_pattern = Graph()
        elif type(self.graph) == DiGraph:
            graph_of_pattern = DiGraph()
        elif type(self.graph) == MultiGraph:
            graph_of_pattern = MultiGraph()
        elif type(self.graph) == MultiDiGraph:
            graph_of_pattern = MultiDiGraph()
        else:
            msg = f"Unknown graph type"
            ValueError(msg)

        edges_list = self.get_edges_in_pattern(pattern)
        graph_of_pattern.add_edges_from(edges_list)

        return graph_of_pattern

    def get_edges_in_pattern(self, pattern) -> list:
        edges = []

        for edge in list(self.graph.edges(data=True)):
            edge_in_pattern = True
            for sel in pattern:
                if edge[2][sel.attribute] != sel.value:
                    edge_in_pattern = False
                    break

            if edge_in_pattern:
                edges.append(edge)

        return edges

    @staticmethod
    def count_edges(graph):
        """

        This used to be called function 'm'

        """
        count = 0
        for node1 in graph.nodes:
            for node2 in graph.nodes:
                if node1 != node2:
                    count += (graph.number_of_edges(node1, node2) - 1)
        return count


class DigraphSDMining(GraphSDMiningBase):

    def __init__(self,
                 n_bins=3,
                 n_samples=100,
                 metric='mean',
                 mode="comparison",
                 random_state=None,
                 n_jobs=1
                 ):
        super().__init__(
            n_bins=n_bins,
            n_samples=n_samples,
            metric=metric,
            mode=mode,
            random_state=random_state,
            n_jobs=n_jobs
        )

        self.n_bins = n_bins
        self.n_samples = n_samples
        self.metric = metric
        self.mode = mode
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.graph = None
        self.social_data = None

        self.graph_type = "digraph"

    def read_data(self, position_data, social_data, time_step=10):
        position_data = addVelXY(position_data)

        counter = count_interactions_digraph(position_data, proximity=1, time_step=time_step)
        ids = position_data.id.unique()

        self._create_graph(counter, ids)
        self.social_data = social_data

        return counter

    def _create_graph(self, counter, ids):
        graph = DiGraph()
        graph.add_nodes_from(ids)
        graph.add_weighted_edges_from(getWEdges(counter))

        self.graph = graph
        self.graph.max_edges = self.graph.number_of_nodes() * (self.graph.number_of_nodes() - 1)

    def _get_transactions(self, mode):
        transactions = self.set_attributes(mode=mode)
        return transactions

    def set_attributes(self, signed_graph=None, mode="comparison"):
        attributes = self.social_data.drop(['id'], axis=1).columns

        attr = {}
        transactions = []

        for edge1, edge2 in list(self.graph.edges()):
            tr = []
            edge_attr = {}
            for att in attributes:
                if mode == "to":
                    edge_attr[att] = self.social_data[self.social_data.id == edge2][att].item()
                elif mode == "from":
                    edge_attr[att] = self.social_data[self.social_data.id == edge1][att].item()
                elif mode == "comparison":
                    if isinstance(self.social_data[self.social_data.id == edge1][att].item(), str):
                        edge_attr[att] = str((self.social_data[self.social_data.id == edge1][att].item(),
                                              self.social_data[self.social_data.id == edge2][att].item()))
                    elif self.social_data[self.social_data.id == edge1][att].item() == \
                            self.social_data[self.social_data.id == edge2][att].item():
                        edge_attr[att] = "EQ"
                    elif self.social_data[self.social_data.id == edge1][att].item() > \
                            self.social_data[self.social_data.id == edge2][att].item():
                        edge_attr[att] = ">"
                    else:
                        edge_attr[att] = "<"
                else:
                    msg = "Unknown mode. Current ones are: ['to','from','comparison']"
                    ValueError(msg)

                tr.append(NominalSelector(att, edge_attr[att]))

            if signed_graph is not None:
                if not signed_graph.has_edge(edge1, edge2):
                    edge_attr['weight'] = 0

            attr[(edge1, edge2)] = edge_attr
            transactions.append(tr)

        nx.set_edge_attributes(self.graph, attr)

        return transactions

    @staticmethod
    def to_dataframe(subgroups):
        col_names = ['Pattern', 'Nodes', 'in', 'out', 'Edges', 'Mean Weight', 'Score']
        dataframe = pd.DataFrame(columns=col_names)
        for p in subgroups:
            if type(p) == Pattern:
                in_nodes = len([y for (x, y) in list(p.graph.in_degree()) if y > 0])
                out_nodes = len([y for (x, y) in list(p.graph.out_degree()) if y > 0])
                dataframe_extension = pd.DataFrame(
                    {'Pattern': p.name, 'Nodes': p.graph.number_of_nodes(), 'in': in_nodes, 'out': out_nodes,
                     'Edges': p.graph.number_of_edges(),
                     'Mean Weight': round(p.weight, 1), 'Score': round(p.quality, 1)
                     })

                dataframe = pd.concat([dataframe, dataframe_extension], ignore_index=True)

        return dataframe

class MultiDigraphSDMining(GraphSDMiningBase):
    def __init__(self,
                 n_bins=3,
                 n_samples=100,
                 metric='mean',
                 mode="comparison",
                 random_state=None,
                 n_jobs=1
                 ):
        super().__init__(
            n_bins=n_bins,
            n_samples=n_samples,
            metric=metric,
            mode=mode,
            random_state=random_state,
            n_jobs=n_jobs
        )
        self.n_bins = n_bins
        self.n_samples = n_samples
        self.metric = metric
        self.mode = mode
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.graph = None
        self.social_data = None

        self.multi = True

    def read_data(self, position_data, social_data, time_step=10):
        position_data = addVelXY(position_data)

        counter = count_interactions_multi_digraph(position_data, proximity=1, time_step=time_step)
        ids = position_data.id.unique()

        self._create_graph(counter, ids)
        self.social_data = social_data

        return counter

    def _create_graph(self, counter, ids):
        graph = MultiDiGraph()
        graph.add_nodes_from(ids)
        graph.add_weighted_edges_from(counter)

        self.graph = graph
        self.graph.max_edges = self.graph.number_of_nodes() * (self.graph.number_of_nodes() - 1)
        # TODO: Estimating the max edges of a multi graph is not 100% correct
        self.graph.max_edges += self.count_edges(self.graph)

    def _get_transactions(self, mode):
        transactions = self.set_attributes(mode=mode)
        return transactions

    def set_attributes(self, signed_graph=None, mode="comparison"):

        attributes = self.social_data.drop(['id'], axis=1).columns

        transactions = []
        i = 0
        for nid1, nid2, ekey, edict in list(self.graph.edges(keys=True, data=True)):
            tr = []
            # edge_attr = {}
            for att in attributes:
                if mode == "to":
                    edict[att] = self.social_data[self.social_data.id == nid2][att].item()
                elif mode == "from":
                    edict[att] = self.social_data[self.social_data.id == nid1][att].item()
                elif mode == "comparison":
                    if isinstance(self.social_data[self.social_data.id == nid1][att].item(), str):
                        edict[att] = str(
                            (self.social_data[self.social_data.id == nid1][att].item(),
                             self.social_data[self.social_data.id == nid2][att].item()))
                    elif self.social_data[self.social_data.id == nid1][att].item() == \
                            self.social_data[self.social_data.id == nid2][att].item():
                        # edge_attr[att] = "EQ"
                        edict[att] = "EQ"
                    elif self.social_data[self.social_data.id == nid1][att].item() > \
                            self.social_data[self.social_data.id == nid2][att].item():
                        # edge_attr[att] = ">"
                        edict[att] = ">"
                    else:
                        # edge_attr[att] = "<"
                        edict[att] = "<"
                tr.append(NominalSelector(att, edict[att]))

            # attr[e] = edge_attr
            transactions.append(tr)
            i += 1

            if signed_graph is not None:
                if not signed_graph.has_edge(nid1, nid2):
                    edict['weight'] = 0

        # nx.set_edge_attributes(G, attr)
        return transactions


class GraphSDMining(GraphSDMiningBase):

    def __init__(self,
                 n_bins=3,
                 n_samples=100,
                 metric='mean',
                 random_state=None,
                 n_jobs=1
                 ):
        super().__init__(
            n_bins=n_bins,
            n_samples=n_samples,
            metric=metric,
            random_state=random_state,
            n_jobs=n_jobs
        )

        self.n_bins = n_bins
        self.n_samples = n_samples
        self.metric = metric
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.graph = None
        self.social_data = None

        self.graph_type = "digraph"

    def read_data(self, position_data, social_data, time_step=10):

        counter = count_interactions(position_data, proximity=1, time_step=time_step)
        ids = position_data.id.unique()

        self._create_graph(counter, ids)
        self.social_data = social_data

        return counter

    def _create_graph(self, counter, ids):
        graph = Graph()
        graph.add_nodes_from(ids)
        graph.add_weighted_edges_from(counter)

        self.graph = graph
        self.graph.max_edges = (self.graph.number_of_nodes() * (self.graph.number_of_nodes() - 1)) / 2

    def _get_transactions(self, mode=None):
        transactions = self.set_attributes()
        return transactions

    def set_attributes(self, signed_graph=None):
        attributes = self.social_data.drop(['id'], axis=1).columns

        attr = {}
        transactions = []

        for nid1, nid2 in list(self.graph.edges()):
            tr = []
            edge_attr = {}
            for att in attributes:
                if self.social_data[self.social_data.id == nid1][att].item() == \
                        self.social_data[self.social_data.id == nid2][att].item():
                    edge_attr[att] = "EQ"
                else:
                    edge_attr[att] = "NEQ"
                tr.append(NominalSelector(att, edge_attr[att]))

            if signed_graph is not None:
                if not signed_graph.has_edge(nid1, nid2):
                    edge_attr['weight'] = 0

            attr[(nid1, nid2)] = edge_attr
            transactions.append(tr)

        nx.set_edge_attributes(self.graph, attr)
        return transactions
