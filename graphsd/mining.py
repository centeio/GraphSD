from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool

import numpy as np
from networkx import Graph, DiGraph, MultiGraph, MultiDiGraph
from orangecontrib.associate.fpgrowth import frequent_itemsets

from graphsd.graph import count_interactions_digraph, count_interactions_multi_digraph, \
    set_attributes_diedges, getWEdges, set_attributes_multi_diedges
from graphsd.utils import Pattern, addVelXY


class GraphSDMining(object):

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
        self.transactions = None
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
        frequent_items = self.get_frequent_items(transactions, min_support)

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
            for k, _ in frequent_items.items():
                pool.apply_async(quality_function, args=(k, metric, quality_measure),
                                 callback=subgroups.append)
            pool.close()
            pool.join()
        else:
            subgroups = [quality_function(k, metric=metric, measure_type=quality_measure)
                         for k, _ in frequent_items.items()]

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


class DigraphSDMining(GraphSDMining):

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
        self.transactions = None
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
        transactions = set_attributes_diedges(self.graph, self.social_data, mode=mode)
        self.transactions = transactions
        return transactions


class MultiDigraphSDMining(GraphSDMining):
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
        self.transactions = None
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
        # TODO: Estimating the max edges of a multi graph is not simple
        self.graph.max_edges += self.count_edges(self.graph)

    def _get_transactions(self, mode):
        transactions = set_attributes_multi_diedges(self.graph, self.social_data, mode=mode)
        self.transactions = transactions
        return transactions
