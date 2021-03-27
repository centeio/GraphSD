from orangecontrib.associate.fpgrowth import frequent_itemsets
from multiprocessing import Pool
from networkx import Graph, DiGraph, MultiGraph, MultiDiGraph

from graphsd.sd2 import *
from graphsd.graph import count_interactions_digraph, count_interactions_multi_digraph, set_attributes_diedges


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

        self.quality_measure_aux = None

    @staticmethod
    def get_frequent_edges(transactions, min_support):
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
        frequent_items = self.get_frequent_edges(transactions, min_support)

        if quality_measure == 'qP':
            quality_function = self.quality_measure_p
            self.quality_measure_aux = self.quality_measure_p_aux
        elif quality_measure == 'qS':
            quality_function = self.quality_measure_s
            self.quality_measure_aux = self.quality_measure_s_aux
        else:
            msg = "Unknown quality function. Current ones are: ['qP','qS']"
            ValueError(msg)

        if self.n_jobs > 1:
            pool = Pool(self.n_jobs)
            subgroups = []
            for k, _ in frequent_items.items():
                pool.apply_async(quality_function, args=(k, metric),
                                 callback=subgroups.append)
            pool.close()
            pool.join()
        else:
            subgroups = [quality_function(k, metric=metric) for k, _ in frequent_items.items()]

        subgroups.sort(reverse=True)

        return subgroups

    def quality_measure_s(self, pattern, metric):

        total_edges = self.graph.number_of_nodes() * (self.graph.number_of_nodes() - 1)

        print("Is measuring quality S")
        if type(self.graph) == Graph:
            graph_of_pattern = Graph()
        elif type(self.graph) == DiGraph:
            graph_of_pattern = DiGraph()
        elif type(self.graph) == MultiGraph:
            graph_of_pattern = MultiGraph()
            total_edges += self.count_edges(self.graph, self.graph.nodes)
            print(total_edges)
        elif type(self.graph) == MultiDiGraph:
            graph_of_pattern = MultiDiGraph()
            total_edges += self.count_edges(self.graph, self.graph.nodes)
            print(total_edges)

        edges = self.get_edges_in_pattern(self.graph, pattern)
        graph_of_pattern.add_edges_from(edges)

        w = self.quality_measure_aux(edges, metric)
        subgroup = Pattern(pattern, graph_of_pattern, w)

        mean, std = self.statistical_validation(self.n_samples, interval=total_edges,
                                                pattern_size=len(edges), metric=metric)

        subgroup.quality = (subgroup.weight - mean) / std

        return subgroup

    def statistical_validation(self, n_samples, interval, pattern_size, metric):
        """

        This function randomly generates graphs and measures their score

        Parameters
        ----------
        pattern_size
        n_samples
        interval
        size
        metric

        Returns
        -------
        mean
        std
        """
        sample = []

        pool = ThreadPool(2)
        list_of_edges = list(self.graph.edges(data=True))
        graph_size = len(list_of_edges)

        for r in range(n_samples):
            # indexes = np.random.choice(range(interval), pattern_size, replace=False)
            # random_edges = [list_of_edges[i] for i in indexes]
            # TODO: Needs to be confirmed if commnented version is the correct or not!
            indexes = np.random.choice(range(interval), pattern_size, replace=False)
            random_edges = [list_of_edges[i] for i in indexes if i < graph_size]
            pool.apply_async(self.quality_measure_aux, args=(random_edges, metric), callback=sample.append)

        pool.close()
        pool.join()

        mean = np.mean(sample)
        std = np.std(sample)

        return mean, std

    # TODO: Merge with S
    def quality_measure_p(self, pattern, metric):

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

        edges = self.get_edges_in_pattern(self.graph, pattern)
        graph_of_pattern.add_edges_from(edges)

        w = self.quality_measure_aux(edges, metric)
        subgroup = Pattern(pattern, graph_of_pattern, w)

        mean, std = self.statistical_validation(self.n_samples, interval=len(list(self.graph.edges())),
                                                pattern_size=len(edges), metric=metric)

        subgroup.quality = (subgroup.weight - mean) / std

        return subgroup

    def quality_measure_s_aux(self, edges, metric):
        nodes = set()
        weights = []
        for e in edges:
            nodes = nodes | {e[0], e[1]}
            weights += [e[2]['weight']]
        num_edges_in_pattern = (len(nodes) * 1.0)  # number of nodes covered by a pattern P
        max_num_edges = num_edges_in_pattern * (num_edges_in_pattern - 1)  # number of all possible edges

        if max_num_edges == 0:
            quality = 0
        else:
            if self.multi:
                max_num_edges += self.count_edges(self.graph, nodes)
            mean = sum(weights) / max_num_edges
            if metric == 'mean':
                quality = mean
            elif metric == 'var':
                var = sum((np.array(weights) - mean) ** 2) / max_num_edges
                quality = var
        return quality

    @staticmethod
    def quality_measure_p_aux(edges, metric='mean'):
        weights = [e[2]['weight'] for e in edges]

        if metric == 'mean':
            quality = np.mean(weights)
        elif metric == 'var':
            quality = np.var(weights)
        return quality

    @staticmethod
    def get_edges_in_pattern(graph, pattern):
        edges = []

        for edge in list(graph.edges(data=True)):
            edge_in_pattern = True
            for sel in pattern:
                if edge[2][sel.attribute] != sel.value:
                    edge_in_pattern = False
                    break

            if edge_in_pattern:
                edges.append(edge)

        return edges

    @staticmethod
    def count_edges(graph, nodes):
        """

        This used to be called function 'm'

        """

        count = 0
        for node1 in nodes:
            for node2 in nodes:
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
        graph = nx.MultiDiGraph()
        graph.add_nodes_from(ids)
        graph.add_weighted_edges_from(counter)

        self.graph = graph

    def _get_transactions(self, mode):
        transactions = set_attributes_multi_diedges(self.graph, self.social_data, mode=mode)
        self.transactions = transactions
        return transactions


class OutlierSDMining(object):

    def __init__(self,
                 quality_measure='qS',
                 n_bins=3,
                 n_samples=100,
                 metric='mean',
                 mode="comparison",
                 random_state=None,
                 n_jobs=1
                 ):
        self.quality_measure = quality_measure
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

    def get_several_lof(self, position_data, socialData, start_time, end_time, nseconds=1, k=5, contamination=.1):
        start_window = pd.Timestamp(start_time)
        areas_df = pd.DataFrame()
        ids = position_data.id.unique()

        while start_window <= pd.Timestamp(end_time):
            temp_df = socialData[socialData['id'].isin(ids)].copy()
            position = position_data[str(start_window)]
            pids = [row.id for index, row in position.iterrows()]
            # print(getAreas(positions))
            # try:
            areasp = get_local_outlier_factor_scores(position[['x', 'y']], k=k, contamination=contamination)
            # print(areasp)
            # return areasp

            tempareas = []
            temptimes = []

            try:
                i = 0
                for pid in ids:
                    temptimes += [start_window]
                    if pid in pids:
                        tempareas += [areasp[i]]
                        i += 1
                    else:
                        tempareas += [np.nan]

                temp_df['lof'] = tempareas
                temp_df['timestamp'] = temptimes

            except:
                print("An exception occurred at: ", start_window)

            areas_df = pd.concat([areas_df, temp_df[np.isfinite(temp_df['lof'])]])
            start_window = start_window + pd.Timedelta(seconds=nseconds)

        # maxW = max(list(counter.values()))

        # counter = counter/count
        return areas_df

    def subgroup_discovery(self, mode="comparison", min_support=0.10, metric='mean'):

        # TODO: random_state is not working
        np.random.seed(self.random_state)
        transactions = self._get_transactions(mode)
        frequent_items = self.get_frequent_edges(transactions, min_support)
        self._set_quality_function()

        if self.n_jobs > 1:
            pool = Pool(self.n_jobs)
            qs = []
            for k, _ in frequent_items.items():
                pool.apply_async(quality_function, args=(self.graph, k, self.n_samples, metric),
                                 callback=qs.append)
            pool.close()
            pool.join()
        else:
            qs = [quality_function(self.graph, k, self.n_samples, metric) for k, _ in frequent_items.items()]

        qs.sort(reverse=True)

        return qs
