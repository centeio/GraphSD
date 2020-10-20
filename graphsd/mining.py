from graphsd.sd import *

from graphsd.graph import count_interactions_digraph, count_interactions_multi_digraph, set_attributes_diedges


class GraphSDMining(object):

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
        self.quality_function = None
        self.subgroups = None
        self.frequent_items = None
        self.social_data = None

        self.graph_type = None

    def _set_quality_function(self):
        if self.quality_measure == 'qP':
            self.quality_function = qP
        elif self.quality_measure == 'qS':
            self.quality_function = qS

    def get_frequent_edges(self, transactions, min_support):
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

        self.frequent_items = frequent_items

        return frequent_items

    def subgroup_discovery(self, mode="comparison", min_support=0.10, metric='mean'):

        # TODO: random_state is not working
        np.random.seed(self.random_state)
        transactions = self._get_transactions(mode)
        frequent_items = self.get_frequent_edges(transactions, min_support)
        self._set_quality_function()

        if self.n_jobs > 1:
            pool = mp.Pool(self.n_jobs)
            qs = []
            for k, _ in frequent_items.items():
                pool.apply_async(self.quality_function, args=(self.graph, k, self.n_samples, metric),
                                 callback=qs.append)
            pool.close()
            pool.join()
        else:
            qs = [self.quality_function(self.graph, k, self.n_samples, metric) for k, _ in frequent_items.items()]

        qs.sort(reverse=True)
        self.subgroups = qs

        return qs


class DigraphSDMining(GraphSDMining):

    def __init__(self,
                 quality_measure='qS',
                 n_bins=3,
                 n_samples=100,
                 metric='mean',
                 mode="comparison",
                 random_state=None,
                 n_jobs=1
                 ):
        super().__init__(
            quality_measure=quality_measure,
            n_bins=n_bins,
            n_samples=n_samples,
            metric=metric,
            mode=mode,
            random_state=random_state,
            n_jobs=n_jobs
        )

        self.quality_measure = quality_measure
        self.n_bins = n_bins
        self.n_samples = n_samples
        self.metric = metric
        self.mode = mode
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.graph = None
        self.transactions = None
        self.quality_function = None
        self.subgroups = None
        self.frequent_items = None
        self.social_data = None

        self.graph_type = "digraph"

    def read_data(self, position_dataframe, social_data, time_step=10):
        position_dataframe = addVelXY(position_dataframe)

        counter = count_interactions_digraph(position_dataframe, proximity=1, time_step=time_step)
        ids = position_dataframe.id.unique()

        self.graph = self.create_graph(counter, ids)
        self.social_data = social_data

        return counter

    def create_graph(self, counter, ids):
        graph = nx.DiGraph()
        graph.add_nodes_from(ids)
        graph.add_weighted_edges_from(getWEdges(counter))

        return graph

    def _get_transactions(self, mode):
        transactions = set_attributes_diedges(self.graph, self.social_data, mode=mode)
        self.transactions = transactions
        return transactions


class MultiDigraphSDMining(GraphSDMining):
    def __init__(self,
                 quality_measure='qS',
                 n_bins=3,
                 n_samples=100,
                 metric='mean',
                 mode="comparison",
                 random_state=None,
                 n_jobs=1
                 ):
        super().__init__(
            quality_measure=quality_measure,
            n_bins=n_bins,
            n_samples=n_samples,
            metric=metric,
            mode=mode,
            random_state=random_state,
            n_jobs=n_jobs
        )

        self.quality_measure = quality_measure
        self.n_bins = n_bins
        self.n_samples = n_samples
        self.metric = metric
        self.mode = mode
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.graph = None
        self.transactions = None
        self.quality_function = None
        self.subgroups = None
        self.frequent_items = None
        self.social_data = None

    def read_data(self, position_dataframe, social_data, time_step=10):
        position_dataframe = addVelXY(position_dataframe)

        counter = count_interactions_multi_digraph(position_dataframe, proximity=1, time_step=time_step)
        ids = position_dataframe.id.unique()

        self.graph = self.create_graph(counter, ids)
        self.social_data = social_data

        return counter

    def create_graph(self, counter, ids):
        graph = nx.MultiDiGraph()
        graph.add_nodes_from(ids)
        graph.add_weighted_edges_from(counter)

        return graph

    def _get_transactions(self, mode):
        transactions = set_attributes_multi_diedges(self.graph, self.social_data, mode=mode)
        self.transactions = transactions
        return transactions
