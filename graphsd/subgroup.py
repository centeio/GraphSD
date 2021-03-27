from graphsd.sd import *
# from process_data import load_data

from graphsd.graph import count_interactions, set_attributes_diedges

np.random.seed(1234)

# transactionsComp, transactionsFrom, transactionsTo = load_data()


class Subgroup(object):

    def __init__(self,
                 quality_measure='qP'
                 ):
        self.quality_measure = quality_measure
        self.n_bins = n_bins
        self.n_samples = n_samples
        self.metric = metric
        self.mode = mode

        self.graph = None
        self.transactions = None
        self.quality_function = None
        self.subgroups = None
        self.nodes = None
        self.social_data = None

    def read(self, graph):



        return self

