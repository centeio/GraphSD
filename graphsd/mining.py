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
    """
        Base class for subgroup discovery on graphs. Handles graph construction, attribute assignment,
        pattern discovery, and quality scoring.

        Attributes:
            graph (nx.Graph): The graph to mine.
            social_data (pd.DataFrame): Attributes used to annotate edges.
            n_bins (int): Number of bins for discretization.
            n_samples (int): Number of samples for statistical testing.
            metric (str): Quality metric ('mean' or 'var').
            mode (str): Attribute assignment mode ('to', 'from', 'comparison').
            random_state (int): Random seed.
            n_jobs (int): Number of parallel jobs.
    """
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
        """
            Computes frequent itemsets from transaction data using FP-growth.

            Parameters:
                transactions (List[List[str]]): List of attribute-value transactions.
                min_support (float): Minimum support threshold (0–1).

            Returns:
                Dict[Tuple[str], int]: Frequent itemsets with their support counts.
        """
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
        """
            Placeholder for graph-specific transaction conversion method.

            Parameters:
                mode (str): Mode for how attributes are extracted.

            Returns:
                List[List[NominalSelector]]: Edge-annotated transactions.
        """
        pass

    def subgroup_discovery(self, mode="comparison", min_support=0.10, metric='mean', quality_measure='qS'):
        """
            Performs subgroup discovery by mining frequent patterns and evaluating their quality.

            Parameters:
                mode (str): Attribute assignment mode ('to', 'from', or 'comparison').
                min_support (float): Minimum support threshold for frequent itemsets.
                metric (str): Quality evaluation metric ('mean' or 'var').
                quality_measure (str): Quality scoring method ('qS' or 'qP').

            Returns:
                List[Pattern]: Ranked list of patterns with quality scores.
        """
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
        """
            Computes the quality score of a pattern's induced subgraph.

            Parameters:
                pattern (Tuple[str]): Attribute-value selectors defining the pattern.
                metric (str): Evaluation metric ('mean' or 'var').
                measure_type (str): Quality scoring strategy ('qS' or 'qP').

            Returns:
                Pattern: A Pattern object containing the subgraph and computed quality.
        """
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
            Estimates mean and standard deviation of scores from random subgraphs for normalization.

            Parameters:
                n_samples (int): Number of random samples to generate.
                max_n_edges (int): Total number of possible edges in the graph.
                pattern_size (int): Number of edges in the pattern's subgraph.
                metric (str): Evaluation metric ('mean' or 'var').
                max_pattern_edges (int): Theoretical maximum number of edges in the pattern.

            Returns:
                Tuple[float, float]: Mean and standard deviation of scores from random subgraphs.
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
        """
            Computes a numeric quality score for a given subgraph using the selected metric.

            Parameters:
                graph_of_pattern (nx.Graph): Subgraph induced by the pattern.
                metric (str): The scoring metric ('mean' or 'var').
                max_pattern_edges (int, optional): Max number of edges used for normalization.

            Returns:
                float: The computed score (density or variance-based).
        """
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
        """
            Builds a subgraph containing only edges that match the attribute-value selectors.

            Parameters:
                pattern (Tuple[str]): A tuple of attribute-value selectors.

            Returns:
                nx.Graph: Subgraph of the main graph that satisfies the pattern.
        """
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
        """
        Extracts edges from the main graph that satisfy all conditions in the pattern.

        Parameters:
            pattern (Tuple[NominalSelector]): Selectors defining the pattern.

        Returns:
            list: List of edge tuples that match the pattern.
        """
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
        Counts the number of multi-edges beyond the basic single-edge count.

        Parameters:
            graph (nx.MultiGraph or MultiDiGraph): Graph to evaluate.

        Returns:
            int: Count of excess edges beyond one per node pair.
        """
        count = 0
        for node1 in graph.nodes:
            for node2 in graph.nodes:
                if node1 != node2:
                    count += (graph.number_of_edges(node1, node2) - 1)
        return count

class DigraphSDMining(GraphSDMiningBase):
    """
    Subclass for directed graphs. Implements methods specific to directed edge interaction modeling.
    """
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
        """
        Constructs a directed graph from position data and social attributes.

        Parameters:
            position_data (pd.DataFrame): Positional tracking data (must include 'id', 'x', 'y', and 'timestamp').
            social_data (pd.DataFrame): Node-level attribute data (must include 'id').
            time_step (int): Temporal granularity for edge construction.

        Returns:
            Counter: Edge occurrence counts between individuals.
        """
        position_data = addVelXY(position_data)

        counter = count_interactions_digraph(position_data, proximity=1, time_step=time_step)
        ids = position_data.id.unique()

        self._create_graph(counter, ids)
        self.social_data = social_data

        return counter

    def _create_graph(self, counter, ids):
        """
        Initializes the directed graph with nodes and weighted edges.

        Parameters:
            counter (Counter): Edge weights from interaction events.
            ids (list): Node identifiers.

        Returns:
            None
        """
        graph = DiGraph()
        graph.add_nodes_from(ids)
        graph.add_weighted_edges_from(getWEdges(counter))

        self.graph = graph
        self.graph.max_edges = self.graph.number_of_nodes() * (self.graph.number_of_nodes() - 1)

    def _get_transactions(self, mode):
        """
        Generates transactions by assigning edge attributes based on the selected mode.

        Parameters:
            mode (str): One of 'to', 'from', or 'comparison', indicating how to label edges.

        Returns:
            List[List[NominalSelector]]: Transactions for pattern mining.
        """
        transactions = self.set_attributes(mode=mode)
        return transactions

    def set_attributes(self, signed_graph=None, mode="comparison"):
        """
        Assigns edge attributes from node-level data depending on the mode of comparison.

        Parameters:
            signed_graph (nx.Graph, optional): If provided, ensures edge weights are 0 for missing edges.
            mode (str): One of 'to', 'from', or 'comparison'.

        Returns:
            List[List[NominalSelector]]: List of edge transactions with attribute-value selectors.
        """
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
        """
        Summarizes a list of discovered patterns in a tabular DataFrame (for directed graphs).

        Parameters:
            subgroups (List[Pattern]): List of discovered subgraphs.

        Returns:
            pd.DataFrame: Summary including node and edge counts, mean weight, and quality.
        """
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
    """
        Subclass for multi-directed graphs (parallel edges allowed).
    """
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
        """
        Constructs a MultiDiGraph from spatial-temporal interactions and social metadata.

        Parameters:
            position_data (pd.DataFrame): Position tracking data with 'id', 'x', 'y', 'timestamp'.
            social_data (pd.DataFrame): Node-level attributes.
            time_step (int): Temporal granularity for graph edges.

        Returns:
            Counter: Edge interaction counts.
        """
        position_data = addVelXY(position_data)

        counter = count_interactions_multi_digraph(position_data, proximity=1, time_step=time_step)
        ids = position_data.id.unique()

        self._create_graph(counter, ids)
        self.social_data = social_data

        return counter

    def _create_graph(self, counter, ids):
        """
        Creates a MultiDiGraph from a weighted edge counter.

        Parameters:
            counter (Counter): Edge weights between node pairs.
            ids (List[int]): List of node identifiers.

        Returns:
            None
        """
        graph = MultiDiGraph()
        graph.add_nodes_from(ids)
        graph.add_weighted_edges_from(counter)

        self.graph = graph
        self.graph.max_edges = self.graph.number_of_nodes() * (self.graph.number_of_nodes() - 1)
        # TODO: Estimating the max edges of a multi graph is not 100% correct
        self.graph.max_edges += self.count_edges(self.graph)

    def _get_transactions(self, mode):
        """
        Extracts edge-level transactions for pattern mining using multi-directed graph structure.

        Parameters:
            mode (str): Attribute assignment mode ('to', 'from', or 'comparison').

        Returns:
            List[List[NominalSelector]]: List of transactions per edge.
        """
        transactions = self.set_attributes(mode=mode)
        return transactions

    def set_attributes(self, signed_graph=None, mode="comparison"):
        """
        Annotates each edge with attributes derived from node-level metadata, handling multiple edges.

        Parameters:
            signed_graph (nx.Graph, optional): Graph used to enforce zero-weight for missing edges.
            mode (str): One of 'to', 'from', or 'comparison'.

        Returns:
            List[List[NominalSelector]]: Transaction list of attribute-value selectors per edge.
        """
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
    """
    Subclass for undirected graphs. Implements edge construction and attribute mapping for symmetric interactions.
    """
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
        """
        Constructs an undirected interaction graph based on co-location in space and time.

        Parameters:
            position_data (pd.DataFrame): Position data including columns 'id', 'x', 'y', 'timestamp'.
            social_data (pd.DataFrame): Node-level attribute data.
            time_step (int): Temporal bin size for detecting interactions.

        Returns:
            Counter: Edge weights representing frequency of interaction.
        """
        counter = count_interactions(position_data, proximity=1, time_step=time_step)
        ids = position_data.id.unique()

        self._create_graph(counter, ids)
        self.social_data = social_data

        return counter

    def _create_graph(self, counter, ids):
        """
        Initializes the undirected graph using provided edges and node list.

        Parameters:
            counter (Counter): Edge weights.
            ids (list): Node identifiers.

        Returns:
            None
        """
        graph = Graph()
        graph.add_nodes_from(ids)
        graph.add_weighted_edges_from(counter)

        self.graph = graph
        self.graph.max_edges = (self.graph.number_of_nodes() * (self.graph.number_of_nodes() - 1)) / 2

    def _get_transactions(self, mode=None):
        """
        Generates transactions by comparing node attributes on undirected edges.

        Parameters:
            mode (str, optional): Unused for undirected graphs but kept for interface consistency.

        Returns:
            List[List[NominalSelector]]: List of transactions for each edge.
        """
        transactions = self.set_attributes()
        return transactions

    def set_attributes(self, signed_graph=None):
        """
        Assigns equality-based attributes to each edge in the undirected graph.

        Parameters:
            signed_graph (nx.Graph, optional): If provided, missing edges are given weight 0.

        Returns:
            List[List[NominalSelector]]: Attribute-value selectors per edge.
        """
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
