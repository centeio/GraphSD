from typing import List, Dict, Tuple, Any, Optional

import networkx as nx
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.preprocessing import TransactionEncoder
from networkx import Graph, DiGraph, MultiGraph, MultiDiGraph

from graphsd.patterns import Pattern, NominalSelector
from graphsd.utils import compute_velocities, count_interactions


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
        self.npr = np.random.RandomState(self.random_state)
        self.n_jobs = n_jobs

        self.graph = None
        self.social_data = None

        self.graph_type = None
        self.multi = False

    @staticmethod
    def get_frequent_items(
            transactions: List[List[NominalSelector]],
            min_support: float
    ) -> Dict[Tuple[NominalSelector, ...], int]:
        """
        Computes frequent itemsets from transactions using FP-Growth via mlxtend.

        Parameters:
            transactions (List[List[NominalSelector]]): Lists of NominalSelector items per transaction.
            min_support (float): Minimum support (as a proportion between 0 and 1).

        Returns:
            Dict[Tuple[NominalSelector, ...], int]: Frequent itemsets and their absolute support counts.
        """
        te = TransactionEncoder()
        encoded = te.fit(transactions).transform(transactions)
        df = pd.DataFrame(encoded, columns=te.columns_)

        itemsets_df = fpgrowth(df, min_support=min_support, use_colnames=True)

        return {
            tuple(itemset): int(support * len(transactions))
            for itemset, support in zip(itemsets_df['itemsets'], itemsets_df['support'])
        }

    def _get_transactions(self, mode: str) -> List[List[NominalSelector]]:
        """
            Placeholder for graph-specific transaction conversion method.

            Parameters:
                mode (str): Mode for how attributes are extracted.

            Returns:
                List[List[NominalSelector]]: Edge-annotated transactions.
        """
        pass

    def _create_graph(self, edge_list: List[Tuple[int, int, Dict[str, Any]]]) -> None:
        """
        Builds a NetworkX graph using the declared graph type and edge list.

        Parameters:
            edge_list (List[Tuple[int, int, Dict[str, Any]]]): Edge data with 'weight' or other attributes.

        Returns:
            None
        """
        G = self.graph_type()
        G.add_edges_from(edge_list)
        self.graph = G

        n = G.number_of_nodes()
        if isinstance(G, nx.DiGraph) or isinstance(G, nx.MultiDiGraph):
            self.graph.max_edges = n * (n - 1)
        else:
            self.graph.max_edges = n * (n - 1) // 2

        if self.multi:
            self.graph.max_edges += self.count_edges(G)

    def read_data(
            self,
            position_data: pd.DataFrame,
            social_data: pd.DataFrame,
            time_step: int = 10,
            proximity: float = 1.0
    ) -> None:
        """
        Processes positional data into a proximity-based interaction graph.

        Parameters:
            position_data (pd.DataFrame): Includes 'id', 'x', 'y', 'time'. Velocities will be computed if needed.
            social_data (Any): Optional metadata.
            time_step (int): Time window size for interaction aggregation.
            proximity (float): Max spatial distance for interactions.

        Returns:
            None
        """
        directed = issubclass(self.graph_type, (nx.DiGraph, nx.MultiDiGraph))
        include_all = self.multi

        if directed:
            position_data = compute_velocities(position_data)

        edge_list = count_interactions(
            position_data,
            proximity=proximity,
            time_step=time_step,
            directed=directed,
            include_all_pairs=include_all
        )

        self._create_graph(edge_list)
        self.social_data = social_data

    @staticmethod
    def _evaluate_quality(args):
        quality_fn, itemset, metric, quality_measure = args
        return quality_fn(itemset, metric, quality_measure)

    def subgroup_discovery(
            self,
            mode: str = "comparison",
            min_support: float = 0.1,
            metric: str = "mean",
            quality_measure: str = "qS"
    ) -> List[Pattern]:
        """
        Discovers high-quality subgroups based on frequent patterns and interaction quality.

        Parameters:
            mode (str): Strategy for extracting attribute transactions ('comparison', 'composition', etc.)
            min_support (float): Minimum frequency threshold (0 < s ≤ 1).
            metric (str): Target metric to evaluate subgraphs ('mean', 'var', etc.)
            quality_measure (str): Quality function to use ('qS', 'qP', etc.)

        Returns:
            List[Pattern]: Sorted list of patterns ranked by quality.
        """
        np.random.seed(self.random_state)

        transactions = self._get_transactions(mode)
        itemsets = self.get_frequent_items(transactions, min_support)

        if not itemsets:
            return []

        if self.n_jobs > 1:
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(self.quality_measure_base)(itemset, metric, quality_measure)
                for itemset in itemsets
            )
        else:
            results = [self.quality_measure_base(itemset, metric, quality_measure) for itemset in itemsets]

        return sorted(results, key=lambda p: p.quality, reverse=True)

    from typing import List, Tuple, Any

    def extract_edge_keys(
            self,
            edges_with_data: List[Tuple]
    ) -> List[Tuple]:
        """
        Extracts edge identifiers from edge tuples that include data dictionaries,
        returning keys suitable for use with NetworkX's edge_subgraph() method.

        This method supports both standard and multi-edge graph types.

        Parameters:
            edges_with_data (List[Tuple]): A list of edge tuples that include attributes.
                - For standard graphs: (u, v, data)
                - For multi-edge graphs: (u, v, key, data)

        Returns:
            List[Tuple]: A list of edge identifiers:
                - (u, v) for Graph / DiGraph
                - (u, v, key) for MultiGraph / MultiDiGraph
        """
        if self.multi:
            return [(u, v, k) for u, v, k, _ in edges_with_data]
        else:
            return [(u, v) for u, v, _ in edges_with_data]

    def quality_measure_base(
            self,
            pattern: Tuple[NominalSelector, ...],
            metric: str,
            measure_type: str = "qS"
    ) -> Pattern:
        """
        Computes the normalized quality score of a pattern's induced subgraph.

        Parameters:
            pattern (Tuple[str]): The selectors defining the subgroup pattern.
            metric (str): Evaluation metric to apply to the subgraph ('mean', 'var', etc.).
            measure_type (str): Quality scoring strategy ('qS' = structural, 'qP' = proportional).

        Returns:
            Pattern: A Pattern object containing the induced subgraph and its quality score.
        """
        raw_edges = self.get_edges_in_pattern(pattern)
        edges = self.extract_edge_keys(raw_edges)
        graph_of_pattern = self.graph.edge_subgraph(edges).copy()

        if measure_type == 'qS':
            n_nodes = len(graph_of_pattern)
            max_pattern_edges = float(n_nodes * (n_nodes - 1))
            if self.multi:
                max_pattern_edges += self.count_edges(graph_of_pattern)
        elif measure_type == 'qP':
            max_pattern_edges = None
        else:
            raise ValueError(f"Unsupported quality measure: {measure_type}")

        score = self.measure_score(graph_of_pattern, metric, max_pattern_edges)
        subgroup = Pattern(pattern, graph_of_pattern, score)

        mean, std = self.statistical_validation(
            n_samples=self.n_samples,
            pattern_size=graph_of_pattern.number_of_edges(),
            metric=metric,
            max_pattern_edges=max_pattern_edges
        )

        subgroup.quality = (subgroup.weight - mean) / std

        return subgroup

    def statistical_validation(
            self,
            n_samples: int,
            pattern_size: int,
            metric: str,
            max_pattern_edges: float
    ) -> Tuple[float, float]:
        """
        Estimates the mean and standard deviation of a quality score under the null hypothesis,
        by sampling random subgraphs with a fixed number of edges.

        Parameters:
            n_samples (int): Number of random subgraphs to sample.
            pattern_size (int): Number of edges in each sampled subgraph.
            metric (str): Quality metric to compute ('mean', 'var').
            max_pattern_edges (float): Normalization constant based on the pattern's node set.

        Returns:
            Tuple[float, float]: Mean and standard deviation of quality scores from sampled subgraphs.
        """
        list_of_edges = list(self.graph.edges(keys=True) if self.multi else self.graph.edges())
        graph_size = len(list_of_edges)

        def sample_and_score() -> float:
            if graph_size < pattern_size:
                return 0.0

            indices = self.npr.choice(graph_size, size=pattern_size, replace=False)
            sampled_edges = [list_of_edges[i] for i in indices]
            subgraph = self.graph.edge_subgraph(sampled_edges)
            return self.measure_score(subgraph, metric, max_pattern_edges)

        scores = (
            Parallel(n_jobs=self.n_jobs)(
                delayed(sample_and_score)() for _ in range(n_samples)
            ) if self.n_jobs > 1 else
            [sample_and_score() for _ in range(n_samples)]
        )

        return float(np.mean(scores)), float(np.std(scores))

    @staticmethod
    def measure_score(
            graph_of_pattern: nx.Graph,
            metric: str,
            max_pattern_edges: Optional[float] = None
    ) -> float:
        """
        Computes a normalized quality score for a given subgraph using the specified metric.

        Parameters:
            graph_of_pattern (nx.Graph): The subgraph induced by a pattern.
            metric (str): The scoring metric to use ('mean' or 'var').
            max_pattern_edges (float, optional): Maximum number of edges for normalization.
                Defaults to the current subgraph size.

        Returns:
            float: The computed quality score (e.g., mean weight density or normalized variance).
        """
        if max_pattern_edges is None:
            max_pattern_edges = graph_of_pattern.size()

        if max_pattern_edges == 0:
            return 0.0

        total_weight = graph_of_pattern.size(weight="weight")
        mean_weight = total_weight / max_pattern_edges

        if metric == 'mean':
            return mean_weight

        elif metric == 'var':
            weights = [d.get('weight', 0.0) for _, _, d in graph_of_pattern.edges(data=True)]
            variance = np.var(weights) / max_pattern_edges
            return variance

        else:
            raise ValueError(f"Unsupported metric: {metric}")

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

    @staticmethod
    def to_dataframe(subgroups: List[Pattern]) -> pd.DataFrame:
        """
        Summarizes a list of Pattern objects into a DataFrame.

        Args:
            subgroups (List[Pattern]): Patterns to summarize.

        Returns:
            pd.DataFrame: Summary table.
        """
        col_names = ['Pattern', 'Nodes', 'In Degree > 0', 'Out Degree > 0', 'Edges', 'Mean Weight', 'Score']
        dataframe = pd.DataFrame(columns=col_names)

        for p in subgroups:
            if not isinstance(p, Pattern):
                continue

            G = p.graph

            if hasattr(G, 'in_degree') and hasattr(G, 'out_degree'):
                # Directed or multi-directed
                in_nodes = sum(1 for _, deg in G.in_degree() if deg > 0)
                out_nodes = sum(1 for _, deg in G.out_degree() if deg > 0)
            else:
                # Undirected fallback
                in_nodes = out_nodes = sum(1 for _, deg in G.degree() if deg > 0)

            dataframe = pd.concat([
                dataframe,
                pd.DataFrame([{
                    'Pattern': p.name,
                    'Nodes': G.number_of_nodes(),
                    'In Degree > 0': in_nodes,
                    'Out Degree > 0': out_nodes,
                    'Edges': G.number_of_edges(),
                    'Mean Weight': round(p.weight, 1),
                    'Score': round(p.quality, 1)
                }])
            ], ignore_index=True)

        return dataframe


class DigraphSDMining(GraphSDMiningBase):
    """
    Subclass for directed graphs. Implements methods specific to directed edge interaction modeling.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.graph_type = nx.DiGraph
        self.multi = False

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


class MultiDigraphSDMining(GraphSDMiningBase):
    """
        Subclass for multi-directed graphs (parallel edges allowed).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.graph_type = nx.MultiDiGraph
        self.multi = True

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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.graph_type = nx.Graph
        self.multi = False

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
