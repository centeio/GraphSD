from typing import List, Dict, Tuple, Union, Optional, Any

import networkx as nx
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.preprocessing import TransactionEncoder

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
        self.directed: bool = False
        self.multi: bool = False

    QUALITY_MEASURES = {
        "qS": "relative_density",
        "qP": "global_proportion",
        "relative_density": "relative_density",
        "global_proportion": "global_proportion"
    }

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
        Builds a NetworkX graph using the declared graph type and provided edge list.

        Also computes the theoretical and observed number of edges for later use in quality scoring.

        Parameters:
            edge_list (List[Tuple[int, int, Dict[str, Any]]]): Edge data with 'weight' or other attributes.

        Returns:
            None
        """
        G = self.graph_type()
        G.add_edges_from(edge_list)
        self.graph = G

        n = G.number_of_nodes()

        # Theoretical maximum number of edges (no self-loops)
        if isinstance(G, (nx.DiGraph, nx.MultiDiGraph)):
            G.theoretical_edges = float(n * (n - 1))  # directed
        else:
            G.theoretical_edges = float(n * (n - 1) // 2)  # undirected

        # Actual observed number of edges (multi-edge-aware)
        G.observed_edges = float(G.number_of_edges())

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
            quality_measure: str = "relative_density"
    ) -> List[Pattern]:
        """
        Discovers high-quality subgroups based on frequent patterns and interaction quality.

        Parameters:
            mode (str): Strategy for extracting attribute transactions ('comparison', 'composition', etc.)
            min_support (float): Minimum frequency threshold (0 < s ≤ 1).
            metric (str): Target metric to evaluate subgraphs ('mean', 'var', etc.)
            quality_measure (str): Quality function to use ('relative_density', 'global_proportion', etc.)

        Returns:
            List[Pattern]: Sorted list of patterns ranked by quality.
        """
        self.npr.seed(self.random_state)

        self.annotate_edges(mode=mode)
        transactions = self.extract_transactions()
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
            measure_type: str = "relative_density"
    ) -> Pattern:
        """
        Computes the normalized quality score of a pattern's induced subgraph.

        Parameters:
            pattern (Tuple[NominalSelector, ...]): Attribute-value selectors defining the pattern.
            metric (str): Evaluation metric ('mean', 'var', etc.).
            measure_type (str): Quality scoring strategy.
                Supported:
                    - 'relative_density' (qS): based on structural potential of the pattern subgraph
                    - 'global_proportion' (qP): based on total graph edge mass

        Returns:
            Pattern: A Pattern object with the subgraph and its computed quality score.
        """
        raw_edges = self.get_edges_in_pattern(pattern)
        edges = self.extract_edge_keys(raw_edges)
        graph_of_pattern = self.graph.edge_subgraph(edges).copy()

        # Normalize shorthand identifiers
        measure_type = self.QUALITY_MEASURES.get(measure_type, None)
        if measure_type is None:
            raise ValueError(
                "Unsupported quality measure. Use one of: "
                f"{list(self.QUALITY_MEASURES.keys())}"
            )

        if measure_type == 'relative_density':
            n_nodes = graph_of_pattern.number_of_nodes()
            max_pattern_edges = float(n_nodes * (n_nodes - 1))
        elif measure_type == 'global_proportion':
            max_pattern_edges = None

        score = self.measure_score(graph_of_pattern, metric, max_pattern_edges)
        subgroup = Pattern(pattern, graph_of_pattern, score)

        mean, std = self.statistical_validation(
            n_samples=self.n_samples,
            pattern_size=graph_of_pattern.number_of_edges(),
            metric=metric,
            max_pattern_edges=max_pattern_edges
        )

        subgroup.quality = (subgroup.weight - mean) / std if std > 0 else 0.0

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

    def get_edges_in_pattern(
            self,
            pattern: Tuple[NominalSelector, ...]
    ) -> List[Union[
        Tuple[int, int, dict],  # For Graph / DiGraph
        Tuple[int, int, int, dict]  # For MultiGraph / MultiDiGraph
    ]]:
        """
        Extracts edges from the main graph that satisfy all attribute-value selectors in the pattern.

        Parameters:
            pattern (Tuple[NominalSelector, ...]): Selectors defining the edge pattern to match.

        Returns:
            List[Tuple]: A list of edges (u, v, data) or (u, v, key, data), depending on graph type.
        """
        if self.multi:
            edge_iter = self.graph.edges(keys=True, data=True)
        else:
            edge_iter = self.graph.edges(data=True)

        matching_edges = []

        for edge in edge_iter:
            data = edge[-1]  # data dict is always the last element
            if all(data.get(sel.attribute) == sel.value for sel in pattern):
                matching_edges.append(edge)

        return matching_edges

    def extract_transactions(self) -> List[List[NominalSelector]]:
        """
        Extracts transactions (attribute-value itemsets) from annotated edges in the graph.

        Returns:
            List[List[NominalSelector]]: One transaction per edge, based on its annotated attributes.
        """
        attributes = self.social_data.drop(columns=["id"]).columns
        transactions = []

        edges = (
            self.graph.edges(keys=True, data=True)
            if self.multi else self.graph.edges(data=True)
        )

        for edge in edges:
            edict = edge[-1]
            tr = [NominalSelector(att, edict[att]) for att in attributes]
            transactions.append(tr)

        return transactions

    def annotate_edges(self, mode: str = "comparison") -> None:
        """
        Annotates each edge in the graph with attributes derived from node-level social data.

        Parameters:
            mode (str): Strategy for encoding node attributes on edges.
                - "to": takes attributes from the target node (only for directed graphs)
                - "from": takes attributes from the source node (only for directed graphs)
                - "comparison": encodes the relationship between source and target attributes

        Raises:
            ValueError: If mode is invalid for the graph type.
        """
        if not self.directed and mode in {"to", "from"}:
            # No direction in undirected graphs — fall back to symmetric comparison
            print(f"[INFO] Mode '{mode}' is not applicable to undirected graphs — falling back to 'comparison'.")
            mode = "comparison"

        attributes = self.social_data.drop(columns=["id"]).columns

        edges = (
            self.graph.edges(keys=True, data=True)
            if self.multi else self.graph.edges(data=True)
        )

        for edge in edges:
            u, v = edge[:2]
            edict = edge[-1]

            for att in attributes:
                val_u = self.social_data[self.social_data.id == u][att].item()
                val_v = self.social_data[self.social_data.id == v][att].item()

                if mode == "to":
                    val = val_v
                elif mode == "from":
                    val = val_u
                elif mode == "comparison":
                    if isinstance(val_u, str):
                        val = str((val_u, val_v))
                    elif val_u == val_v:
                        val = "EQ"
                    elif val_u > val_v:
                        val = ">"
                    else:
                        val = "<"
                else:
                    raise ValueError("Invalid mode. Use one of: 'to', 'from', 'comparison'.")

                edict[att] = val

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
        self.directed = True
        self.multi = False


class MultiDigraphSDMining(GraphSDMiningBase):
    """
        Subclass for multi-directed graphs (parallel edges allowed).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.graph_type = nx.MultiDiGraph
        self.directed = True
        self.multi = True

class GraphSDMining(GraphSDMiningBase):
    """
    Subclass for undirected graphs. Implements edge construction and attribute mapping for symmetric interactions.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.graph_type = nx.Graph
        self.directed = False
        self.multi = False
