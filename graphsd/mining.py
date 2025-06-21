from typing import List, Dict, Tuple, Union, Optional, Any

import networkx as nx
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.preprocessing import TransactionEncoder

from graphsd.patterns import Pattern, NominalSelector
from graphsd.utils import compute_velocities, count_interactions

import logging

logger = logging.getLogger(__name__)

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
        self._edge_attributes = None
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

    def _max_possible_edges(self, num_nodes: int) -> float:
        """
        Compute the maximum number of edges possible for the graph type.

        Args:
            num_nodes (int): Number of nodes.

        Returns:
            float: Theoretical upper bound of edges.
        """
        max_edges = float(num_nodes * (num_nodes - 1)) if self.directed else float(num_nodes * (num_nodes - 1) // 2)
        logger.debug(f"Computed max possible edges for n={num_nodes}: {max_edges}")
        return max_edges

    def _create_graph(self, edge_list: List[Tuple]) -> None:
        """
        Initialize a NetworkX graph from a list of edges with attributes.

        Sets:
            self.graph, self.theoretical_edges, self.observed_edges
        """
        G = self.graph_type()
        G.add_edges_from(edge_list)
        self.graph = G

        n = G.number_of_nodes()
        self.theoretical_edges = self._max_possible_edges(n)
        self.observed_edges = float(G.number_of_edges())

        logger.info(f"Created graph with {n} nodes and {self.observed_edges:.0f} edges.")
        logger.debug(f"Theoretical edges: {self.theoretical_edges:.0f}, Observed edges: {self.observed_edges:.0f}")

    def read_data(
            self,
            position_data: pd.DataFrame,
            social_data: pd.DataFrame,
            time_step: int = 10,
            proximity: float = 1.0
    ) -> None:
        """
        Processes positional data into a proximity-based interaction graph.

        Args:
            position_data (pd.DataFrame): Must include 'id', 'x', 'y', 'time'. Velocities are computed if needed.
            social_data (pd.DataFrame): Node-level or edge-level metadata. Should align with graph structure.
            time_step (int): Time window size for interaction aggregation.
            proximity (float): Max spatial distance to consider two entities as interacting.

        Returns:
            None
        """
        logger.info("Reading input data and constructing interaction graph...")

        if not isinstance(position_data, pd.DataFrame):
            raise ValueError("position_data must be a pandas DataFrame.")
        if not isinstance(social_data, pd.DataFrame):
            raise ValueError("social_data must be a pandas DataFrame.")

        directed = issubclass(self.graph_type, (nx.DiGraph, nx.MultiDiGraph))
        include_all = self.multi

        if directed:
            logger.debug("Graph is directed. Computing velocities...")
            position_data = compute_velocities(position_data)
            # Ensure columns use vel_x / vel_y instead of old velX / velY
            position_data.rename(columns={"velX": "vel_x", "velY": "vel_y"}, inplace=True)

        logger.debug("Computing interaction edges...")
        edge_list = count_interactions(
            position_data,
            proximity=proximity,
            time_step=time_step,
            directed=directed,
            include_all_pairs=include_all
        )

        logger.debug(f"Creating graph from {len(edge_list)} edges...")
        self._create_graph(edge_list)
        self.social_data = social_data.copy()
        logger.info("Graph construction complete.")

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
        Discovers high-quality subgroups based on frequent patterns and edge-level attributes.

        Args:
            mode (str): Strategy for extracting transactions ('comparison', 'to', 'from', etc.).
            min_support (float): Minimum support threshold (between 0 and 1).
            metric (str): Metric to evaluate on ('mean', 'var', etc.).
            quality_measure (str): Pattern quality function ('relative_density', 'qP', etc.).

        Returns:
            List[Pattern]: Patterns ranked by quality score.
        """
        if not (0 < min_support <= 1):
            raise ValueError("min_support must be in (0, 1].")

        logger.info(
            f"Starting subgroup discovery with mode='{mode}', min_support={min_support}, metric='{metric}', quality='{quality_measure}'")

        self.npr.seed(self.random_state)

        logger.debug("Annotating graph edges with social attributes...")
        self.annotate_edges(mode=mode)

        logger.debug("Extracting transactions from graph...")
        transactions = self.extract_transactions()

        logger.debug("Finding frequent itemsets...")
        itemsets = self.get_frequent_items(transactions, min_support)
        if not itemsets:
            logger.warning("No frequent itemsets found. Returning empty list.")
            return []

        logger.debug(f"Scoring {len(itemsets)} itemsets with '{quality_measure}'...")
        if self.n_jobs > 1:
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(self.quality_measure_base)(itemset, metric, quality_measure) for itemset in itemsets
            )
        else:
            results = [self.quality_measure_base(itemset, metric, quality_measure) for itemset in itemsets]

        sorted_results = sorted(results, key=lambda p: p.quality, reverse=True)
        logger.info(f"Finished subgroup discovery. Returned {len(sorted_results)} patterns.")
        return sorted_results

    def extract_edge_keys(self, edges_with_data: List[Tuple]) -> List[Tuple]:
        """
        Extract edge keys (u, v) or (u, v, k) from edge tuples with data.

        Args:
            edges_with_data (List[Tuple]): Edge tuples from the graph.

        Returns:
            List[Tuple]: Keys for edge_subgraph.
        """
        idx = 3 if self.multi else 2
        keys = [edge[:idx] for edge in edges_with_data]
        logger.debug(f"Extracted {len(keys)} edge keys from {len(edges_with_data)} input edges.")
        return keys

    def quality_measure_base(
            self,
            pattern: Tuple[NominalSelector, ...],
            metric: str,
            measure_type: str = "relative_density"
    ) -> Pattern:
        """
        Computes the quality of a pattern's induced subgraph.

        Args:
            pattern (Tuple[NominalSelector, ...]): Attribute-value selectors defining the pattern.
            metric (str): Metric to apply over the induced subgraph (e.g., 'mean').
            measure_type (str): One of 'relative_density' or 'global_proportion'.

        Returns:
            Pattern: A Pattern instance with computed weight and quality.
        """
        if not isinstance(pattern, tuple):
            raise TypeError("Pattern must be a tuple of NominalSelectors.")

        logger.debug(f"Scoring pattern: {pattern} with metric='{metric}', measure='{measure_type}'")

        raw_edges = self.get_edges_in_pattern(pattern)
        edges = self.extract_edge_keys(raw_edges)
        G_sub = self.graph.edge_subgraph(edges).copy()

        measure_mode = self.QUALITY_MEASURES.get(measure_type)
        if measure_mode is None:
            raise ValueError(f"Unsupported quality measure '{measure_type}'. "
                             f"Choose from {list(self.QUALITY_MEASURES.keys())}")

        max_edges = None
        if measure_mode == "relative_density":
            n = G_sub.number_of_nodes()
            max_edges = float(n * (n - 1))  # assuming directed
            logger.debug(f"Relative density: max_edges = {max_edges}")
        elif measure_mode == "global_proportion":
            logger.debug("Global proportion: using total graph mass as denominator")

        score = self.measure_score(G_sub, metric, max_edges)
        subgroup = Pattern(pattern, G_sub, score)

        mean, std = self.statistical_validation(
            n_samples=self.n_samples,
            pattern_size=G_sub.number_of_edges(),
            metric=metric,
            max_pattern_edges=max_edges
        )

        subgroup.quality = (subgroup.weight - mean) / std if std > 0 else 0.0
        logger.debug(f"Pattern score: {subgroup.weight:.3f}, normalized quality: {subgroup.quality:.3f}")

        return subgroup

    def statistical_validation(
            self,
            n_samples: int,
            pattern_size: int,
            metric: str,
            max_pattern_edges: float
    ) -> Tuple[float, float]:
        """
        Estimate the expected quality score and its variance under the null model by
        sampling subgraphs with a fixed number of edges.

        Args:
            n_samples (int): Number of samples to draw.
            pattern_size (int): Number of edges in each sampled subgraph.
            metric (str): Metric to evaluate ('mean', 'var').
            max_pattern_edges (float): Normalization factor for scoring.

        Returns:
            Tuple[float, float]: (mean score, std deviation score)
        """
        if pattern_size <= 0 or n_samples <= 0:
            return 0.0, 0.0

        edge_list = list(self.graph.edges(keys=True) if self.multi else self.graph.edges())
        total_edges = len(edge_list)

        if total_edges < pattern_size:
            logger.warning("Pattern size exceeds total number of edges. Returning zero stats.")
            return 0.0, 0.0

        def sample_score() -> float:
            sampled = self.npr.choice(total_edges, size=pattern_size, replace=False)
            sub_edges = [edge_list[i] for i in sampled]
            subgraph = self.graph.edge_subgraph(sub_edges)
            return self.measure_score(subgraph, metric, max_pattern_edges)

        logger.debug(f"Sampling {n_samples} subgraphs of {pattern_size} edges each...")

        scores = (
            Parallel(n_jobs=self.n_jobs)(
                delayed(sample_score)() for _ in range(n_samples)
            ) if self.n_jobs > 1 else
            [sample_score() for _ in range(n_samples)]
        )

        return float(np.mean(scores)), float(np.std(scores))

    @staticmethod
    def measure_score(
            graph_of_pattern: nx.Graph,
            metric: str,
            max_pattern_edges: Optional[float] = None
    ) -> float:
        """
        Compute a normalized score for a pattern's subgraph using the specified metric.

        Args:
            graph_of_pattern (nx.Graph): The induced subgraph of the pattern.
            metric (str): Scoring metric ('mean' or 'var').
            max_pattern_edges (float, optional): Normalization factor for edge-based metrics.
                If None, defaults to actual edge count.

        Returns:
            float: The normalized quality score.
        """
        if graph_of_pattern.number_of_edges() == 0:
            return 0.0

        if max_pattern_edges is None:
            max_pattern_edges = graph_of_pattern.number_of_edges()
        if max_pattern_edges == 0:
            return 0.0

        weights = [d.get("weight", 0.0) for _, _, d in graph_of_pattern.edges(data=True)]

        if metric == "mean":
            return sum(weights) / max_pattern_edges

        if metric == "var":
            return np.var(weights) / max_pattern_edges

        raise ValueError(f"Unsupported metric: '{metric}'. Choose 'mean' or 'var'.")

    def get_edges_in_pattern(self, pattern: Tuple[NominalSelector, ...]) -> list:
        """
        Return all edges that match all attribute-value selectors in the pattern.

        Args:
            pattern (Tuple[NominalSelector, ...]): The selectors to match.

        Returns:
            list: Matching edges.
        """
        if not isinstance(pattern, tuple):
            raise TypeError("Pattern must be a tuple of NominalSelectors.")

        selector_map = {sel.attribute: sel.value for sel in pattern}
        edge_iter = self.graph.edges(keys=True, data=True) if self.multi else self.graph.edges(data=True)

        matches = [
            edge for edge in edge_iter
            if all(edge[-1].get(attr) == val for attr, val in selector_map.items())
        ]

        logger.debug(f"Pattern {pattern} matched {len(matches)} edges.")
        return matches

    def extract_transactions(self) -> List[List[NominalSelector]]:
        """
        Extract transactions from annotated edges.

        Each transaction is a list of NominalSelectors, one per attribute,
        representing the attribute values attached to an edge.

        Returns:
            List[List[NominalSelector]]: One transaction per edge.
        """
        if not hasattr(self, "_edge_attributes"):
            raise RuntimeError("Edge attributes not initialized. Run annotate_edges() first.")

        attributes = self._edge_attributes
        transactions = []

        edge_iter = (
            self.graph.edges(keys=True, data=True)
            if self.multi else self.graph.edges(data=True)
        )

        for edge in edge_iter:
            edict = edge[-1]
            try:
                transaction = [NominalSelector(att, edict[att]) for att in attributes]
                transactions.append(transaction)
            except KeyError as e:
                logger.debug(f"Skipping edge {edge[:2]} due to missing attribute: {e}")

        logger.info(f"Extracted {len(transactions)} transactions from {self.graph.number_of_edges()} edges.")
        return transactions

    def annotate_edges(self, mode: str = "comparison") -> None:
        """
        Annotate graph edges with social attributes from node-level data.

        Args:
            mode (str): 'to', 'from', or 'comparison'.
        """
        if mode not in {"to", "from", "comparison"}:
            raise ValueError("mode must be one of: 'to', 'from', 'comparison'.")

        if not self.directed and mode in {"to", "from"}:
            logger.warning(f"Mode '{mode}' is invalid for undirected graphs. Using 'comparison' instead.")
            mode = "comparison"

        if "id" not in self.social_data.columns:
            raise ValueError("social_data must contain an 'id' column.")

        attributes = [col for col in self.social_data.columns if col != "id"]

        # Use flat lookup: {attribute: {node_id: value}}
        attr_lookup = {
            att: self.social_data.set_index("id")[att].to_dict()
            for att in attributes
        }

        edge_iter = (
            self.graph.edges(keys=True, data=True) if self.multi
            else self.graph.edges(data=True)
        )

        for edge in edge_iter:
            u, v = edge[:2]
            edict = edge[-1]

            for att in attributes:
                val_u = attr_lookup[att].get(u)
                val_v = attr_lookup[att].get(v)

                if val_u is None or val_v is None:
                    logger.debug(f"Skipping edge ({u}, {v}): missing value for '{att}'")
                    continue

                edict[att] = self._resolve_edge_value(val_u, val_v, mode)

        self._edge_attributes = attributes

    @staticmethod
    def _resolve_edge_value(val_u: Any, val_v: Any, mode: str) -> Any:
        """Resolve the attribute value to assign to an edge based on mode."""
        if mode == "to":
            return val_v
        if mode == "from":
            return val_u
        if isinstance(val_u, str) or isinstance(val_v, str):
            return str((val_u, val_v))
        if val_u == val_v:
            return "EQ"
        return ">" if val_u > val_v else "<"

    @staticmethod
    def to_dataframe(subgroups: List[Pattern]) -> pd.DataFrame:
        """
        Convert a list of Pattern objects into a summary DataFrame.

        Args:
            subgroups (List[Pattern]): The discovered patterns.

        Returns:
            pd.DataFrame: Summary of each pattern's structure and quality.
        """
        rows = []
        for p in subgroups:
            if not isinstance(p, Pattern):
                continue
            G = p.graph

            if hasattr(G, "in_degree") and hasattr(G, "out_degree"):
                in_nodes = sum(1 for _, d in G.in_degree() if d > 0)
                out_nodes = sum(1 for _, d in G.out_degree() if d > 0)
            else:
                degree_counts = [deg for _, deg in G.degree()]
                in_nodes = out_nodes = sum(1 for d in degree_counts if d > 0)

            rows.append({
                "Pattern": p.name,
                "Nodes": G.number_of_nodes(),
                "In Degree > 0": in_nodes,
                "Out Degree > 0": out_nodes,
                "Edges": G.number_of_edges(),
                "Mean Weight": round(p.weight, 1),
                "Score": round(p.quality, 1)
            })

        return pd.DataFrame(rows)


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
