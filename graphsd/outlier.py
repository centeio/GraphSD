from typing import List, Optional, Tuple

import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import Voronoi
from shapely.geometry import MultiPoint, Point, Polygon
from sklearn.neighbors import LocalOutlierFactor

from graphsd.utils import make_bins
from graphsd import PatternWithoutGraph, NominalSelector

logger = logging.getLogger(__name__)

# The class definition and previous functions remain unchanged

class OutlierSDMining:
    def __init__(self, random_state: Optional[int] = None, n_jobs: int = 1):
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.npr = np.random.RandomState(self.random_state)

    def compute_voronoi_areas(
        self,
        points: List[Tuple[float, float]],
        print_mode: bool = False,
    ) -> List[float]:
        """
        Compute areas of Voronoi cells for a set of 2D points.

        The Voronoi diagram partitions space such that each cell contains the region closest
        to one point. Infinite regions are clipped using a bounding box mask.

        Args:
            points (List[Tuple[float, float]]): List of (x, y) coordinates.
            print_mode (bool): Whether to plot the resulting Voronoi regions.

        Returns:
            List[float]: Area of each Voronoi cell corresponding to input points.
        """
        vor = Voronoi(points)
        regions, vertices = voronoi_finite_polygons_2d(vor)

        pts = MultiPoint([Point(p) for p in points])
        minx, miny, maxx, maxy = pts.bounds
        margin = 0.1 * (maxx - minx)
        mask = Polygon(
            [[minx - margin, miny - margin], [minx - margin, maxy + margin],
             [maxx + margin, maxy + margin], [maxx + margin, miny - margin]]
        )

        areas = []
        for region in regions:
            polygon = vertices[region]
            poly = Polygon(np.append(polygon, polygon[0]).reshape(-1, 2)).intersection(mask)
            if print_mode:
                plt.fill(*zip(*poly.exterior.coords), alpha=0.4)
            areas.append(poly.area)

        return areas

    def get_voronoi_areas(
        self,
        position_data: pd.DataFrame,
        social_data: pd.DataFrame,
        time_step: int = 1,
    ) -> pd.DataFrame:
        """
        Compute Voronoi cell area dynamics across sliding time windows.

        For each time step, the average location of each individual is used to compute
        Voronoi cells. The area of each cell is then assigned back to each user.

        Args:
            position_data (pd.DataFrame): Input data with 'id', 'x', 'y', 'time'.
            social_data (pd.DataFrame): Metadata with individual attributes.
            time_step (int): Window size in seconds.

        Returns:
            pd.DataFrame: Time-indexed Voronoi areas for all individuals.
        """
        start = min(position_data.time)
        end = max(position_data.time)
        areas_df = pd.DataFrame()
        ids = position_data.id.unique()

        while start <= end:
            delta = pd.Timedelta(seconds=time_step)
            window = position_data[
                (position_data["time"] >= start) & (position_data["time"] <= start + delta)
            ]

            temp_df = social_data[social_data["id"].isin(ids)].copy()
            position = (
                window
                .dropna(subset=["x", "y"])
                .groupby("id")
                .mean()
            )
            positions = position[["x", "y"]].values.tolist()
            pids = position.index.tolist()

            try:
                areas = self.compute_voronoi_areas(positions)
                temp_df["area"] = [areas[pids.index(pid)] if pid in pids else np.nan for pid in ids]
                temp_df["timestamp"] = start
                areas_df = pd.concat([areas_df, temp_df[np.isfinite(temp_df["area"])]])
            except Exception as e:
                logger.warning(f"Voronoi computation failed at {start}: {e}")

            start += delta

        return areas_df

    @staticmethod
    def get_lof(
        position_data: pd.DataFrame,
        social_data: pd.DataFrame,
        time_step: int = 1,
        k: int = 5,
        contamination: float = 0.1,
    ) -> pd.DataFrame:
        """
        Compute Local Outlier Factor (LOF) scores over sliding time windows.
        """
        from graphsd.outlier import compute_local_outlier_factor_scores

        start = min(position_data.time)
        end = max(position_data.time)
        lof_df = pd.DataFrame()
        ids = position_data.id.unique()

        while start <= end:
            delta = pd.Timedelta(seconds=time_step)
            window = position_data[
                (position_data["time"] >= start) & (position_data["time"] <= start + delta)
            ]

            temp_df = social_data[social_data["id"].isin(ids)].copy()
            position = (
                window
                .groupby("id")[["x", "y"]]
                .mean()
            )
            pids = position.index.tolist()

            if len(position) >= k:
                scores = compute_local_outlier_factor_scores(position, k=k, contamination=contamination)
            else:
                scores = [np.nan] * len(position)

            temp_df["lof"] = [scores[pids.index(pid)] if pid in pids else np.nan for pid in ids]
            temp_df["timestamp"] = start

            lof_df = pd.concat([lof_df, temp_df[np.isfinite(temp_df["lof"])]])
            start += delta

        return lof_df


# Utility function to reconstruct Voronoi polygons
def voronoi_finite_polygons_2d(vor: Voronoi, radius: Optional[float] = None) -> Tuple[List[List[int]], np.ndarray]:
    """
    Convert infinite Voronoi regions to finite polygons by bounding with a circular region.

    Args:
        vor (Voronoi): Scipy Voronoi object based on 2D points.
        radius (float, optional): Bounding radius for infinite edges.

    Returns:
        Tuple[List[List[int]], np.ndarray]:
            - List of indices representing each finite region.
            - Corresponding array of 2D vertices.
    """
    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()
    center = vor.points.mean(axis=0)
    radius = radius or vor.points.ptp().max()

    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]
        if all(v >= 0 for v in vertices):
            new_regions.append(vertices)
            continue

        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                continue

            t = vor.points[p2] - vor.points[p1]
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_regions.append(np.array(new_region)[np.argsort(angles)].tolist())

    return new_regions, np.asarray(new_vertices)


def compute_local_outlier_factor_scores(
    position_df: pd.DataFrame, k: int = 5, contamination: float = 0.1
) -> List[float]:
    """
    Compute negative LOF scores for given 2D positions.

    Args:
        position_df (pd.DataFrame): DataFrame with 'x' and 'y' columns.
        k (int): Number of neighbors for LOF.
        contamination (float): Expected proportion of outliers.

    Returns:
        List[float]: LOF anomaly scores (higher is more anomalous).
    """
    clf = LocalOutlierFactor(n_neighbors=k, contamination=contamination)
    clf.fit_predict(position_df)
    return [-x for x in clf.negative_outlier_factor_]


def qs_nodes_aux(dataframe: pd.DataFrame, eids: List[int], target: str) -> float:
    """
    Compute the average target value for a list of entity indices.

    Args:
        dataframe (pd.DataFrame): The input dataset.
        eids (List[int]): Row indices of selected entities.
        target (str): Column to average.

    Returns:
        float: Average of the target values.
    """
    return dataframe.loc[eids, target].mean()


def set_attributes(dataset: pd.DataFrame, attributes: List[str]) -> List[List[NominalSelector]]:
    """
    Encode a dataset into transaction format for subgroup discovery.

    Each transaction is a list of NominalSelectors corresponding to a single entity,
    based on the specified attribute columns.

    Args:
        dataset (pd.DataFrame): Input DataFrame with an 'id' column and feature columns.
        attributes (List[str]): List of column names to include as features.

    Returns:
        List[List[NominalSelector]]: List of transactions.
    """
    grouped = dataset.groupby("id", as_index=False)[attributes].first()
    return [
        [NominalSelector(attr, row[attr]) for attr in attributes]
        for _, row in grouped.iterrows()
    ]


def filter_pattern_and_score(
    dataset: pd.DataFrame,
    pattern: List[NominalSelector],
    target: str
) -> PatternWithoutGraph:
    """
    Filter the dataset according to a given pattern and compute the mean of a target column.

    Args:
        dataset (pd.DataFrame): Input data.
        pattern (List[NominalSelector]): A list of attribute-value conditions.
        target (str): Column to aggregate (e.g., 'lof', 'area').

    Returns:
        PatternWithoutGraph: Encapsulates the matched pattern, covered IDs, and mean target value.
    """
    try:
        query = " & ".join([
            f"{sel.attribute} == '{sel.value}'" if isinstance(sel.value, str)
            else f"{sel.attribute} == {sel.value}"
            for sel in pattern
        ])

        matched = dataset.query(query)
        ids = matched.index.tolist()
        weight = matched[target].mean() if not matched.empty else 0.0

        return PatternWithoutGraph(pattern, ids, weight)
    except Exception as e:
        logger.warning(f"Failed to filter or score pattern: {e}")
        return PatternWithoutGraph(pattern, [], 0.0)


def evaluate_pattern_significance(
    dataset: pd.DataFrame,
    pattern_selectors: List[NominalSelector],
    frequency: int,
    n_samples: int,
    target: str,
    random_state: Optional[int] = None
) -> PatternWithoutGraph:
    """
    Evaluate the quality of a pattern via Monte Carlo simulation.

    Args:
        dataset (pd.DataFrame): Data with feature and target columns.
        pattern_selectors (List[NominalSelector]): Conditions used to select rows.
        frequency (int): Number of covered instances (unused).
        n_samples (int): Number of random samples to draw.
        target (str): Target column name.
        random_state (Optional[int]): Seed for reproducibility.

    Returns:
        PatternWithoutGraph: The pattern with assigned quality score.
    """
    rng = np.random.RandomState(random_state)  # avoid using self in non-method
    pattern = filter_pattern_and_score(dataset, pattern_selectors, target)

    sampled_scores = [
        dataset.loc[rng.choice(dataset.index, size=len(pattern.ids), replace=False), target].mean()
        for _ in range(n_samples)
    ]

    sample_array = np.array(sampled_scores)
    sample_std = sample_array.std()
    pattern.quality = (pattern.weight - sample_array.mean()) / sample_std if sample_std > 0 else 0.0
    return pattern


def evaluate_tree_patterns(
    dataset: pd.DataFrame,
    node_patterns: dict,
    target: str
) -> List[PatternWithoutGraph]:
    """
    Evaluate a dictionary of patterns by computing quality scores for each.

    Args:
        dataset (pd.DataFrame): Full dataset.
        node_patterns (dict): Keys are pattern lists (NominalSelector), values are frequencies.
        target (str): Name of column used for evaluation.

    Returns:
        List[PatternWithoutGraph]: All valid patterns scored.
    """
    dataset = dataset.reset_index(drop=True)
    results = []

    for pattern, freq in node_patterns.items():
        try:
            results.append(evaluate_pattern_significance(dataset, pattern, freq, 100, target))
        except ZeroDivisionError:
            logger.warning(f"Zero division encountered for pattern: {pattern}")
            continue

    return results
