import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import Voronoi
from shapely.geometry import MultiPoint, Point, Polygon
from sklearn.neighbors import LocalOutlierFactor
from typing import List, Optional
from graphsd.utils import get_rng
from graphsd import PatternWithoutGraph, NominalSelector


class OutlierSDMining(object):
    """
    Outlier mining class for spatial-temporal trajectory data.

    Provides functionality to compute:
    - Local Outlier Factor (LOF) scores over time
    - Voronoi cell area dynamics
    - Subgroup discovery on node-level anomaly indicators

    Attributes:
        quality_measure (str): Quality function type (e.g., 'qS').
        n_bins (int): Number of bins for discretization.
        n_samples (int): Number of samples for significance testing.
        metric (str): Metric used in quality evaluation.
        mode (str): Attribute comparison mode.
        rng (int or None): Random seed.
        n_jobs (int): Number of parallel processes to use.

        graph (nx.Graph): Not used in LOF/area directly, placeholder.
        transactions (Any): Placeholder for transaction data.
        social_data (pd.DataFrame): Metadata for individuals.
        graph_type (str): Not used directly; reserved for compatibility.
    """
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
        self.rng = get_rng(random_state)
        self.n_jobs = n_jobs

        self.graph = None
        self.transactions = None
        self.social_data = None

        self.graph_type = None

    def get_lof(self, position_data, social_data, time_step=1, k=5, contamination=.1):
        """
        Calculates the Local Outlier Factor (LOF) score for each individual across time windows.

        Parameters:
            position_data (pd.DataFrame): DataFrame with 'x', 'y', 'time', and 'id'.
            social_data (pd.DataFrame): DataFrame with 'id' and metadata per individual.
            time_step (int): Size of the sliding time window (in seconds).
            k (int): Number of neighbors to use in LOF.
            contamination (float): LOF contamination parameter.

        Returns:
            pd.DataFrame: Time-indexed LOF scores per individual.
        """
        start = min(position_data.time)
        end = start + time_step
        areas_df = pd.DataFrame()
        ids = position_data.id.unique()

        while start <= pd.Timestamp(end):
            temp_df = social_data[social_data['id'].isin(ids)].copy()
            position = position_data.query("@start <= time <= @end").groupby(['id']).mean()
            pids = [row.id for index, row in position.iterrows()]

            areasp = self.compute_local_outlier_factor_scores(position[['x', 'y']], k=k, contamination=contamination)

            temp_areas = []
            temp_times = []

            try:
                i = 0
                for pid in ids:
                    temp_times += [start]
                    if pid in pids:
                        temp_areas += [areasp[i]]
                        i += 1
                    else:
                        temp_areas += [np.nan]

                temp_df['lof'] = temp_areas
                temp_df['timestamp'] = temp_times

            except:
                print("An exception occurred at: ", start)

            areas_df = pd.concat([areas_df, temp_df[np.isfinite(temp_df['lof'])]])
            start = start + pd.Timedelta(seconds=n_seconds)

        return areas_df

    @staticmethod
    def compute_local_outlier_factor_scores(position_df, k=5, contamination=.1):
        """
        Computes negative LOF scores for given positions.

        Parameters:
            position_df (pd.DataFrame): DataFrame with 'x' and 'y' columns.
            k (int): Number of neighbors for LOF.
            contamination (float): LOF contamination parameter.

        Returns:
            List[float]: LOF scores (higher = more anomalous).
        """
        clf = LocalOutlierFactor(n_neighbors=k, contamination=contamination)
        clf.fit_predict(position_df)
        lof_scores = clf.negative_outlier_factor_

        return [-x for x in lof_scores]

    @staticmethod
    def compute_voronoi_areas(points, pids, print_mode=False):
        """
        Computes Voronoi cell areas for a given set of 2D points.

        Parameters:
            points (List[Tuple[float, float]]): List of 2D coordinates.
            pids (List[Any]): Point IDs (same order as `points`).
            print_mode (bool): If True, plots filled Voronoi cells.

        Returns:
            List[float]: Area of each cell.
        """
        vor = Voronoi(points)
        new_vertices = []

        regions, vertices = voronoi_finite_polygons_2d(vor)

        pts = MultiPoint([Point(i) for i in points])
        minx, miny, maxx, maxy = pts.bounds

        factor = 0.1
        margin = minx * factor
        mask = MultiPoint([Point(i) for i in
                           [[minx - margin, miny - margin], [minx - margin, maxy + margin],
                            [maxx + margin, miny - margin],
                            [maxx + margin, maxy + margin]]]).convex_hull

        areas = []
        for region in regions:
            polygon = vertices[region]
            shape = list(polygon.shape)
            shape[0] += 1
            p = Polygon(np.append(polygon, polygon[0]).reshape(*shape)).intersection(mask)
            areas += [p.area]

            poly = np.array(list(zip(p.boundary.coords.xy[0][:-1], p.boundary.coords.xy[1][:-1])))
            new_vertices.append(poly)
            if print_mode:
                plt.fill(*zip(*poly), alpha=0.4)

        return areas

    def get_voronoi_areas(self, position_data, social_data, time_step=1):
        """
        Computes Voronoi cell areas over time for each individual.

        Parameters:
            position_data (pd.DataFrame): DataFrame with 'x', 'y', 'time', and 'id'.
            social_data (pd.DataFrame): DataFrame with 'id' and attributes.
            time_step (int): Time window size in seconds.

        Returns:
            pd.DataFrame: Time-indexed Voronoi areas per individual.
        """
        # nseconds = 1
        start = min(position_data.time)
        end = start + time_step
        areas_df = pd.DataFrame()
        counter = 0
        ids = position_data.id.unique()

        while start <= pd.Timestamp(end):
            temp_df = social_data[social_data['id'].isin(ids)].copy()
            position = position_data.query("@start <= time <= @end").query("x != 'NaN'").groupby(['id']).mean()
            positions = [[row.x, row.y] for index, row in position.iterrows()]
            pids = [row.id for index, row in position.iterrows()]

            try:
                areasp = self.compute_voronoi_areas(positions, pids, print_mode=False)
                temp_areas = []
                temp_times = []

                i = 0
                for pid in ids:
                    temp_times += [start]
                    if pid in pids:
                        temp_areas += [areasp[i]]
                        i += 1
                    else:
                        temp_areas += [np.nan]

                temp_df['area'] = temp_areas
                temp_df['timestamp'] = temp_times

            except:
                print("An exception occurred at: ", start)
                start = start + pd.Timedelta(seconds=time_step)
                continue

            areas_df = pd.concat([areas_df, temp_df[np.isfinite(temp_df['area'])]])
            start = start + pd.Timedelta(seconds=time_step)
            counter += 1

        return areas_df


def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstructs infinite Voronoi regions in 2D to finite ones.

    Code adapted from: https://gist.github.com/pv/8036995

    Parameters:
        vor (scipy.spatial.Voronoi): Input Voronoi diagram.
        radius (float, optional): Distance to 'points at infinity'.

    Returns:
        Tuple[List[List[int]], np.ndarray]:
            - List of region vertex indices (clipped to finite polygons).
            - Array of polygon vertices.
    """
    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        # print(p1, region)
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)


def setAtts(dataset, atts):
    """
    Converts a dataset into transactions based on selected attributes.

    Parameters:
        dataset (pd.DataFrame): DataFrame with entity rows and named attributes.
        atts (List[str]): List of attribute column names to encode.

    Returns:
        List[List[NominalSelector]]: One list of selectors per row.
    """
    ids = dataset.id.unique()
    transactions = []
    tr = []
    for nid in ids:
        tr = []
        eattr = {}
        for att in atts:
            res = list(dataset[dataset.id == nid][att])[0]
            tr.append(NominalSelector(att, res))

        transactions.append(tr)

    return transactions


def idsInP(dataset, P, target):
    """
    Filters rows matching a pattern and computes the average target value.

    Parameters:
        dataset (pd.DataFrame): DataFrame with data to filter.
        P (List[NominalSelector]): Pattern of attribute-value conditions.
        target (str): Column name for the target variable (e.g., 'lof' or 'area').

    Returns:
        PatternWithoutGraph: A pattern object with matching IDs and computed weight.
    """
    pids = []
    wsum = 0

    query = ""
    first = True

    for sel in P:
        if first:
            first = False
        else:
            query += " & "
        if type(sel.value) == str:
            query += sel.attribute + " == '" + sel.value + "'"
        else:
            query += sel.attribute + " == " + str(sel.value)

    print(query)

    for index, row in dataset.query(query).iterrows():
        pids += [index]
        wsum += row[target]

    nEp = len(pids)  # number of nodes covered by a pattern P
    nE = len(dataset)  # number of all possible edges

    if nE == 0:
        w = 0
    else:
        w = wsum / nEp

    print(w)
    pat = PatternWithoutGraph(P, pids, w)

    return pat


# Measure quality of subgroups
def qs_nodes_aux(dataframe, eids, target):
    """
    Computes the mean target value for a list of entity IDs.

    Parameters:
        dataframe (pd.DataFrame): Source data.
        eids (List[int]): List of row indices.
        target (str): Target column to average.

    Returns:
        float: Mean target value.
    """
    wsum = 0

    for index, row in dataframe.iloc[eids, :].iterrows():
        wsum += row[target]

    return wsum / len(eids)


# def qs_nodes(dataset, P, freq, n_samples, target):
#     """
#     Evaluates pattern quality using Monte Carlo sampling over entities.
#
#     Parameters:
#         dataset (pd.DataFrame): Full dataset.
#         P (List[NominalSelector]): Pattern conditions.
#         freq (int): Frequency of the pattern (not used).
#         n_samples (int): Number of random samples for null distribution.
#         target (str): Target variable for quality scoring.
#
#     Returns:
#         PatternWithoutGraph: Pattern object with quality score set.
#     """
#     pat = idsInP(dataset, P, target)
#
#     # totalE = round((len(list(G.nodes())) * (len(list(G.nodes())) - 1)))
#
#     temp_ids = dataset.index.values.tolist()
#     sample = []
#     rng = get_rng(random_state)
#
#     for r in range(n_samples):
#         indxs = np.random.choice(temp_ids, size=len(pat.ids), replace=False)
#         tempres = qs_nodes_aux(dataset, indxs, target)
#         sample = np.append(sample, [tempres])
#
#     mean = np.mean(sample)
#     std = np.std(sample)
#     pat.quality = (pat.weight - mean) / std
#
#     return pat

def qs_nodes(
    dataset: pd.DataFrame,
    pattern_selectors: List['NominalSelector'],
    frequency: int,
    n_samples: int,
    target: str,
    random_state: Optional[int] = None
) -> PatternWithoutGraph:
    """
    Evaluates pattern quality using Monte Carlo sampling over entities.

    Parameters
    ----------
    dataset : pd.DataFrame
        Full dataset.
    pattern_selectors : List[NominalSelector]
        Pattern conditions to apply.
    frequency : int
        Frequency of the pattern (currently unused).
    n_samples : int
        Number of random samples for the null distribution.
    target : str
        Target variable used for quality scoring.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    PatternWithoutGraph
        Pattern object with updated quality score.
    """
    rng = get_rng(random_state)
    pattern = idsInP(dataset, pattern_selectors, target)

    temp_ids = dataset.index.tolist()
    results = []

    for _ in range(n_samples):
        sampled_indices = rng.choice(temp_ids, size=len(pattern.ids), replace=False)
        results.append(qs_nodes_aux(dataset, sampled_indices, target))

    sample_array = np.array(results)
    mean = np.mean(sample_array)
    std = np.std(sample_array)

    pattern.quality = (pattern.weight - mean) / std if std != 0 else 0.0
    return pattern

def treeQuality_nodes(dataset, nodes, target):
    """
    Applies `qs_nodes` to a set of pattern nodes and computes their quality scores.

    Parameters:
        dataset (pd.DataFrame): The full dataset of entities and attributes.
        nodes (dict): Dictionary where keys are patterns (List[NominalSelector])
                      and values are frequencies.
        target (str): Name of the column to evaluate (e.g., 'lof', 'area').

    Returns:
        List[PatternWithoutGraph]: List of patterns with computed quality scores.
    """
    dataset = dataset.reset_index(drop=True)
    qs = []
    for k, val in nodes.items():
        try:
            pat = qs_nodes(dataset, k, val, 100, target)
            qs += [pat]
        except ZeroDivisionError:
            continue

    return qs
