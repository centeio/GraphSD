import math
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
from scipy.spatial import distance


def bin_column(series: pd.Series, n_bins=3, strategy: str = "uniform") -> pd.Series:
    """
    Bins a numeric series using the specified strategy.

    Args:
        series (pd.Series): Numeric column to bin.
        n_bins (int or list): Number of bins or list of labels.
        strategy (str): "quantile" or "uniform" binning.

    Returns:
        pd.Series: Binned column.
    """
    if isinstance(n_bins, list):
        return pd.qcut(series, q=len(n_bins), labels=n_bins, duplicates="drop")

    try:
        if strategy == "quantile":
            return pd.qcut(series, q=n_bins, labels=False, duplicates="drop")
        elif strategy == "uniform":
            return pd.cut(series, bins=n_bins, labels=False)
        else:
            raise ValueError("Invalid strategy. Choose 'quantile' or 'uniform'.")
    except ValueError as e:
        print(f"[WARN] Fallback to uniform binning for column '{series.name}' due to: {e}")
        return pd.cut(series, bins=n_bins, labels=False)


def make_bins(dataframe: pd.DataFrame, n_bins=3, strategy: str = "uniform") -> pd.DataFrame:
    """
    Discretizes numeric columns of a DataFrame into bins, skipping low-variance columns.

    Args:
        dataframe (pd.DataFrame): Input data.
        n_bins (int or list): Number of bins or labels.
        strategy (str): "uniform" (default) or "quantile".

    Returns:
        pd.DataFrame: DataFrame with binned columns.
    """
    numeric_cols = dataframe.drop(columns=["id"], errors="ignore").select_dtypes(include="number").columns

    for col in numeric_cols:
        series = dataframe[col]
        if series.nunique() <= n_bins:
            print(f"[INFO] Skipping column '{col}' — only {series.nunique()} unique values (≤ bins: {n_bins})")
            continue

        try:
            dataframe[col] = bin_column(series, n_bins=n_bins, strategy=strategy)
        except ValueError as e:
            print(f"[WARN] Failed to bin column '{col}': {e}. Skipping.")
            continue

    return dataframe


def compute_velocities(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Computes velocity components in the X and Y directions for a trajectory DataFrame.
    The DataFrame must contain 'id', 'x', 'y', and 'time' columns.

    Velocity is calculated as the first-order difference in position
    within each 'id' group, assuming ascending timestamps (time).

    Args:
        dataframe (pd.DataFrame): A DataFrame with columns 'id', 'x', 'y', and 'timestamp'.

    Returns:
        pd.DataFrame: The input DataFrame with added 'velX' and 'velY' columns.
    """
    required_columns = {'id', 'x', 'y', 'time'}
    missing_columns = required_columns - set(dataframe.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    dataframe = dataframe.sort_values(by=['id', 'time']).reset_index(drop=True)

    dataframe['velX'] = dataframe.groupby('id')['x'].diff().fillna(0)
    dataframe['velY'] = dataframe.groupby('id')['y'].diff().fillna(0)

    return dataframe


def count_interactions(
        dataframe: pd.DataFrame,
        proximity: float = 1.0,
        time_step: int = 10,
        directed: bool = False,
        include_all_pairs: bool = False
) -> List[Tuple[int, int, Dict[str, Any]]]:
    """
    Computes pairwise interactions based on spatial proximity and optional directionality
    inferred from velocity alignment.

    Parameters:
        dataframe (pd.DataFrame): Must contain 'id', 'x', 'y', 'time', and optionally 'velX', 'velY'.
        proximity (float): Maximum distance to consider an interaction.
        time_step (int): Size of the time window (in time units).
        directed (bool): If True, treat interactions as directional based on cosine similarity.
        include_all_pairs (bool): Include all node pairs in the result, even with zero interactions.

    Returns:
        List[Tuple[int, int, Dict[str, Any]]]: List of edges with 'weight' attribute, suitable for NetworkX.
    """
    ids = dataframe['id'].unique()
    counter: Dict[Tuple[int, int], int] = {}

    if include_all_pairs:
        for id1 in ids:
            for id2 in ids:
                if id1 != id2:
                    counter[(id1, id2)] = 0

    start = dataframe['time'].min()
    end = start + time_step
    max_time = dataframe['time'].max()

    while start <= max_time:
        window = dataframe.query("@start <= time <= @end").groupby('id').mean()
        if window.empty:
            start = end
            end += time_step
            continue

        coords = window[['x', 'y']].to_numpy()
        dists = distance.cdist(coords, coords, 'euclidean')
        np.fill_diagonal(dists, np.inf)

        xs, ys = np.where(dists <= proximity)
        seen_pairs = set()

        for i in range(len(xs)):
            id_x = window.index[xs[i]]
            id_y = window.index[ys[i]]

            if id_x == id_y:
                continue

            if not directed:
                key = tuple(sorted((id_x, id_y)))
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
            else:
                key = (id_x, id_y)
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)

                try:
                    vel_x, vel_y = window.loc[id_x, ['velX', 'velY']]
                    dx = window.loc[id_y, 'x'] - window.loc[id_x, 'x']
                    dy = window.loc[id_y, 'y'] - window.loc[id_x, 'y']

                    dot = vel_x * dx + vel_y * dy
                    norm_product = math.sqrt(vel_x ** 2 + vel_y ** 2) * math.sqrt(dx ** 2 + dy ** 2)
                    cosine = dot / norm_product if norm_product else 0.0

                    if cosine < 0:
                        continue
                except KeyError:
                    continue

            counter[key] = counter.get(key, 0) + 1

        start = end
        end += time_step

    return [(a, b, {'weight': c}) for (a, b), c in counter.items()]
