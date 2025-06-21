import math
from typing import List, Tuple, Dict, Any, Union

import numpy as np
import pandas as pd
from scipy.spatial import distance
from itertools import permutations

import logging

logger = logging.getLogger(__name__)

def bin_column(
    series: pd.Series,
    n_bins: Union[int, List[str]] = 3,
    strategy: str = "uniform"
) -> pd.Series:
    """
    Bins a numeric pandas Series using a specified strategy.

    Args:
        series (pd.Series): Numeric column to bin.
        n_bins (int or list): Number of bins (int) or custom bin labels (list).
        strategy (str): "quantile" or "uniform" (default).

    Returns:
        pd.Series: Binned series.

    Raises:
        ValueError: If the strategy is not recognized.
    """
    logger.debug(f"Binning series '{series.name}' with strategy='{strategy}' and n_bins={n_bins}")

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
        logger.warning(f"Fallback to uniform binning for column '{series.name}' due to: {e}")
        return pd.cut(series, bins=n_bins, labels=False)


def make_bins(
    dataframe: pd.DataFrame,
    n_bins: Union[int, List[str]] = 3,
    strategy: str = "uniform"
) -> pd.DataFrame:
    """
    Discretizes numeric columns in the DataFrame into bins using the specified strategy.
    Columns with fewer unique values than the number of bins are skipped.
    The column 'id' is excluded automatically if present.

    Args:
        dataframe (pd.DataFrame): Input DataFrame.
        n_bins (int or list): Number of bins (int) or custom labels (list).
        strategy (str): "uniform" or "quantile".

    Returns:
        pd.DataFrame: A copy of the DataFrame with binned numeric columns.
    """
    df = dataframe.copy()
    numeric_cols = df.drop(columns=["id"], errors="ignore").select_dtypes(include="number").columns

    logger.debug(f"Numeric columns selected for binning: {list(numeric_cols)}")

    for col in numeric_cols:
        series = df[col]
        bin_count = len(n_bins) if isinstance(n_bins, list) else n_bins

        if series.nunique() <= bin_count:
            logger.info(f"Skipping column '{col}' — only {series.nunique()} unique values (≤ bins: {bin_count})")
        else:
            try:
                logger.debug(f"Binning column '{col}'")
                df[col] = bin_column(series, n_bins=n_bins, strategy=strategy)
            except ValueError as e:
                logger.warning(f"Failed to bin column '{col}': {e}. Skipping.")

    logger.info("Completed binning of numeric columns.")
    return df


def compute_velocities(
    dataframe: pd.DataFrame,
    id_col: str = "id",
    x_col: str = "x",
    y_col: str = "y",
    time_col: str = "time"
) -> pd.DataFrame:
    """
    Computes velocity components (first-order differences) in the X and Y directions.
    Column names are configurable to support varied input schemas.

    Args:
        dataframe (pd.DataFrame): Input data with coordinate and time columns.
        id_col (str): Column representing entity identity.
        x_col (str): Column representing X coordinate.
        y_col (str): Column representing Y coordinate.
        time_col (str): Column representing timestamp.

    Returns:
        pd.DataFrame: A copy of the DataFrame with added 'velX' and 'velY' columns.

    Raises:
        ValueError:
            If any required columns are missing.
    """
    required_columns = {id_col, x_col, y_col, time_col}
    logger.debug(f"Checking for required columns: {required_columns}")
    missing = required_columns - set(dataframe.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = dataframe.copy()
    logger.debug(f"Sorting DataFrame by '{id_col}' and '{time_col}'")
    df = df.sort_values(by=[id_col, time_col]).reset_index(drop=True)

    logger.debug(f"Computing differences for '{x_col}' and '{y_col}' grouped by '{id_col}'")
    df["velX"] = df.groupby(id_col)[x_col].diff().fillna(0)
    df["velY"] = df.groupby(id_col)[y_col].diff().fillna(0)

    logger.info("Velocity columns 'velX' and 'velY' computed successfully.")
    return df


def is_directionally_valid(src: int, tgt: int, window: pd.DataFrame) -> bool:
    try:
        vel_x, vel_y = window.loc[src, ['velX', 'velY']]
        dx = window.loc[tgt, 'x'] - window.loc[src, 'x']
        dy = window.loc[tgt, 'y'] - window.loc[src, 'y']
        dot = vel_x * dx + vel_y * dy
        norm = math.hypot(vel_x, vel_y) * math.hypot(dx, dy)
        return (dot / norm if norm else 0.0) >= 0
    except KeyError:
        logger.debug(f"Missing velocity data for {src} → {tgt}")
        return False


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

    Args:
        dataframe (pd.DataFrame): Must contain 'id', 'x', 'y', 'time', and optionally 'vel_x', 'vel_y'.
        proximity (float): Maximum distance to consider an interaction.
        time_step (int): Size of the time window (in time units).
        directed (bool): If True, treat interactions as directional based on cosine similarity.
        include_all_pairs (bool): Include all node pairs in the result, even with zero interactions.

    Returns:
        List[Tuple[int, int, Dict[str, Any]]]: List of edges with 'weight' attribute, suitable for NetworkX.
    """
    logger.info("Starting interaction count...")
    ids = dataframe["id"].unique()
    logger.debug(f"Found {len(ids)} unique IDs")

    counter: Dict[Tuple[int, int], int] = {
        pair: 0 for pair in permutations(ids, 2) if include_all_pairs and pair[0] != pair[1]
    }

    start, max_time = dataframe["time"].min(), dataframe["time"].max()
    end = start + time_step
    logger.debug(f"Time range: [{start}, {max_time}], step: {time_step}")

    while start <= max_time:
        logger.debug(f"Processing window [{start}, {end}]")
        mask = (dataframe["time"] >= start) & (dataframe["time"] <= end)
        grouped = dataframe.loc[mask].groupby("id")

        # Dynamically include velocity columns if present
        agg_columns = {"x": "mean", "y": "mean"}
        if {"vel_x", "vel_y"}.issubset(dataframe.columns):
            agg_columns.update({"vel_x": "mean", "vel_y": "mean"})

        window = grouped.agg(agg_columns)

        if directed and not {"vel_x", "vel_y"}.issubset(window.columns):
            logger.warning("Directed interactions requested but velocity columns not found. Directionality will be skipped.")

        if window.empty:
            logger.debug("Window is empty, skipping.")
            start = end
            end += time_step
            continue

        coords = window[["x", "y"]].to_numpy()
        dists = distance.cdist(coords, coords, "euclidean")
        np.fill_diagonal(dists, np.inf)

        xs, ys = np.where(dists <= proximity)
        logger.debug(f"Detected {len(xs)} close pairs")

        seen_pairs = set()

        for i in range(len(xs)):
            id_x = window.index[xs[i]]
            id_y = window.index[ys[i]]
            if id_x == id_y:
                continue

            key = (id_x, id_y) if directed else tuple(sorted((id_x, id_y)))
            if key in seen_pairs:
                continue
            seen_pairs.add(key)

            if directed:
                try:
                    vel_x, vel_y = window.loc[id_x, ["vel_x", "vel_y"]]
                    dx = window.loc[id_y, "x"] - window.loc[id_x, "x"]
                    dy = window.loc[id_y, "y"] - window.loc[id_x, "y"]
                    dot = vel_x * dx + vel_y * dy
                    norm = math.hypot(vel_x, vel_y) * math.hypot(dx, dy)
                    cosine = dot / norm if norm else 0.0
                    if cosine < 0:
                        continue
                except KeyError:
                    logger.debug(f"Missing velocity for directed pair: {id_x} → {id_y}")
                    continue

            counter[key] = counter.get(key, 0) + 1

        start = end
        end += time_step

    logger.info(f"Completed interaction counting. {len(counter)} unique pairs found.")
    return [(a, b, {'weight': c}) for (a, b), c in counter.items()]


__all__ = [
    "bin_column",
    "make_bins",
    "compute_velocities",
    "is_directionally_valid",
    "count_interactions"
]
