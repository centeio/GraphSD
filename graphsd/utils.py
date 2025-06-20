import pandas as pd
from dataclasses import dataclass

@dataclass(frozen=True, order=True)
class NominalSelector:
    """
    Represents a nominal selector used in pattern descriptions for subgroup discovery.

    Attributes:
        attribute (str): Attribute name.
        value (Any): Attribute value.
    """
    attribute: str
    value: any

    def __str__(self):
        return f"({self.attribute}, {self.value})"


class Pattern:
    """
    Represents a discovered pattern in a graph.

    Attributes:
        name (List[NominalSelector])
        graph (nx.Graph)
        weight (float)
        quality (float)
    """
    def __init__(self, name: list, graph, weight: float):
        self.name = name
        self.graph = graph
        self.weight = weight
        self.quality = 0.0

    def __repr__(self):
        return str(self.name)

    def __eq__(self, other):
        return isinstance(other, Pattern) and set(self.name) == set(other.name)

    def __lt__(self, other):
        return self.quality < other.quality


class PatternWithoutGraph:
    """
    Pattern representation for cases where a graph is not required.

    Attributes:
        name (List[NominalSelector])
        ids (Set[Any])
        weight (float)
        quality (float)
    """
    def __init__(self, name: list, ids: set, weight: float):
        self.name = name
        self.ids = ids
        self.weight = weight
        self.quality = 0.0

    def __repr__(self):
        return str(self.name)

    def __eq__(self, other):
        return isinstance(other, PatternWithoutGraph) and set(self.name) == set(other.name)

    def __lt__(self, other):
        return self.quality < other.quality


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

# def freqItemsets(transactions, prun=20):
#     intTransactionsDict = {}
#     last = 1
#     intT = []
#
#     for trans in transactions:
#         temp = []
#         for att in trans:
#             if att not in intTransactionsDict:
#                 intTransactionsDict[att] = last
#                 last += 1
#             temp += [intTransactionsDict[att]]
#         intT += [temp]
#
#     inv_intTransactionsDict = {v: k for k, v in intTransactionsDict.items()}
#
#     itemsets = list(frequent_itemsets(intT, prun))
#
#     newTransactions = {}
#     for fset, count in itemsets:
#         first = True
#         for n in fset:
#             if first:
#                 temp = (inv_intTransactionsDict[n],)
#                 first = False
#             else:
#                 temp += (inv_intTransactionsDict[n],)
#
#         newTransactions[temp] = count
#
#     return newTransactions


# Only works for bi directional graphs
