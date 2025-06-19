import math
import pandas as pd


class NominalSelector:
    """
        Represents a nominal selector used in pattern descriptions for subgroup discovery.

        Attributes:
            attribute (str): Attribute name.
            value (Any): Attribute value.
    """
    def __init__(self, attribute, value):
        self.attribute = attribute
        self.value = value

    def __repr__(self):
        return "(" + self.attribute + ", " + str(self.value) + ")"

    def __str__(self):
        return "(" + self.attribute + ", " + str(self.value) + ")"

    def __eq__(self, other):
        return self.attribute == other.attribute and self.value == other.value

    def __lt__(self, other):
        if self.attribute != other.attribute:
            return self.attribute < other.attribute
        else:
            return self.value < other.value

    def __hash__(self):
        return hash(str(self))


class Pattern:
    """
        Represents a discovered pattern in the graph.

        Attributes:
            name (List[NominalSelector]): List of conditions.
            graph (nx.Graph): Subgraph induced by the pattern.
            weight (float): Mean edge weight.
            quality (float): Quality score of the pattern.
    """
    def __init__(self, name, graph, weight):  # name has to be of type list of NominalSelector
        self.name = name
        self.graph = graph
        self.weight = weight
        self.quality = 0

    def __repr__(self):
        return str(self.name)

    def __str__(self):
        return str(self.name)

    def __eq__(self, other):
        set(self.name) == set(other.name)

    def __lt__(self, other):
        return self.quality < other.quality


class NoGPattern:
    """
        Pattern representation for cases where a graph is not required.

        Attributes:
            name (List[NominalSelector]): List of conditions.
            ids (Set[Any]): Set of entity IDs matching the pattern.
            weight (float): Mean weight.
            quality (float): Pattern's quality score.
    """
    def __init__(self, name, ids, weight):  # name has to be of type list of NominalSelector
        self.name = name
        self.weight = weight
        self.quality = 0
        self.ids = ids

    def __repr__(self):
        return str(self.name)

    def __str__(self):
        return str(self.name)

    def __eq__(self, other):
        set(self.name) == set(other.name)

    def __lt__(self, other):
        return self.quality < other.quality


def make_bins(dataframe, n_bins=3):
    """
        Discretizes numeric columns of a DataFrame into bins.

        Parameters:
            dataframe (pd.DataFrame): DataFrame with numeric columns.
            n_bins (int): Number of bins to use.

        Returns:
            pd.DataFrame: DataFrame with binned columns.
    """
    columns = dataframe.drop(['id'], axis=1)._get_numeric_data().columns

    for c in columns:
        if isinstance(n_bins, int):
            dataframe[c] = get_bins_2(dataframe[c].values, n_bins)
        elif isinstance(n_bins, list):
            dataframe[c] = pd.qcut(dataframe[c], len(n_bins), labels=n_bins)

    return dataframe


def get_bins(data, n_bins=3):
    """
        Calculates equal-width bin edges for numeric data.

        Parameters:
            data (array-like): Input numeric data.
            n_bins (int): Number of bins.

        Returns:
            List[float]: Bin edges.
    """
    data = list(data)

    data_points_per_bin = math.ceil(len(data) / n_bins)
    sortedData = data.copy()
    sortedData.sort()

    binList = data.copy()

    limitinf = min(data)
    for el in range(len(data)):
        if data[el] == limitinf:
            binList[el] = 0

    for j in range(n_bins):
        limitsup = sortedData[data_points_per_bin * (j + 1) - 1]
        for el in range(len(data)):
            if limitinf < data[el] <= limitsup:
                binList[el] = j

        limitinf = limitsup

    return binList


def get_bins_2(data, n_bins=3):
    """
        Alternate binning strategy using quantiles.

        Parameters:
            data (array-like): Input data.
            n_bins (int): Number of bins.

        Returns:
            List[float]: Quantile-based bin edges.
    """
    data_points_per_bin = math.ceil(len(data) / n_bins)
    sortedData = data.copy()
    sortedData.sort()

    binList = data.copy()

    limitinf = min(data)
    for el in range(len(data)):
        if data[el] == limitinf:
            binList[el] = 0

    for j in range(n_bins):
        pos = data_points_per_bin * (j + 1) - 1

        if pos >= len(sortedData):
            pos = len(sortedData) - 1
        limitsup = sortedData[pos]
        for el in range(len(data)):
            if limitinf < data[el] <= limitsup:
                binList[el] = j

        limitinf = limitsup

    return binList


def addVelXY(dataframe):  # has to be sorted ascending by timestamp!!
    """
        Adds velocity in X and Y direction to a trajectory DataFrame.

        Parameters:
            dataframe (pd.DataFrame): DataFrame containing 'x' and 'y' columns representing position.

        Returns:
            pd.DataFrame: The same DataFrame with added 'velX' and 'velY' columns.
    """
    first = True
    ids = dataframe.id.unique()

    for i in ids:
        tempdf = dataframe.query("id == @i").copy()

        x1 = list(tempdf.x)
        x2 = list(tempdf.x)

        x2.pop(0)
        x2 += [x1[-1]]

        tempdf['velX'] = [px1 - px2 for (px1, px2) in zip(x2, x1)]

        y1 = list(tempdf.y)
        y2 = list(tempdf.y)
        y2.pop(0)
        y2 += [y1[-1]]

        tempdf['velY'] = [py1 - py2 for (py1, py2) in zip(y2, y1)]

        if first:
            resdf = tempdf.copy()
            first = False
        else:
            resdf = pd.concat([resdf, tempdf])

    return resdf


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
def to_dataframe(subgroups):
    """
        Converts a list of Pattern objects into a summary DataFrame. Only works for bidirectional graphs.

        Parameters:
            subgroups (List[Pattern]): List of Pattern instances.

        Returns:
            pd.DataFrame: DataFrame summarizing each pattern's graph statistics including node counts,
                          edge count, mean edge weight, and pattern score.
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
