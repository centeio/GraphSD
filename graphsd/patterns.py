from dataclasses import dataclass
from typing import List


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
