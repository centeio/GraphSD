from dataclasses import dataclass
from typing import Any, List, Optional
import networkx as nx


@dataclass(frozen=True, order=True)
class NominalSelector:
    """
    Represents a nominal selector used in pattern descriptions for subgroup discovery.

    Attributes:
        attribute (str): Attribute name.
        value (Any): Attribute value.
    """
    attribute: str
    value: Any

    def __str__(self) -> str:
        return f"({self.attribute}, {self.value})"

    def __repr__(self) -> str:
        return f"NominalSelector(attribute={self.attribute!r}, value={self.value!r})"


class Pattern:
    """
    Represents a discovered pattern in a graph.

    Attributes:
        name (List[NominalSelector]): Descriptive selectors for the pattern.
        graph (nx.Graph): The subgraph this pattern represents.
        weight (float): Frequency or coverage of the pattern.
        quality (float): Optional quality metric (default = 0.0).
    """

    def __init__(self, name: List[NominalSelector], graph: nx.Graph, weight: float) -> None:
        self.name: List[NominalSelector] = name
        self.graph: nx.Graph = graph
        self.weight: float = weight
        self.quality: float = 0.0

    def __repr__(self) -> str:
        return f"Pattern(name={self.name}, weight={self.weight}, quality={self.quality})"

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Pattern):
            return False
        return set(self.name) == set(other.name)

    def __lt__(self, other: Any) -> bool:
        return sorted(self.name) < sorted(other.name)

    def __hash__(self) -> int:
        return hash(frozenset(self.name))


class PatternWithoutGraph:
    """
    Represents a pattern without storing the graph structure (for memory efficiency or serialization).

    Attributes:
        name (List[NominalSelector]): Descriptive selectors.
        weight (float): Pattern frequency.
        quality (float): Pattern quality.
    """

    def __init__(self, name: List[NominalSelector], weight: float, quality: float = 0.0) -> None:
        self.name: List[NominalSelector] = name
        self.weight: float = weight
        self.quality: float = quality

    def __repr__(self) -> str:
        return f"PatternWithoutGraph(name={self.name}, weight={self.weight}, quality={self.quality})"

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, PatternWithoutGraph):
            return False
        return set(self.name) == set(other.name)

    def __lt__(self, other: Any) -> bool:
        return sorted(self.name) < sorted(other.name)

    def __hash__(self) -> int:
        return hash(frozenset(self.name))
