import matplotlib.pyplot as plt
import networkx as nx
from typing import Optional, Callable, Any, Tuple


def graph_viz(
    graph: nx.Graph,
    layout: Optional[Callable[[nx.Graph], dict]] = nx.spring_layout,
    width: float = 2.0,
    node_size: int = 300,
    node_color: str = "skyblue",
    edge_color_attr: Optional[str] = "weight",
    edge_cmap: Any = plt.cm.Blues,
    with_labels: bool = True,
    figsize: Tuple[int, int] = (6, 6),
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None
) -> None:
    """
    Visualize a NetworkX graph with customizable layout and styling options.

    Args:
        graph (nx.Graph): The graph to visualize.
        layout (Callable, optional): A layout function like nx.spring_layout. Defaults to nx.circular_layout.
        width (float): Width of edges.
        node_size (int): Size of nodes.
        node_color (str): Color of nodes.
        edge_color_attr (str, optional): Edge attribute to color edges by. If None, uses solid black.
        edge_cmap (Any): Colormap for edge weights.
        with_labels (bool): Whether to show node labels.
        figsize (tuple): Figure size in inches.
        title (str, optional): Title of the plot.
        ax (matplotlib.axes.Axes, optional): Optional axis to draw on (useful for subplots).

    Returns:
        None
    """
    pos = layout(graph) if layout else nx.circular_layout(graph)

    edge_colors = "black"
    if edge_color_attr:
        edge_attr = nx.get_edge_attributes(graph, edge_color_attr)
        edge_colors = [edge_attr.get(edge, 0.0) for edge in graph.edges]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    nx.draw(
        graph,
        pos,
        ax=ax,
        node_size=node_size,
        node_color=node_color,
        edge_color=edge_colors,
        width=width,
        with_labels=with_labels,
        edge_cmap=edge_cmap,
    )

    if title:
        ax.set_title(title)

    ax.set_axis_off()
    plt.tight_layout()
