import matplotlib.pyplot as plt
import networkx as nx
from typing import Optional, Callable, Any

def graph_viz(
        graph: nx.Graph,
        layout: Optional[Callable[[nx.Graph], dict]] = None,
        width: float = 2.0,
        node_size: int = 300,
        node_color: str = "skyblue",
        edge_color_attr: Optional[str] = "weight",
        edge_cmap: Any = plt.cm.Blues,
        with_labels: bool = True,
        figsize: tuple = (6, 6),
        title: Optional[str] = None,
) -> None:
    """
    Visualize a graph with customizable layout and styling options.

    Args:
        graph (nx.Graph): The graph to visualize.
        layout (Callable, optional): A layout function from NetworkX (e.g., nx.spring_layout).
                                     Defaults to circular layout if not provided.
        width (float): Width of the edges.
        node_size (int): Size of the nodes.
        node_color (str): Color of the nodes.
        edge_color_attr (str, optional): Edge attribute to use for coloring. If None, all edges are same color.
        edge_cmap (Colormap): Colormap for edge coloring.
        with_labels (bool): Whether to show node labels.
        figsize (tuple): Size of the plot figure.
        title (str, optional): Optional plot title.

    Returns:
        None
    """
    # Determine node positions
    pos = layout(graph) if layout else nx.circular_layout(graph)

    # Determine edge colors
    if edge_color_attr:
        edge_attrs = nx.get_edge_attributes(graph, edge_color_attr)
        edge_colors = list(edge_attrs.values())
    else:
        edge_colors = "gray"

    # Plotting
    plt.figure(figsize=figsize)
    nx.draw(
        graph,
        pos,
        node_size=node_size,
        node_color=node_color,
        edge_color=edge_colors,
        width=width,
        edge_cmap=edge_cmap,
        with_labels=with_labels,
    )

    if title:
        plt.title(title)

    plt.axis("off")
    plt.tight_layout()
    plt.show()
