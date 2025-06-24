import networkx as nx

from graphsd.datasets import load_data
from graphsd import DigraphSDMining, MultiDigraphSDMining, GraphSDMining
from graphsd.viz import graph_viz
from graphsd.utils import make_bins
import logging
import matplotlib.pyplot as plt  # Needed for plt.show()


logging.basicConfig(level=logging.INFO)
logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)


if __name__ == '__main__':
    # Load datasets
    position_data, social_data = load_data("playground_a")

    # Create bins for social data
    social_data = make_bins(social_data, n_bins=3, strategy="quantile")

    # Choose a random seed for reproducibility
    #dig = GraphSDMining(random_state=1234)
    dig = DigraphSDMining(random_state=1234, n_jobs=4)
    #dig = MultiDigraphSDMining(random_state=12345)

    dig.read_data(position_data, social_data, time_step=10)

    # Run subgroup discovery on the digraph.
    subgroups_to = dig.subgroup_discovery(
        mode="to",
        min_support=0.20,
        metric='mean',
        quality_measure="qP"
    )

    # Convert results to DataFrame
    print(dig.to_dataframe(subgroups_to)[:5])

    subgroups_from = dig.subgroup_discovery(mode="from", min_support=0.20, metric='mean', quality_measure="qP")
    print(dig.to_dataframe(subgroups_from))

    subgroups_comp = dig.subgroup_discovery(mode="comparison", min_support=0.20, quality_measure="qP")
    print(dig.to_dataframe(subgroups_comp))

    # --- Visualize the first discovered pattern ---
    if subgroups_to:
        pattern = subgroups_to[0]
        graph_viz(
            pattern.graph,
            layout=nx.circular_layout,
            title=str(pattern.name),
            node_color="lightblue"
        )
        plt.show()
    else:
        print("No patterns found to visualize.")
