import pandas as pd

from graphsd import DigraphSDMining
from graphsd.utils import make_bins, to_dataframe

if __name__ == '__main__':
    position_data = pd.read_csv("../position_data_anonymized.csv")
    social_data = pd.read_csv("../social_data_anonymized.csv")
    social_data = make_bins(social_data)

    dgm = DigraphSDMining(random_state=1234, n_jobs=3)

    dgm.read_data(position_data, social_data, time_step=10)

    # Subgroup Discovery

    # change argument mode to ['to', 'from', 'comparison']
    subgroups_both = dgm.subgroup_discovery(mode="comparison", min_support=0.10)

    to_dataframe(subgroups_both).to_csv('output/Comp_qSD_mean.csv', index=True)

    # Using variance as metric

    subgroups_both_var = dgm.subgroup_discovery(mode="comparison", min_support=0.10, metric='var')

    to_dataframe(subgroups_both_var).to_csv('output/Comp_qSD_var.csv', index=True)

    # visualize

    # displayGender(GComp,ids, socialData, filtere=4)
    # graphViz(compTQ[0].graph)

    # TODO show example printpositions and printpositionsG
