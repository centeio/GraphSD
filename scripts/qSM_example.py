import pandas as pd

from graphsd import MultiDigraphSDMining
from graphsd.utils import make_bins, to_dataframe

if __name__ == '__main__':

    position_data = pd.read_csv("../position_data_anonymized.csv")
    social_data = pd.read_csv("../social_data_anonymized.csv")
    social_data = make_bins(social_data)

    mdgm = MultiDigraphSDMining(random_state=1234, n_jobs=3, n_samples=100)

    mdgm.read_data(position_data, social_data, time_step=10)

    #### Subgroup Discovery

    # change argument mode to ['to', 'from', 'both']
    subgroups_both = mdgm.subgroup_discovery(mode="both", min_support=100)

    to_dataframe(subgroups_both).to_csv('output/Comp_qSM_mean.csv', index=True)

    # Using variance as metric
    subgroups_both_var = dgm.subgroup_discovery(mode="both", min_support=100, metric='var')

    to_dataframe(subgroups_both_var).to_csv('output/Comp_qSM_var.csv', index=True)
