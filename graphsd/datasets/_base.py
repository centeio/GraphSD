import pandas as pd
from os.path import dirname

module_path = dirname(__file__)


def load_playground():

    position_data = pd.read_csv(module_path + "./data/playground_position_data_anonymized.csv")
    social_data = pd.read_csv(module_path + "./data/playground_social_data_anonymized.csv")

    return position_data, social_data
