import pandas as pd
from os.path import dirname

module_path = dirname(__file__)


def load_playground_a():

    position_data = pd.read_csv(module_path + "./data/playground_a_position_data_anonymized.csv")
    social_data = pd.read_csv(module_path + "./data/playground_a_social_data_anonymized.csv")

    return position_data, social_data


def load_playground_b():

    position_data = pd.read_csv(module_path + "./data/playground_b_position_data_anonymized.csv")
    social_data = pd.read_csv(module_path + "./data/playground_b_social_data_anonymized.csv")

    return position_data, social_data
