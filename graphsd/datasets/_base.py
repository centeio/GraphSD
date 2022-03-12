import pandas as pd
from os.path import dirname

module_path = dirname(__file__)


def load_playgroundA():

    position_data = pd.read_csv(module_path + "./data/playgrounda_position_data_anonymized.csv")
    social_data = pd.read_csv(module_path + "./data/playgrounda_social_data_anonymized.csv")

    return position_data, social_data

def load_playgroundB():

    position_data = pd.read_csv(module_path + "./data/playgroundb_position_data_anonymized.csv")
    social_data = pd.read_csv(module_path + "./data/playgroundb_social_data_anonymized.csv")

    return position_data, social_data