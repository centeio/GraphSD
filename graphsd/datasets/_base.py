import pandas as pd
from pathlib import Path

module_path = Path(__file__).parent


def load_playground_a():
    """
    Loads anonymized position and social data for Playground A.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Position data and social metadata.
    """
    position_data = pd.read_csv(module_path / "data/playground_a_position_data_anonymized.csv")
    social_data = pd.read_csv(module_path / "data/playground_a_social_data_anonymized.csv")

    return position_data, social_data


def load_playground_b():
    """
    Loads anonymized position and social data for Playground B.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Position data and social metadata.
    """
    position_data = pd.read_csv(module_path + "./data/playground_b_position_data_anonymized.csv")
    social_data = pd.read_csv(module_path + "./data/playground_b_social_data_anonymized.csv")

    return position_data, social_data
