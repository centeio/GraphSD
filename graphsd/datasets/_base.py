from pathlib import Path
from typing import Tuple

import pandas as pd

module_path = Path(__file__).parent


def load_data(dataset_name: str = "playground_a") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads anonymized position and social data for a given playground dataset.

    Args:
        dataset_name (str): Name of the dataset. Options are "a" or "b".

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Position data and social metadata.

    Raises:
        ValueError: If an unknown dataset name is provided.
    """
    if dataset_name == "playground_a":
        position_data = pd.read_csv(module_path / "data/playground_a_position_data_anonymized.csv")
        social_data = pd.read_csv(module_path / "data/playground_a_social_data_anonymized.csv")
    elif dataset_name == "playground_b":
        position_data = pd.read_csv(module_path / "data/playground_b_position_data_anonymized.csv")
        social_data = pd.read_csv(module_path / "data/playground_b_social_data_anonymized.csv")
    else:
        raise ValueError(f"Unknown dataset name '{dataset_name}'. Choose 'a' or 'b'.")

    return position_data, social_data
