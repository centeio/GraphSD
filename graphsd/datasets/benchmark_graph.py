"""
Defines a small benchmark directed graph and associated social attributes for testing or demonstration.
"""
import pandas as pd

digraph1 = {
    'nodes': ['1', '2', '3', '4', '5', '6', '7'],
    'edges': [
        ('3', '7', 1),
        ('4', '7', 1),
        ('5', '7', 1),
        ('6', '7', 1),
        ('7', '1', 2),
        ('7', '2', 2)
    ],
    'social_data': pd.DataFrame(
        {
            'id': ['1', '2', '3', '4', '5', '6', '7'],
            'gender': ['b', 'b', 'g', 'g', 'b', 'b', 'g'],
            'age': [2, 2, 2, 1, 1, 2, 1]
        }
    )
}
