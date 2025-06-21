"""
GraphSD: A package for subgroup discovery and pattern mining on graphs.

This module exposes the main components of the library via the top-level package API.
"""

from importlib.metadata import version as _version

from .patterns import Pattern, PatternWithoutGraph, NominalSelector
from .mining import GraphSDMining, DigraphSDMining, MultiDigraphSDMining
from .viz import graph_viz
#from .outlier import OutlierDetector

__version__ = _version("graph-sd")
get_version = lambda: __version__

__all__ = [
    "GraphSDMining",
    "DigraphSDMining",
    "MultiDigraphSDMining",
    "Pattern",
    "PatternWithoutGraph",
    "NominalSelector",
    "graph_viz",
    "get_version",
    "__version__",
]
