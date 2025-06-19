"""
GraphSD: A package for subgroup discovery and pattern mining on graphs.

This module exposes the main components of the library via the top-level package API.
"""
# from .outlier import *
from importlib.metadata import version

from .graph import *
from .mining import *
from .utils import *

__version__ = version("graph-sd")

__all__ = ['GraphSDMining', 'DigraphSDMining', 'MultiDigraphSDMining', '__version__']
