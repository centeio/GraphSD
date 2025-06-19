"""
GraphSD: A package for subgroup discovery and pattern mining on graphs.

This module exposes the main components of the library via the top-level package API.
"""
from .utils import *
from .graph import *
from .mining import *
# from .outlier import *

from ._version import __version__

__all__ = ['GraphSDMining', 'DigraphSDMining', 'MultiDigraphSDMining', '__version__']
