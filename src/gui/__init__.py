"""
GUI Module

This module contains the graphical user interface components for the
information retrieval system.
"""

from .ir_engine import IREngine, SearchModel
from .gui import IRMainWindow

__all__ = [
    'IREngine',
    'SearchModel',
    'IRMainWindow'
] 