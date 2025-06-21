"""
GUI Module

This module contains the graphical user interface components for the
information retrieval system.
"""

from .ir_engine import IREngine, SearchModel
from .gui import IRGUI

__all__ = [
    'IREngine',
    'SearchModel',
    'IRGUI'
] 