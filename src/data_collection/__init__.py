"""Data collection modules for market indicators."""

from src.data_collection.fred_collector import FREDCollector
from src.data_collection.yahoo_collector import YahooCollector
from src.data_collection.alternative_collector import AlternativeCollector

__all__ = [
    'FREDCollector',
    'YahooCollector',
    'AlternativeCollector',
]

