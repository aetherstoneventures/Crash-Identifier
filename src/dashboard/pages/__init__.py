"""Dashboard pages module."""

from .overview import OverviewPage
from .crash_predictions import CrashPredictionsPage
from .bottom_predictions import BottomPredictionsPage
from .indicators import IndicatorsPage
from .settings import SettingsPage

__all__ = [
    'OverviewPage',
    'CrashPredictionsPage',
    'BottomPredictionsPage',
    'IndicatorsPage',
    'SettingsPage',
]

