"""Walk-forward validation for time-series models."""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class WalkForwardValidator:
    """Implements walk-forward validation for time-series models."""
    
    def __init__(self, window_size=252, step_size=21, min_train_size=1260):
        self.window_size = window_size
        self.step_size = step_size
        self.min_train_size = min_train_size
    
    def expanding_window_split(self, X, y):
        """Generate expanding window splits."""
        n_samples = len(X)
        splits = []
        train_end = self.min_train_size
        
        while train_end + self.window_size <= n_samples:
            train_indices = np.arange(0, train_end)
            test_start = train_end
            test_end = min(train_end + self.window_size, n_samples)
            test_indices = np.arange(test_start, test_end)
            splits.append((train_indices, test_indices))
            train_end += self.step_size
        
        logger.info(f"Generated {len(splits)} expanding window splits")
        return splits
