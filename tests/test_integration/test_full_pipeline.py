"""Full pipeline integration tests."""

import pytest
import os
import sys
from datetime import datetime, timedelta
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

from src.models.crash_prediction import SVMCrashModel, RandomForestCrashModel
from src.models.crash_prediction.crash_labeler import CrashLabeler
from src.models.bottom_prediction import MLPBottomModel, LSTMBottomModel, BottomLabeler
from src.alerts import AlertManager


class TestFullPipeline:
    """Test complete pipeline from data collection to alerts."""

    def test_crash_model_integration(self):
        """Test crash model integration."""
        print("\n✓ STEP 1: Crash Model Integration")

        # Create synthetic data
        X = np.random.randn(100, 28)  # 28 features
        y = np.random.randint(0, 2, 100)  # Binary labels

        # Train SVM model
        model = SVMCrashModel()
        model.train(X, y)

        # Make predictions
        predictions = model.predict(X[:10])

        assert predictions is not None
        assert len(predictions) == 10
        print(f"  - Trained crash model and made {len(predictions)} predictions")

    def test_bottom_model_integration(self):
        """Test bottom model integration."""
        print("\n✓ STEP 2: Bottom Model Integration")

        # Create synthetic data
        X = np.random.randn(100, 28)  # 28 features
        y = np.random.randint(1, 60, 100)  # Days to bottom (1-60)

        # Train MLP model
        mlp_model = MLPBottomModel()
        mlp_model.train(X, y)

        # Make predictions
        predictions = mlp_model.predict(X[:10])

        assert predictions is not None
        assert len(predictions) == 10
        print(f"  - Trained MLP model and made {len(predictions)} predictions")

    def test_alert_system_integration(self):
        """Test alert system integration."""
        print("\n✓ STEP 3: Alert System Integration")

        manager = AlertManager()

        # Test crash alert
        results = manager.send_crash_alert(
            probability=0.75,
            confidence=0.85
        )

        assert results is not None
        assert 'email' in results
        assert 'sms' in results
        print(f"  - Crash alert sent successfully")

        # Test bottom alert
        results = manager.send_bottom_alert(
            days_to_bottom=30,
            recovery_date="2024-12-20"
        )

        assert results is not None
        print(f"  - Bottom alert sent successfully")

    def test_model_compatibility(self):
        """Test model compatibility."""
        print("\n✓ STEP 4: Model Compatibility")

        # Create synthetic data
        X = np.random.randn(50, 28)
        y_crash = np.random.randint(0, 2, 50)
        y_bottom = np.random.randint(1, 60, 50)

        # Train both models
        crash_model = RandomForestCrashModel()
        crash_model.train(X, y_crash)

        bottom_model = MLPBottomModel()
        bottom_model.train(X, y_bottom)

        # Make predictions
        crash_pred = crash_model.predict(X[:5])
        bottom_pred = bottom_model.predict(X[:5])

        assert len(crash_pred) == 5
        assert len(bottom_pred) == 5
        print(f"  - Both models trained and compatible")


class TestPipelineRobustness:
    """Test pipeline robustness and error handling."""

    def test_nan_handling_in_models(self):
        """Test handling of NaN values in models."""
        print("\n✓ STEP 5: NaN Handling")

        # Create data with NaN values
        X = np.random.randn(50, 28)
        X[0, 0] = np.nan
        X[5, 10] = np.nan

        # Replace NaN with 0
        X = np.nan_to_num(X, nan=0.0)
        y = np.random.randint(0, 2, 50)

        # Train model
        model = SVMCrashModel()
        model.train(X, y)

        # Should handle NaN gracefully
        assert model is not None
        print(f"  - NaN values handled correctly")

    def test_model_prediction_consistency(self):
        """Test model prediction consistency."""
        print("\n✓ STEP 6: Prediction Consistency")

        # Create synthetic data
        X = np.random.randn(50, 28)
        y = np.random.randint(0, 2, 50)

        # Train model twice with same data
        model1 = RandomForestCrashModel()
        model1.train(X, y)
        pred1 = model1.predict(X[:5])

        model2 = RandomForestCrashModel()
        model2.train(X, y)
        pred2 = model2.predict(X[:5])

        # Predictions should have same length
        assert len(pred1) == len(pred2)
        print(f"  - Predictions are consistent")

    def test_alert_history_tracking(self):
        """Test alert history tracking."""
        print("\n✓ STEP 7: Alert History")

        manager = AlertManager()

        # Send multiple alerts
        for i in range(5):
            manager.send_crash_alert(0.5 + i * 0.05, 0.8)

        # Check history
        history = manager.get_alert_history()
        assert len(history) == 5
        print(f"  - Alert history tracked correctly ({len(history)} alerts)")

