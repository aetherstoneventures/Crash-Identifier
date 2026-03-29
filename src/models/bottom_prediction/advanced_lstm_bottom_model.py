"""Advanced LSTM Bottom Prediction Model with Deep Learning.

This model predicts:
- Days to market bottom
- Recovery time
- Bottom severity

Uses bidirectional LSTM with attention mechanism.
"""

import logging
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd

from tensorflow import keras
from tensorflow.keras import layers, callbacks, regularizers
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.models.bottom_prediction.base_bottom_model import BaseBottomModel
from src.utils.config import (
    LSTM_UNITS,
    LSTM_LAYERS,
    LSTM_DROPOUT,
    LSTM_SEQUENCE_LENGTH,
    LSTM_BATCH_SIZE,
    LSTM_EPOCHS,
    LSTM_LEARNING_RATE,
    RANDOM_STATE
)

logger = logging.getLogger(__name__)


class AdvancedLSTMBottomModel(BaseBottomModel):
    """Advanced LSTM model for bottom prediction with attention mechanism."""
    
    def __init__(
        self,
        sequence_length: int = LSTM_SEQUENCE_LENGTH,
        units: int = LSTM_UNITS,
        num_layers: int = LSTM_LAYERS,
        dropout: float = LSTM_DROPOUT,
        use_attention: bool = True
    ):
        """Initialize advanced LSTM bottom model.
        
        Args:
            sequence_length: Length of input sequences
            units: Number of LSTM units per layer
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            use_attention: Whether to use attention mechanism
        """
        super().__init__("Advanced_LSTM_Bottom")
        self.sequence_length = sequence_length
        self.units = units
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_attention = use_attention
        
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
    
    def _create_sequences(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM input.
        
        Args:
            X: Features array
            y: Target array
            
        Returns:
            Tuple of (X_sequences, y_sequences)
        """
        X_seq = []
        y_seq = []
        
        for i in range(len(X) - self.sequence_length):
            X_seq.append(X[i:i + self.sequence_length])
            y_seq.append(y[i + self.sequence_length])
        
        return np.array(X_seq), np.array(y_seq)
    
    def _build_model(self, input_shape: Tuple[int, int]) -> keras.Model:
        """Build LSTM model architecture.
        
        Args:
            input_shape: Shape of input (sequence_length, n_features)
            
        Returns:
            Compiled Keras model
        """
        inputs = keras.Input(shape=input_shape)
        x = inputs
        
        # Bidirectional LSTM layers
        for i in range(self.num_layers):
            return_sequences = (i < self.num_layers - 1) or self.use_attention
            
            x = layers.Bidirectional(
                layers.LSTM(
                    self.units,
                    return_sequences=return_sequences,
                    dropout=self.dropout,
                    recurrent_dropout=self.dropout * 0.5,
                    kernel_regularizer=regularizers.l2(0.01)
                )
            )(x)
            
            x = layers.BatchNormalization()(x)
        
        # Attention mechanism
        if self.use_attention:
            # Attention weights
            attention = layers.Dense(1, activation='tanh')(x)
            attention = layers.Flatten()(attention)
            attention = layers.Activation('softmax')(attention)
            attention = layers.RepeatVector(self.units * 2)(attention)  # *2 for bidirectional
            attention = layers.Permute([2, 1])(attention)
            
            # Apply attention
            x = layers.Multiply()([x, attention])
            x = layers.Lambda(lambda xin: keras.backend.sum(xin, axis=1))(x)
        
        # Dense layers for regression
        x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
        x = layers.Dropout(self.dropout)(x)
        x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
        x = layers.Dropout(self.dropout)(x)
        
        # Output layer (predicting days to bottom)
        outputs = layers.Dense(1, activation='linear')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        
        # Compile
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=LSTM_LEARNING_RATE),
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        return model
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> Dict:
        """Train the LSTM model.
        
        Args:
            X_train: Training features
            y_train: Training targets (days to bottom)
            X_val: Validation features
            y_val: Validation targets
            
        Returns:
            Dictionary with training metrics
        """
        logger.info(f"Training {self.name} model...")
        
        # Store feature names
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = X_train.columns.tolist()
            X_train = X_train.values
        
        if isinstance(X_val, pd.DataFrame):
            X_val = X_val.values
        
        if isinstance(y_train, pd.Series):
            y_train = y_train.values
        
        if isinstance(y_val, pd.Series):
            y_val = y_val.values
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Create sequences
        X_train_seq, y_train_seq = self._create_sequences(X_train_scaled, y_train)
        
        logger.info(f"Created {len(X_train_seq)} training sequences")
        
        # Build model
        input_shape = (self.sequence_length, X_train.shape[1])
        self.model = self._build_model(input_shape)
        
        logger.info(f"Model architecture: {self.model.summary()}")
        
        # Callbacks
        callback_list = [
            callbacks.EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=20,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=10,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Prepare validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            X_val_seq, y_val_seq = self._create_sequences(X_val_scaled, y_val)
            validation_data = (X_val_seq, y_val_seq)
            logger.info(f"Created {len(X_val_seq)} validation sequences")
        
        # Train
        history = self.model.fit(
            X_train_seq,
            y_train_seq,
            batch_size=LSTM_BATCH_SIZE,
            epochs=LSTM_EPOCHS,
            validation_data=validation_data,
            callbacks=callback_list,
            verbose=1
        )
        
        self.is_trained = True
        
        # Calculate metrics
        y_train_pred = self.model.predict(X_train_seq).flatten()
        
        metrics = {
            'train_mse': mean_squared_error(y_train_seq, y_train_pred),
            'train_mae': mean_absolute_error(y_train_seq, y_train_pred),
            'train_r2': r2_score(y_train_seq, y_train_pred),
            'final_loss': history.history['loss'][-1]
        }
        
        if validation_data is not None:
            y_val_pred = self.model.predict(X_val_seq).flatten()
            metrics.update({
                'val_mse': mean_squared_error(y_val_seq, y_val_pred),
                'val_mae': mean_absolute_error(y_val_seq, y_val_pred),
                'val_r2': r2_score(y_val_seq, y_val_pred),
                'final_val_loss': history.history['val_loss'][-1]
            })
        
        logger.info(f"Training complete. Metrics: {metrics}")
        
        return {
            'metrics': metrics,
            'history': history.history
        }
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions.
        
        Args:
            X: Features DataFrame
            
        Returns:
            Predicted days to bottom
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Scale
        X_scaled = self.scaler.transform(X)
        
        # Create sequences (pad if necessary)
        if len(X_scaled) < self.sequence_length:
            # Pad with zeros
            padding = np.zeros((self.sequence_length - len(X_scaled), X_scaled.shape[1]))
            X_scaled = np.vstack([padding, X_scaled])
        
        # Create sequences
        X_seq, _ = self._create_sequences(X_scaled, np.zeros(len(X_scaled)))
        
        if len(X_seq) == 0:
            # Return last sequence
            X_seq = X_scaled[-self.sequence_length:].reshape(1, self.sequence_length, -1)
        
        # Predict
        predictions = self.model.predict(X_seq, verbose=0).flatten()
        
        return predictions
    
    def save(self, path: str):
        """Save model to disk.
        
        Args:
            path: Path to save model
        """
        if self.model is not None:
            self.model.save(f"{path}_keras_model.h5")
        
        # Save scaler and metadata
        import joblib
        joblib.dump({
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'sequence_length': self.sequence_length,
            'units': self.units,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'use_attention': self.use_attention
        }, f"{path}_metadata.pkl")
    
    def load(self, path: str):
        """Load model from disk.
        
        Args:
            path: Path to load model from
        """
        self.model = keras.models.load_model(f"{path}_keras_model.h5")
        
        # Load scaler and metadata
        import joblib
        metadata = joblib.load(f"{path}_metadata.pkl")
        self.scaler = metadata['scaler']
        self.feature_names = metadata['feature_names']
        self.sequence_length = metadata['sequence_length']
        self.units = metadata['units']
        self.num_layers = metadata['num_layers']
        self.dropout = metadata['dropout']
        self.use_attention = metadata['use_attention']
        
        self.is_trained = True

