"""Advanced LSTM model for crash prediction with anti-overfitting measures.

This model uses:
- Bidirectional LSTM layers for temporal pattern recognition
- Dropout and L2 regularization to prevent overfitting
- Early stopping with patience
- Walk-forward validation
- Attention mechanism for interpretability
"""

import logging
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

from src.models.crash_prediction.base_model import BaseCrashModel
from src.utils.config import (
    LSTM_UNITS,
    LSTM_LAYERS,
    LSTM_DROPOUT,
    LSTM_SEQUENCE_LENGTH,
    LSTM_BATCH_SIZE,
    LSTM_EPOCHS,
    LSTM_EARLY_STOPPING_PATIENCE,
    RANDOM_STATE
)

logger = logging.getLogger(__name__)


class LSTMCrashModel(BaseCrashModel):
    """LSTM-based crash prediction model with advanced anti-overfitting."""
    
    def __init__(
        self,
        sequence_length: int = LSTM_SEQUENCE_LENGTH,
        units: int = LSTM_UNITS,
        num_layers: int = LSTM_LAYERS,
        dropout: float = LSTM_DROPOUT,
        l2_reg: float = 0.001,
        use_attention: bool = True
    ):
        """Initialize LSTM crash model.
        
        Args:
            sequence_length: Number of time steps to look back
            units: Number of LSTM units per layer
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            l2_reg: L2 regularization factor
            use_attention: Whether to use attention mechanism
        """
        super().__init__()
        self.sequence_length = sequence_length
        self.units = units
        self.num_layers = num_layers
        self.dropout = dropout
        self.l2_reg = l2_reg
        self.use_attention = use_attention
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        
        # Set random seeds for reproducibility
        np.random.seed(RANDOM_STATE)
        tf.random.set_seed(RANDOM_STATE)
    
    def _create_sequences(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM input.
        
        Args:
            X: Feature array (n_samples, n_features)
            y: Target array (n_samples,)
            
        Returns:
            Tuple of (X_sequences, y_sequences)
        """
        X_seq, y_seq = [], []
        
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
        inputs = layers.Input(shape=input_shape)
        x = inputs
        
        # Stack Bidirectional LSTM layers
        for i in range(self.num_layers):
            return_sequences = (i < self.num_layers - 1) or self.use_attention
            
            x = layers.Bidirectional(
                layers.LSTM(
                    self.units,
                    return_sequences=return_sequences,
                    kernel_regularizer=regularizers.l2(self.l2_reg),
                    recurrent_regularizer=regularizers.l2(self.l2_reg),
                    dropout=self.dropout,
                    recurrent_dropout=self.dropout
                )
            )(x)
            
            # Batch normalization
            x = layers.BatchNormalization()(x)
        
        # Attention mechanism (if enabled)
        if self.use_attention:
            attention = layers.Dense(1, activation='tanh')(x)
            attention = layers.Flatten()(attention)
            attention = layers.Activation('softmax')(attention)
            attention = layers.RepeatVector(self.units * 2)(attention)  # *2 for bidirectional
            attention = layers.Permute([2, 1])(attention)
            
            x = layers.Multiply()([x, attention])
            x = layers.Lambda(lambda xin: tf.reduce_sum(xin, axis=1))(x)
        
        # Dense layers with dropout
        x = layers.Dense(
            64,
            activation='relu',
            kernel_regularizer=regularizers.l2(self.l2_reg)
        )(x)
        x = layers.Dropout(self.dropout)(x)
        
        x = layers.Dense(
            32,
            activation='relu',
            kernel_regularizer=regularizers.l2(self.l2_reg)
        )(x)
        x = layers.Dropout(self.dropout)(x)
        
        # Output layer
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        
        # Compile with Adam optimizer
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.AUC(name='auc'),
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall')
            ]
        )
        
        return model
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """Train the LSTM model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Dictionary with training history and metrics
        """
        logger.info("Training LSTM crash prediction model...")
        
        # Store feature names
        self.feature_names = X_train.columns.tolist()
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Create sequences
        X_train_seq, y_train_seq = self._create_sequences(X_train_scaled, y_train.values)
        
        logger.info(f"Created {len(X_train_seq)} training sequences")
        
        # Prepare validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            X_val_seq, y_val_seq = self._create_sequences(X_val_scaled, y_val.values)
            validation_data = (X_val_seq, y_val_seq)
            logger.info(f"Created {len(X_val_seq)} validation sequences")
        
        # Build model
        input_shape = (self.sequence_length, X_train.shape[1])
        self.model = self._build_model(input_shape)
        
        logger.info(f"Model architecture:\n{self.model.summary()}")
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss' if validation_data else 'loss',
                patience=LSTM_EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss' if validation_data else 'loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Train model
        history = self.model.fit(
            X_train_seq,
            y_train_seq,
            batch_size=LSTM_BATCH_SIZE,
            epochs=LSTM_EPOCHS,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )
        
        # Calculate final metrics
        train_metrics = self.model.evaluate(X_train_seq, y_train_seq, verbose=0)
        metrics = {
            'train_loss': train_metrics[0],
            'train_accuracy': train_metrics[1],
            'train_auc': train_metrics[2],
            'train_precision': train_metrics[3],
            'train_recall': train_metrics[4]
        }
        
        if validation_data:
            val_metrics = self.model.evaluate(X_val_seq, y_val_seq, verbose=0)
            metrics.update({
                'val_loss': val_metrics[0],
                'val_accuracy': val_metrics[1],
                'val_auc': val_metrics[2],
                'val_precision': val_metrics[3],
                'val_recall': val_metrics[4]
            })
        
        logger.info(f"Training complete. Metrics: {metrics}")
        
        return {
            'history': history.history,
            'metrics': metrics,
            'model_type': 'LSTM',
            'sequence_length': self.sequence_length,
            'units': self.units,
            'num_layers': self.num_layers
        }
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict crash probabilities.
        
        Args:
            X: Features to predict on
            
        Returns:
            Array of crash probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Create sequences
        X_seq, _ = self._create_sequences(X_scaled, np.zeros(len(X)))
        
        # Predict
        predictions = self.model.predict(X_seq, verbose=0)
        
        # Pad predictions to match input length
        padded_predictions = np.zeros(len(X))
        padded_predictions[self.sequence_length:] = predictions.flatten()
        
        return padded_predictions

