"""Monitoring and metrics collection using Prometheus.

Provides:
- System health metrics
- Model performance tracking
- Data collection metrics
- Alert generation
"""

import logging
import time
from typing import Dict, Any, Optional
from datetime import datetime
from functools import wraps

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    Summary,
    CollectorRegistry,
    generate_latest,
    CONTENT_TYPE_LATEST
)

from src.utils.config import ENABLE_MONITORING, ALERT_THRESHOLD

logger = logging.getLogger(__name__)

# Create registry
registry = CollectorRegistry()

# ============================================================================
# METRICS DEFINITIONS
# ============================================================================

# Data Collection Metrics
data_collection_total = Counter(
    'data_collection_total',
    'Total number of data collection runs',
    ['source', 'status'],
    registry=registry
)

data_collection_duration = Histogram(
    'data_collection_duration_seconds',
    'Duration of data collection',
    ['source'],
    registry=registry
)

indicators_collected = Gauge(
    'indicators_collected_total',
    'Total number of indicators in database',
    registry=registry
)

# Model Training Metrics
model_training_total = Counter(
    'model_training_total',
    'Total number of model training runs',
    ['model_name', 'status'],
    registry=registry
)

model_training_duration = Histogram(
    'model_training_duration_seconds',
    'Duration of model training',
    ['model_name'],
    registry=registry
)

# Prediction Metrics
predictions_total = Counter(
    'predictions_total',
    'Total number of predictions made',
    ['model_name'],
    registry=registry
)

crash_probability = Gauge(
    'crash_probability',
    'Current crash probability',
    ['model_name'],
    registry=registry
)

prediction_latency = Summary(
    'prediction_latency_seconds',
    'Latency of prediction requests',
    ['model_name'],
    registry=registry
)

# Model Performance Metrics
model_auc = Gauge(
    'model_auc',
    'Model AUC score',
    ['model_name', 'dataset'],
    registry=registry
)

model_accuracy = Gauge(
    'model_accuracy',
    'Model accuracy',
    ['model_name', 'dataset'],
    registry=registry
)

model_precision = Gauge(
    'model_precision',
    'Model precision',
    ['model_name', 'dataset'],
    registry=registry
)

model_recall = Gauge(
    'model_recall',
    'Model recall',
    ['model_name', 'dataset'],
    registry=registry
)

# System Health Metrics
database_connections = Gauge(
    'database_connections',
    'Number of active database connections',
    registry=registry
)

api_requests_total = Counter(
    'api_requests_total',
    'Total number of API requests',
    ['endpoint', 'method', 'status'],
    registry=registry
)

api_request_duration = Histogram(
    'api_request_duration_seconds',
    'Duration of API requests',
    ['endpoint', 'method'],
    registry=registry
)

# Alert Metrics
alerts_triggered = Counter(
    'alerts_triggered_total',
    'Total number of alerts triggered',
    ['alert_type', 'severity'],
    registry=registry
)


# ============================================================================
# MONITORING DECORATORS
# ============================================================================

def monitor_data_collection(source: str):
    """Decorator to monitor data collection functions.
    
    Args:
        source: Data source name (e.g., 'fred', 'yahoo')
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not ENABLE_MONITORING:
                return func(*args, **kwargs)
            
            start_time = time.time()
            status = 'success'
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                status = 'failure'
                raise
            finally:
                duration = time.time() - start_time
                data_collection_total.labels(source=source, status=status).inc()
                data_collection_duration.labels(source=source).observe(duration)
                
                logger.info(f"Data collection [{source}] completed in {duration:.2f}s - {status}")
        
        return wrapper
    return decorator


def monitor_model_training(model_name: str):
    """Decorator to monitor model training functions.
    
    Args:
        model_name: Name of the model being trained
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not ENABLE_MONITORING:
                return func(*args, **kwargs)
            
            start_time = time.time()
            status = 'success'
            
            try:
                result = func(*args, **kwargs)
                
                # Extract metrics from result if available
                if isinstance(result, dict) and 'metrics' in result:
                    metrics = result['metrics']
                    dataset = 'validation'
                    
                    if 'val_auc' in metrics:
                        model_auc.labels(model_name=model_name, dataset=dataset).set(metrics['val_auc'])
                    if 'val_accuracy' in metrics:
                        model_accuracy.labels(model_name=model_name, dataset=dataset).set(metrics['val_accuracy'])
                    if 'val_precision' in metrics:
                        model_precision.labels(model_name=model_name, dataset=dataset).set(metrics['val_precision'])
                    if 'val_recall' in metrics:
                        model_recall.labels(model_name=model_name, dataset=dataset).set(metrics['val_recall'])
                
                return result
            except Exception as e:
                status = 'failure'
                raise
            finally:
                duration = time.time() - start_time
                model_training_total.labels(model_name=model_name, status=status).inc()
                model_training_duration.labels(model_name=model_name).observe(duration)
                
                logger.info(f"Model training [{model_name}] completed in {duration:.2f}s - {status}")
        
        return wrapper
    return decorator


def monitor_prediction(model_name: str):
    """Decorator to monitor prediction functions.
    
    Args:
        model_name: Name of the model making predictions
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not ENABLE_MONITORING:
                return func(*args, **kwargs)
            
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                
                # Track prediction
                predictions_total.labels(model_name=model_name).inc()
                
                # Update crash probability if result is a probability
                if isinstance(result, (int, float)):
                    crash_probability.labels(model_name=model_name).set(result)
                
                return result
            finally:
                duration = time.time() - start_time
                prediction_latency.labels(model_name=model_name).observe(duration)
        
        return wrapper
    return decorator


def monitor_api_request(endpoint: str, method: str):
    """Decorator to monitor API requests.
    
    Args:
        endpoint: API endpoint path
        method: HTTP method
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not ENABLE_MONITORING:
                return func(*args, **kwargs)
            
            start_time = time.time()
            status = '200'
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                status = '500'
                raise
            finally:
                duration = time.time() - start_time
                api_requests_total.labels(endpoint=endpoint, method=method, status=status).inc()
                api_request_duration.labels(endpoint=endpoint, method=method).observe(duration)
        
        return wrapper
    return decorator


# ============================================================================
# ALERT FUNCTIONS
# ============================================================================

def trigger_alert(alert_type: str, severity: str, message: str):
    """Trigger an alert.
    
    Args:
        alert_type: Type of alert (e.g., 'high_crash_probability', 'model_degradation')
        severity: Severity level ('info', 'warning', 'critical')
        message: Alert message
    """
    if not ENABLE_MONITORING:
        return
    
    alerts_triggered.labels(alert_type=alert_type, severity=severity).inc()
    
    logger.log(
        logging.CRITICAL if severity == 'critical' else
        logging.WARNING if severity == 'warning' else
        logging.INFO,
        f"ALERT [{severity.upper()}] {alert_type}: {message}"
    )


def check_crash_probability_alert(probability: float, model_name: str):
    """Check if crash probability exceeds alert threshold.
    
    Args:
        probability: Crash probability (0-1)
        model_name: Name of the model
    """
    if probability >= ALERT_THRESHOLD:
        trigger_alert(
            alert_type='high_crash_probability',
            severity='critical',
            message=f"Crash probability from {model_name} is {probability:.2%} (threshold: {ALERT_THRESHOLD:.2%})"
        )


# ============================================================================
# METRICS EXPORT
# ============================================================================

def get_metrics() -> bytes:
    """Get Prometheus metrics in text format.
    
    Returns:
        Metrics in Prometheus text format
    """
    return generate_latest(registry)


def get_metrics_content_type() -> str:
    """Get content type for Prometheus metrics.
    
    Returns:
        Content type string
    """
    return CONTENT_TYPE_LATEST


# ============================================================================
# HEALTH CHECK
# ============================================================================

def get_system_health() -> Dict[str, Any]:
    """Get system health status.
    
    Returns:
        Dictionary with health metrics
    """
    return {
        'timestamp': datetime.now().isoformat(),
        'monitoring_enabled': ENABLE_MONITORING,
        'metrics_available': True
    }

