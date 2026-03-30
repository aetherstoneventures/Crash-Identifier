"""FastAPI REST API for Market Crash Prediction System.

This API provides:
- Real-time crash probability predictions
- Historical predictions and analysis
- Model performance metrics
- Health checks and monitoring
"""

import logging
from datetime import datetime, date
from typing import List, Optional, Dict, Any
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np

from src.utils.config import API_HOST, API_PORT, validate_config
from src.utils.database import DatabaseManager, Indicator, Prediction, CrashEvent
from src.utils.mlflow_utils import MLflowModelManager
from src.feature_engineering.feature_pipeline import FeaturePipeline

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Market Crash Prediction API",
    description="REST API for crash probability predictions and analysis",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
db_manager = DatabaseManager()
mlflow_manager = MLflowModelManager()
feature_pipeline = FeaturePipeline()

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: datetime
    database: str
    mlflow: str
    config_valid: bool
    warnings: List[str] = []


class PredictionResponse(BaseModel):
    """Crash prediction response."""
    date: date
    crash_probability: float = Field(..., ge=0.0, le=1.0)
    model_name: str
    confidence: Optional[float] = None
    risk_level: str


class HistoricalPredictionsResponse(BaseModel):
    """Historical predictions response."""
    predictions: List[PredictionResponse]
    count: int
    start_date: date
    end_date: date


class ModelMetricsResponse(BaseModel):
    """Model performance metrics."""
    model_name: str
    auc: Optional[float] = None
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    last_updated: datetime


class CrashEventResponse(BaseModel):
    """Historical crash event."""
    start_date: date
    end_date: date
    trough_date: date
    max_drawdown: float
    recovery_months: Optional[int] = None


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_risk_level(probability: float) -> str:
    """Convert probability to risk level.
    
    Args:
        probability: Crash probability (0-1)
        
    Returns:
        Risk level string
    """
    if probability >= 0.8:
        return "CRITICAL"
    elif probability >= 0.6:
        return "HIGH"
    elif probability >= 0.4:
        return "MODERATE"
    elif probability >= 0.2:
        return "LOW"
    else:
        return "MINIMAL"


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "message": "Market Crash Prediction API",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    # Validate configuration
    config_validation = validate_config()
    
    # Check database
    try:
        with db_manager.get_session() as session:
            count = session.query(Indicator).count()
        db_status = f"OK ({count} indicators)"
    except Exception as e:
        db_status = f"ERROR: {str(e)}"
    
    # Check MLflow
    try:
        experiments = mlflow_manager.client.search_experiments()
        mlflow_status = f"OK ({len(experiments)} experiments)"
    except Exception as e:
        mlflow_status = f"ERROR: {str(e)}"
    
    return HealthResponse(
        status="healthy" if config_validation['valid'] else "degraded",
        timestamp=datetime.now(),
        database=db_status,
        mlflow=mlflow_status,
        config_valid=config_validation['valid'],
        warnings=config_validation['warnings']
    )


@app.get("/predictions/latest", response_model=PredictionResponse)
async def get_latest_prediction(
    model_name: str = Query("crash_predictor_xgboost", description="Model name to use")
):
    """Get the latest crash prediction."""
    try:
        with db_manager.get_session() as session:
            # Get latest prediction
            prediction = session.query(Prediction).order_by(
                Prediction.prediction_date.desc()
            ).first()
            
            if not prediction:
                raise HTTPException(status_code=404, detail="No predictions found")
            
            session.expunge_all()
        
        probability = prediction.crash_probability or 0.0
        
        return PredictionResponse(
            date=prediction.prediction_date,
            crash_probability=probability,
            model_name=model_name,
            confidence=None,  # Could calculate from model uncertainty
            risk_level=get_risk_level(probability)
        )
    
    except Exception as e:
        logger.error(f"Error getting latest prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/predictions/historical", response_model=HistoricalPredictionsResponse)
async def get_historical_predictions(
    start_date: Optional[date] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[date] = Query(None, description="End date (YYYY-MM-DD)"),
    model_name: str = Query("crash_predictor_xgboost", description="Model name"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of results")
):
    """Get historical predictions."""
    try:
        with db_manager.get_session() as session:
            query = session.query(Prediction).order_by(Prediction.prediction_date.desc())
            
            if start_date:
                query = query.filter(Prediction.prediction_date >= start_date)
            if end_date:
                query = query.filter(Prediction.prediction_date <= end_date)
            
            predictions = query.limit(limit).all()
            session.expunge_all()
        
        if not predictions:
            raise HTTPException(status_code=404, detail="No predictions found")
        
        # Convert to response format
        prediction_list = []
        for pred in predictions:
            probability = pred.crash_probability or 0.0
            
            prediction_list.append(PredictionResponse(
                date=pred.prediction_date,
                crash_probability=probability,
                model_name=model_name,
                risk_level=get_risk_level(probability)
            ))
        
        return HistoricalPredictionsResponse(
            predictions=prediction_list,
            count=len(prediction_list),
            start_date=predictions[-1].prediction_date,
            end_date=predictions[0].prediction_date
        )
    
    except Exception as e:
        logger.error(f"Error getting historical predictions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/crashes/historical", response_model=List[CrashEventResponse])
async def get_historical_crashes():
    """Get historical crash events."""
    try:
        with db_manager.get_session() as session:
            crashes = session.query(CrashEvent).order_by(CrashEvent.start_date.desc()).all()
            session.expunge_all()
        
        return [
            CrashEventResponse(
                start_date=crash.start_date,
                end_date=crash.end_date,
                trough_date=crash.trough_date,
                max_drawdown=crash.max_drawdown,
                recovery_months=crash.recovery_months
            )
            for crash in crashes
        ]
    
    except Exception as e:
        logger.error(f"Error getting historical crashes: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models/list", response_model=List[str])
async def list_models():
    """List available models."""
    try:
        # Get registered models from MLflow
        models = mlflow_manager.client.search_registered_models()
        return [model.name for model in models]
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models/metrics/{model_name}", response_model=ModelMetricsResponse)
async def get_model_metrics(model_name: str):
    """Get metrics for a specific model."""
    try:
        # Get latest run for this model
        history = mlflow_manager.get_model_history(model_name, limit=1)
        
        if not history:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
        
        latest_run = history[0]
        
        return ModelMetricsResponse(
            model_name=model_name,
            auc=latest_run.get('metrics.val_auc'),
            accuracy=latest_run.get('metrics.val_accuracy'),
            precision=latest_run.get('metrics.val_precision'),
            recall=latest_run.get('metrics.val_recall'),
            f1_score=latest_run.get('metrics.val_f1'),
            last_updated=datetime.fromtimestamp(latest_run.get('start_time', 0) / 1000)
        )
    
    except Exception as e:
        logger.error(f"Error getting model metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models/compare", response_model=Dict[str, Dict[str, Optional[float]]])
async def compare_models(
    models: List[str] = Query(
        ["crash_predictor_lstm", "crash_predictor_xgboost", "crash_predictor_statistical"],
        description="Models to compare"
    )
):
    """Compare multiple models."""
    try:
        comparison = mlflow_manager.compare_models(
            model_names=models,
            metrics=["val_auc", "val_accuracy", "val_precision", "val_recall", "val_f1"]
        )
        return comparison
    except Exception as e:
        logger.error(f"Error comparing models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# STARTUP/SHUTDOWN EVENTS
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Run on application startup."""
    logger.info("Starting Market Crash Prediction API...")
    logger.info(f"API Host: {API_HOST}")
    logger.info(f"API Port: {API_PORT}")
    
    # Validate configuration
    config_validation = validate_config()
    if not config_validation['valid']:
        logger.error("Configuration validation failed!")
        for error in config_validation['errors']:
            logger.error(f"  - {error}")
    
    for warning in config_validation['warnings']:
        logger.warning(f"  - {warning}")


@app.on_event("shutdown")
async def shutdown_event():
    """Run on application shutdown."""
    logger.info("Shutting down Market Crash Prediction API...")
    db_manager.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=API_HOST, port=API_PORT)

