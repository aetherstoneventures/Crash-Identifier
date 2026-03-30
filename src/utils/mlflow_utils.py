"""MLflow utilities for model versioning, tracking, and registry.

This module provides a centralized interface for:
- Experiment tracking
- Model versioning and registry
- Hyperparameter logging
- Metrics tracking
- Model deployment and rollback
"""

import os
import logging
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from datetime import datetime
import pickle
import json

import mlflow
import mlflow.sklearn
import mlflow.keras
import mlflow.pytorch
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType

from src.utils.config import (
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_REGISTRY_URI,
    MODELS_DIR
)

logger = logging.getLogger(__name__)


class MLflowModelManager:
    """Manages ML model lifecycle with MLflow."""
    
    def __init__(self, experiment_name: str = MLFLOW_EXPERIMENT_NAME):
        """Initialize MLflow model manager.
        
        Args:
            experiment_name: Name of the MLflow experiment
        """
        self.experiment_name = experiment_name
        self.client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        
        # Set tracking and registry URIs
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_registry_uri(MLFLOW_REGISTRY_URI)
        
        # Create or get experiment
        try:
            self.experiment = mlflow.get_experiment_by_name(experiment_name)
            if self.experiment is None:
                self.experiment_id = mlflow.create_experiment(experiment_name)
                logger.info(f"Created MLflow experiment: {experiment_name}")
            else:
                self.experiment_id = self.experiment.experiment_id
                logger.info(f"Using existing MLflow experiment: {experiment_name}")
        except Exception as e:
            logger.error(f"Failed to initialize MLflow experiment: {e}")
            raise
    
    def start_run(self, run_name: str, tags: Optional[Dict[str, str]] = None) -> mlflow.ActiveRun:
        """Start a new MLflow run.
        
        Args:
            run_name: Name for the run
            tags: Optional tags for the run
            
        Returns:
            Active MLflow run context
        """
        return mlflow.start_run(
            experiment_id=self.experiment_id,
            run_name=run_name,
            tags=tags or {}
        )
    
    def log_model(
        self,
        model: Any,
        model_name: str,
        model_type: str,
        metrics: Dict[str, float],
        params: Dict[str, Any],
        artifacts: Optional[Dict[str, Any]] = None,
        register: bool = True
    ) -> str:
        """Log a model with MLflow.
        
        Args:
            model: The trained model object
            model_name: Name for the model
            model_type: Type of model (sklearn, keras, pytorch, etc.)
            metrics: Dictionary of metrics to log
            params: Dictionary of parameters to log
            artifacts: Optional additional artifacts to log
            register: Whether to register the model in the model registry
            
        Returns:
            Run ID of the logged model
        """
        with self.start_run(run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
            # Log parameters
            mlflow.log_params(params)
            
            # Log metrics (filter out NaN values which MLflow rejects)
            clean_metrics = {k: v for k, v in metrics.items()
                            if v is not None and not (isinstance(v, float) and (v != v))}
            mlflow.log_metrics(clean_metrics)
            
            # Log model based on type
            if model_type == "sklearn":
                inner = model.model if hasattr(model, 'model') else model
                mlflow.sklearn.log_model(inner, "model", registered_model_name=model_name if register else None)
            elif model_type == "keras":
                inner = model.model if hasattr(model, 'model') else model
                mlflow.keras.log_model(inner, "model", registered_model_name=model_name if register else None)
            elif model_type == "pytorch":
                inner = model.model if hasattr(model, 'model') else model
                mlflow.pytorch.log_model(inner, "model", registered_model_name=model_name if register else None)
            else:
                # Generic pickle logging
                model_path = MODELS_DIR / f"{model_name}.pkl"
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                mlflow.log_artifact(str(model_path))
            
            # Log additional artifacts
            if artifacts:
                for name, artifact in artifacts.items():
                    artifact_path = MODELS_DIR / f"{model_name}_{name}.pkl"
                    with open(artifact_path, 'wb') as f:
                        pickle.dump(artifact, f)
                    mlflow.log_artifact(str(artifact_path))
            
            # Log model metadata
            mlflow.set_tags({
                "model_type": model_type,
                "model_name": model_name,
                "timestamp": datetime.now().isoformat()
            })
            
            logger.info(f"Logged model {model_name} with run_id: {run.info.run_id}")
            return run.info.run_id
    
    def load_model(
        self,
        model_name: str,
        version: Optional[str] = None,
        stage: str = "Production"
    ) -> Any:
        """Load a model from MLflow registry.
        
        Args:
            model_name: Name of the registered model
            version: Specific version to load (if None, loads latest from stage)
            stage: Stage to load from (Production, Staging, None)
            
        Returns:
            Loaded model object
        """
        try:
            if version:
                model_uri = f"models:/{model_name}/{version}"
            else:
                model_uri = f"models:/{model_name}/{stage}"
            
            model = mlflow.pyfunc.load_model(model_uri)
            logger.info(f"Loaded model {model_name} from {model_uri}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise
    
    def get_best_model(
        self,
        metric_name: str,
        model_name_filter: Optional[str] = None,
        maximize: bool = True
    ) -> Tuple[str, Dict[str, Any]]:
        """Get the best model based on a metric.
        
        Args:
            metric_name: Name of the metric to optimize
            model_name_filter: Optional filter for model name
            maximize: Whether to maximize the metric (True) or minimize (False)
            
        Returns:
            Tuple of (run_id, run_data)
        """
        runs = mlflow.search_runs(
            experiment_ids=[self.experiment_id],
            filter_string=f"tags.model_name = '{model_name_filter}'" if model_name_filter else "",
            order_by=[f"metrics.{metric_name} {'DESC' if maximize else 'ASC'}"],
            max_results=1
        )
        
        if runs.empty:
            raise ValueError(f"No runs found for metric {metric_name}")
        
        best_run = runs.iloc[0]
        return best_run['run_id'], best_run.to_dict()
    
    def promote_model(
        self,
        model_name: str,
        version: str,
        stage: str = "Production"
    ) -> None:
        """Promote a model version to a specific stage.
        
        Args:
            model_name: Name of the registered model
            version: Version to promote
            stage: Target stage (Production, Staging, Archived)
        """
        try:
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage
            )
            logger.info(f"Promoted {model_name} version {version} to {stage}")
        except Exception as e:
            logger.error(f"Failed to promote model: {e}")
            raise
    
    def compare_models(
        self,
        model_names: List[str],
        metrics: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """Compare multiple models across metrics.
        
        Args:
            model_names: List of model names to compare
            metrics: List of metrics to compare
            
        Returns:
            Dictionary mapping model names to their metrics
        """
        comparison = {}
        
        for model_name in model_names:
            try:
                runs = mlflow.search_runs(
                    experiment_ids=[self.experiment_id],
                    filter_string=f"tags.model_name = '{model_name}'",
                    order_by=["start_time DESC"],
                    max_results=1
                )
                
                if not runs.empty:
                    run = runs.iloc[0]
                    comparison[model_name] = {
                        metric: run.get(f"metrics.{metric}", None)
                        for metric in metrics
                    }
            except Exception as e:
                logger.warning(f"Failed to get metrics for {model_name}: {e}")
                comparison[model_name] = {metric: None for metric in metrics}
        
        return comparison
    
    def get_model_history(
        self,
        model_name: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get training history for a model.
        
        Args:
            model_name: Name of the model
            limit: Maximum number of runs to return
            
        Returns:
            List of run dictionaries with metrics and parameters
        """
        runs = mlflow.search_runs(
            experiment_ids=[self.experiment_id],
            filter_string=f"tags.model_name = '{model_name}'",
            order_by=["start_time DESC"],
            max_results=limit
        )
        
        return runs.to_dict('records')
    
    def cleanup_old_runs(
        self,
        days_to_keep: int = 30,
        keep_production: bool = True
    ) -> int:
        """Clean up old MLflow runs.
        
        Args:
            days_to_keep: Number of days of runs to keep
            keep_production: Whether to keep runs with production models
            
        Returns:
            Number of runs deleted
        """
        cutoff_time = datetime.now().timestamp() - (days_to_keep * 24 * 3600)
        
        runs = mlflow.search_runs(
            experiment_ids=[self.experiment_id],
            filter_string=f"attributes.start_time < {cutoff_time}",
            run_view_type=ViewType.ACTIVE_ONLY
        )
        
        deleted_count = 0
        for _, run in runs.iterrows():
            if keep_production and run.get('tags.stage') == 'Production':
                continue
            
            try:
                self.client.delete_run(run['run_id'])
                deleted_count += 1
            except Exception as e:
                logger.warning(f"Failed to delete run {run['run_id']}: {e}")
        
        logger.info(f"Deleted {deleted_count} old runs")
        return deleted_count

