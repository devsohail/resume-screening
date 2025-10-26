"""
MLflow integration for experiment tracking
Tracks model training, parameters, metrics, and artifacts
"""

import logging
import mlflow
import mlflow.pytorch
from typing import Dict, Any, Optional
from pathlib import Path

from backend.core.config import settings
from backend.core.exceptions import ModelError

logger = logging.getLogger(__name__)


class MLflowTracker:
    """
    MLflow experiment tracker
    Tracks training experiments, model versions, and metrics
    """
    
    def __init__(self, experiment_name: Optional[str] = None):
        """
        Initialize MLflow tracker
        
        Args:
            experiment_name: Name of MLflow experiment
        """
        try:
            # Set tracking URI
            mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
            
            # Set experiment
            self.experiment_name = experiment_name or settings.mlflow_experiment_name
            mlflow.set_experiment(self.experiment_name)
            
            logger.info(f"Initialized MLflow tracker for experiment: {self.experiment_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize MLflow: {e}")
            raise ModelError(f"MLflow initialization failed: {e}")
    
    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None):
        """
        Start a new MLflow run
        
        Args:
            run_name: Name for the run
            tags: Dictionary of tags
        """
        mlflow.start_run(run_name=run_name, tags=tags or {})
        logger.info(f"Started MLflow run: {run_name}")
    
    def end_run(self):
        """End current MLflow run"""
        mlflow.end_run()
        logger.info("Ended MLflow run")
    
    def log_params(self, params: Dict[str, Any]):
        """
        Log parameters
        
        Args:
            params: Dictionary of parameters
        """
        for key, value in params.items():
            mlflow.log_param(key, value)
        logger.debug(f"Logged {len(params)} parameters")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log metrics
        
        Args:
            metrics: Dictionary of metrics
            step: Step number (for time series)
        """
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)
        logger.debug(f"Logged {len(metrics)} metrics")
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """
        Log artifact file
        
        Args:
            local_path: Local file path
            artifact_path: Artifact directory path in MLflow
        """
        mlflow.log_artifact(local_path, artifact_path=artifact_path)
        logger.debug(f"Logged artifact: {local_path}")
    
    def log_model(
        self,
        model,
        artifact_path: str = "model",
        registered_model_name: Optional[str] = None
    ):
        """
        Log PyTorch model
        
        Args:
            model: PyTorch model
            artifact_path: Path within run's artifact directory
            registered_model_name: Name for model registry
        """
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path=artifact_path,
            registered_model_name=registered_model_name
        )
        logger.info(f"Logged model to: {artifact_path}")
    
    def load_model(self, model_uri: str):
        """
        Load model from MLflow
        
        Args:
            model_uri: MLflow model URI
            
        Returns:
            Loaded model
        """
        model = mlflow.pytorch.load_model(model_uri)
        logger.info(f"Loaded model from: {model_uri}")
        return model
    
    def track_training(
        self,
        model,
        params: Dict[str, Any],
        metrics: Dict[str, float],
        artifacts: Optional[Dict[str, str]] = None,
        run_name: Optional[str] = None,
        register_model: bool = False
    ) -> str:
        """
        Complete tracking workflow for a training run
        
        Args:
            model: Trained model
            params: Training parameters
            metrics: Final metrics
            artifacts: Dictionary of artifact paths
            run_name: Name for the run
            register_model: Whether to register model
            
        Returns:
            Run ID
        """
        with mlflow.start_run(run_name=run_name) as run:
            # Log parameters
            self.log_params(params)
            
            # Log metrics
            self.log_metrics(metrics)
            
            # Log artifacts
            if artifacts:
                for artifact_name, artifact_path in artifacts.items():
                    self.log_artifact(artifact_path, artifact_path=artifact_name)
            
            # Log model
            registered_name = "resume_classifier" if register_model else None
            self.log_model(model, registered_model_name=registered_name)
            
            run_id = run.info.run_id
            logger.info(f"Completed tracking for run: {run_id}")
            
            return run_id


def get_mlflow_tracker(experiment_name: Optional[str] = None) -> MLflowTracker:
    """
    Get MLflow tracker instance
    
    Args:
        experiment_name: Experiment name
        
    Returns:
        MLflowTracker instance
    """
    return MLflowTracker(experiment_name=experiment_name)

