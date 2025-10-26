"""
Inference handler for the trained classifier
Loads model and performs predictions
"""

import logging
import torch
import numpy as np
from typing import Dict, Optional, Tuple
from pathlib import Path

from backend.ml.classifier.model import ResumeClassifier
from backend.core.config import settings
from backend.core.exceptions import ModelError
from backend.core.models import ClassifierResult, ScreeningDecision

logger = logging.getLogger(__name__)


class ClassifierInference:
    """
    Inference engine for resume classification
    Loads trained model and performs predictions
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = None,
        threshold: float = None
    ):
        """
        Initialize inference engine
        
        Args:
            model_path: Path to trained model checkpoint
            device: Device to run inference on ('cuda' or 'cpu')
            threshold: Classification threshold (default from settings)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.threshold = threshold or settings.classifier_threshold
        self.model = None
        self.model_path = model_path
        
        if model_path:
            self.load_model(model_path)
        
        logger.info(f"Initialized inference engine on {self.device} with threshold {self.threshold}")
    
    def load_model(self, model_path: str):
        """
        Load trained model from checkpoint
        
        Args:
            model_path: Path to model checkpoint
        """
        try:
            if not Path(model_path).exists():
                raise ModelError(f"Model file not found: {model_path}")
            
            # Load checkpoint
            model, _, epoch, metrics = ResumeClassifier.load_checkpoint(
                model_path,
                map_location=self.device
            )
            
            model.to(self.device)
            model.eval()  # Set to evaluation mode
            
            self.model = model
            self.model_path = model_path
            
            logger.info(
                f"Loaded model from {model_path} "
                f"(epoch {epoch}, metrics: {metrics})"
            )
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise ModelError(f"Failed to load model: {e}")
    
    def predict(
        self,
        resume_embedding: np.ndarray,
        job_embedding: np.ndarray,
        return_probability: bool = True
    ) -> ClassifierResult:
        """
        Predict whether to shortlist a candidate
        
        Args:
            resume_embedding: Resume embedding vector
            job_embedding: Job embedding vector
            return_probability: Whether to return probability
            
        Returns:
            ClassifierResult with prediction and probability
        """
        if self.model is None:
            raise ModelError("Model not loaded. Call load_model() first.")
        
        try:
            # Convert to tensors
            resume_tensor = torch.FloatTensor(resume_embedding).unsqueeze(0).to(self.device)
            job_tensor = torch.FloatTensor(job_embedding).unsqueeze(0).to(self.device)
            
            # Inference
            with torch.no_grad():
                probability = self.model(resume_tensor, job_tensor).item()
            
            # Make decision
            prediction = ScreeningDecision.SHORTLIST if probability >= self.threshold else ScreeningDecision.REJECT
            
            # Calculate confidence (distance from threshold)
            confidence = abs(probability - self.threshold) / self.threshold
            confidence = min(confidence, 1.0)  # Cap at 1.0
            
            logger.debug(
                f"Classifier prediction: {prediction.value} "
                f"(probability={probability:.4f}, confidence={confidence:.4f})"
            )
            
            return ClassifierResult(
                probability=probability,
                prediction=prediction,
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise ModelError(f"Prediction failed: {e}")
    
    def predict_batch(
        self,
        resume_embeddings: np.ndarray,
        job_embeddings: np.ndarray
    ) -> list[ClassifierResult]:
        """
        Batch prediction for multiple resume-job pairs
        
        Args:
            resume_embeddings: Array of resume embeddings (n_samples, embedding_dim)
            job_embeddings: Array of job embeddings (n_samples, embedding_dim)
            
        Returns:
            List of ClassifierResult objects
        """
        if self.model is None:
            raise ModelError("Model not loaded. Call load_model() first.")
        
        try:
            # Convert to tensors
            resume_tensors = torch.FloatTensor(resume_embeddings).to(self.device)
            job_tensors = torch.FloatTensor(job_embeddings).to(self.device)
            
            # Batch inference
            with torch.no_grad():
                probabilities = self.model(resume_tensors, job_tensors).cpu().numpy().flatten()
            
            # Create results
            results = []
            for probability in probabilities:
                prediction = ScreeningDecision.SHORTLIST if probability >= self.threshold else ScreeningDecision.REJECT
                confidence = abs(probability - self.threshold) / self.threshold
                confidence = min(confidence, 1.0)
                
                results.append(ClassifierResult(
                    probability=float(probability),
                    prediction=prediction,
                    confidence=float(confidence)
                ))
            
            logger.info(f"Batch prediction completed for {len(results)} samples")
            return results
            
        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            raise ModelError(f"Batch prediction failed: {e}")
    
    def update_threshold(self, new_threshold: float):
        """
        Update classification threshold
        
        Args:
            new_threshold: New threshold value (0-1)
        """
        if not 0 <= new_threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1")
        
        self.threshold = new_threshold
        logger.info(f"Updated threshold to {new_threshold}")
    
    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model
        
        Returns:
            Dictionary with model information
        """
        if self.model is None:
            return {"status": "not_loaded"}
        
        return {
            "status": "loaded",
            "model_path": self.model_path,
            "device": str(self.device),
            "threshold": self.threshold,
            "num_parameters": self.model.get_model_size(),
            "embedding_dim": self.model.embedding_dim
        }


# Singleton instance
_inference_instance = None


def get_inference_engine(model_path: Optional[str] = None) -> ClassifierInference:
    """
    Get singleton instance of ClassifierInference
    
    Args:
        model_path: Path to model checkpoint (only used on first call)
        
    Returns:
        ClassifierInference instance
    """
    global _inference_instance
    
    if _inference_instance is None:
        _inference_instance = ClassifierInference(model_path=model_path)
    elif model_path and model_path != _inference_instance.model_path:
        # Reload if different model requested
        _inference_instance.load_model(model_path)
    
    return _inference_instance

