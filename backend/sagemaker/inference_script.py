"""
SageMaker inference script
Serves the trained model on SageMaker endpoint
"""

import json
import logging
import torch
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model instance
model = None
device = None


def model_fn(model_dir):
    """
    Load model for inference
    
    Args:
        model_dir: Directory containing model artifacts
        
    Returns:
        Loaded model
    """
    global model, device
    
    logger.info(f"Loading model from {model_dir}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Import model class
    from backend.ml.classifier.model import ResumeClassifier
    
    # Load model
    model_path = Path(model_dir) / "model.pt"
    model, _, _, _ = ResumeClassifier.load_checkpoint(str(model_path), map_location=device)
    model.eval()
    
    logger.info(f"Model loaded successfully on {device}")
    
    return model


def input_fn(request_body, content_type):
    """
    Deserialize input data
    
    Args:
        request_body: Request body
        content_type: Content type
        
    Returns:
        Parsed input data
    """
    if content_type == 'application/json':
        data = json.loads(request_body)
        
        resume_embedding = np.array(data['resume_embedding'], dtype=np.float32)
        job_embedding = np.array(data['job_embedding'], dtype=np.float32)
        
        return {
            'resume_embedding': resume_embedding,
            'job_embedding': job_embedding,
            'threshold': data.get('threshold', 0.7)
        }
    else:
        raise ValueError(f"Unsupported content type: {content_type}")


def predict_fn(input_data, model):
    """
    Make prediction
    
    Args:
        input_data: Input data dict
        model: Loaded model
        
    Returns:
        Prediction result
    """
    global device
    
    # Convert to tensors
    resume_tensor = torch.FloatTensor(input_data['resume_embedding']).unsqueeze(0).to(device)
    job_tensor = torch.FloatTensor(input_data['job_embedding']).unsqueeze(0).to(device)
    threshold = input_data['threshold']
    
    # Inference
    with torch.no_grad():
        probability = model(resume_tensor, job_tensor).item()
    
    # Make decision
    prediction = "shortlist" if probability >= threshold else "reject"
    confidence = abs(probability - threshold) / threshold
    
    return {
        'probability': float(probability),
        'prediction': prediction,
        'confidence': float(confidence)
    }


def output_fn(prediction, accept):
    """
    Serialize output
    
    Args:
        prediction: Prediction result
        accept: Accept header
        
    Returns:
        Serialized response
    """
    if accept == 'application/json':
        return json.dumps(prediction), accept
    else:
        raise ValueError(f"Unsupported accept type: {accept}")

