"""
PyTorch binary classifier model for resume screening
Neural network that takes resume and job embeddings as input
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class ResumeClassifier(nn.Module):
    """
    Binary classifier for resume screening
    
    Architecture:
    - Input: Concatenated resume and job embeddings (2 * embedding_dim)
    - Hidden layers with dropout for regularization
    - Output: Single sigmoid unit (probability of shortlisting)
    """
    
    def __init__(
        self,
        embedding_dim: int = 1536,  # Bedrock Titan dimension
        hidden_dims: list = [512, 256, 128],
        dropout_rate: float = 0.3
    ):
        """
        Initialize classifier model
        
        Args:
            embedding_dim: Dimension of input embeddings
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout rate for regularization
        """
        super(ResumeClassifier, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.input_dim = embedding_dim * 2  # Concatenated resume + job embeddings
        
        # Build layers
        layers = []
        prev_dim = self.input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
        
        logger.info(
            f"Initialized ResumeClassifier with "
            f"embedding_dim={embedding_dim}, "
            f"hidden_dims={hidden_dims}, "
            f"dropout={dropout_rate}"
        )
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, resume_embedding: torch.Tensor, job_embedding: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            resume_embedding: Resume embedding tensor (batch_size, embedding_dim)
            job_embedding: Job embedding tensor (batch_size, embedding_dim)
            
        Returns:
            Probability of shortlisting (batch_size, 1)
        """
        # Concatenate embeddings
        x = torch.cat([resume_embedding, job_embedding], dim=1)
        
        # Forward through network
        logits = self.network(x)
        
        # Apply sigmoid activation
        probabilities = torch.sigmoid(logits)
        
        return probabilities
    
    def predict(self, resume_embedding: torch.Tensor, job_embedding: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Make binary predictions
        
        Args:
            resume_embedding: Resume embedding
            job_embedding: Job embedding
            threshold: Classification threshold
            
        Returns:
            Binary predictions (0 or 1)
        """
        self.eval()
        with torch.no_grad():
            probabilities = self.forward(resume_embedding, job_embedding)
            predictions = (probabilities >= threshold).long()
        
        return predictions
    
    def get_model_size(self) -> int:
        """
        Calculate total number of parameters
        
        Returns:
            Number of parameters
        """
        return sum(p.numel() for p in self.parameters())
    
    def save_checkpoint(self, filepath: str, optimizer=None, epoch: int = 0, metrics: dict = None):
        """
        Save model checkpoint
        
        Args:
            filepath: Path to save checkpoint
            optimizer: Optimizer state (optional)
            epoch: Current epoch
            metrics: Training metrics
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'embedding_dim': self.embedding_dim,
            'model_config': {
                'embedding_dim': self.embedding_dim,
                'input_dim': self.input_dim
            },
            'metrics': metrics or {}
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        torch.save(checkpoint, filepath)
        logger.info(f"Saved checkpoint to {filepath}")
    
    @classmethod
    def load_checkpoint(cls, filepath: str, map_location=None):
        """
        Load model from checkpoint
        
        Args:
            filepath: Path to checkpoint
            map_location: Device to load model to
            
        Returns:
            Tuple of (model, optimizer_state, epoch, metrics)
        """
        checkpoint = torch.load(filepath, map_location=map_location)
        
        # Create model with same config
        model = cls(embedding_dim=checkpoint['model_config']['embedding_dim'])
        model.load_state_dict(checkpoint['model_state_dict'])
        
        optimizer_state = checkpoint.get('optimizer_state_dict')
        epoch = checkpoint.get('epoch', 0)
        metrics = checkpoint.get('metrics', {})
        
        logger.info(f"Loaded checkpoint from {filepath} (epoch {epoch})")
        
        return model, optimizer_state, epoch, metrics


class EnsembleClassifier(nn.Module):
    """
    Ensemble of multiple classifiers for improved performance
    """
    
    def __init__(self, models: list):
        """
        Initialize ensemble
        
        Args:
            models: List of ResumeClassifier models
        """
        super(EnsembleClassifier, self).__init__()
        self.models = nn.ModuleList(models)
        self.num_models = len(models)
    
    def forward(self, resume_embedding: torch.Tensor, job_embedding: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ensemble
        
        Args:
            resume_embedding: Resume embedding
            job_embedding: Job embedding
            
        Returns:
            Average probability across all models
        """
        outputs = []
        for model in self.models:
            output = model(resume_embedding, job_embedding)
            outputs.append(output)
        
        # Average predictions
        ensemble_output = torch.stack(outputs).mean(dim=0)
        
        return ensemble_output
    
    def predict(self, resume_embedding: torch.Tensor, job_embedding: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Make ensemble predictions
        
        Args:
            resume_embedding: Resume embedding
            job_embedding: Job embedding
            threshold: Classification threshold
            
        Returns:
            Binary predictions
        """
        self.eval()
        with torch.no_grad():
            probabilities = self.forward(resume_embedding, job_embedding)
            predictions = (probabilities >= threshold).long()
        
        return predictions

