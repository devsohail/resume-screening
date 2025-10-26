"""
Abstract base class for text embeddings
Allows for different embedding providers (Bedrock, OpenAI, etc.)
"""

from abc import ABC, abstractmethod
from typing import List, Union
import numpy as np


class BaseEmbedder(ABC):
    """Abstract base class for text embedding generation"""
    
    @abstractmethod
    async def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text
        
        Args:
            text: Input text string
            
        Returns:
            numpy array of embedding vectors
        """
        pass
    
    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts
        
        Args:
            texts: List of input text strings
            
        Returns:
            List of numpy arrays containing embeddings
        """
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by this embedder
        
        Returns:
            Integer dimension size
        """
        pass
    
    @staticmethod
    def cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score (0-1)
        """
        # Normalize vectors
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Calculate cosine similarity
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        
        # Clamp to [0, 1] range
        return float(max(0.0, min(1.0, similarity)))

