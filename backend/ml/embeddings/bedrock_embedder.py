"""
AWS Bedrock Titan Embeddings integration
Generates text embeddings using Amazon Bedrock's Titan model
"""

import json
import logging
from typing import List
import numpy as np
import boto3
from botocore.exceptions import ClientError
from tenacity import retry, stop_after_attempt, wait_exponential

from backend.core.config import settings
from backend.core.exceptions import EmbeddingError, ExternalServiceError
from backend.ml.embeddings.base import BaseEmbedder

logger = logging.getLogger(__name__)


class BedrockEmbedder(BaseEmbedder):
    """
    AWS Bedrock Titan Embeddings implementation
    Uses amazon.titan-embed-text-v1 model for generating embeddings
    """
    
    def __init__(self):
        """Initialize Bedrock client"""
        try:
            self.client = boto3.client(
                service_name='bedrock-runtime',
                region_name=settings.bedrock_region,
                aws_access_key_id=settings.aws_access_key_id,
                aws_secret_access_key=settings.aws_secret_access_key
            )
            self.model_id = settings.bedrock_model_id
            self.dimension = settings.pinecone_dimension  # Titan v1 produces 1536-dim embeddings
            
            logger.info(f"Initialized BedrockEmbedder with model: {self.model_id}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Bedrock client: {e}")
            raise EmbeddingError(f"Failed to initialize Bedrock client: {e}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text using AWS Bedrock
        
        Args:
            text: Input text string
            
        Returns:
            numpy array of embedding vector (1536 dimensions)
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding")
            return np.zeros(self.dimension)
        
        try:
            # Prepare request body
            body = json.dumps({
                "inputText": text.strip()
            })
            
            # Call Bedrock API
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=body,
                contentType='application/json',
                accept='application/json'
            )
            
            # Parse response
            response_body = json.loads(response['body'].read())
            embedding = response_body.get('embedding')
            
            if not embedding:
                raise EmbeddingError("No embedding returned from Bedrock")
            
            # Convert to numpy array
            embedding_array = np.array(embedding, dtype=np.float32)
            
            logger.debug(f"Generated embedding with shape: {embedding_array.shape}")
            
            return embedding_array
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_message = e.response['Error']['Message']
            logger.error(f"Bedrock API error: {error_code} - {error_message}")
            
            raise ExternalServiceError(
                service="AWS Bedrock",
                message=f"{error_code}: {error_message}",
                details={"model_id": self.model_id}
            )
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}", exc_info=True)
            raise EmbeddingError(f"Failed to generate embedding: {e}")
    
    async def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts
        
        Note: Bedrock Titan doesn't support native batch processing,
        so we process texts sequentially with retry logic
        
        Args:
            texts: List of input text strings
            
        Returns:
            List of numpy arrays containing embeddings
        """
        embeddings = []
        
        for i, text in enumerate(texts):
            try:
                embedding = await self.embed_text(text)
                embeddings.append(embedding)
                
                # Log progress for large batches
                if (i + 1) % 10 == 0:
                    logger.info(f"Generated {i + 1}/{len(texts)} embeddings")
                    
            except Exception as e:
                logger.error(f"Failed to embed text at index {i}: {e}")
                # Return zero vector for failed embeddings
                embeddings.append(np.zeros(self.dimension))
        
        logger.info(f"Successfully generated {len(embeddings)} embeddings")
        return embeddings
    
    def get_dimension(self) -> int:
        """
        Get embedding dimension
        
        Returns:
            1536 (dimension of Titan Embeddings V1)
        """
        return self.dimension
    
    async def embed_documents(
        self,
        documents: List[str],
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Embed multiple documents and return as 2D array
        
        Args:
            documents: List of document strings
            show_progress: Whether to log progress
            
        Returns:
            2D numpy array of shape (n_documents, embedding_dim)
        """
        embeddings = await self.embed_batch(documents)
        return np.vstack(embeddings)
    
    def test_connection(self) -> bool:
        """
        Test connection to Bedrock service
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Try to embed a simple test string
            import asyncio
            test_embedding = asyncio.run(self.embed_text("test"))
            return len(test_embedding) == self.dimension
        except Exception as e:
            logger.error(f"Bedrock connection test failed: {e}")
            return False


# Singleton instance
_embedder_instance = None


def get_embedder() -> BedrockEmbedder:
    """
    Get singleton instance of BedrockEmbedder
    
    Returns:
        BedrockEmbedder instance
    """
    global _embedder_instance
    
    if _embedder_instance is None:
        _embedder_instance = BedrockEmbedder()
    
    return _embedder_instance

