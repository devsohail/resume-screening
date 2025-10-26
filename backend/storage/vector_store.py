"""
Vector database integration using Pinecone
Stores and retrieves resume embeddings for similarity search
"""

import logging
from typing import List, Dict, Optional, Any
import numpy as np
from pinecone import Pinecone, ServerlessSpec
from tenacity import retry, stop_after_attempt, wait_exponential

from backend.core.config import settings
from backend.core.exceptions import VectorStoreError

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Pinecone vector database client
    Manages resume embeddings for efficient similarity search
    """
    
    def __init__(self):
        """Initialize Pinecone client"""
        # Check if Pinecone is configured
        if not settings.pinecone_api_key or settings.pinecone_api_key == "your-pinecone-api-key-here":
            logger.warning("Pinecone not configured - vector search will be disabled")
            self.enabled = False
            self.pc = None
            self.index = None
            return
        
        try:
            # Initialize Pinecone
            self.pc = Pinecone(api_key=settings.pinecone_api_key)
            self.index_name = settings.pinecone_index_name
            self.dimension = settings.pinecone_dimension
            self.metric = settings.pinecone_metric
            self.enabled = True
            
            # Create index if it doesn't exist
            self._ensure_index_exists()
            
            # Connect to index
            self.index = self.pc.Index(self.index_name)
            
            logger.info(f"Connected to Pinecone index: {self.index_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone client: {e}")
            logger.warning("Continuing without vector search capabilities")
            self.enabled = False
            self.pc = None
            self.index = None
    
    def _ensure_index_exists(self):
        """Create Pinecone index if it doesn't exist"""
        try:
            # Check if index exists
            existing_indexes = self.pc.list_indexes()
            index_names = [idx['name'] for idx in existing_indexes]
            
            if self.index_name not in index_names:
                logger.info(f"Creating Pinecone index: {self.index_name}")
                
                # Create serverless index
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric=self.metric,
                    spec=ServerlessSpec(
                        cloud='aws',
                        region=settings.pinecone_environment
                    )
                )
                
                logger.info(f"Successfully created index: {self.index_name}")
            else:
                logger.info(f"Index already exists: {self.index_name}")
                
        except Exception as e:
            logger.error(f"Failed to ensure index exists: {e}")
            raise VectorStoreError(f"Failed to create/check index: {e}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def upsert_resume(
        self,
        resume_id: str,
        embedding: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Insert or update resume embedding
        
        Args:
            resume_id: Unique resume identifier
            embedding: Resume embedding vector
            metadata: Additional metadata (skills, experience, etc.)
        """
        if not self.enabled:
            logger.debug("Vector store disabled - skipping upsert")
            return
        
        try:
            # Prepare vector for upsert
            vector = {
                'id': resume_id,
                'values': embedding.tolist() if isinstance(embedding, np.ndarray) else embedding,
                'metadata': metadata or {}
            }
            
            # Upsert to Pinecone
            self.index.upsert(vectors=[vector])
            
            logger.debug(f"Upserted resume: {resume_id}")
            
        except Exception as e:
            logger.error(f"Failed to upsert resume {resume_id}: {e}")
            raise VectorStoreError(f"Failed to upsert resume: {e}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def upsert_batch(
        self,
        resume_ids: List[str],
        embeddings: List[np.ndarray],
        metadata_list: Optional[List[Dict[str, Any]]] = None
    ):
        """
        Batch insert/update multiple resumes
        
        Args:
            resume_ids: List of resume IDs
            embeddings: List of embedding vectors
            metadata_list: List of metadata dictionaries
        """
        if not self.enabled:
            logger.debug("Vector store disabled - skipping batch upsert")
            return
        
        try:
            if metadata_list is None:
                metadata_list = [{}] * len(resume_ids)
            
            # Prepare vectors
            vectors = []
            for resume_id, embedding, metadata in zip(resume_ids, embeddings, metadata_list):
                vectors.append({
                    'id': resume_id,
                    'values': embedding.tolist() if isinstance(embedding, np.ndarray) else embedding,
                    'metadata': metadata
                })
            
            # Batch upsert (Pinecone recommends batches of 100)
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                self.index.upsert(vectors=batch)
                logger.info(f"Upserted batch {i // batch_size + 1}: {len(batch)} resumes")
            
            logger.info(f"Successfully upserted {len(vectors)} resumes")
            
        except Exception as e:
            logger.error(f"Failed to batch upsert: {e}")
            raise VectorStoreError(f"Batch upsert failed: {e}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def search_similar(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Search for similar resumes
        
        Args:
            query_embedding: Query embedding vector (job description)
            top_k: Number of results to return
            filter_dict: Metadata filters (e.g., {'experience_years': {'$gte': 5}})
            include_metadata: Whether to include metadata in results
            
        Returns:
            List of similar resumes with scores
        """
        if not self.enabled:
            logger.debug("Vector store disabled - returning empty results")
            return []
        
        try:
            # Query Pinecone
            results = self.index.query(
                vector=query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding,
                top_k=top_k,
                filter=filter_dict,
                include_metadata=include_metadata
            )
            
            # Format results
            matches = []
            for match in results.get('matches', []):
                matches.append({
                    'resume_id': match['id'],
                    'score': match['score'],
                    'metadata': match.get('metadata', {}) if include_metadata else {}
                })
            
            logger.info(f"Found {len(matches)} similar resumes")
            return matches
            
        except Exception as e:
            logger.error(f"Failed to search similar resumes: {e}")
            raise VectorStoreError(f"Similarity search failed: {e}")
    
    def get_resume(self, resume_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific resume by ID
        
        Args:
            resume_id: Resume identifier
            
        Returns:
            Resume data or None if not found
        """
        try:
            result = self.index.fetch(ids=[resume_id])
            
            if resume_id in result.get('vectors', {}):
                vector_data = result['vectors'][resume_id]
                return {
                    'id': resume_id,
                    'values': vector_data.get('values', []),
                    'metadata': vector_data.get('metadata', {})
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to fetch resume {resume_id}: {e}")
            return None
    
    def delete_resume(self, resume_id: str):
        """
        Delete a resume from the vector store
        
        Args:
            resume_id: Resume identifier
        """
        try:
            self.index.delete(ids=[resume_id])
            logger.info(f"Deleted resume: {resume_id}")
            
        except Exception as e:
            logger.error(f"Failed to delete resume {resume_id}: {e}")
            raise VectorStoreError(f"Delete failed: {e}")
    
    def delete_batch(self, resume_ids: List[str]):
        """
        Delete multiple resumes
        
        Args:
            resume_ids: List of resume IDs to delete
        """
        try:
            self.index.delete(ids=resume_ids)
            logger.info(f"Deleted {len(resume_ids)} resumes")
            
        except Exception as e:
            logger.error(f"Failed to delete batch: {e}")
            raise VectorStoreError(f"Batch delete failed: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get index statistics
        
        Returns:
            Dictionary with index stats
        """
        try:
            stats = self.index.describe_index_stats()
            return {
                'total_vectors': stats.get('total_vector_count', 0),
                'dimension': stats.get('dimension', 0),
                'index_fullness': stats.get('index_fullness', 0)
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {}


# Singleton instance
_vector_store_instance = None


def get_vector_store() -> VectorStore:
    """
    Get singleton instance of VectorStore
    
    Returns:
        VectorStore instance
    """
    global _vector_store_instance
    
    if _vector_store_instance is None:
        _vector_store_instance = VectorStore()
    
    return _vector_store_instance

