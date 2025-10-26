"""
Unit tests for embeddings module
"""

import pytest
import numpy as np
from backend.ml.embeddings.bedrock_embedder import BedrockEmbedder


@pytest.mark.asyncio
async def test_embed_text():
    """Test embedding generation"""
    embedder = BedrockEmbedder()
    text = "This is a test resume with Python and AWS skills"
    
    embedding = await embedder.embed_text(text)
    
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape[0] == embedder.get_dimension()


@pytest.mark.asyncio
async def test_embed_batch():
    """Test batch embedding"""
    embedder = BedrockEmbedder()
    texts = ["Resume 1", "Resume 2", "Resume 3"]
    
    embeddings = await embedder.embed_batch(texts)
    
    assert len(embeddings) == len(texts)
    assert all(isinstance(emb, np.ndarray) for emb in embeddings)


def test_cosine_similarity():
    """Test cosine similarity calculation"""
    from backend.ml.embeddings.base import BaseEmbedder
    
    vec1 = np.array([1, 0, 0])
    vec2 = np.array([1, 0, 0])
    vec3 = np.array([0, 1, 0])
    
    # Identical vectors
    sim1 = BaseEmbedder.cosine_similarity(vec1, vec2)
    assert abs(sim1 - 1.0) < 0.001
    
    # Orthogonal vectors
    sim2 = BaseEmbedder.cosine_similarity(vec1, vec3)
    assert abs(sim2 - 0.0) < 0.001

