"""
Similarity scoring engine
Calculates semantic and feature-based similarity between resume and job description
"""

import logging
from typing import Dict, List, Optional
import numpy as np

from backend.ml.embeddings.base import BaseEmbedder
from backend.ml.preprocessing.feature_extractor import FeatureExtractor
from backend.core.models import SimilarityScore

logger = logging.getLogger(__name__)


class SimilarityScorer:
    """
    Calculate similarity scores between resumes and job descriptions
    Uses combination of semantic embeddings and feature matching
    """
    
    def __init__(
        self,
        embedder: BaseEmbedder,
        feature_extractor: FeatureExtractor,
        semantic_weight: float = 0.5,
        skills_weight: float = 0.3,
        experience_weight: float = 0.2
    ):
        """
        Initialize similarity scorer
        
        Args:
            embedder: Embedding model (e.g., BedrockEmbedder)
            feature_extractor: Feature extraction utility
            semantic_weight: Weight for semantic similarity (0-1)
            skills_weight: Weight for skills match (0-1)
            experience_weight: Weight for experience match (0-1)
        """
        self.embedder = embedder
        self.feature_extractor = feature_extractor
        
        # Normalize weights
        total_weight = semantic_weight + skills_weight + experience_weight
        self.semantic_weight = semantic_weight / total_weight
        self.skills_weight = skills_weight / total_weight
        self.experience_weight = experience_weight / total_weight
        
        logger.info(
            f"Initialized SimilarityScorer with weights: "
            f"semantic={self.semantic_weight:.2f}, "
            f"skills={self.skills_weight:.2f}, "
            f"experience={self.experience_weight:.2f}"
        )
    
    async def calculate_semantic_similarity(
        self,
        resume_text: str,
        job_text: str
    ) -> float:
        """
        Calculate semantic similarity using embeddings
        
        Args:
            resume_text: Resume text
            job_text: Job description text
            
        Returns:
            Cosine similarity score (0-1)
        """
        try:
            # Generate embeddings
            resume_embedding = await self.embedder.embed_text(resume_text)
            job_embedding = await self.embedder.embed_text(job_text)
            
            # Calculate cosine similarity
            similarity = self.embedder.cosine_similarity(resume_embedding, job_embedding)
            
            logger.debug(f"Semantic similarity: {similarity:.4f}")
            return similarity
            
        except Exception as e:
            logger.error(f"Failed to calculate semantic similarity: {e}")
            # Return neutral score on failure
            return 0.5
    
    def calculate_skills_similarity(
        self,
        resume_skills: List[str],
        required_skills: List[str],
        preferred_skills: Optional[List[str]] = None
    ) -> float:
        """
        Calculate skills match score
        
        Args:
            resume_skills: Skills from resume
            required_skills: Required skills from job
            preferred_skills: Preferred skills from job
            
        Returns:
            Skills match score (0-1)
        """
        skills_match = self.feature_extractor.calculate_skills_match(
            resume_skills=resume_skills,
            required_skills=required_skills,
            preferred_skills=preferred_skills
        )
        
        # Required skills have more weight
        required_score = skills_match['required_match_percentage'] / 100
        preferred_score = skills_match['preferred_match_percentage'] / 100
        
        # Weighted combination: 70% required, 30% preferred
        combined_score = 0.7 * required_score + 0.3 * preferred_score
        
        logger.debug(
            f"Skills similarity: required={required_score:.2f}, "
            f"preferred={preferred_score:.2f}, combined={combined_score:.2f}"
        )
        
        return combined_score
    
    def calculate_education_similarity(
        self,
        resume_education: List[str],
        job_education: List[str]
    ) -> float:
        """
        Calculate education match score
        
        Args:
            resume_education: Education from resume
            job_education: Education requirements from job
            
        Returns:
            Education match score (0-1)
        """
        if not job_education:
            return 1.0  # No specific requirements
        
        if not resume_education:
            return 0.3  # Penalty for missing education info
        
        # Check for any overlap in education levels
        resume_set = set(edu.lower() for edu in resume_education)
        job_set = set(edu.lower() for edu in job_education)
        
        overlap = resume_set & job_set
        
        if overlap:
            return 1.0
        
        # Partial credit for higher education
        # e.g., if job requires Bachelor's but candidate has Master's
        higher_education = {'phd', 'ph.d', 'doctorate', 'masters', 'master', 'msc', 'ma', 'mba'}
        mid_education = {'bachelor', 'bachelors', 'bs', 'ba', 'bsc'}
        
        resume_has_higher = bool(resume_set & higher_education)
        job_requires_mid = bool(job_set & mid_education)
        
        if resume_has_higher and job_requires_mid:
            return 1.0  # Over-qualified is acceptable
        
        return 0.5  # Partial match
    
    async def calculate_overall_similarity(
        self,
        resume_text: str,
        job_text: str,
        resume_skills: Optional[List[str]] = None,
        required_skills: Optional[List[str]] = None,
        preferred_skills: Optional[List[str]] = None,
        resume_experience: Optional[float] = None,
        min_experience: Optional[float] = None,
        max_experience: Optional[float] = None,
        resume_education: Optional[List[str]] = None,
        job_education: Optional[List[str]] = None
    ) -> SimilarityScore:
        """
        Calculate comprehensive similarity score
        
        Args:
            resume_text: Full resume text
            job_text: Full job description text
            resume_skills: Pre-extracted resume skills (optional)
            required_skills: Required skills list
            preferred_skills: Preferred skills list
            resume_experience: Years of experience
            min_experience: Minimum required experience
            max_experience: Maximum preferred experience
            resume_education: Resume education
            job_education: Job education requirements
            
        Returns:
            SimilarityScore object with detailed scores
        """
        # Extract features if not provided
        if resume_skills is None:
            resume_skills = self.feature_extractor.extract_skills(resume_text)
        
        if required_skills is None:
            required_skills = self.feature_extractor.extract_skills(job_text)
        
        if resume_education is None:
            resume_education = self.feature_extractor.extract_education(resume_text)
        
        if job_education is None:
            job_education = self.feature_extractor.extract_education(job_text)
        
        # Calculate semantic similarity
        semantic_sim = await self.calculate_semantic_similarity(resume_text, job_text)
        
        # Calculate skills similarity
        skills_sim = self.calculate_skills_similarity(
            resume_skills=resume_skills,
            required_skills=required_skills,
            preferred_skills=preferred_skills
        )
        
        # Calculate experience match
        experience_score = self.feature_extractor.calculate_experience_match(
            resume_experience=resume_experience,
            min_experience=min_experience,
            max_experience=max_experience
        )
        experience_sim = experience_score / 100  # Convert to 0-1 scale
        
        # Calculate education match
        education_sim = self.calculate_education_similarity(
            resume_education=resume_education,
            job_education=job_education
        )
        
        # Calculate weighted overall score
        overall_score = (
            self.semantic_weight * semantic_sim +
            self.skills_weight * skills_sim +
            self.experience_weight * experience_sim
        ) * 100  # Convert to 0-100 scale
        
        logger.info(
            f"Overall similarity: {overall_score:.2f} "
            f"(semantic={semantic_sim:.2f}, skills={skills_sim:.2f}, "
            f"experience={experience_sim:.2f}, education={education_sim:.2f})"
        )
        
        return SimilarityScore(
            overall_score=round(overall_score, 2),
            skills_match=round(skills_sim * 100, 2),
            experience_match=round(experience_score, 2),
            education_match=round(education_sim * 100, 2),
            semantic_similarity=round(semantic_sim, 4)
        )


def create_similarity_scorer(
    embedder: BaseEmbedder,
    feature_extractor: FeatureExtractor
) -> SimilarityScorer:
    """
    Factory function to create SimilarityScorer instance
    
    Args:
        embedder: Embedding model
        feature_extractor: Feature extractor
        
    Returns:
        Configured SimilarityScorer instance
    """
    return SimilarityScorer(
        embedder=embedder,
        feature_extractor=feature_extractor,
        semantic_weight=0.5,
        skills_weight=0.3,
        experience_weight=0.2
    )

