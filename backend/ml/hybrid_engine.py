"""
Hybrid decision engine
Combines similarity scoring and binary classification for final screening decision
"""

import logging
from typing import Dict, Optional, List
import numpy as np

from backend.core.config import settings
from backend.core.models import (
    ScreeningDecision, ScreeningResult, SimilarityScore,
    ClassifierResult, ScreeningStatus
)
from backend.ml.similarity.scorer import SimilarityScorer
from backend.ml.classifier.inference import ClassifierInference

logger = logging.getLogger(__name__)


class HybridScreeningEngine:
    """
    Hybrid screening engine that combines:
    1. Similarity scoring (semantic + feature-based)
    2. Binary classification (PyTorch model)
    
    Produces a final weighted score and decision
    """
    
    def __init__(
        self,
        similarity_scorer: SimilarityScorer,
        classifier: Optional[ClassifierInference] = None,
        similarity_weight: float = None,
        classifier_weight: float = None,
        similarity_threshold: float = None,
        use_classifier: bool = True,
        use_similarity: bool = True
    ):
        """
        Initialize hybrid engine
        
        Args:
            similarity_scorer: Similarity scoring engine
            classifier: Classifier inference engine
            similarity_weight: Weight for similarity score (0-1)
            classifier_weight: Weight for classifier score (0-1)
            similarity_threshold: Minimum similarity threshold
            use_classifier: Whether to use classifier
            use_similarity: Whether to use similarity
        """
        self.similarity_scorer = similarity_scorer
        self.classifier = classifier
        
        # Weights
        self.similarity_weight = similarity_weight or settings.hybrid_similarity_weight
        self.classifier_weight = classifier_weight or settings.hybrid_classifier_weight
        
        # Normalize weights
        total_weight = self.similarity_weight + self.classifier_weight
        self.similarity_weight /= total_weight
        self.classifier_weight /= total_weight
        
        # Thresholds
        self.similarity_threshold = similarity_threshold or settings.similarity_threshold
        
        # Flags
        self.use_classifier = use_classifier and classifier is not None
        self.use_similarity = use_similarity
        
        logger.info(
            f"Initialized HybridScreeningEngine with "
            f"similarity_weight={self.similarity_weight:.2f}, "
            f"classifier_weight={self.classifier_weight:.2f}, "
            f"similarity_threshold={self.similarity_threshold}, "
            f"use_classifier={self.use_classifier}, "
            f"use_similarity={self.use_similarity}"
        )
    
    async def screen_candidate(
        self,
        resume_id: str,
        job_id: str,
        resume_text: str,
        job_text: str,
        resume_embedding: Optional[np.ndarray] = None,
        job_embedding: Optional[np.ndarray] = None,
        resume_skills: Optional[List[str]] = None,
        required_skills: Optional[List[str]] = None,
        preferred_skills: Optional[List[str]] = None,
        resume_experience: Optional[float] = None,
        min_experience: Optional[float] = None,
        max_experience: Optional[float] = None,
        resume_education: Optional[List[str]] = None,
        job_education: Optional[List[str]] = None
    ) -> ScreeningResult:
        """
        Screen a candidate using hybrid approach
        
        Args:
            resume_id: Resume identifier
            job_id: Job identifier
            resume_text: Full resume text
            job_text: Full job description
            resume_embedding: Resume embedding vector
            job_embedding: Job embedding vector
            resume_skills: Resume skills (optional)
            required_skills: Required skills
            preferred_skills: Preferred skills
            resume_experience: Years of experience
            min_experience: Minimum required experience
            max_experience: Maximum preferred experience
            resume_education: Resume education
            job_education: Job education requirements
            
        Returns:
            ScreeningResult with final decision and scores
        """
        import time
        start_time = time.time()
        
        similarity_result = None
        classifier_result = None
        final_score = 0.0
        decision = ScreeningDecision.REJECT
        explanation_parts = []
        matched_skills = []
        missing_skills = []
        
        try:
            # 1. Calculate similarity score
            if self.use_similarity:
                similarity_result = await self.similarity_scorer.calculate_overall_similarity(
                    resume_text=resume_text,
                    job_text=job_text,
                    resume_skills=resume_skills,
                    required_skills=required_skills,
                    preferred_skills=preferred_skills,
                    resume_experience=resume_experience,
                    min_experience=min_experience,
                    max_experience=max_experience,
                    resume_education=resume_education,
                    job_education=job_education
                )
                
                logger.info(f"Similarity score: {similarity_result.overall_score:.2f}")
                
                # Extract skill matching info
                from backend.ml.preprocessing.feature_extractor import get_feature_extractor
                extractor = get_feature_extractor()
                
                if resume_skills is None:
                    resume_skills = extractor.extract_skills(resume_text)
                if required_skills is None:
                    required_skills = extractor.extract_skills(job_text)
                
                skills_match = extractor.calculate_skills_match(
                    resume_skills=resume_skills,
                    required_skills=required_skills,
                    preferred_skills=preferred_skills
                )
                
                matched_skills = skills_match['matched_required'] + skills_match.get('matched_preferred', [])
                missing_skills = skills_match['missing_required']
                
                explanation_parts.append(
                    f"Similarity score: {similarity_result.overall_score:.1f}/100 "
                    f"(Skills: {similarity_result.skills_match:.1f}%, "
                    f"Experience: {similarity_result.experience_match:.1f}%, "
                    f"Semantic: {similarity_result.semantic_similarity:.2f})"
                )
            
            # 2. Run classifier prediction
            if self.use_classifier and resume_embedding is not None and job_embedding is not None:
                try:
                    classifier_result = self.classifier.predict(
                        resume_embedding=resume_embedding,
                        job_embedding=job_embedding
                    )
                    
                    logger.info(
                        f"Classifier prediction: {classifier_result.prediction.value} "
                        f"(probability={classifier_result.probability:.4f})"
                    )
                    
                    explanation_parts.append(
                        f"Classifier probability: {classifier_result.probability:.2f} "
                        f"(confidence: {classifier_result.confidence:.2f})"
                    )
                except Exception as e:
                    logger.warning(f"Classifier prediction failed: {e}. Using similarity-only scoring.")
                    self.use_classifier = False  # Disable for this session
            
            # 3. Calculate final weighted score
            if self.use_similarity and self.use_classifier:
                # Hybrid: combine both scores
                similarity_normalized = similarity_result.overall_score / 100  # Convert to 0-1
                final_score = (
                    self.similarity_weight * similarity_normalized +
                    self.classifier_weight * classifier_result.probability
                ) * 100  # Convert back to 0-100
                
                # Decision logic: Both should agree for shortlist
                similarity_pass = similarity_result.overall_score >= self.similarity_threshold
                classifier_pass = classifier_result.prediction == ScreeningDecision.SHORTLIST
                
                if similarity_pass and classifier_pass:
                    decision = ScreeningDecision.SHORTLIST
                elif similarity_pass or classifier_pass:
                    decision = ScreeningDecision.REVIEW  # Conflicting signals
                else:
                    decision = ScreeningDecision.REJECT
                
                explanation = (
                    f"Hybrid decision: {decision.value}. "
                    f"Final score: {final_score:.1f}/100. "
                    + " ".join(explanation_parts)
                )
                
            elif self.use_similarity:
                # Similarity only
                final_score = similarity_result.overall_score
                decision = (
                    ScreeningDecision.SHORTLIST
                    if final_score >= self.similarity_threshold
                    else ScreeningDecision.REJECT
                )
                explanation = f"Similarity-based decision: {decision.value}. " + " ".join(explanation_parts)
                
            elif self.use_classifier:
                # Classifier only
                final_score = classifier_result.probability * 100
                decision = classifier_result.prediction
                explanation = f"Classifier-based decision: {decision.value}. " + " ".join(explanation_parts)
                
            else:
                # Neither enabled (shouldn't happen)
                explanation = "No screening method enabled"
            
            # Add skill matching details to explanation
            if matched_skills:
                explanation += f" Matched skills: {', '.join(matched_skills[:5])}"
            if missing_skills:
                explanation += f" Missing skills: {', '.join(missing_skills[:5])}"
            
            # Calculate processing time
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            logger.info(
                f"Screening completed for {resume_id}: {decision.value} "
                f"(score={final_score:.2f}, time={processing_time_ms}ms)"
            )
            
            # Create result
            from datetime import datetime
            from uuid import uuid4
            
            result = ScreeningResult(
                id=str(uuid4()),
                resume_id=resume_id,
                job_id=job_id,
                status=ScreeningStatus.COMPLETED,
                decision=decision,
                final_score=round(final_score, 2),
                similarity_result=similarity_result,
                classifier_result=classifier_result,
                explanation=explanation,
                matched_skills=matched_skills,
                missing_skills=missing_skills,
                completed_at=datetime.utcnow(),
                processing_time_ms=processing_time_ms,
                model_version="1.0.0"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Screening failed for {resume_id}: {e}", exc_info=True)
            
            # Return failed result
            from datetime import datetime
            from uuid import uuid4
            
            return ScreeningResult(
                id=str(uuid4()),
                resume_id=resume_id,
                job_id=job_id,
                status=ScreeningStatus.FAILED,
                decision=None,
                final_score=None,
                explanation=f"Screening failed: {str(e)}",
                matched_skills=[],
                missing_skills=[]
            )
    
    async def screen_batch(
        self,
        screening_requests: List[Dict]
    ) -> List[ScreeningResult]:
        """
        Screen multiple candidates in batch
        
        Args:
            screening_requests: List of screening request dictionaries
            
        Returns:
            List of ScreeningResult objects
        """
        results = []
        
        for i, request in enumerate(screening_requests):
            try:
                result = await self.screen_candidate(**request)
                results.append(result)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{len(screening_requests)} screenings")
                    
            except Exception as e:
                logger.error(f"Failed to screen candidate {i}: {e}")
                # Add failed result
                from datetime import datetime
                from uuid import uuid4
                
                results.append(ScreeningResult(
                    id=str(uuid4()),
                    resume_id=request.get('resume_id', 'unknown'),
                    job_id=request.get('job_id', 'unknown'),
                    status=ScreeningStatus.FAILED,
                    decision=None,
                    final_score=None,
                    explanation=f"Screening failed: {str(e)}",
                    matched_skills=[],
                    missing_skills=[]
                ))
        
        logger.info(f"Batch screening completed: {len(results)} results")
        return results


def create_hybrid_engine(
    similarity_scorer: SimilarityScorer,
    classifier: Optional[ClassifierInference] = None
) -> HybridScreeningEngine:
    """
    Factory function to create HybridScreeningEngine
    
    Args:
        similarity_scorer: Similarity scorer instance
        classifier: Classifier inference instance
        
    Returns:
        Configured HybridScreeningEngine
    """
    return HybridScreeningEngine(
        similarity_scorer=similarity_scorer,
        classifier=classifier,
        use_classifier=settings.enable_classifier,
        use_similarity=settings.enable_similarity
    )

