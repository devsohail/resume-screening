"""
Ranking utilities for screening results
Sorts and ranks candidates based on similarity scores
"""

import logging
from typing import List, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RankedCandidate:
    """Represents a ranked candidate"""
    resume_id: str
    score: float
    rank: int
    metadata: Dict[str, Any]


class CandidateRanker:
    """Ranks candidates based on screening scores"""
    
    @staticmethod
    def rank_candidates(
        results: List[Dict[str, Any]],
        score_key: str = 'final_score',
        reverse: bool = True
    ) -> List[RankedCandidate]:
        """
        Rank candidates by score
        
        Args:
            results: List of screening results
            score_key: Key to use for scoring
            reverse: Sort in descending order (highest scores first)
            
        Returns:
            List of ranked candidates
        """
        # Sort results by score
        sorted_results = sorted(
            results,
            key=lambda x: x.get(score_key, 0),
            reverse=reverse
        )
        
        # Create ranked candidates
        ranked = []
        for rank, result in enumerate(sorted_results, start=1):
            ranked.append(RankedCandidate(
                resume_id=result.get('resume_id', ''),
                score=result.get(score_key, 0),
                rank=rank,
                metadata=result
            ))
        
        logger.info(f"Ranked {len(ranked)} candidates")
        return ranked
    
    @staticmethod
    def filter_by_threshold(
        ranked_candidates: List[RankedCandidate],
        threshold: float
    ) -> List[RankedCandidate]:
        """
        Filter candidates by minimum score threshold
        
        Args:
            ranked_candidates: List of ranked candidates
            threshold: Minimum score threshold
            
        Returns:
            Filtered list of candidates
        """
        filtered = [c for c in ranked_candidates if c.score >= threshold]
        logger.info(f"Filtered {len(filtered)}/{len(ranked_candidates)} candidates above threshold {threshold}")
        return filtered
    
    @staticmethod
    def get_top_n(
        ranked_candidates: List[RankedCandidate],
        n: int
    ) -> List[RankedCandidate]:
        """
        Get top N candidates
        
        Args:
            ranked_candidates: List of ranked candidates
            n: Number of top candidates to return
            
        Returns:
            Top N candidates
        """
        return ranked_candidates[:n]

