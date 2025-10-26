"""
Analytics API routes
Provides metrics and statistics for dashboard
"""

import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, Query

from backend.core.models import AnalyticsMetrics
from backend.storage.db_handler import get_db_handler

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/analytics", tags=["Analytics"])


@router.get("/metrics", response_model=AnalyticsMetrics)
async def get_analytics_metrics(
    job_id: Optional[str] = Query(None, description="Filter by job ID")
):
    """
    Get analytics metrics for dashboard
    
    Args:
        job_id: Optional job ID to filter metrics
        
    Returns:
        Analytics metrics
    """
    try:
        db = get_db_handler()
        
        # Get metrics from database
        metrics = db.get_analytics_metrics(job_id=job_id)
        
        # Get job and resume counts
        jobs = db.list_jobs(limit=10000)
        resumes = db.list_resumes(limit=10000)
        
        # Build response
        analytics = AnalyticsMetrics(
            total_jobs=len(jobs),
            total_resumes=len(resumes),
            total_screenings=metrics['total_screenings'],
            shortlist_rate=metrics['shortlist_rate'],
            average_score=metrics['average_score'],
            average_processing_time_ms=metrics['average_processing_time_ms'],
            screenings_by_day={},  # TODO: Implement time series
            decisions_distribution={
                'shortlist': metrics['shortlist_count'],
                'reject': metrics['reject_count'],
                'review': metrics['review_count']
            }
        )
        
        return analytics
        
    except Exception as e:
        logger.error(f"Failed to get analytics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

