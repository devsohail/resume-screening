"""
Feedback API routes
Allows users to confirm or correct AI screening decisions
This feedback improves training data quality
"""

import logging
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from backend.core.models import ScreeningDecision
from backend.storage.db_handler import get_db_handler

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/feedback", tags=["Feedback"])


class FeedbackRequest(BaseModel):
    """Request model for feedback"""
    screening_result_id: str
    human_decision: ScreeningDecision
    notes: str = ""


class FeedbackResponse(BaseModel):
    """Response model for feedback"""
    success: bool
    message: str
    ready_for_training: bool
    total_samples: int


@router.post("/submit", response_model=FeedbackResponse)
async def submit_feedback(feedback: FeedbackRequest):
    """
    Submit human feedback on AI screening decision
    
    This is used to:
    1. Improve model training with confirmed labels
    2. Track model accuracy
    3. Trigger retraining when enough feedback is collected
    """
    try:
        db = get_db_handler()
        
        # Get screening result
        result = db.get_screening_result(feedback.screening_result_id)
        if not result:
            raise HTTPException(status_code=404, detail="Screening result not found")
        
        # Update with human feedback
        update_data = {
            'human_decision': feedback.human_decision.value,
            'human_notes': feedback.notes,
            'human_reviewed': True,
            'human_reviewed_at': 'now()'  # Will be handled by updated_at
        }
        
        # If human decision differs from AI, mark for retraining
        if result.decision and result.decision.value != feedback.human_decision.value:
            update_data['needs_retraining'] = True
            logger.info(
                f"Human corrected AI decision: {result.decision.value} -> {feedback.human_decision.value}"
            )
        
        db.update_screening_result(feedback.screening_result_id, update_data)
        
        # Count total reviewed samples
        with db.get_session() as session:
            from backend.storage.db_models import ScreeningResult
            total_reviewed = session.query(ScreeningResult).filter(
                ScreeningResult.human_reviewed == True
            ).count()
        
        # Check if ready for training (100+ reviewed samples)
        ready_for_training = total_reviewed >= 100
        
        message = "Feedback submitted successfully!"
        if ready_for_training:
            message += f" You have {total_reviewed} reviewed samples - ready for model training!"
        
        logger.info(
            f"Feedback submitted for {feedback.screening_result_id}: "
            f"{feedback.human_decision.value} (total reviewed: {total_reviewed})"
        )
        
        return FeedbackResponse(
            success=True,
            message=message,
            ready_for_training=ready_for_training,
            total_samples=total_reviewed
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to submit feedback: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/training-status")
async def get_training_status():
    """
    Check if system has enough data for training
    
    Returns:
        - total_samples: Total reviewed samples
        - ready: Whether ready for training
        - needed: Samples needed to reach threshold
    """
    try:
        db = get_db_handler()
        
        with db.get_session() as session:
            from backend.storage.db_models import ScreeningResult
            
            # Count total reviewed
            total_reviewed = session.query(ScreeningResult).filter(
                ScreeningResult.human_reviewed == True
            ).count()
            
            # Count positives and negatives
            positives = session.query(ScreeningResult).filter(
                ScreeningResult.human_reviewed == True,
                ScreeningResult.human_decision == 'shortlist'
            ).count()
            
            negatives = session.query(ScreeningResult).filter(
                ScreeningResult.human_reviewed == True,
                ScreeningResult.human_decision == 'reject'
            ).count()
        
        threshold = 100
        ready = total_reviewed >= threshold
        needed = max(0, threshold - total_reviewed)
        
        return {
            "total_samples": total_reviewed,
            "positive_samples": positives,
            "negative_samples": negatives,
            "ready_for_training": ready,
            "samples_needed": needed,
            "threshold": threshold,
            "message": (
                f"Ready to train! ({total_reviewed} samples)"
                if ready
                else f"Need {needed} more samples to train"
            )
        }
        
    except Exception as e:
        logger.error(f"Failed to get training status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/trigger-training")
async def trigger_training():
    """
    Trigger automatic model training
    
    This endpoint:
    1. Exports reviewed data
    2. Starts training pipeline
    3. Returns job status
    """
    try:
        db = get_db_handler()
        
        # Check if enough data
        with db.get_session() as session:
            from backend.storage.db_models import ScreeningResult
            total_reviewed = session.query(ScreeningResult).filter(
                ScreeningResult.human_reviewed == True
            ).count()
        
        if total_reviewed < 100:
            raise HTTPException(
                status_code=400,
                detail=f"Not enough data. Have {total_reviewed}, need 100 samples."
            )
        
        # Trigger training (async)
        import subprocess
        from datetime import datetime
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = f"logs/training_{timestamp}.log"
        
        # Run training in background
        cmd = [
            "python", "scripts/export_training_data.py",
            "--output", f"training_data/auto_{timestamp}.csv"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise Exception(f"Export failed: {result.stderr}")
        
        # Start training
        cmd = [
            "python", "scripts/auto_train.py",
            "--data", f"training_data/auto_{timestamp}.csv"
        ]
        
        # Run in background
        process = subprocess.Popen(
            cmd,
            stdout=open(log_file, 'w'),
            stderr=subprocess.STDOUT
        )
        
        logger.info(f"Started training job PID {process.pid}")
        
        return {
            "success": True,
            "message": "Training started!",
            "job_id": process.pid,
            "log_file": log_file,
            "status": "Training in progress... Check logs for details"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to trigger training: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

