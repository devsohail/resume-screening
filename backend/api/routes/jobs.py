"""
Jobs API routes
Handles job description management
"""

import logging
from uuid import uuid4
from typing import List, Optional
from fastapi import APIRouter, HTTPException, status, Query

from backend.core.models import JobDescription, JobDescriptionCreate, JobDescriptionUpdate
from backend.storage.db_handler import get_db_handler

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/jobs", tags=["Jobs"])


@router.post("", response_model=JobDescription, status_code=status.HTTP_201_CREATED)
async def create_job(job_data: JobDescriptionCreate):
    """Create a new job description"""
    try:
        db = get_db_handler()
        
        # Create job data
        job_dict = job_data.dict()
        job_dict['id'] = str(uuid4())
        
        job = db.create_job(job_dict)
        
        logger.info(f"Created job: {job.id}")
        
        return JobDescription.from_orm(job)
        
    except Exception as e:
        logger.error(f"Failed to create job: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("", response_model=List[JobDescription])
async def list_jobs(
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0)
):
    """List all jobs with optional filtering"""
    try:
        db = get_db_handler()
        
        jobs = db.list_jobs(status=status, limit=limit, offset=offset)
        
        return [JobDescription.from_orm(job) for job in jobs]
        
    except Exception as e:
        logger.error(f"Failed to list jobs: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{job_id}", response_model=JobDescription)
async def get_job(job_id: str):
    """Get a specific job by ID"""
    try:
        db = get_db_handler()
        
        job = db.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        return JobDescription.from_orm(job)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get job: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{job_id}", response_model=JobDescription)
async def update_job(job_id: str, job_update: JobDescriptionUpdate):
    """Update a job description"""
    try:
        db = get_db_handler()
        
        # Check if job exists
        existing_job = db.get_job(job_id)
        if not existing_job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        # Update job
        update_data = job_update.dict(exclude_unset=True)
        updated_job = db.update_job(job_id, update_data)
        
        logger.info(f"Updated job: {job_id}")
        
        return JobDescription.from_orm(updated_job)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update job: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_job(job_id: str):
    """Delete a job description"""
    try:
        db = get_db_handler()
        
        success = db.delete_job(job_id)
        if not success:
            raise HTTPException(status_code=404, detail="Job not found")
        
        logger.info(f"Deleted job: {job_id}")
        
        return None
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete job: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

