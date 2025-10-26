"""
Database handler
SQLAlchemy session management and CRUD operations
"""

import logging
from typing import List, Optional, Dict, Any
from contextlib import contextmanager
from sqlalchemy import create_engine, and_
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError

from backend.core.config import settings
from backend.core.exceptions import DatabaseError
from backend.storage.db_models import Base, Job, Resume, ScreeningResult, ModelMetadata

logger = logging.getLogger(__name__)


class DatabaseHandler:
    """
    Database handler for PostgreSQL operations
    Manages connections, sessions, and CRUD operations
    """
    
    def __init__(self):
        """Initialize database connection"""
        try:
            # Create engine
            self.engine = create_engine(
                settings.database_url,
                pool_size=settings.db_pool_size,
                max_overflow=settings.db_max_overflow,
                pool_pre_ping=True,  # Verify connections before using
                echo=settings.api_debug  # Log SQL queries in debug mode
            )
            
            # Create session factory
            self.SessionLocal = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine
            )
            
            logger.info("Initialized DatabaseHandler")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise DatabaseError(f"Database initialization failed: {e}")
    
    def create_tables(self):
        """Create all database tables"""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Created database tables")
        except Exception as e:
            logger.error(f"Failed to create tables: {e}")
            raise DatabaseError(f"Table creation failed: {e}")
    
    def drop_tables(self):
        """Drop all database tables (use with caution!)"""
        try:
            Base.metadata.drop_all(bind=self.engine)
            logger.warning("Dropped all database tables")
        except Exception as e:
            logger.error(f"Failed to drop tables: {e}")
            raise DatabaseError(f"Table drop failed: {e}")
    
    @contextmanager
    def get_session(self) -> Session:
        """
        Get database session context manager
        
        Yields:
            SQLAlchemy session
        """
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise DatabaseError(f"Database operation failed: {e}")
        finally:
            session.close()
    
    # Job CRUD operations
    def create_job(self, job_data: Dict[str, Any]) -> Job:
        """Create a new job"""
        with self.get_session() as session:
            job = Job(**job_data)
            session.add(job)
            session.flush()
            session.refresh(job)
            session.expunge(job)  # Detach to keep accessible after session closes
            logger.info(f"Created job: {job.id}")
            return job
    
    def get_job(self, job_id: str) -> Optional[Job]:
        """Get job by ID"""
        with self.get_session() as session:
            job = session.query(Job).filter(Job.id == job_id).first()
            if job:
                # Expunge to detach from session but keep data accessible
                session.expunge(job)
            return job
    
    def list_jobs(
        self,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Job]:
        """List jobs with filters"""
        with self.get_session() as session:
            query = session.query(Job)
            
            if status:
                query = query.filter(Job.status == status)
            
            jobs = query.order_by(Job.created_at.desc()).limit(limit).offset(offset).all()
            
            # Expunge all objects to make them accessible after session closes
            for job in jobs:
                session.expunge(job)
            
            return jobs
    
    def update_job(self, job_id: str, update_data: Dict[str, Any]) -> Optional[Job]:
        """Update job"""
        with self.get_session() as session:
            job = session.query(Job).filter(Job.id == job_id).first()
            if job:
                for key, value in update_data.items():
                    if hasattr(job, key):
                        setattr(job, key, value)
                session.flush()
                session.refresh(job)
                session.expunge(job)  # Detach to keep accessible after session closes
                logger.info(f"Updated job: {job_id}")
            return job
    
    def delete_job(self, job_id: str) -> bool:
        """Delete job"""
        with self.get_session() as session:
            job = session.query(Job).filter(Job.id == job_id).first()
            if job:
                session.delete(job)
                logger.info(f"Deleted job: {job_id}")
                return True
            return False
    
    # Resume CRUD operations
    def create_resume(self, resume_data: Dict[str, Any]) -> Resume:
        """Create a new resume"""
        with self.get_session() as session:
            resume = Resume(**resume_data)
            session.add(resume)
            session.flush()
            session.refresh(resume)
            logger.info(f"Created resume: {resume.id}")
            return resume
    
    def get_resume(self, resume_id: str) -> Optional[Resume]:
        """Get resume by ID"""
        with self.get_session() as session:
            resume = session.query(Resume).filter(Resume.id == resume_id).first()
            return resume
    
    def list_resumes(self, limit: int = 100, offset: int = 0) -> List[Resume]:
        """List resumes"""
        with self.get_session() as session:
            resumes = (
                session.query(Resume)
                .order_by(Resume.created_at.desc())
                .limit(limit)
                .offset(offset)
                .all()
            )
            return resumes
    
    def update_resume(self, resume_id: str, update_data: Dict[str, Any]) -> Optional[Resume]:
        """Update resume"""
        with self.get_session() as session:
            resume = session.query(Resume).filter(Resume.id == resume_id).first()
            if resume:
                for key, value in update_data.items():
                    if hasattr(resume, key):
                        setattr(resume, key, value)
                session.flush()
                session.refresh(resume)
                logger.info(f"Updated resume: {resume_id}")
            return resume
    
    def delete_resume(self, resume_id: str) -> bool:
        """Delete resume"""
        with self.get_session() as session:
            resume = session.query(Resume).filter(Resume.id == resume_id).first()
            if resume:
                session.delete(resume)
                logger.info(f"Deleted resume: {resume_id}")
                return True
            return False
    
    # Screening Result CRUD operations
    def create_screening_result(self, result_data: Dict[str, Any]) -> ScreeningResult:
        """Create a new screening result"""
        with self.get_session() as session:
            result = ScreeningResult(**result_data)
            session.add(result)
            session.flush()
            session.refresh(result)
            session.expunge(result)  # Detach to keep accessible after session closes
            logger.info(f"Created screening result: {result.id}")
            return result
    
    def get_screening_result(self, result_id: str) -> Optional[ScreeningResult]:
        """Get screening result by ID"""
        with self.get_session() as session:
            result = session.query(ScreeningResult).filter(ScreeningResult.id == result_id).first()
            if result:
                session.expunge(result)
            return result
    
    def list_screening_results(
        self,
        job_id: Optional[str] = None,
        resume_id: Optional[str] = None,
        decision: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[ScreeningResult]:
        """List screening results with filters"""
        with self.get_session() as session:
            query = session.query(ScreeningResult)
            
            if job_id:
                query = query.filter(ScreeningResult.job_id == job_id)
            if resume_id:
                query = query.filter(ScreeningResult.resume_id == resume_id)
            if decision:
                query = query.filter(ScreeningResult.decision == decision)
            
            results = query.order_by(ScreeningResult.created_at.desc()).limit(limit).offset(offset).all()
            
            # Expunge all objects to make them accessible after session closes
            for result in results:
                session.expunge(result)
            
            return results
    
    def update_screening_result(
        self,
        result_id: str,
        update_data: Dict[str, Any]
    ) -> Optional[ScreeningResult]:
        """Update screening result"""
        with self.get_session() as session:
            result = session.query(ScreeningResult).filter(ScreeningResult.id == result_id).first()
            if result:
                for key, value in update_data.items():
                    if hasattr(result, key):
                        setattr(result, key, value)
                session.flush()
                session.refresh(result)
                logger.info(f"Updated screening result: {result_id}")
            return result
    
    # Analytics queries
    def get_analytics_metrics(self, job_id: Optional[str] = None) -> Dict[str, Any]:
        """Get analytics metrics"""
        with self.get_session() as session:
            # Base query
            query = session.query(ScreeningResult)
            if job_id:
                query = query.filter(ScreeningResult.job_id == job_id)
            
            # Count metrics
            total_screenings = query.count()
            completed_screenings = query.filter(ScreeningResult.status == "completed").count()
            
            shortlist_count = query.filter(ScreeningResult.decision == "shortlist").count()
            reject_count = query.filter(ScreeningResult.decision == "reject").count()
            review_count = query.filter(ScreeningResult.decision == "review").count()
            
            # Calculate rates
            shortlist_rate = (shortlist_count / completed_screenings * 100) if completed_screenings > 0 else 0
            
            # Average scores
            avg_score_result = query.filter(ScreeningResult.final_score.isnot(None)).all()
            avg_score = sum(r.final_score for r in avg_score_result) / len(avg_score_result) if avg_score_result else 0
            
            # Average processing time
            avg_time_result = query.filter(ScreeningResult.processing_time_ms.isnot(None)).all()
            avg_time = sum(r.processing_time_ms for r in avg_time_result) / len(avg_time_result) if avg_time_result else 0
            
            return {
                "total_screenings": total_screenings,
                "completed_screenings": completed_screenings,
                "shortlist_count": shortlist_count,
                "reject_count": reject_count,
                "review_count": review_count,
                "shortlist_rate": round(shortlist_rate, 2),
                "average_score": round(avg_score, 2),
                "average_processing_time_ms": round(avg_time, 2)
            }


# Singleton instance
_db_handler_instance = None


def get_db_handler() -> DatabaseHandler:
    """
    Get singleton instance of DatabaseHandler
    
    Returns:
        DatabaseHandler instance
    """
    global _db_handler_instance
    
    if _db_handler_instance is None:
        _db_handler_instance = DatabaseHandler()
    
    return _db_handler_instance

