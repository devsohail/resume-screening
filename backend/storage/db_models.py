"""
SQLAlchemy database models
PostgreSQL schema for resumes, jobs, and screening results
"""

from datetime import datetime
from sqlalchemy import Column, String, Integer, Float, Boolean, DateTime, Text, JSON, ForeignKey, Enum as SQLEnum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
import enum
import uuid

Base = declarative_base()


class JobStatusEnum(str, enum.Enum):
    """Job status enumeration"""
    ACTIVE = "active"
    CLOSED = "closed"
    DRAFT = "draft"


class ScreeningStatusEnum(str, enum.Enum):
    """Screening status enumeration"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ScreeningDecisionEnum(str, enum.Enum):
    """Screening decision enumeration"""
    SHORTLIST = "shortlist"
    REJECT = "reject"
    REVIEW = "review"


class Job(Base):
    """Job description table"""
    __tablename__ = "jobs"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    title = Column(String(200), nullable=False, index=True)
    company = Column(String(200), nullable=False)
    description = Column(Text, nullable=False)
    required_skills = Column(JSON, default=list)
    preferred_skills = Column(JSON, default=list)
    min_experience_years = Column(Integer, nullable=True)
    max_experience_years = Column(Integer, nullable=True)
    education_requirements = Column(Text, nullable=True)
    location = Column(String(200), nullable=True)
    salary_range = Column(String(100), nullable=True)
    status = Column(SQLEnum(JobStatusEnum), default=JobStatusEnum.ACTIVE, index=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    screening_results = relationship("ScreeningResult", back_populates="job", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Job(id={self.id}, title={self.title}, company={self.company})>"


class Resume(Base):
    """Resume table"""
    __tablename__ = "resumes"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    candidate_name = Column(String(200), nullable=True, index=True)
    candidate_email = Column(String(200), nullable=True, index=True)
    file_path = Column(String(500), nullable=False)  # S3 path
    extracted_text = Column(Text, nullable=True)
    skills = Column(JSON, default=list)
    experience_years = Column(Float, nullable=True)
    education = Column(Text, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    screening_results = relationship("ScreeningResult", back_populates="resume", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Resume(id={self.id}, name={self.candidate_name})>"


class ScreeningResult(Base):
    """Screening result table"""
    __tablename__ = "screening_results"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    resume_id = Column(String(36), ForeignKey("resumes.id"), nullable=False, index=True)
    job_id = Column(String(36), ForeignKey("jobs.id"), nullable=False, index=True)
    
    # Status and decision
    status = Column(SQLEnum(ScreeningStatusEnum), default=ScreeningStatusEnum.PENDING, nullable=False, index=True)
    decision = Column(SQLEnum(ScreeningDecisionEnum), nullable=True, index=True)
    final_score = Column(Float, nullable=True)
    
    # Similarity results
    similarity_overall_score = Column(Float, nullable=True)
    similarity_skills_match = Column(Float, nullable=True)
    similarity_experience_match = Column(Float, nullable=True)
    similarity_education_match = Column(Float, nullable=True)
    similarity_semantic = Column(Float, nullable=True)
    
    # Classifier results
    classifier_probability = Column(Float, nullable=True)
    classifier_confidence = Column(Float, nullable=True)
    
    # Details
    explanation = Column(Text, nullable=True)
    matched_skills = Column(JSON, default=list)
    missing_skills = Column(JSON, default=list)
    
    # Metadata
    model_version = Column(String(50), nullable=True)
    processing_time_ms = Column(Integer, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    completed_at = Column(DateTime, nullable=True)
    
    # Human feedback for training
    human_reviewed = Column(Boolean, default=False, index=True)
    human_decision = Column(String(20), nullable=True)  # Human-confirmed decision
    human_notes = Column(Text, nullable=True)
    human_reviewed_at = Column(DateTime, nullable=True)
    needs_retraining = Column(Boolean, default=False, index=True)
    
    # Relationships
    resume = relationship("Resume", back_populates="screening_results")
    job = relationship("Job", back_populates="screening_results")
    
    def __repr__(self):
        return f"<ScreeningResult(id={self.id}, resume_id={self.resume_id}, decision={self.decision})>"


class ModelMetadata(Base):
    """Model metadata table"""
    __tablename__ = "model_metadata"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    model_type = Column(String(50), nullable=False)  # 'classifier', 'similarity', etc.
    version = Column(String(50), nullable=False, unique=True, index=True)
    s3_path = Column(String(500), nullable=False)
    
    # Metrics
    accuracy = Column(Float, nullable=True)
    precision = Column(Float, nullable=True)
    recall = Column(Float, nullable=True)
    f1_score = Column(Float, nullable=True)
    
    # Status
    is_active = Column(Boolean, default=False, index=True)
    
    # Timestamps
    training_date = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    def __repr__(self):
        return f"<ModelMetadata(version={self.version}, is_active={self.is_active})>"

