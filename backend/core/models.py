"""
Pydantic models for request/response validation
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum
from pydantic import BaseModel, Field, EmailStr, validator


# Enums
class ScreeningStatus(str, Enum):
    """Status of screening process"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ScreeningDecision(str, Enum):
    """Final screening decision"""
    SHORTLIST = "shortlist"
    REJECT = "reject"
    REVIEW = "review"


class JobStatus(str, Enum):
    """Job posting status"""
    ACTIVE = "active"
    CLOSED = "closed"
    DRAFT = "draft"


# Request Models
class JobDescriptionCreate(BaseModel):
    """Request model for creating a job description"""
    title: str = Field(..., min_length=1, max_length=200, description="Job title")
    company: str = Field(..., min_length=1, max_length=200, description="Company name")
    description: str = Field(..., min_length=10, description="Full job description")
    required_skills: List[str] = Field(default_factory=list, description="Required skills")
    preferred_skills: List[str] = Field(default_factory=list, description="Preferred skills")
    min_experience_years: Optional[int] = Field(None, ge=0, description="Minimum years of experience")
    max_experience_years: Optional[int] = Field(None, ge=0, description="Maximum years of experience")
    education_requirements: Optional[str] = Field(None, description="Education requirements")
    location: Optional[str] = Field(None, description="Job location")
    salary_range: Optional[str] = Field(None, description="Salary range")
    
    @validator("required_skills", "preferred_skills")
    def lowercase_skills(cls, v):
        """Convert skills to lowercase for consistency"""
        return [skill.lower().strip() for skill in v if skill.strip()]
    
    class Config:
        json_schema_extra = {
            "example": {
                "title": "Senior Python Developer",
                "company": "TechCorp Inc.",
                "description": "We are looking for an experienced Python developer...",
                "required_skills": ["python", "fastapi", "aws", "postgresql"],
                "preferred_skills": ["machine learning", "docker"],
                "min_experience_years": 5,
                "max_experience_years": 10,
                "education_requirements": "Bachelor's in Computer Science or related field",
                "location": "Remote",
                "salary_range": "$120k - $160k"
            }
        }


class JobDescriptionUpdate(BaseModel):
    """Request model for updating a job description"""
    title: Optional[str] = Field(None, min_length=1, max_length=200)
    company: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = Field(None, min_length=10)
    required_skills: Optional[List[str]] = None
    preferred_skills: Optional[List[str]] = None
    min_experience_years: Optional[int] = Field(None, ge=0)
    max_experience_years: Optional[int] = Field(None, ge=0)
    education_requirements: Optional[str] = None
    location: Optional[str] = None
    salary_range: Optional[str] = None
    status: Optional[JobStatus] = None


class ResumeUpload(BaseModel):
    """Request model for resume upload"""
    candidate_name: Optional[str] = Field(None, max_length=200, description="Candidate name")
    candidate_email: Optional[EmailStr] = Field(None, description="Candidate email")
    job_id: str = Field(..., description="Job ID to screen against")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")


class ScreeningRequest(BaseModel):
    """Request model for screening a resume against a job"""
    resume_id: str = Field(..., description="Resume ID")
    job_id: str = Field(..., description="Job ID")
    use_classifier: bool = Field(default=True, description="Use binary classifier")
    use_similarity: bool = Field(default=True, description="Use similarity scoring")


class BatchScreeningRequest(BaseModel):
    """Request model for batch screening"""
    resume_ids: List[str] = Field(..., min_items=1, max_items=100, description="List of resume IDs")
    job_id: str = Field(..., description="Job ID")
    use_classifier: bool = Field(default=True, description="Use binary classifier")
    use_similarity: bool = Field(default=True, description="Use similarity scoring")


# Response Models
class ResumeInfo(BaseModel):
    """Resume information"""
    id: str = Field(..., description="Resume ID")
    candidate_name: Optional[str] = Field(None, description="Candidate name")
    candidate_email: Optional[EmailStr] = Field(None, description="Candidate email")
    file_path: str = Field(..., description="S3 file path")
    extracted_text: Optional[str] = Field(None, description="Extracted text from resume")
    skills: List[str] = Field(default_factory=list, description="Extracted skills")
    experience_years: Optional[float] = Field(None, description="Years of experience")
    education: Optional[str] = Field(None, description="Education background")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    
    class Config:
        from_attributes = True


class JobDescription(BaseModel):
    """Job description response model"""
    id: str = Field(..., description="Job ID")
    title: str = Field(..., description="Job title")
    company: str = Field(..., description="Company name")
    description: str = Field(..., description="Full job description")
    required_skills: List[str] = Field(default_factory=list, description="Required skills")
    preferred_skills: List[str] = Field(default_factory=list, description="Preferred skills")
    min_experience_years: Optional[int] = None
    max_experience_years: Optional[int] = None
    education_requirements: Optional[str] = None
    location: Optional[str] = None
    salary_range: Optional[str] = None
    status: JobStatus = Field(default=JobStatus.ACTIVE, description="Job status")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
    
    class Config:
        from_attributes = True


class SimilarityScore(BaseModel):
    """Similarity scoring result"""
    overall_score: float = Field(..., ge=0, le=100, description="Overall similarity score (0-100)")
    skills_match: float = Field(..., ge=0, le=100, description="Skills match percentage")
    experience_match: float = Field(..., ge=0, le=100, description="Experience match score")
    education_match: float = Field(..., ge=0, le=100, description="Education match score")
    semantic_similarity: float = Field(..., ge=0, le=1, description="Cosine similarity of embeddings")


class ClassifierResult(BaseModel):
    """Binary classifier result"""
    probability: float = Field(..., ge=0, le=1, description="Probability of shortlisting")
    prediction: ScreeningDecision = Field(..., description="Binary prediction")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence")


class ScreeningResult(BaseModel):
    """Complete screening result"""
    id: str = Field(..., description="Screening result ID")
    resume_id: str = Field(..., description="Resume ID")
    job_id: str = Field(..., description="Job ID")
    status: ScreeningStatus = Field(..., description="Screening status")
    decision: Optional[ScreeningDecision] = Field(None, description="Final decision")
    final_score: Optional[float] = Field(None, ge=0, le=100, description="Final combined score")
    
    # Component scores
    similarity_result: Optional[SimilarityScore] = Field(None, description="Similarity scoring result")
    classifier_result: Optional[ClassifierResult] = Field(None, description="Classifier result")
    
    # Explanation
    explanation: Optional[str] = Field(None, description="Explanation of the decision")
    matched_skills: List[str] = Field(default_factory=list, description="Matched skills")
    missing_skills: List[str] = Field(default_factory=list, description="Missing required skills")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Screening start time")
    completed_at: Optional[datetime] = Field(None, description="Screening completion time")
    
    # Metadata
    model_version: Optional[str] = Field(None, description="Model version used")
    processing_time_ms: Optional[int] = Field(None, description="Processing time in milliseconds")
    
    class Config:
        from_attributes = True


class ScreeningResultList(BaseModel):
    """List of screening results with pagination"""
    results: List[ScreeningResult] = Field(..., description="List of results")
    total: int = Field(..., description="Total number of results")
    page: int = Field(default=1, ge=1, description="Current page")
    page_size: int = Field(default=20, ge=1, le=100, description="Results per page")
    has_more: bool = Field(..., description="Whether there are more results")


class AnalyticsMetrics(BaseModel):
    """Analytics metrics for dashboard"""
    total_jobs: int = Field(..., description="Total number of jobs")
    total_resumes: int = Field(..., description="Total number of resumes")
    total_screenings: int = Field(..., description="Total screenings performed")
    shortlist_rate: float = Field(..., ge=0, le=100, description="Percentage of shortlisted candidates")
    average_score: float = Field(..., ge=0, le=100, description="Average screening score")
    average_processing_time_ms: float = Field(..., description="Average processing time")
    
    # Time series data
    screenings_by_day: Dict[str, int] = Field(default_factory=dict, description="Screenings per day")
    decisions_distribution: Dict[str, int] = Field(default_factory=dict, description="Distribution of decisions")
    
    # Model performance
    model_accuracy: Optional[float] = Field(None, description="Model accuracy (if available)")
    model_version: Optional[str] = Field(None, description="Current model version")


class ModelInfo(BaseModel):
    """ML model information"""
    model_id: str = Field(..., description="Model ID")
    version: str = Field(..., description="Model version")
    model_type: str = Field(..., description="Model type (classifier/similarity)")
    accuracy: Optional[float] = Field(None, description="Model accuracy")
    precision: Optional[float] = Field(None, description="Model precision")
    recall: Optional[float] = Field(None, description="Model recall")
    f1_score: Optional[float] = Field(None, description="Model F1 score")
    training_date: datetime = Field(..., description="Training date")
    is_active: bool = Field(default=False, description="Whether this is the active model")
    s3_path: str = Field(..., description="S3 path to model artifacts")


class HealthCheck(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Current timestamp")
    services: Dict[str, str] = Field(default_factory=dict, description="Status of dependent services")


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request ID for tracking")

