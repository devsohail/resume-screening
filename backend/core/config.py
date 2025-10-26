"""
Centralized configuration management using Pydantic Settings
Loads configuration from environment variables and .env file
"""

from typing import List, Optional
from functools import lru_cache
from pydantic import Field, validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with validation"""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # AWS Configuration
    aws_region: str = Field(default="us-east-1", description="AWS Region")
    aws_access_key_id: Optional[str] = Field(default=None, description="AWS Access Key")
    aws_secret_access_key: Optional[str] = Field(default=None, description="AWS Secret Key")
    
    # S3 Buckets
    s3_bucket_resumes: str = Field(default="resume-screening-resumes", description="S3 bucket for resumes")
    s3_bucket_models: str = Field(default="resume-screening-models", description="S3 bucket for ML models")
    s3_bucket_data: str = Field(default="resume-screening-data", description="S3 bucket for training data")
    
    # AWS Bedrock
    bedrock_model_id: str = Field(default="amazon.titan-embed-text-v1", description="Bedrock embedding model")
    bedrock_region: str = Field(default="us-east-1", description="Bedrock region")
    bedrock_max_retries: int = Field(default=3, description="Max retries for Bedrock API")
    
    # Vector Database - Pinecone
    pinecone_api_key: Optional[str] = Field(default=None, description="Pinecone API key")
    pinecone_environment: str = Field(default="us-east-1-aws", description="Pinecone environment")
    pinecone_index_name: str = Field(default="resume-embeddings", description="Pinecone index name")
    pinecone_dimension: int = Field(default=1536, description="Embedding dimension for Titan")
    pinecone_metric: str = Field(default="cosine", description="Distance metric")
    
    # Database
    database_url: str = Field(
        default="postgresql://postgres:postgres@localhost:5432/resume_screening",
        description="PostgreSQL connection URL"
    )
    db_host: str = Field(default="localhost", description="Database host")
    db_port: int = Field(default=5432, description="Database port")
    db_name: str = Field(default="resume_screening", description="Database name")
    db_user: str = Field(default="postgres", description="Database user")
    db_password: str = Field(default="postgres", description="Database password")
    db_pool_size: int = Field(default=10, description="Database connection pool size")
    db_max_overflow: int = Field(default=20, description="Max overflow connections")
    
    # API Configuration
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8000, description="API port")
    api_version: str = Field(default="v1", description="API version")
    api_title: str = Field(default="Resume Screening API", description="API title")
    api_debug: bool = Field(default=False, description="Debug mode")
    api_reload: bool = Field(default=False, description="Auto-reload on code changes")
    
    # Security
    secret_key: str = Field(default="change-me-in-production", description="Secret key for encryption")
    jwt_secret: str = Field(default="change-me-in-production", description="JWT secret key")
    jwt_algorithm: str = Field(default="HS256", description="JWT algorithm")
    jwt_expiration_minutes: int = Field(default=60, description="JWT token expiration")
    
    # CORS
    cors_origins: str = Field(
        default="http://localhost:3000,http://localhost:8000",
        description="Allowed CORS origins (comma-separated)"
    )
    
    @property
    def cors_origins_list(self) -> List[str]:
        """Parse CORS origins from string to list"""
        if isinstance(self.cors_origins, str):
            return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]
        return self.cors_origins if isinstance(self.cors_origins, list) else []
    
    # MLflow
    mlflow_tracking_uri: str = Field(default="http://localhost:5000", description="MLflow tracking server")
    mlflow_experiment_name: str = Field(default="resume-screening", description="MLflow experiment name")
    mlflow_s3_endpoint_url: Optional[str] = Field(default=None, description="S3 endpoint for MLflow artifacts")
    
    # Model Configuration
    similarity_threshold: float = Field(default=70.0, description="Minimum similarity score (0-100)")
    classifier_threshold: float = Field(default=0.7, description="Classifier probability threshold")
    hybrid_similarity_weight: float = Field(default=0.4, description="Weight for similarity in hybrid model")
    hybrid_classifier_weight: float = Field(default=0.6, description="Weight for classifier in hybrid model")
    
    @validator("hybrid_similarity_weight", "hybrid_classifier_weight")
    def validate_weights(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Weights must be between 0 and 1")
        return v
    
    # Lambda Configuration
    lambda_function_name: str = Field(default="resume-screening-inference", description="Lambda function name")
    lambda_memory_size: int = Field(default=1024, description="Lambda memory in MB")
    lambda_timeout: int = Field(default=30, description="Lambda timeout in seconds")
    
    # SageMaker
    sagemaker_role: Optional[str] = Field(default=None, description="SageMaker execution role ARN")
    sagemaker_instance_type: str = Field(default="ml.m5.xlarge", description="SageMaker instance type")
    sagemaker_endpoint_name: str = Field(default="resume-classifier-endpoint", description="SageMaker endpoint")
    sagemaker_training_instance_type: str = Field(default="ml.m5.xlarge", description="Training instance type")
    sagemaker_training_volume_size: int = Field(default=30, description="Training volume size in GB")
    
    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(default="json", description="Log format: json or text")
    
    # Environment
    environment: str = Field(default="development", description="Environment: development, staging, production")
    
    # Feature Flags
    enable_classifier: bool = Field(default=True, description="Enable binary classifier")
    enable_similarity: bool = Field(default=True, description="Enable similarity scoring")
    enable_monitoring: bool = Field(default=True, description="Enable Prometheus monitoring")
    enable_auto_retraining: bool = Field(default=False, description="Enable automatic model retraining")
    
    # Processing
    max_file_size_mb: int = Field(default=10, description="Max upload file size in MB")
    batch_size: int = Field(default=32, description="Batch size for inference")
    max_workers: int = Field(default=4, description="Max worker threads")
    
    @property
    def database_url_async(self) -> str:
        """Get async database URL for asyncpg"""
        return self.database_url.replace("postgresql://", "postgresql+asyncpg://")
    
    @property
    def is_production(self) -> bool:
        """Check if running in production"""
        return self.environment == "production"
    
    @property
    def is_development(self) -> bool:
        """Check if running in development"""
        return self.environment == "development"


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance
    Use lru_cache to avoid reading .env file multiple times
    """
    return Settings()


# Global settings instance
settings = get_settings()

