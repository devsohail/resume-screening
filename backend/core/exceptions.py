"""
Custom exceptions for the Resume Screening System
"""

from typing import Any, Dict, Optional


class ResumeScreeningException(Exception):
    """Base exception for all custom exceptions"""
    
    def __init__(
        self,
        message: str,
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)


class ConfigurationError(ResumeScreeningException):
    """Raised when configuration is invalid or missing"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=500, details=details)


class ValidationError(ResumeScreeningException):
    """Raised when input validation fails"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=422, details=details)


class NotFoundError(ResumeScreeningException):
    """Raised when a resource is not found"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=404, details=details)


class AuthenticationError(ResumeScreeningException):
    """Raised when authentication fails"""
    
    def __init__(self, message: str = "Authentication failed", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=401, details=details)


class AuthorizationError(ResumeScreeningException):
    """Raised when authorization fails"""
    
    def __init__(self, message: str = "Insufficient permissions", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=403, details=details)


class EmbeddingError(ResumeScreeningException):
    """Raised when embedding generation fails"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=500, details=details)


class ModelError(ResumeScreeningException):
    """Raised when ML model operations fail"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=500, details=details)


class StorageError(ResumeScreeningException):
    """Raised when storage operations fail"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=500, details=details)


class DatabaseError(ResumeScreeningException):
    """Raised when database operations fail"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=500, details=details)


class VectorStoreError(ResumeScreeningException):
    """Raised when vector database operations fail"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=500, details=details)


class FileProcessingError(ResumeScreeningException):
    """Raised when file processing fails"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=422, details=details)


class RateLimitError(ResumeScreeningException):
    """Raised when rate limit is exceeded"""
    
    def __init__(self, message: str = "Rate limit exceeded", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=429, details=details)


class ExternalServiceError(ResumeScreeningException):
    """Raised when external service call fails"""
    
    def __init__(self, service: str, message: str, details: Optional[Dict[str, Any]] = None):
        details = details or {}
        details["service"] = service
        super().__init__(f"{service} error: {message}", status_code=502, details=details)

