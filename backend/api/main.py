"""
Main FastAPI application
Entry point for the Resume Screening API
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from backend.core.config import settings
from backend.core.exceptions import ResumeScreeningException
from backend.core.logging import setup_logging
from backend.api.middleware.logging import LoggingMiddleware
from backend.api.routes import health, screening, jobs, analytics, feedback

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events
    """
    # Startup
    logger.info(f"Starting {settings.api_title} v{settings.api_version}")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Debug mode: {settings.api_debug}")
    
    # Initialize database
    try:
        from backend.storage.db_handler import get_db_handler
        db = get_db_handler()
        db.create_tables()
        logger.info("Database tables created/verified")
    except Exception as e:
        logger.warning(f"Database initialization warning: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down application")


# Create FastAPI app
app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description="Intelligent Resume Screening System API",
    docs_url=f"/api/{settings.api_version}/docs",
    redoc_url=f"/api/{settings.api_version}/redoc",
    openapi_url=f"/api/{settings.api_version}/openapi.json",
    lifespan=lifespan
)


# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Logging Middleware
app.add_middleware(LoggingMiddleware)


# Exception Handlers
@app.exception_handler(ResumeScreeningException)
async def resume_screening_exception_handler(request: Request, exc: ResumeScreeningException):
    """Handle custom application exceptions"""
    logger.error(
        f"Application error: {exc.message}",
        extra={
            "error_type": exc.__class__.__name__,
            "status_code": exc.status_code,
            "details": exc.details,
            "path": request.url.path
        }
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.__class__.__name__,
            "message": exc.message,
            "details": exc.details
        }
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle Pydantic validation errors"""
    logger.warning(
        f"Validation error: {exc}",
        extra={
            "path": request.url.path,
            "errors": exc.errors()
        }
    )
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "ValidationError",
            "message": "Request validation failed",
            "details": exc.errors()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions"""
    logger.error(
        f"Unexpected error: {exc}",
        extra={"path": request.url.path},
        exc_info=True
    )
    
    # Don't expose internal errors in production
    message = str(exc) if settings.api_debug else "Internal server error"
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "InternalServerError",
            "message": message
        }
    )


# Include routers
app.include_router(health.router, prefix=f"/api/{settings.api_version}")
app.include_router(screening.router, prefix=f"/api/{settings.api_version}")
app.include_router(jobs.router, prefix=f"/api/{settings.api_version}")
app.include_router(analytics.router, prefix=f"/api/{settings.api_version}")
app.include_router(feedback.router, prefix=f"/api/{settings.api_version}")


# Root endpoint
@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint - redirect to docs"""
    return {
        "message": f"Welcome to {settings.api_title}",
        "version": settings.api_version,
        "docs": f"/api/{settings.api_version}/docs"
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "backend.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
        log_level=settings.log_level.lower()
    )
