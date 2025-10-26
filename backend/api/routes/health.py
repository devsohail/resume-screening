"""
Health check endpoints
"""

import logging
from datetime import datetime
from fastapi import APIRouter, status
from backend.core.models import HealthCheck
from backend.core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/health", tags=["Health"])


@router.get("/", response_model=HealthCheck, status_code=status.HTTP_200_OK)
async def health_check():
    """
    Health check endpoint
    Returns the current status of the service and its dependencies
    """
    services_status = {}
    
    # Check AWS connectivity (basic)
    try:
        import boto3
        boto3.client('sts', region_name=settings.aws_region).get_caller_identity()
        services_status["aws"] = "healthy"
    except Exception as e:
        logger.warning(f"AWS health check failed: {e}")
        services_status["aws"] = "unhealthy"
    
    # Overall status
    overall_status = "healthy" if all(s == "healthy" for s in services_status.values()) else "degraded"
    
    return HealthCheck(
        status=overall_status,
        version=settings.api_version,
        timestamp=datetime.utcnow(),
        services=services_status
    )


@router.get("/liveness", status_code=status.HTTP_200_OK)
async def liveness():
    """
    Kubernetes liveness probe
    Returns 200 if the service is alive
    """
    return {"status": "alive"}


@router.get("/readiness", status_code=status.HTTP_200_OK)
async def readiness():
    """
    Kubernetes readiness probe
    Returns 200 if the service is ready to accept traffic
    """
    # Add checks for critical dependencies
    return {"status": "ready"}

