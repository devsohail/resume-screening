"""
Authentication and authorization middleware
JWT token validation and user context
"""

from typing import Optional
from fastapi import HTTPException, Security, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from datetime import datetime, timedelta

from backend.core.config import settings
from backend.core.exceptions import AuthenticationError, AuthorizationError

security = HTTPBearer(auto_error=False)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token"""
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.jwt_expiration_minutes)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.jwt_secret, algorithm=settings.jwt_algorithm)
    
    return encoded_jwt


def verify_token(token: str) -> dict:
    """Verify and decode JWT token"""
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
        return payload
    except JWTError as e:
        raise AuthenticationError(f"Invalid token: {str(e)}")


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security)
) -> Optional[dict]:
    """
    Get current authenticated user from JWT token
    Returns None if no token provided (for optional authentication)
    """
    if not credentials:
        return None
    
    token = credentials.credentials
    payload = verify_token(token)
    
    # Extract user info from payload
    user_id = payload.get("sub")
    if not user_id:
        raise AuthenticationError("Invalid token payload")
    
    return {
        "user_id": user_id,
        "email": payload.get("email"),
        "roles": payload.get("roles", [])
    }


async def get_current_user_required(
    current_user: Optional[dict] = Depends(get_current_user)
) -> dict:
    """Get current user (required - raises error if not authenticated)"""
    if not current_user:
        raise AuthenticationError("Authentication required")
    return current_user


def require_role(required_role: str):
    """Dependency to check if user has required role"""
    async def role_checker(current_user: dict = Depends(get_current_user_required)):
        user_roles = current_user.get("roles", [])
        if required_role not in user_roles and "admin" not in user_roles:
            raise AuthorizationError(f"Role '{required_role}' required")
        return current_user
    
    return role_checker

