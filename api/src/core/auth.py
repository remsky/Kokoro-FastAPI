"""Authentication utilities for the API"""

from fastapi import Depends, HTTPException, Header, status
from fastapi.security import APIKeyHeader
from typing import Optional

from .config import settings

# Define API key header
api_key_header = APIKeyHeader(name="Authorization", auto_error=False)


async def verify_api_key(
    authorization: Optional[str] = Depends(api_key_header),
) -> Optional[str]:
    """
    Verify the API key from the Authorization header.
    
    Args:
        authorization: The Authorization header value
        
    Returns:
        The API key if valid
        
    Raises:
        HTTPException: If authentication is enabled and the API key is invalid
    """
    # If authentication is disabled, allow all requests
    if not settings.enable_auth:
        return None
    
    # Check if Authorization header is present
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": "authentication_error",
                "message": "API key is required",
                "type": "unauthorized",
            },
        )
    
    # Extract the API key from the Authorization header
    # Support both "Bearer sk-xxx" and "sk-xxx" formats
    api_key = authorization
    if authorization.lower().startswith("bearer "):
        api_key = authorization[7:].strip()
    
    # Check if the API key is valid
    if not settings.api_keys or api_key not in settings.api_keys:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": "authentication_error",
                "message": "Invalid API key",
                "type": "unauthorized",
            },
        )
    
    return api_key 