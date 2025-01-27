"""
Authentication Module for FastAPI Application

This module handles user authentication, JWT token management, and password hashing.
It provides a secure authentication layer using OAuth2 with Password Bearer tokens.

Features:
    - Password hashing with bcrypt
    - JWT token generation and validation
    - OAuth2 password flow
    - User authentication middleware
    
Security Configuration:
    - JWT signing with HS256 algorithm
    - 30-minute token expiration
    - Secure password hashing
    - Bearer token authentication
    
Environment Variables Required:
    - SECRET_KEY: Key for JWT signing
        Example: openssl rand -hex 32
        
Dependencies:
    - fastapi: Web framework components
    - passlib: Password hashing
    - python-jose: JWT handling
    - python-dotenv: Environment management
    
"""
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from passlib.context import CryptContext
from datetime import datetime, timedelta
from jose import JWTError, jwt
from typing import Optional
import os
from dotenv import load_dotenv

load_dotenv()

# Security configurations
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Mock user database - Replace with real database in production
fake_users_db = {
    "amna@example.com": {
        "email": "amna@example.com",
        "hashed_password": pwd_context.hash("amna123"),
    }
}

def verify_password(plain_password, hashed_password):
    """
    Verify a plain password against a hashed password.
    
    Args:
        plain_password: The password to verify
        hashed_password: The hashed password to check against
        
    Returns:
        bool: True if password matches, False otherwise
        
    Example:
        >>> hashed = pwd_context.hash("secret123")
        >>> verify_password("secret123", hashed)
        True
    """
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """
    Create a JWT access token.
    
    Args:
        data: Payload to encode in the token
        expires_delta: Optional expiration time override
        
    Returns:
        str: Encoded JWT token
        
    Example:
        >>> token = create_access_token(
        ...     data={"sub": "user@example.com"},
        ...     expires_delta=timedelta(minutes=30)
        ... )
    
    """
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


async def get_current_user(token: str = Depends(oauth2_scheme)):
    """
    Dependency for validating and extracting user from JWT token.
    
    Args:
        token: JWT token from Authorization header
        
    Returns:
        dict: User information if token is valid
        
    Raises:
        HTTPException (401): If token is invalid or user not found
        
    Security:
        - Validates token signature
        - Checks token expiration
        - Verifies user exists
        - Returns 401 for any validation failure
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid authentication credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = fake_users_db.get(email)
    if user is None:
        raise credentials_exception
    return user