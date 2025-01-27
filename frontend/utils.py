"""
Utility functions for the RAG application frontend.

This module provides functions for interacting with the backend API,
handling authentication, document management, and query processing.

The module uses Streamlit for UI feedback and requests for API communication.

Constants:
    API_URL (str): Base URL for the backend API

Functions:
    login: Authenticate user and get JWT token
    upload_files: Upload documents to the vector store
    query_documents: Process queries using RAG pipeline
    delete_all_documents: Remove all documents from the store

Usage:
    Import specific functions:
    >>> from frontend.utils import login, query_documents

    Or import everything:
    >>> from frontend.utils import *

Dependencies:
    - streamlit
    - requests
    - typing

Authentication:
    Most functions require a JWT token obtained via the login function.
    Tokens are typically stored in Streamlit's session state.

Error Handling:
    All functions handle their own exceptions and display errors via
    Streamlit's error messaging system.
"""
import streamlit as st
import requests
from typing import List, Optional

API_URL = "http://localhost:8000"

def login(email: str, password: str) -> Optional[str]:
    """
    Authenticate user with email and password.
    
    Args:
        email: User's email address
        password: User's password
        
    Returns:
        str: JWT access token if authentication successful
        None: If authentication fails or server error occurs
        
    Raises:
        No exceptions are raised; all errors are handled internally
        and displayed via Streamlit's error messages
        
    Side Effects:
        - Displays error message via st.error() if authentication fails
        - Displays error message if server connection fails
        
    Example:
        >>> token = login("user@example.com", "password123")
        >>> if token:
        >>>     st.session_state.token = token
        >>>     st.session_state.logged_in = True
    """
    try:
        response = requests.post(
            f"{API_URL}/token",
            data={"username": email, "password": password}
        )
        if response.status_code == 200:
            return response.json()["access_token"]
        else:
            st.error("Invalid email or password")
            return None
    except Exception as e:
        st.error(f"Login failed: {str(e)}")
        return None

def upload_files(files: List[str], token: str) -> dict:
    """
    Upload multiple files to the backend server.
    
    Args:
        files: List of file objects from Streamlit's file_uploader
        token: JWT authentication token
        
    Returns:
        dict: Response from the server containing upload status
            Example: {"status": "success", "message": "2 files uploaded"}
        None: If no files were provided
        
    Raises:
        Exception: If upload fails or server returns an error
        
    Supported file types:
        - PDF (.pdf)
        - Markdown (.md)
        
    Example:
        >>> uploaded_files = st.file_uploader("Upload files", accept_multiple_files=True)
        >>> if uploaded_files:
        >>>     result = upload_files(uploaded_files, st.session_state.token)
        >>>     if result:
        >>>         st.success(result["message"])
    """
    if not files:
        return None
    
    files_to_upload = []
    for file in files:
        content_type = 'application/pdf' if file.name.endswith('.pdf') else 'text/markdown'
        files_to_upload.append(
            ('files', (file.name, file, content_type))
        )
    
    try:
        response = requests.post(f"{API_URL}/upload", files=files_to_upload, headers={"Authorization": f"Bearer {token}"})
        response.raise_for_status() 
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error uploading files: {str(e)}")
        return None
    
# def get_all_documents():
#     try:
#         response = requests.get(
#             f"{API_URL}/documents",
#             headers={"Authorization": f"Bearer {st.session_state.token}"}
#         )
#         if response.status_code == 200:
#             documents = response.json()
#             st.write("All Documents:")
#             for doc in documents["documents"]:
#                 st.write(doc["content"])
#         else:
#             st.error("Failed to fetch documents")
#     except Exception as e:
#         st.error(f"Error: {str(e)}")

def query_documents(query: str, token: str):
    """
    Send query to backend RAG pipeline and get response with context.
    
    Args:
        query: User's question to be answered using stored documents
        token: JWT authentication token for API access
        
    Returns:
        dict: Response containing answer and context if successful
            Format:
            {
                "response": str,  # Generated answer
                "context": List[Dict],  # Retrieved documents with relevance scores
                    [
                        {
                            "content": str,
                            "relevance_score": float
                        },
                        ...
                    ]
            }
        None: If any error occurs
        
    Raises:
        No exceptions are raised; all errors are handled internally
        and displayed via Streamlit's error messages:
        - requests.RequestException: For API communication errors
        - KeyError: For unexpected response format
        - Exception: For any other unexpected errors
        
    Side Effects:
        Displays specific error messages via st.error() for:
        - API communication issues
        - Response format issues
        - Unexpected errors
        
    Example:
        >>> result = query_documents("What is RAG?", "jwt_token_here")
        >>> if result:
        >>>     st.write("Answer:", result["response"])
        >>>     st.write("Sources:", result["context"])
    """
    try:
        response = requests.post(
            f"{API_URL}/query",
            json={"query": query},  # Make sure to use the correct JSON structure
            headers={"Authorization": f"Bearer {token}",
                     "Content-Type": "application/json"}
        )

        response.raise_for_status()
        result = response.json()
        return result;
        
        
    except requests.RequestException as e:
        st.error(f"API Error: {str(e)}")
    except KeyError as e:
        st.error(f"Unexpected response format: {str(e)}")
    except Exception as e:
        st.error(f"Error: {str(e)}")

def delete_all_documents(token: str):
    """
    Delete all documents from the vector store database.
    
    Args:
        token: JWT authentication token for API access
        
    Returns:
        None
        
    Raises:
        No exceptions are raised; all errors are handled internally
        and displayed via Streamlit's error messages
        
    Side Effects:
        - Displays success message via st.success() when documents are deleted
        - Displays error message via st.error() if deletion fails
        - Permanently removes all documents from the vector store
        
    Warning:
        This is a destructive operation that cannot be undone.
        All documents in the database will be permanently deleted.
        
    Example:
        >>> if st.button("Delete All Documents"):
        >>>    delete_all_documents(st.session_state.token)
    """
    try:
        response = requests.delete(
            f"{API_URL}/documents",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            }
        )
        response.raise_for_status()
        st.success("All documents deleted successfully")
    except Exception as e:
        st.error(f"Error deleting documents: {str(e)}")