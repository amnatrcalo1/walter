"""
FastAPI Backend for RAG (Retrieval-Augmented Generation) Application

This module implements a REST API for document management and question-answering
using RAG architecture with vector storage and LLM integration.

Core Features:
    - User Authentication (JWT)
    - Document Upload and Processing
    - Vector Storage Management
    - RAG-based Question Answering
    - System Health Monitoring

API Endpoints:
    Authentication:
        POST /token
            - User login
            - Returns JWT token
            
    Document Management:
        POST /upload
            - Upload PDF/MD files
            - Process and store in vector DB
            
        DELETE /documents
            - Remove all documents
            - Requires authentication
            
    Query Processing:
        POST /query
            - Process natural language queries
            - Returns AI-generated responses
            
    System Status:
        GET /healthcheck
            - System component status
            - Resource utilization
            
Security:
    - JWT-based authentication
    - Password hashing
    - Token expiration
    - Protected endpoints
    
Dependencies:
    - FastAPI: Web framework
    - Weaviate: Vector database
    - OpenAI: Embeddings and LLM
    - PyPDF2: PDF processing
    - python-jose: JWT handling
    - passlib: Password hashing
    
Environment Variables:
    Required:
        - OPENAI_API_KEY: OpenAI API credentials
        - WEAVIATE_URL: Vector DB connection
        - SECRET_KEY: JWT signing key
        
    Optional:
        - LOG_LEVEL: Logging configuration
        - TOKEN_EXPIRE_MINUTES: JWT expiry
        
Error Handling:
    - Consistent error response format
    - Detailed error logging
    - Appropriate HTTP status codes
    
Logging:
    - Request processing
    - Error details
    - System operations
    - Performance metrics
    
Example Usage:
    ```bash
    # Start the server
    uvicorn main:app --reload
    
    # Environment setup
    export OPENAI_API_KEY="your-key"
    export WEAVIATE_URL="http://localhost:8080"
    export SECRET_KEY="your-secret-key"
    ```
    
Performance Considerations:
    - Batch processing for uploads
    - Efficient vector search
    - Connection pooling
    - Resource monitoring
    
Future Improvements:
    - Rate limiting
    - Caching layer
    - Batch query processing
    - Enhanced error handling
    - User management
    - Role-based access
"""
from datetime import datetime, timedelta
import io
from typing import List
from PyPDF2 import PdfReader
from fastapi import Depends, FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
from fastapi.security import OAuth2PasswordRequestForm
from langchain_openai import OpenAIEmbeddings
from pydantic import BaseModel
import uvicorn
from utils import get_text_chunks, process_query
from preprocessing import preprocess
from auth import *
from vector_store import create_vector_store, delete_all_documents, get_weaviate_client
import psutil
import platform

class QueryRequest(BaseModel):
    query: str

app = FastAPI()

origins = [
    "http://localhost:8000",
    "http://localhost:8501",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Authenticate user and generate JWT access token.
    
    Args:
        form_data: OAuth2 password request form containing:
            - username: User's email address
            - password: User's password
            
    Returns:
        Dict[str, str]: Token response
            Format:
            {
                "access_token": str,  # JWT token
                "token_type": "bearer"  # Always "bearer"
            }
            
    Raises:
        HTTPException (400): If credentials are invalid
            - Incorrect email
            - Incorrect password
            - User not found
            
    Authentication Flow:
        1. Validate user exists in database
        2. Verify password hash matches
        3. Generate JWT token with 30-minute expiry
        4. Return token with bearer type
        
    Security:
        - Passwords are hashed (not stored in plain text)
        - Tokens expire after 30 minutes
        - Uses standard OAuth2 password flow
        - Bearer token authentication
        
    Example Usage:
        ```
        POST /token
        Content-Type: application/x-www-form-urlencoded
        
        username=user@example.com&password=secretpass
        ```
        
    Response Example:
        ```json
        {
            "access_token": "eyJhbGciOiJIUzI1NiIs...",
            "token_type": "bearer"
        }
        ```
        
    Note:
        - Username field expects email address
        - Token includes user email in sub claim
        - Uses OAuth2 password flow for standard compatibility
        - 30-minute token expiry for security
    """
    user = fake_users_db.get(form_data.username)
    if not user or not verify_password(form_data.password, user["hashed_password"]):
        raise HTTPException(
            status_code=400,
            detail="Incorrect email or password"
        )
    
    access_token = create_access_token(
        data={"sub": user["email"]},
        expires_delta=timedelta(minutes=30)
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/upload")
async def upload_documents(files: List[UploadFile] = File(...), current_user: dict = Depends(get_current_user)): # This will automatically verify the token
    """
    Process and store documents in the vector database with authentication.
    
    Args:
        files: List of files to process
            Supported formats:
            - PDF (.pdf)
            - Markdown (.md)
        current_user: Authenticated user info (automatically injected)
    
    Returns:
        Dict[str, Any]: Upload status and processing details
            Format:
            {
                "status": "success",
                "message": str,  # e.g., "Processed 3 files"
                "processed_files": List[Dict]  # Processing details per file
                    [
                        {
                            "filename": str,
                            "processed_at": str,  # ISO format timestamp
                            "metadata": Dict  # Additional file metadata
                        },
                        ...
                    ]
            }
    
    Raises:
        HTTPException:
            - 400: Unsupported file type
            - 401: Authentication failed
            - 500: Processing or storage error
    
    Processing Steps:
        1. Authentication check
        2. File content extraction
           - PDF: Extract text from all pages
           - Markdown: Decode UTF-8 content
        3. Text preprocessing
        4. Chunk generation
        5. Vector storage
    
    Example:
        ```python
        # Using requests
        files = [
            ('files', ('doc1.pdf', open('doc1.pdf', 'rb'))),
            ('files', ('doc2.md', open('doc2.md', 'rb')))
        ]
        headers = {'Authorization': 'Bearer your-token'}
        response = requests.post('/upload', files=files, headers=headers)
        ```
    
    Security:
        - Requires valid JWT token
        - User authentication via get_current_user dependency
        - File type validation
    
    Performance:
        - Processes multiple files in one request
        - Combines text for efficient chunking
        - Batch vector storage
    
    Logging:
        - File count received
        - Chunk generation count
        - Vector storage status
        - Error details
    
    Note:
        - Large files may take longer to process
        - Combined text is chunked together
        - Existing vectors are preserved
        - Supports concurrent uploads
    """
    logger.info(f"Received {len(files)} files") 
    try:
        combined_text = ""
        processed_files = []

        for file in files:
            content = await file.read()
            
            if file.filename.endswith('.pdf'):
                pdf_reader = PdfReader(io.BytesIO(content))
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
                
            elif file.filename.endswith('.md'):
                text = content.decode('utf-8')
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.filename}")
            
            combined_text += text

            processed = preprocess(combined_text)

            # print(processed)
            
            processed_files.append({
                "filename": file.filename,
                "processed_at": datetime.now().isoformat(),
                "metadata": processed['metadata'],
            })

            # Get chunks after processing all files
        chunks = get_text_chunks(combined_text)
        logger.info(f"Created {len(chunks)} chunks from the documents")

        # Store chunks in vector database
        try:
            create_vector_store(chunks)
            logger.info("Successfully stored vectors in Weaviate")
        except Exception as e:
            logger.error(f"Error storing vectors: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to store vectors")

        return {
            "status": "success",
            "message": f"Processed {len(processed_files)} files",
            "processed_files": processed_files,
        }
    except Exception as e:
        logger.error(f"Error processing files: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))  



# @app.get("/documents")
# async def get_documents(current_user: dict = Depends(get_current_user)):
#     """Retrieve all documents from the vector store."""
#     try:
#         documents = get_all_documents()
#         return {"documents": documents}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
    

@app.post("/query")
async def query_documents(
    request: QueryRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Process a natural language query using RAG (Retrieval-Augmented Generation).
    
    Args:
        request: QueryRequest object containing:
            - query: str, The user's question
        current_user: Authenticated user info (automatically injected)
        
    Returns:
        Dict[str, Any]: Query results with context
            Format:
            {
                "status": "success",
                "response": str,  # AI-generated answer
                "context": List[Dict],  # Retrieved relevant documents
                    [
                        {
                            "content": str,  # Document content
                            "relevance_score": float  # Similarity score
                        },
                        ...
                    ]
            }
            
    Raises:
        HTTPException:
            - 401: Authentication failed
            - 500: Query processing error
            
    Processing Pipeline:
        1. Authentication verification
        2. Vector similarity search
        3. Context retrieval
        4. LLM response generation
        
    Example:
        ```python
        # Using requests
        headers = {
            'Authorization': 'Bearer your-token',
            'Content-Type': 'application/json'
        }
        data = {
            "query": "What is RAG?"
        }
        response = requests.post('/query', json=data, headers=headers)
        ```
        
    Security:
        - Requires valid JWT token
        - User authentication via get_current_user
        
    Performance:
        - Retrieves top-k most relevant documents
        - Uses efficient vector similarity search
        - Optimized context window for LLM
        
    Logging:
        - Query processing errors
        - Exception details for debugging
        
    Note:
        - Response quality depends on:
            - Document content in vector store
            - Query relevance to stored content
            - LLM generation parameters
        - Context helps verify response accuracy
    """
    try:
        result = process_query(request.query)
        return {
            "status": "success",
            "response": result["response"],
            "context": result["context"],
        }
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.delete("/documents")
async def delete_documents(current_user: dict = Depends(get_current_user)):
    """
    Delete all documents from the vector store database.
    
    Args:
        current_user: Authenticated user info (automatically injected)
        
    Returns:
        Dict[str, str]: Deletion status
            Format:
            {
                "status": "success",
                "message": "All documents deleted successfully"
            }
            
    Raises:
        HTTPException:
            - 401: Authentication failed
            - 500: Deletion operation failed
            
    Security:
        - Requires valid JWT token
        - User authentication via get_current_user
        - Consider adding additional authorization checks
            for this destructive operation
            
    Warning:
        This is a destructive operation that:
        - Permanently deletes ALL documents
        - Cannot be undone
        - Affects all users of the system
        - Removes all vectors and content
        
    Example:
        ```python
        # Using requests
        headers = {
            'Authorization': 'Bearer your-token'
        }
        response = requests.delete('/documents', headers=headers)
        ```
        
    Side Effects:
        - Removes all documents from vector store
        - Clears all embeddings
        - Resets document count to zero
        - Logs deletion operation
        
    Best Practices:
        - Implement confirmation mechanism
        - Add role-based access control
        - Create backup before deletion
        - Log user who performed deletion
        
    Note:
        Consider implementing:
        - Selective deletion
        - Soft delete option
        - Backup mechanism
        - Audit logging
    """
    try:
        delete_all_documents()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/healthcheck")
async def healthcheck():
    """
    Comprehensive system health check endpoint that monitors all critical components.
    
    Returns:
        Dict[str, Any]: Health status of all system components
            Format:
            {
                "status": str,  # "ok", "degraded", or "error"
                "timestamp": str,  # ISO format UTC timestamp
                "components": {
                    "weaviate": {
                        "status": str,  # "ok" or "error"
                        "version": str,
                        "document_count": int,
                        "error": str,  # Only if status is "error"
                    },
                    "openai": {
                        "status": str,  # "ok" or "error"
                        "model": str,
                        "error": str,  # Only if status is "error"
                    }
                },
                "system": {
                    "cpu_usage": str,
                    "memory_usage": str,
                    "disk_usage": str,
                    "python_version": str,
                    "platform": str
                },
                "environment": {
                    "status": str,  # "ok" or "error"
                    "missing_variables": List[str]
                }
            }
            
    Raises:
        HTTPException: 500 status code if overall health check fails
            
    Checks Performed:
        1. Weaviate Vector Store:
           - Connection status
           - Version information
           - Document count
           
        2. OpenAI API:
           - Connection status
           - Embedding model availability
           
        3. System Resources:
           - CPU usage percentage
           - Memory usage percentage
           - Disk usage percentage
           - Python version
           - Platform information
           
        4. Environment Variables:
           - WEAVIATE_URL
           - OPENAI_API_KEY
           
    Status Levels:
        - "ok": All components functioning normally
        - "degraded": Some components have issues
        - "error": Critical system failure
        
    Example Response:
        {
            "status": "ok",
            "timestamp": "2024-03-14T12:00:00Z",
            "components": {
                "weaviate": {
                    "status": "ok",
                    "version": "1.21.5",
                    "document_count": 42
                },
                "openai": {
                    "status": "ok",
                    "model": "text-embedding-ada-002"
                }
            },
            "system": {
                "cpu_usage": "45%",
                "memory_usage": "60%",
                "disk_usage": "75%",
                "python_version": "3.9.5",
                "platform": "Linux-5.4.0-x86_64"
            },
            "environment": {
                "status": "ok",
                "missing_variables": []
            }
        }
        
    Note:
        - Performs lightweight test operations
        - Document count limited to 100 for performance
        - Component failures mark system as "degraded"
        - Missing environment variables mark as "error"
    """
    health_status = {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "components": {}
    }
    
    try:
        # 1. Check Weaviate
        try:
            client = get_weaviate_client()
            weaviate_health = client.get_meta()
            
            # Get document count
            result = (
                client.query
                .get("Document", ["content"])
                .with_limit(100)
                .do()
            )
            doc_count = len(result.get("data", {}).get("Get", {}).get("Document", []))
            
            health_status["components"]["weaviate"] = {
                "status": "ok",
                "version": weaviate_health.get("version", "unknown"),
                "document_count": doc_count
            }
        except Exception as e:
            health_status["components"]["weaviate"] = {
                "status": "error",
                "error": str(e)
            }
            health_status["status"] = "degraded"

        # 2. Check OpenAI connection
        try:
            embeddings = OpenAIEmbeddings()
            _ = embeddings.embed_query("test")
            health_status["components"]["openai"] = {
                "status": "ok",
                "model": "text-embedding-ada-002"
            }
        except Exception as e:
            health_status["components"]["openai"] = {
                "status": "error",
                "error": str(e)
            }
            health_status["status"] = "degraded"

        # 3. System Information
        health_status["system"] = {
            "cpu_usage": f"{psutil.cpu_percent()}%",
            "memory_usage": f"{psutil.virtual_memory().percent}%",
            "disk_usage": f"{psutil.disk_usage('/').percent}%",
            "python_version": platform.python_version(),
            "platform": platform.platform()
        }

        # 4. Environment Check
        required_env_vars = ["WEAVIATE_URL", "OPENAI_API_KEY"]
        missing_vars = [var for var in required_env_vars if not os.getenv(var)]
        
        health_status["environment"] = {
            "status": "ok" if not missing_vars else "error",
            "missing_variables": missing_vars
        }

        return health_status

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e)
            }
        )


  
    
    


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

