from datetime import datetime, timedelta
import io
from typing import Dict, List
from PyPDF2 import PdfReader
from fastapi import Depends, FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
from fastapi.security import OAuth2PasswordRequestForm
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel
import uvicorn
from utils import get_text_chunks, process_query
from preprocessing import preprocess
from auth import *
from vector_store import create_vector_store, delete_all_documents
from fastapi.responses import JSONResponse

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
    # If the token is invalid, get_current_user will raise an HTTPException
    # If we get here, the user is authenticated
    
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
    """Query the vector store and get AI-generated response"""
    try:
        result = process_query(request.query)
        return {
            "status": "success",
            "response": result["response"],
            "context": result["context"]
        }
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.delete("/documents")
async def delete_documents(current_user: dict = Depends(get_current_user)):
    """Delete all documents from the vector store"""
    try:
        result = delete_all_documents()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


  
    
    


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

