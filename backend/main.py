from datetime import datetime, timedelta
import io
from typing import List
from PyPDF2 import PdfReader
from fastapi import Depends, FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
from fastapi.security import OAuth2PasswordRequestForm
import uvicorn
from preprocessing import preprocess
from auth import *


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
            
            processed_files.append({
                "filename": file.filename,
                "processed_at": datetime.datetime.now().isoformat(),
                "metadata": processed['metadata'],
            })

        return {
            "status": "success",
            "message": f"Processed {len(processed_files)} files",
            "processed_files": processed_files
        }
    except Exception as e:
        logger.error(f"Error processing files: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

