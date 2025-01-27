## Install dependencies

```bash
cd backend
pip install -r requirements.txt
```

```bash
cd frontend
pip install -r requirements.txt
```

## Run the frontend

```bash
cd frontend
streamlit run app.py
```

Frontend is running on port 8501.

## Run the backend

```bash
cd backend
uvicorn main:app --reload
```

Backend is running on port 8000.

API documentation is available at http://localhost:8000/docs.

## Run the Weaviate

```bash
docker-compose up -d
```

Weaviate is running on port 8080.

## Environment Configuration

### Required Environment Variables (.env)

Create a `.env` file in the root directory with the following configurations:

```bash
# .env

# OpenAI Configuration
OPENAI_API_KEY="sk-..."  # Your OpenAI API key

# Weaviate Configuration
WEAVIATE_URL="http://localhost:8080"  # Weaviate instance URL

# Authentication
SECRET_KEY="your-secret-key-here" 
```

## Check the database

To see a visual representation of the database, go to http://localhost:8080/v1/objects?class=Document

## Healthcheck

To check the health of the system, go to http://localhost:8000/healthcheck

## Brief technical description

The system is built using FastAPI for the backend, Streamlit for the frontend, and Weaviate for the vector database. The backend handles authentication, document processing, and API endpoints. The frontend provides a user interface for uploading documents, querying the database, and viewing results. The vector database is used to store and retrieve document embeddings for efficient similarity search.

The system uses the following technologies:

- FastAPI: A modern, fast (high-performance), web framework for building APIs with Python 3.10+
- Streamlit: An open-source app framework for building custom web apps in pure Python
- Weaviate: A cloud-native, open-source vector database for storing and querying embeddings
- LangChain: A framework for building applications with AI
- OpenAI: A cloud-based AI service for generating embeddings

### Embedding and Language Model Strategy

### Model Selection
I use two different OpenAI models for distinct purposes:

1. **Embeddings: text-embedding-ada-002**
```python
embeddings = OpenAIEmbeddings()  # Uses Ada by default
```
Selected for:
- Optimized for vector similarity search
- Cost-effective ($0.0001/1K tokens)
- Fast processing speed
- High-quality semantic embeddings
- 1536-dimensional vectors

2. **LLM: GPT-3.5-turbo**
```python
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
```
Selected for:
- Strong reasoning capabilities
- Good cost-performance ratio ($0.0015/1K tokens)
- Reliable context understanding
- Temperature of 0.7 balances creativity and accuracy

### Chunking Strategy

Our text chunking implementation:
```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,    # Characters per chunk
    chunk_overlap=200,  # Overlap between chunks
    length_function=len # Simple character count
)
```

Key design decisions:

1. **Chunk Size (1000 characters)**
   - Balances context preservation and token limits
   - Approximately 250-300 tokens per chunk
   - Allows multiple chunks in context window
   - Maintains readable semantic units

2. **Overlap (200 characters)**
   - 20% overlap between chunks
   - Prevents context loss at boundaries
   - Helps maintain coherent sentences
   - Improves retrieval accuracy

3. **RecursiveCharacterTextSplitter Benefits**
   - Splits on paragraph boundaries first
   - Falls back to sentence boundaries
   - Last resort: character-level splits
   - Preserves semantic meaning
   - Handles various document formats

### Technical challenges

One of my biggest challenges was resolving version compatibility issues between LangChain, Weaviate client, and the vector store implementation.

##### Initial Problem
```python
# This caused version conflicts
from langchain.vectorstores import Weaviate  # Older version
import weaviate  # Client v3.x

# Error encountered:
# AttributeError: module 'weaviate' has no attribute 'Client'
```

#### Solution
Specific versions of Weaviate client was required to resolve the version compatibility issues.




