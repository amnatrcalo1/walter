import os
from typing import List, Dict, Any
from langchain_community.vectorstores import Weaviate
import weaviate
from langchain_openai import OpenAIEmbeddings
import logging

logger = logging.getLogger(__name__)

# Returns raw Weaviate database client
# Used for direct database operations (create/update/delete)
# Needed for low-level operations
# What we've been using for storing documents
def get_weaviate_client():
    """Get connected Weaviate client"""
    return weaviate.Client(
        url=os.getenv("WEAVIATE_URL"),
        additional_headers={
            "X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")
        }
    )

# def get_langchain_client():
#     """Get Weaviate client compatible with LangChain"""
#     auth_config = weaviate.auth.AuthApiKey(api_key=os.getenv("WEAVIATE_API_KEY", ""))
    
#     return weaviate.Client(
#         url=os.getenv("WEAVIATE_URL"),
#         auth_client_secret=auth_config,
#         additional_headers={
#             "X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")
#         }
#     )

# Returns LangChain's wrapper around Weaviate
# Integrates with other LangChain components
# Handles embeddings automatically
# Better for RAG operations (retrieval and querying)
def get_vectorstore():
    """Get Weaviate vectorstore with LangChain integration"""
    client = get_weaviate_client()
    embeddings = OpenAIEmbeddings()
    
    return Weaviate(
        client=client,
        index_name="Document",
        text_key="content",
        embedding=embeddings,
        attributes=["content"],
        by_text=False
    )

# def get_all_documents() -> List[Dict[str, Any]]:
#     """Retrieve all documents from the vector store"""
#     client = get_weaviate_client()
#     try:
#         # Query syntax for Weaviate v3
#         result = (
#             client.query
#             .get("Document", ["content"])
#             .with_limit(100)
#             .do()
#         )
        
#         # Extract documents from response
#         if result and "data" in result and "Get" in result["data"]:
#             documents = result["data"]["Get"]["Document"]
#             return [{"content": doc.get("content")} for doc in documents]
#         return []
        
#     except Exception as e:
#         logger.error(f"Error retrieving documents: {str(e)}")
#         raise
#     finally:
#         client.close()

def delete_all_documents() -> Dict[str, str]:
    """Delete all documents from the vector store"""
    client = get_weaviate_client()
    try:
        
        client.schema.delete_class("Document")
            
        logger.info("Successfully deleted all documents from the vector store")
        return {"status": "success", "message": "All documents deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting documents: {str(e)}")
        raise
    finally:
        client.close()

def create_vector_store(chunks: List[str]):
    """Create vector store in Weaviate."""
    client = get_weaviate_client()
    try:
        # Configure the schema for Document class if it doesn't exist
        class_obj = {
            "class": "Document",
            "vectorizer": "text2vec-openai",  # Using OpenAI's text vectorizer
            "moduleConfig": {
                "text2vec-openai": {
                    "model": "ada",
                    "modelVersion": "002",
                    "type": "text"
                }
            },
            "properties": [
                {
                    "name": "content",
                    "dataType": ["text"],
                }
            ]
        }

        # Create or get schema
        try:
            client.schema.create_class(class_obj)
            logger.info("Created Document class in Weaviate")
        except Exception as e:
            logger.info(f"Class might already exist: {str(e)}")

        # Create embeddings
        embeddings = OpenAIEmbeddings()
        
        # Batch import data
        with client.batch as batch:
            batch.batch_size = 100
            for chunk in chunks:
                # Create embedding vector
                embedding = embeddings.embed_query(chunk)
                
                # Properties of the object
                properties = {
                    "content": chunk
                }
                
                # Add object to batch
                batch.add_data_object(
                    data_object=properties,
                    class_name="Document",
                    vector=embedding
                )
        
        logger.info(f"Successfully stored {len(chunks)} chunks in vector store")
        
    except Exception as e:
        logger.error(f"Error creating vector store: {str(e)}")
        raise
