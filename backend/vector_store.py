import os
from typing import List, Dict, Any
from weaviate.client import WeaviateClient # dodala weviate.clinet umejsto weaviate
from langchain_community.vectorstores import Weaviate
import weaviate
from weaviate.connect import ConnectionParams
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import logging

logger = logging.getLogger(__name__)

# Returns raw Weaviate database client
# Used for direct database operations (create/update/delete)
# Needed for low-level operations
# What we've been using for storing documents
def get_weaviate_client():
    """Get connected Weaviate client"""
    client = WeaviateClient(
        connection_params=ConnectionParams.from_url(
            url=os.getenv('WEAVIATE_URL'),
            grpc_port=50051
        )
    )
    client.connect()
    return client

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
    )

def get_all_documents() -> List[Dict[str, Any]]:
    """Retrieve all documents from the vector store"""
    client = get_weaviate_client()
    try:
        collection = client.collections.get("Document")
        # Simplified query syntax
        result = collection.query.fetch_objects(
            limit=100  # Adjust limit as needed
        )
        return [{"content": obj.properties.get("content")} for obj in result.objects]
    except Exception as e:
        logger.error(f"Error retrieving documents: {str(e)}")
        raise
    finally:
        client.close()

def delete_all_documents() -> Dict[str, str]:
    """Delete all documents from the vector store"""
    client = get_weaviate_client()
    try:
        client.collections.delete("Document")
            
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
        try:
            collection = client.collections.create(
                name="Document",
                properties=[
                    {
                        "name": "content",
                        "dataType": ["text"],
                    }
                ],
            )
        except Exception:
            # Collection might already exist
            collection = client.collections.get("Document")
        
        embeddings = OpenAIEmbeddings()
        
        # Batch import data
        with collection.batch.dynamic() as batch:
            for chunk in chunks:
                # Create embedding vector
                embedding = embeddings.embed_query(chunk)
                
                # Add object to batch
                batch.add_object(
                    properties={"content": chunk},
                    vector=embedding
                )
        
        logger.info(f"Successfully stored {len(chunks)} chunks in vector store")
        
    except Exception as e:
        logger.error(f"Error creating vector store: {str(e)}")
        raise
    finally:
        client.close()
