import os
from typing import List, Dict, Any
from langchain_community.vectorstores import Weaviate
import weaviate
from langchain_openai import OpenAIEmbeddings
import logging

logger = logging.getLogger(__name__)

def get_weaviate_client():
    """
    Create and return a configured Weaviate client instance.
    Used for direct database operations (create/update/delete)
    
    Returns:
        weaviate.Client: Configured Weaviate client connection
        
    Environment Variables Required:
        - WEAVIATE_URL: URL of the Weaviate instance (e.g., "http://localhost:8080")
        - OPENAI_API_KEY: OpenAI API key for embeddings
        
    Raises:
        weaviate.exceptions.WeaviateConnectionError: If cannot connect to Weaviate
        ValueError: If required environment variables are missing
        
    Example:
        >>> client = get_weaviate_client()
        >>> client.schema.get()  # Check connection
            
    Configuration:
        - No authentication (suitable for development)
        - Default timeout settings
    """
    if not os.getenv("WEAVIATE_URL"):
        raise ValueError("WEAVIATE_URL environment variable is not set")
        
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    return weaviate.Client(
        url=os.getenv("WEAVIATE_URL"),
    )

def get_vectorstore():
    """
    Create a LangChain-integrated Weaviate vector store instance.
    
    Returns:
        Weaviate: Configured vector store with the following setup:
            - Index: "Document" class
            - Text field: "content"
            - Embeddings: OpenAI embeddings
            - Attributes: ["content"]
            - Search: Vector-based (by_text=False)
            
    Components:
        - Client: Weaviate connection from get_weaviate_client()
        - Embeddings: OpenAI's text-embedding-ada-002 model
        - Index: "Document" class for storing text chunks
        
    Configuration:
        - text_key: Field containing the document text ("content")
        - by_text: False (use vector similarity instead of text search)
        - attributes: Only retrieve "content" field
        
    Example:
        >>> vectorstore = get_vectorstore()
        >>> results = vectorstore.similarity_search("query", k=3)
        >>> for doc in results:
        >>>     print(doc.page_content)
        
    Note:
        This setup is optimized for RAG (Retrieval-Augmented Generation):
        - Vector similarity search for semantic matching
        - OpenAI embeddings for high-quality vectors
        - Minimal attribute retrieval for efficiency
        
    Dependencies:
        - OpenAIEmbeddings from langchain_openai
        - Weaviate client configuration
        - "Document" class must exist in Weaviate
    """
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
    """
    Delete all documents by removing the entire Document class from Weaviate.
    
    Returns:
        Dict[str, str]: Status and message
            Format:
            {
                "status": "success",
                "message": "All documents deleted successfully"
            }
            
    Raises:
        Exception: If deletion fails, with error details logged
            - Connection errors
            - Schema deletion errors
            - Class not found errors
            
    Side Effects:
        - Permanently deletes the "Document" class
        - Removes all vectors and content
        - Logs deletion status
        - Automatically closes client connection
        
    Example:
        >>> try:
        >>>     result = delete_all_documents()
        >>>     print(result["message"])
        >>> except Exception as e:
        >>>     print(f"Deletion failed: {e}")
        
    Warning:
        This is a destructive operation that cannot be undone.
        The entire "Document" class will be removed from the schema.
        
    Note:
        The client connection is properly closed in the finally block,
        ensuring cleanup even if an error occurs.
    """
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
    """
    Create and populate a Weaviate vector store with text chunks and their embeddings.
    
    Args:
        chunks: List of text chunks to be stored and vectorized
        
    Returns:
        None
        
    Raises:
        Exception: If any step fails (logged with details)
            - Schema creation errors
            - Embedding generation errors
            - Batch import errors
            
    Process Steps:
        1. Schema Setup:
           - Creates "Document" class if not exists
           - Configures "content" property for text storage
           
        2. Embedding Generation:
           - Uses OpenAI's embedding model
           - Converts each text chunk to a vector
           
        3. Batch Import:
           - Processes chunks in batches of 100
           - Stores both text and vectors
           - Optimized for performance
           
    Example:
        >>> text_chunks = ["chunk1", "chunk2", "chunk3"]
        >>> create_vector_store(text_chunks)
        
    Performance:
        - Uses batch processing for efficient imports
        - Batch size of 100 for optimal throughput
        - Vectors are created and stored in parallel
        
    Schema:
        Document Class:
        {
            "class": "Document",
            "properties": [
                {
                    "name": "content",
                    "dataType": ["text"]
                }
            ]
        }
        
    Logging:
        - Info: Successful creation and import counts
        - Error: Detailed error messages
        - Info: Schema existence checks
        
    Note:
        The function is idempotent for schema creation:
        - Safe to call multiple times
        - Won't duplicate schema if exists
        - Will add new documents each time
    """
    client = get_weaviate_client()
    try:
        # Configure the schema for Document class if it doesn't exist
        class_obj = {
            "class": "Document",
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
