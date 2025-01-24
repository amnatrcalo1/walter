import os
from typing import List, Dict, Any
from weaviate import WeaviateClient
from weaviate.connect import ConnectionParams
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import logging

logger = logging.getLogger(__name__)

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

def retrieve_relevant_context(query: str, top_k: int = 3) -> List[str]:
    """Retrieve most relevant document chunks."""
    client = get_weaviate_client()
    try:
        collection = client.collections.get("Document")
        embeddings = OpenAIEmbeddings()
        query_embedding = embeddings.embed_query(query)
        
        # Updated near_vector syntax
        results = collection.query.near_vector(
            near_vector=query_embedding,  # Changed from vector= to near_vector=
            limit=top_k,
            certainty=0.7  # Changed from distance= to certainty=
        )
        
        # Extract content from results
        contexts = [obj.properties.get("content", "") for obj in results.objects]
        logger.info(f"Retrieved {len(contexts)} relevant chunks for query: {query}")
        return contexts
        
    except Exception as e:
        logger.error(f"Error retrieving context: {str(e)}")
        raise
    finally:
        client.close()

def generate_response(query: str, context: List[str]) -> str:
    """Generate response using retrieved context."""
    try:
        llm = ChatOpenAI(model="gpt-3.5-turbo")
        
        prompt = f"""
        Based on the following context, please answer the query.
        If the context doesn't contain relevant information, say so.
        
        Context: {' '.join(context)}
        
        Query: {query}
        
        Response:"""
        
        response = llm.invoke(prompt).content
        logger.info(f"Generated response for query: {query}")
        return response
        
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        raise