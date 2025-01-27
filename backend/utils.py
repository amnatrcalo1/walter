import logging
from typing import Any, Dict, List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from vector_store import get_vectorstore
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA

logger = logging.getLogger(__name__)

def get_text_chunks(raw_text):
    """
    Split raw text into overlapping chunks for better processing and retrieval.
    
    Args:
        raw_text: The input text to be split into chunks
        
    Returns:
        List[str]: List of text chunks with overlap
        
    Configuration:
        - chunk_size: 1000 characters per chunk
        - chunk_overlap: 200 characters overlap between chunks
        - length_function: Python's len() function for measuring text length
        
    Details:
        Uses LangChain's RecursiveCharacterTextSplitter which:
        - Splits text at sentence/paragraph boundaries when possible
        - Maintains context through chunk overlap
        - Prevents splitting in the middle of important information
        
    Example:
        >>> text = "Long document content here..."
        >>> chunks = get_text_chunks(text)
        >>> print(f"Split into {len(chunks)} chunks")
        
    Note:
        Chunk size and overlap are optimized for:
        - Token limits of LLM models
        - Context preservation
        - Efficient vector storage
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000, # 1000 chars
        chunk_overlap = 200,
        length_function = len
    )

    chunks = text_splitter.split_text(raw_text)
    # print(chunks)
    return chunks


def retrieve_context_with_scoring(
    query: str, 
    top_k: int = 3
) -> List[Dict[str, Any]]:
    """
    Retrieve and score most relevant documents for a given query using vector similarity.
    
    Args:
        query: User's question or search query
        top_k: Number of most relevant documents to retrieve (default: 3)
    
    Returns:
        List[Dict[str, Any]]: List of relevant documents with their scores
            Format:
            [
                {
                    "content": str,  # The document text
                    "relevance_score": float  # Score between 0 and 1
                },
                ...
            ]
            
    Details:
        - Uses vector similarity search to find relevant documents
        - Returns similarity scores (higher is better)
        - Sorts results by relevance (highest score first)
        - Filters out metadata and raw document objects for clean response
        
    Scoring:
        - Scores represent similarity
        - 1.0 = perfect match (most similar)
        - 0.0 = completely different
        - Higher scores indicate higher relevance
        
    Example:
        >>> results = retrieve_context_with_scoring("What is RAG?", top_k=2)
        >>> for doc in results:
        >>>     print(f"Similarity: {doc['relevance_score']:.2f}")
        >>>     print(f"Content: {doc['content'][:100]}...")
    """
    
    # Perform similarity search with scores
    vectorstore = get_vectorstore()
    results = vectorstore.similarity_search_with_score(query, k=top_k)
    
    # Format results with relevance scoring
    scored_contexts = []
    for doc, score in results:
        scored_contexts.append({
           # "document": doc,
            "content": doc.page_content,
            #"metadata": doc.metadata,
            "relevance_score": score 
        })
    
    # Sort by relevance score in descending order
    scored_contexts.sort(key=lambda x: x['relevance_score'], reverse=True)
    
    return scored_contexts

def process_query(query: str) -> Dict[str, Any]:
    """
    Process a user query using the RAG (Retrieval-Augmented Generation) pipeline.
    
    Args:
        query: User's question or query string
        
    Returns:
        Dict[str, Any]: Response containing generated answer and context
            Format:
            {
                "response": str,  # LLM-generated answer
                "context": List[Dict],  # Retrieved documents with scores
                    [
                        {
                            "content": str,
                            "relevance_score": float
                        },
                        ...
                    ]
            }
            
    Raises:
        Exception: If any part of the pipeline fails
            - Vector store retrieval errors
            - LLM generation errors
            - Context processing errors
            
    Pipeline Steps:
        1. Initialize LLM and vector store
        2. Retrieve relevant context with similarity scores
        3. Format context with relevance information
        4. Generate response using custom RAG chain
        
    Components:
        - LLM: GPT-3.5-turbo with temperature 0.7
        - Retriever: Vector store with top-3 document retrieval
        - Prompt: Custom template with context and relevance scores
        - Chain Type: "stuff" (concatenates all context)
        
    Example:
        >>> result = process_query("What is machine learning?")
        >>> print(result["response"])  # Generated answer
        >>> for ctx in result["context"]:
        >>>     print(f"Source (score: {ctx['relevance_score']:.2f}):")
        >>>     print(ctx["content"])
        
    Note:
        The temperature of 0.7 allows for some creativity in responses
        while maintaining factual accuracy based on the context.
    """
    try:
        # Setup LangChain components
        vectorstore = get_vectorstore()
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
        
        # Retrieve context with relevance scoring
        context_results = retrieve_context_with_scoring(query)
        
        # Setup custom prompt with context scoring
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful AI assistant. 
            Use the following context to answer the question. 
            Each document is labeled with its relevance score.
            If the context doesn't contain relevant information, say so.
            
            Context:
            {context}"""),
            ("human", "{question}")
        ])
        
        # Create RAG chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(
                search_kwargs={"k": 3}
            ),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )
        

        result = qa_chain.invoke(query) 
        
        return {
            "response": result["result"],
            "context": context_results,
        }

    except Exception as e:
        logger.error(f"Error in RAG pipeline: {str(e)}")
        raise