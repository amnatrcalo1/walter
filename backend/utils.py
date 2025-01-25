import logging
from typing import Any, Dict, List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from vector_store import get_vectorstore
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA

logger = logging.getLogger(__name__)

def get_text_chunks(raw_text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000, # 1000 chars
        chunk_overlap = 200,
        length_function = len # len fn from python
    )

    chunks = text_splitter.split_text(raw_text)
    # print(chunks)
    return chunks


def retrieve_context_with_scoring(
    query: str, 
    top_k: int = 3
) -> List[Dict[str, Any]]:
    """
    Retrieve context documents with relevance scoring
    
    Args:
        query: Input query string
        top_k: Number of top documents to retrieve
    
    Returns:
        List of dictionaries with document and relevance score
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
            "relevance_score": 1 - score  # Convert distance to similarity score
        })
    
    # Sort by relevance score in descending order
    scored_contexts.sort(key=lambda x: x['relevance_score'], reverse=True)
    
    return scored_contexts

def process_query(query: str) -> Dict[str, Any]:
    """Enhanced RAG pipeline with context relevance scoring"""
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
            "sources": [doc.page_content for doc in result["source_documents"]]
        }

    except Exception as e:
        logger.error(f"Error in RAG pipeline: {str(e)}")
        raise