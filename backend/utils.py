import logging
from typing import Any, Dict
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

def process_query(query: str) -> Dict[str, Any]:
    """RAG pipeline using LangChain components"""
    try:
        # 1. Setup LangChain components
        vectorstore = get_vectorstore()
        embeddings = OpenAIEmbeddings()
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
        query_vector = embeddings.embed_query(query)
        
        # 2. Create retriever
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3, "vector": query_vector}
        )
        
        # 3. Setup custom prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful AI assistant. Use the following context to answer the question.
            If the context doesn't contain relevant information, say so.
            
            Context:
            {context}"""),
            ("human", "{question}")
        ])
        
        # 4. Create RAG chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )
        
        # 5. Run the chain
        result = qa_chain.invoke(query)
        
        return {
            "response": result["result"],
            "context": [doc.page_content for doc in result["source_documents"]]
        }

    except Exception as e:
        logger.error(f"Error in RAG pipeline: {str(e)}")
        raise
