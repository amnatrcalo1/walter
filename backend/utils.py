import logging
from typing import Any, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import ChatOpenAI
from vector_store import get_vectorstore, get_weaviate_client
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

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
    """RAG pipeline: retrieve -> generate"""
    client = None # new
    try:
        client = get_weaviate_client()
        collection = client.collections.get("Document")
        # vectorstore = get_vectorstore()
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

        # docs = vectorstore.similarity_search(
        #     query,
        #     k=3  # Get top 3 most relevant chunks
        # )

          # 2. Retrieve relevant documents using vector search
        results = collection.query.bm25(
            query=query,
            limit=3
        )

        if not results.objects:
            return {
                "response": "No relevant information found in the documents.",
                "context": []
            }

        docs = [obj.properties.get("content", "") for obj in results.objects]
        
        context = "\n\n".join(docs)
        
        # 4. Create prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful AI assistant. Use the following context to answer the question.
            If the context doesn't contain relevant information, say so.
            
            Context:
            {context}"""),
            ("human", "{question}")
        ])

        # 5. Generate response
        messages = prompt.format_messages(
            context=context,
            question=query
        )
        response = llm.invoke(messages)

        return {
            "response": response.content,
            "context": docs
        }

    except Exception as e:
        logger.error(f"Error in RAG pipeline: {str(e)}")
        raise
    finally:
        if client:
            client.close()

