import streamlit as st
import requests
from typing import List, Optional
import json

from utils import get_all_documents, login, query_documents, upload_files


def main():
    st.title("Document Processing System")

    # Initialize token in session state if not already present
    if 'token' not in st.session_state:
        st.session_state.token = None

    # Login section
    if not st.session_state.token:
        st.header("Login")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        
        if st.button("Login"):
            token = login(email, password)
            if token:
                st.session_state.token = token
                st.success("Logged in successfully!")
                st.rerun()

    else:
        if st.sidebar.button("Logout"):
            st.session_state.token = None
            st.rerun()
    
        with st.sidebar:
            st.subheader("Your documents")
            uploaded_files = st.file_uploader(
                "Upload your PDFs or Markdown files",
                type=['pdf', 'md'],
                accept_multiple_files=True
            )
            
            if st.button("Process"):
                if not uploaded_files:
                    st.error("Please upload at least one file first!")
                else:
                    with st.spinner("Processing documents..."):
                        result = upload_files(uploaded_files, st.session_state.token)
                        
                        if result:
                            st.success(f"{len(result['processed_files'])} files are processed.")
                            
                            st.subheader("Details:")
                            for file in result['processed_files']:
                                st.write(f"📄 {file['filename']}")
                                st.write(f"Processed at: {file['processed_at']}")
                                st.write("---")
                            
                            st.session_state.last_process_result = result

        if st.button("Show All Documents"):
                get_all_documents()

        user_question = st.chat_input("How can I help you today?") 
        if user_question:
            with st.spinner("Generating response..."):
                response = query_documents(user_question)
                st.write(response)

    if 'last_process_result' in st.session_state:        
        st.header("Summary")
        for file in st.session_state.last_process_result['processed_files']:
            st.write(f"- {file['filename']}")

            st.write("\nText Analysis:")
            metadata = file['metadata']
            st.write(f"- Number of sentences: {metadata['num_sentences']}")
            st.write(f"- Number of words: {metadata['num_words']}")
            st.write(f"- Number of characters: {metadata['num_characters']}")

    

    


                    
    

if __name__ == "__main__":
    main()