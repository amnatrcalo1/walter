import streamlit as st
import requests
from typing import List, Optional
import json

# API endpoint
API_URL = "http://localhost:8000"

def login(email: str, password: str) -> Optional[str]:
    try:
        response = requests.post(
            f"{API_URL}/token",
            data={"username": email, "password": password}
        )
        if response.status_code == 200:
            return response.json()["access_token"]
        else:
            st.error("Invalid email or password")
            return None
    except Exception as e:
        st.error(f"Login failed: {str(e)}")
        return None

def upload_files(files: List[str], token: str) -> dict:
    if not files:
        return None
    
    files_to_upload = []
    for file in files:
        content_type = 'application/pdf' if file.name.endswith('.pdf') else 'text/markdown'
        files_to_upload.append(
            ('files', (file.name, file, content_type))
        )

        # st.write(files_to_upload)
    
    try:
        response = requests.post(f"{API_URL}/upload", files=files_to_upload, headers={"Authorization": f"Bearer {token}"})
        response.raise_for_status() 
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error uploading files: {str(e)}")
        return None

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
                        result = upload_files(uploaded_files)
                        
                        if result:
                            st.success(f"{len(result['processed_files'])} files are processed.")
                            
                            st.subheader("Details:")
                            for file in result['processed_files']:
                                st.write(f"📄 {file['filename']}")
                                st.write(f"Processed at: {file['processed_at']}")
                                st.write("---")
                            
                            st.session_state.last_process_result = result

    if 'last_process_result' in st.session_state:
        st.header("Summary")
        for file in st.session_state.last_process_result['processed_files']:
            st.write(f"- {file['filename']}")

            st.write("\nText Analysis:")
            metadata = file['metadata']
            st.write(f"- Number of sentences: {metadata['num_sentences']}")
            st.write(f"- Number of words: {metadata['num_words']}")


                    
    

if __name__ == "__main__":
    main()