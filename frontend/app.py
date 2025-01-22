import streamlit as st
import requests
from typing import List
import json

# API endpoint
API_URL = "http://localhost:8000"

def upload_files(files: List[str]) -> dict:
    if not files:
        return None
    
    files_to_upload = []
    for file in files:
        content_type = 'application/pdf' if file.name.endswith('.pdf') else 'text/markdown'
        files_to_upload.append(
            ('files', (file.name, file, content_type))
        )

        st.write(files_to_upload)
    
    try:
        response = requests.post(f"{API_URL}/upload", files=files_to_upload)
        response.raise_for_status() 
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error uploading files: {str(e)}")
        return None

def main():
    st.title("Document Processing System")
    
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


                    
    

if __name__ == "__main__":
    main()