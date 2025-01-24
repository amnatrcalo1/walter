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
    
def get_all_documents():
    try:
        response = requests.get(
            f"{API_URL}/documents",
            headers={"Authorization": f"Bearer {st.session_state.token}"}
        )
        if response.status_code == 200:
            documents = response.json()
            st.write("All Documents:")
            for doc in documents["documents"]:
                st.write(doc["content"])
        else:
            st.error("Failed to fetch documents")
    except Exception as e:
        st.error(f"Error: {str(e)}")

def query_documents(query: str):
    try:
        response = requests.post(
            f"{API_URL}/query",
            json={"query": query},  # Make sure to use the correct JSON structure
            headers={"Authorization": f"Bearer {st.session_state.token}"}
        )
        if response.status_code == 200:
            result = response.json()
            st.write("Response:", result["response"])
            with st.expander("Show context"):
                st.write(result["context"])
        else:
            st.error(f"Error: {response.status_code}")
    except Exception as e:
        st.error(f"Error: {str(e)}")