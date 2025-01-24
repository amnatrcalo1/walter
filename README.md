### Install dependencies

```bash
cd backend
pip install -r requirements.txt
```

```bash
cd frontend
pip install -r requirements.txt
```

### Run the frontend

```bash
cd frontend
streamlit run app.py
```

Frontend is running on port 8501.

### Run the backend

```bash
cd backend
uvicorn main:app --reload
```

Backend is running on port 8000.

API documentation is available at http://localhost:8000/docs.

### Run the Weaviate

```bash
docker-compose up -d
```

Weaviate is running on port 8080.

