from fastapi import FastAPI
from app.api.routes import router

# Application activation point

app = FastAPI(title="AI RAG Backend")

@app.get("/")
def home():
    return {"message": "AI Backend is running"}

# Linking the path
app.include_router(router)
