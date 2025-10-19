from fastapi import FastAPI
from datetime import datetime, timezone
from pydantic import BaseModel

from app.rag_pipeline import load_rag_pipeline
from app.config import OLLAMA_MODEL

app = FastAPI(title="RAG Chatbot API - TinyLlama + FAISS + LoRA")

qa_pipeline = None
MODEL_NAME_DISPLAY = OLLAMA_MODEL
try:
    qa_pipeline, MODEL_NAME_DISPLAY = load_rag_pipeline()
    print(f"✅ RAG pipeline loaded successfully at startup using model: {MODEL_NAME_DISPLAY}")
except Exception as e:
    print("⚠️ Could not load RAG pipeline at startup:", e)
    print("You can still run the server and load it after indexing.")

@app.get("/", status_code=200)
def home():
    return {
        "message": "✅ RAG Chatbot API is running!",
        "endpoints": ["/ask"],
        "docs": "Visit /docs for the API interface",
    }

class QuestionRequest(BaseModel):
    question: str

@app.post("/ask")
async def ask(payload: QuestionRequest):
    global qa_pipeline, MODEL_NAME_DISPLAY 
    question = payload.question.strip()
    if not question:
        return {"error": "No question provided"}

    if qa_pipeline is None:
        try:
            qa_pipeline, MODEL_NAME_DISPLAY = load_rag_pipeline()
            print(f"✅ Pipeline reloaded dynamically using model: {MODEL_NAME_DISPLAY}")
        except Exception as e:
            MODEL_NAME_DISPLAY = OLLAMA_MODEL
            return {
                "error": "RAG pipeline not available. Build index and ensure model is running.",
                "detail": str(e),
            }
    try:
        result = qa_pipeline({"question": question})
        context_sources = []
        if isinstance(result, dict):
            answer = result.get("answer", "")
            context_sources = result.get("context_sources", [])
        else:
            answer = result
            context_sources = []
    except Exception as e:
        return {"error": "Failed to generate answer", "detail": str(e)}

    response = {
        "question": question,
        "answer": answer,
        "context_sources": context_sources,
        "metadata": {
            "model": MODEL_NAME_DISPLAY,
            "retrieval_engine": "FAISS",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    }

    return response