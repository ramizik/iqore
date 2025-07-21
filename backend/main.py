from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import uvicorn
from typing import Dict

app = FastAPI(title="iQore Chatbot Backend", version="1.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root() -> Dict[str, str]:
    """Root endpoint"""
    return {"message": "iQore Chatbot Backend is running"}

@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint for Google Cloud Run"""
    return {"status": "healthy", "service": "iqore-chatbot-backend"}

@app.get("/api/v1/status")
async def api_status() -> Dict[str, str]:
    """API status endpoint"""
    return {
        "api_version": "v1",
        "status": "active",
        "message": "Backend ready for LangChain integration"
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port) 