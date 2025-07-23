from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import uvicorn
from typing import Dict, List, Optional
import logging
import asyncio
from datetime import datetime

# LangChain imports for OpenAI integration
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# MongoDB
from pymongo import MongoClient
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = FastAPI(title="iQore Chatbot Backend", version="1.0.0")

# Add CORS middleware - matching working vocal AI configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class ChatRequest(BaseModel):
    message: str
    chat_history: Optional[List[Dict[str, str]]] = []

class ChatResponse(BaseModel):
    response: str
    sources: Optional[List[Dict[str, str]]] = []
    chat_history: List[Dict[str, str]]

class HealthResponse(BaseModel):
    status: str
    service: str
    document_count: Optional[int] = None
    timestamp: str

# Global variables for chatbot
qa_chain = None
embeddings = None
mongo_client = None

class ChatbotService:
    """
    Service class for handling chatbot functionality
    """
    
    def __init__(self):
        self.embeddings = None
        self.qa_chain = None
        self.mongo_client = None
        self.db = None
        self.pdf_chunks_collection = None
        
    async def initialize(self):
        """Initialize the chatbot service"""
        try:
            # Initialize OpenAI embeddings
            self.embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small",
                openai_api_key=os.getenv('OPENAI_API_KEY')
            )
            
            # Initialize MongoDB client
            mongodb_uri = os.getenv('MONGODB_URI')
            if not mongodb_uri:
                raise ValueError("MONGODB_URI environment variable is required")
            
            self.mongo_client = MongoClient(mongodb_uri)
            db_name = os.getenv('MONGODB_DATABASE')
            if not db_name:
                raise ValueError("MONGODB_DATABASE environment variable is required")
            
            self.db = self.mongo_client[db_name]
            self.pdf_chunks_collection = self.db['pdf_chunks']
            
            # Check if there are documents in the collection
            doc_count = self.pdf_chunks_collection.count_documents({})
            if doc_count == 0:
                logger.warning("No documents found in database. Chatbot will work but without document context.")
                return False
                
            logger.info(f"Found {doc_count} document chunks in database")
            
            # Set up MongoDB Atlas Vector Search
            vector_store = MongoDBAtlasVectorSearch.from_connection_string(
                connection_string=os.getenv("MONGODB_URI"),
                embedding=self.embeddings,
                namespace=f"{os.getenv('MONGODB_DATABASE')}.pdf_chunks",
                text_key="text",
                embedding_key="embedding",
                relevance_score_fn="cosine"
            )
            
            # Create retriever
            retriever = vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            )
            
            # Set up ChatOpenAI LLM
            llm = ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0.7,
                openai_api_key=os.getenv('OPENAI_API_KEY')
            )
            
            # Modern LangChain LCEL implementation (replaces deprecated ConversationalRetrievalChain)
            
            # Step 1: Create history-aware retriever that can rephrase questions based on chat history
            condense_question_system_template = (
                "Given a chat history and the latest user question "
                "which might reference context in the chat history, "
                "formulate a standalone question which can be understood "
                "without the chat history. Do NOT answer the question, "
                "just reformulate it if needed and otherwise return it as is."
            )
            
            condense_question_prompt = ChatPromptTemplate.from_messages([
                ("system", condense_question_system_template),
                ("placeholder", "{chat_history}"),
                ("human", "{input}"),
            ])
            
            history_aware_retriever = create_history_aware_retriever(
                llm, retriever, condense_question_prompt
            )
            
            # Step 2: Create the question-answering chain
            system_prompt = (
                "You are a knowledgeable and professional virtual assistant for iQore, a deep-tech company "
                "pioneering quantum-classical hybrid compute infrastructure. iQore's core innovation lies in its "
                "software-native, platform-agnostic execution layers—iQD (quantum emulator) and iCD (classical "
                "compute distribution)—designed to accelerate performance and scalability of enterprise AI and "
                "simulation workloads.\n\n"
                "You have access to a curated set of official iQore documents and whitepapers, which you use to "
                "answer questions accurately and in detail. When responding, reference the information from these "
                "documents when relevant, but do not fabricate answers if the information is not available.\n\n"
                "Your tone is helpful, confident, and persuasive. You offer technical and business insights, and "
                "you're able to support a range of user types—from curious visitors to experienced engineers and "
                "decision-makers.\n\n"
                "When appropriate, encourage users to:\n"
                "- Request a product demo\n"
                "- Schedule a follow-up meeting\n"
                "- Learn more about specific use cases\n"
                "- Ask deeper questions about the architecture\n\n"
                "Your goal is to inform, engage, and guide potential customers by showcasing the value of iQore's "
                "solutions, while being honest if something is outside your knowledge.\n\n"
                "Use the following pieces of retrieved context to answer the question. If you don't know the answer, "
                "say that you don't know. Keep the answer informative but concise.\n\n"
                "{context}"
            )
            
            qa_prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("placeholder", "{chat_history}"),
                ("human", "{input}"),
            ])
            
            qa_chain = create_stuff_documents_chain(llm, qa_prompt)
            
            # Step 3: Combine retriever and QA chain
            self.qa_chain = create_retrieval_chain(history_aware_retriever, qa_chain)
            
            logger.info("✅ Modern LCEL QA chain initialized successfully")
            
            logger.info("Chatbot service initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing chatbot service: {e}")
            return False
    
    async def get_response(self, message: str, chat_history: List[Dict[str, str]]) -> Dict:
        """Get response from the chatbot"""
        if not self.qa_chain:
            return {
                "response": "I'm sorry, but the chatbot service is not properly initialized. Please contact support.",
                "sources": [],
                "chat_history": chat_history
            }
        
        try:
            # Convert chat_history to the format expected by modern LCEL chains
            langchain_history = []
            for item in chat_history:
                if "user" in item and "assistant" in item:
                    langchain_history.append(HumanMessage(content=item["user"]))
                    langchain_history.append(AIMessage(content=item["assistant"]))
            
            # Invoke the modern LCEL chain
            result = self.qa_chain.invoke({
                "input": message, 
                "chat_history": langchain_history
            })
            
            response = result["answer"]
            source_docs = result.get("context", [])  # LCEL chains use "context" instead of "source_documents"
            
            # Format sources
            sources = []
            for doc in source_docs[:3]:  # Limit to top 3 sources
                source_info = {
                    "source": doc.metadata.get("source", "Unknown"),
                    "chunk_id": str(doc.metadata.get("chunk_id", "N/A")),
                    "content_preview": doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
                }
                sources.append(source_info)
            
            # Update chat history
            updated_history = chat_history.copy()
            updated_history.append({"user": message, "assistant": response})
            
            # Limit chat history to last 10 exchanges
            if len(updated_history) > 10:
                updated_history = updated_history[-10:]
            
            return {
                "response": response,
                "sources": sources,
                "chat_history": updated_history
            }
            
        except Exception as e:
            logger.error(f"Error getting chatbot response: {e}")
            return {
                "response": "I'm sorry, but I encountered an error while processing your request. Please try again.",
                "sources": [],
                "chat_history": chat_history
            }
    
    def get_document_count(self) -> int:
        """Get the number of documents in the database"""
        if self.pdf_chunks_collection is None:
            return 0
        return self.pdf_chunks_collection.count_documents({})
    
    def close_connections(self):
        """Close database connections"""
        if self.mongo_client is not None:
            self.mongo_client.close()
            logger.info("MongoDB connection closed")

# Initialize chatbot service
chatbot_service = ChatbotService()

@app.on_event("startup")
async def startup_event():
    """Initialize the chatbot service on startup"""
    logger.info("Starting iQore Chatbot Backend...")
    try:
        success = await chatbot_service.initialize()
        if success:
            logger.info("✅ Chatbot service initialized successfully")
        else:
            logger.warning("⚠️ Chatbot service initialized but no documents found in database")
    except Exception as e:
        logger.error(f"⚠️ Failed to initialize chatbot service: {e}")
        logger.info("🔄 Server will continue running, but chatbot functionality may be limited")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    chatbot_service.close_connections()
    logger.info("Chatbot backend shutdown complete")

@app.get("/")
async def root() -> Dict[str, str]:
    """Root endpoint"""
    return {
        "message": "iQore Chatbot Backend is running",
        "status": "healthy",
        "version": "1.0.0",
        "port": str(os.environ.get("PORT", "8080"))
    }

@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint for Google Cloud Run"""
    try:
        doc_count = chatbot_service.get_document_count() if chatbot_service else 0
    except Exception as e:
        logger.warning(f"Could not get document count: {e}")
        doc_count = 0
    
    return HealthResponse(
        status="healthy",
        service="iqore-chatbot-backend",
        document_count=doc_count,
        timestamp=datetime.utcnow().isoformat()
    )

@app.get("/api/v1/status")
async def api_status() -> Dict[str, str]:
    """API status endpoint"""
    try:
        doc_count = chatbot_service.get_document_count() if chatbot_service else 0
    except Exception as e:
        logger.warning(f"Could not get document count for status: {e}")
        doc_count = 0
    
    return {
        "api_version": "v1",
        "status": "active",
        "message": "Backend ready for chat interactions",
        "document_count": doc_count
    }

@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest) -> ChatResponse:
    """
    Main chat endpoint for processing user messages
    """
    try:
        logger.info(f"Received chat request: {request.message[:50]}...")
        
        result = await chatbot_service.get_response(request.message, request.chat_history)
        
        response = ChatResponse(
            response=result["response"],
            sources=result["sources"],
            chat_history=result["chat_history"]
        )
        
        logger.info(f"Chat response sent successfully")
        return response
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

if __name__ == "__main__":
    # Get port from environment variable (Google Cloud Run provides this)
    port = int(os.environ.get("PORT", 8080))
    
    # Log startup information
    logger.info(f"🚀 Starting iQore Chatbot Backend on 0.0.0.0:{port}")
    logger.info(f"📊 Environment: {os.environ.get('GAE_ENV', 'local')}")
    logger.info(f"🔧 Port: {port}")
    
    # Start the server with explicit configuration for Google Cloud Run
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port,
        log_level="info",
        access_log=True,
        timeout_keep_alive=30
    ) 