from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import uvicorn
from typing import Dict, List, Optional, Literal, Annotated
import logging
import asyncio
from datetime import datetime
from contextlib import asynccontextmanager
import operator

# LangChain imports for OpenAI integration
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# LangGraph imports for multi-agent system
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

# MongoDB
from pymongo import MongoClient
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Pydantic models for request/response
class ChatRequest(BaseModel):
    message: str
    chat_history: Optional[List[Dict[str, str]]] = []

class ChatResponse(BaseModel):
    response: str
    chat_history: List[Dict[str, str]]

class HealthResponse(BaseModel):
    status: str
    service: str
    document_count: Optional[int] = None
    timestamp: str

# LangGraph State for multi-agent system
class AgentState(TypedDict):
    """State shared between all agents in the system"""
    messages: Annotated[List[BaseMessage], operator.add]
    next: str  # Which agent to call next
    user_intent: str  # Detected user intent (demo, contact, technical, business)
    lead_info: Dict  # Information about potential leads
    chat_history: List[Dict[str, str]]  # Chat history for context

# Global variables for chatbot
qa_chain = None
embeddings = None
mongo_client = None
multi_agent_graph = None

class ChatbotService:
    """
    Service class for handling chatbot functionality with multi-agent support
    """
    
    def __init__(self):
        self.embeddings = None
        self.qa_chain = None
        self.mongo_client = None
        self.db = None
        self.pdf_chunks_collection = None
        self.multi_agent_graph = None
        self.llm = None
        
    async def initialize(self):
        """Initialize the chatbot service with multi-agent system"""
        try:
            # Initialize OpenAI embeddings and LLM
            self.embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small",
                openai_api_key=os.getenv('OPENAI_API_KEY')
            )
            
            self.llm = ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0.7,
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
                logger.warning("No documents found in database. Technical agent will work with limited context.")
                
            logger.info(f"Found {doc_count} document chunks in database")
            
            # Initialize the traditional RAG chain for technical agent
            await self._initialize_rag_chain()
            
            # Initialize multi-agent system
            await self._initialize_multi_agent_system()
            
            logger.info("✅ Multi-agent chatbot service initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing chatbot service: {e}")
            return False
    
    async def _initialize_rag_chain(self):
        """Initialize the traditional RAG chain for technical questions"""
        try:
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
            
            # History-aware retriever setup
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
                self.llm, retriever, condense_question_prompt
            )
            
            # QA chain setup
            technical_system_prompt = (
                "You are the Technical Expert for iQore at a quantum computing convention booth. "
                "You specialize in explaining iQore's quantum-classical hybrid compute infrastructure, "
                "including iQD (quantum emulator) and iCD (classical compute distribution) technologies.\n\n"
                "Keep responses technical but accessible, 2-3 sentences max. Focus on:\n"
                "- Architecture details\n"
                "- Performance capabilities\n" 
                "- Use cases and applications\n"
                "- Technical comparisons\n\n"
                "Use the following context to answer accurately:\n{context}"
            )
            
            qa_prompt = ChatPromptTemplate.from_messages([
                ("system", technical_system_prompt),
                ("placeholder", "{chat_history}"),
                ("human", "{input}"),
            ])
            
            qa_chain = create_stuff_documents_chain(self.llm, qa_prompt)
            self.qa_chain = create_retrieval_chain(history_aware_retriever, qa_chain)
            
            logger.info("✅ RAG chain for technical agent initialized")
            
        except Exception as e:
            logger.error(f"Error initializing RAG chain: {e}")
            self.qa_chain = None
    
    async def _initialize_multi_agent_system(self):
        """Initialize the LangGraph multi-agent system with supervisor pattern"""
        try:
            # Create the workflow graph
            workflow = StateGraph(AgentState)
            
            # Add nodes for each agent
            workflow.add_node("supervisor", self._supervisor_node)
            workflow.add_node("demo_agent", self._demo_agent_node)
            workflow.add_node("contact_agent", self._contact_agent_node)
            workflow.add_node("technical_agent", self._technical_agent_node)
            workflow.add_node("business_agent", self._business_agent_node)
            
            # Add edges - supervisor routes to appropriate agents
            workflow.add_conditional_edges(
                "supervisor",
                self._should_continue,
                {
                    "demo_agent": "demo_agent",
                    "contact_agent": "contact_agent", 
                    "technical_agent": "technical_agent",
                    "business_agent": "business_agent",
                    "END": END,
                }
            )
            
            # All agents return to supervisor for potential follow-up routing
            workflow.add_edge("demo_agent", "supervisor")
            workflow.add_edge("contact_agent", "supervisor") 
            workflow.add_edge("technical_agent", "supervisor")
            workflow.add_edge("business_agent", "supervisor")
            
            # Set entry point
            workflow.set_entry_point("supervisor")
            
            # Compile the graph
            self.multi_agent_graph = workflow.compile()
            
            logger.info("✅ Multi-agent system initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing multi-agent system: {e}")
            self.multi_agent_graph = None
    
    def _detect_user_intent(self, message: str) -> str:
        """
        Analyze user message to determine intent and route to appropriate agent
        Enhanced version of frontend's generateContextualSuggestions logic
        """
        message_lower = message.lower()
        
        # Demo intent detection
        demo_keywords = ['demo', 'demonstration', 'show me', 'see the demo', 'live demo', 'preview']
        if any(keyword in message_lower for keyword in demo_keywords):
            return "demo_agent"
        
        # Contact/Lead intent detection  
        contact_keywords = ['contact', 'meeting', 'schedule', 'talk to someone', 'call', 'reach out', 
                           'speak with', 'connect', 'get in touch', 'sales', 'representative']
        if any(keyword in message_lower for keyword in contact_keywords):
            return "contact_agent"
        
        # Business intent detection
        business_keywords = ['industry', 'enterprise', 'partnership', 'roi', 'cost', 'pricing', 
                           'business case', 'investment', 'market', 'competition', 'solution']
        if any(keyword in message_lower for keyword in business_keywords):
            return "business_agent"
        
        # Technical intent detection (default for technical questions)
        technical_keywords = ['what is', 'how does', 'applications', 'technology', 'quantum', 
                            'classical', 'iqd', 'icd', 'architecture', 'performance', 'algorithm']
        if any(keyword in message_lower for keyword in technical_keywords):
            return "technical_agent"
        
        # Default to technical agent for general questions
        return "technical_agent"
    
    def _supervisor_node(self, state: AgentState) -> AgentState:
        """
        Supervisor node that analyzes user intent and routes to appropriate specialist agent
        """
        try:
            # Get the latest message
            if not state["messages"]:
                return state
                
            latest_message = state["messages"][-1]
            
            # Skip routing if this is not a user message
            if not isinstance(latest_message, HumanMessage):
                state["next"] = "END"
                return state
            
            # Detect user intent
            user_intent = self._detect_user_intent(latest_message.content)
            
            # Update state with detected intent
            state["user_intent"] = user_intent
            state["next"] = user_intent
            
            logger.info(f"Supervisor routing to: {user_intent}")
            return state
            
        except Exception as e:
            logger.error(f"Error in supervisor node: {e}")
            state["next"] = "technical_agent"  # Fallback to technical agent
            return state
    
    def _should_continue(self, state: AgentState) -> str:
        """Determine which agent to call next based on supervisor decision"""
        return state.get("next", "END")
    
    def _demo_agent_node(self, state: AgentState) -> AgentState:
        """Demo Agent - Handles demo requests and product demonstrations"""
        try:
            demo_prompt = ChatPromptTemplate.from_messages([
                ("system", 
                 "You are the Demo Experience Coordinator for iQore's booth at this quantum computing convention. "
                 "Your mission is to get visitors excited about our live demonstration and guide them through the sign-up process.\n\n"
                 "What you showcase:\n"
                 "🎯 LIVE QUANTUM ALGORITHMS: Watch real-time execution of quantum algorithms on iQore's hybrid architecture\n"
                 "🔧 iQD + iCD IN ACTION: See our quantum emulator (iQD) seamlessly integrate with classical compute distribution (iCD)\n"
                 "📊 PERFORMANCE METRICS: Live performance comparisons showing speed, accuracy, and efficiency gains\n"
                 "🏗️ ARCHITECTURE WALKTHROUGH: Visual demonstration of how our quantum-classical hybrid system works\n\n"
                 "Demo Session Experience (15-20 minutes):\n"
                 "• Interactive algorithm selection and execution\n"
                 "• Real-time performance monitoring and analysis\n" 
                 "• Q&A with our quantum engineers\n"
                 "• Hands-on exploration of use cases relevant to their industry\n\n"
                 "Your goal: Get them excited about the technology and signed up for a demo slot! "
                 "Be enthusiastic but professional. Explain the value they'll get from attending. "
                 "Let them know demo slots are limited and popular. "
                 "Keep responses concise (2-3 sentences) and always end with a call-to-action to join the demo queue."),
                ("human", "{input}")
            ])
            
            latest_message = state["messages"][-1]
            formatted_prompt = demo_prompt.format_messages(input=latest_message.content)
            response = self.llm.invoke(formatted_prompt)
            
            # Add agent response to messages
            agent_message = AIMessage(
                content=response.content,
                name="demo_agent"
            )
            state["messages"].append(agent_message)
            state["next"] = "END"  # Demo agent completes the interaction
            
            return state
            
        except Exception as e:
            logger.error(f"Error in demo agent: {e}")
            error_message = AIMessage(
                content="I'd love to show you our demo! Let me connect you with our technical team at the booth for a hands-on demonstration.",
                name="demo_agent"
            )
            state["messages"].append(error_message)
            state["next"] = "END"
            return state
    
    def _contact_agent_node(self, state: AgentState) -> AgentState:
        """Contact Agent - Handles lead capture and meeting scheduling"""
        try:
            contact_prompt = ChatPromptTemplate.from_messages([
                ("system",
                 "You are the Lead Engagement Specialist for iQore, focused on connecting serious prospects with our team. "
                 "Your primary mission is to collect visitor information and schedule personalized follow-up conversations.\n\n"
                 "Information Collection Process:\n"
                 "📋 ESSENTIAL INFO: Name, Company, Role/Title, Email, Phone (optional)\n"
                 "🎯 INTEREST AREAS: Which aspects of quantum computing interest them most?\n"
                 "🏢 BUSINESS CONTEXT: Company size, industry, current computing challenges\n"
                 "⏰ AVAILABILITY: Preferred times/days for calls or meetings\n"
                 "🌍 LOCATION: Time zone for scheduling purposes\n\n"
                 "Scheduling Options:\n"
                 "• 30-minute discovery call with our business development team\n"
                 "• 60-minute technical deep-dive with our quantum engineers\n"
                 "• Executive briefing with our leadership team\n"
                 "• Follow-up email with detailed information and resources\n\n"
                 "Approach: Be warm, professional, and consultative. Make them feel valued as a potential partner. "
                 "Explain that our follow-up conversations are tailored to their specific needs and business challenges. "
                 "Emphasize the value of a personalized discussion over generic information. "
                 "Keep responses conversational (2-3 sentences) and always guide toward information collection and scheduling."),
                ("human", "{input}")
            ])
            
            latest_message = state["messages"][-1]
            formatted_prompt = contact_prompt.format_messages(input=latest_message.content)
            response = self.llm.invoke(formatted_prompt)
            
            # Add agent response to messages
            agent_message = AIMessage(
                content=response.content,
                name="contact_agent"
            )
            state["messages"].append(agent_message)
            state["next"] = "END"  # Contact agent completes the interaction
            
            return state
            
        except Exception as e:
            logger.error(f"Error in contact agent: {e}")
            error_message = AIMessage(
                content="I'd be happy to connect you with our team! Please share your contact information and I'll arrange a meeting with our specialists.",
                name="contact_agent"
            )
            state["messages"].append(error_message)
            state["next"] = "END"
            return state
    
    def _technical_agent_node(self, state: AgentState) -> AgentState:
        """Technical Agent - Uses existing RAG system for detailed technical questions"""
        try:
            latest_message = state["messages"][-1]
            
            # Convert LangGraph messages to format expected by RAG chain
            chat_history = []
            for msg in state["messages"][:-1]:  # Exclude current message
                if isinstance(msg, HumanMessage):
                    chat_history.append({"user": msg.content, "assistant": ""})
                elif isinstance(msg, AIMessage) and chat_history:
                    chat_history[-1]["assistant"] = msg.content
            
            # Use existing RAG chain if available
            if self.qa_chain:
                langchain_history = []
                for item in chat_history:
                    if item["user"] and item["assistant"]:
                        langchain_history.append(HumanMessage(content=item["user"]))
                        langchain_history.append(AIMessage(content=item["assistant"]))
                
                result = self.qa_chain.invoke({
                    "input": latest_message.content,
                    "chat_history": langchain_history
                })
                response_content = result["answer"]
            else:
                # Fallback if RAG chain is not available
                technical_prompt = ChatPromptTemplate.from_messages([
                    ("system",
                     "You are the Technical Expert for iQore at a quantum convention booth. "
                     "Explain iQore's quantum-classical hybrid technology in accessible terms. "
                     "Keep responses technical but clear, 2-3 sentences max."),
                    ("human", "{input}")
                ])
                
                formatted_prompt = technical_prompt.format_messages(input=latest_message.content)
                response = self.llm.invoke(formatted_prompt)
                response_content = response.content
            
            # Add agent response to messages
            agent_message = AIMessage(
                content=response_content,
                name="technical_agent"
            )
            state["messages"].append(agent_message)
            state["next"] = "END"  # Technical agent completes the interaction
            
            return state
            
        except Exception as e:
            logger.error(f"Error in technical agent: {e}")
            error_message = AIMessage(
                content="I'm here to answer technical questions about iQore's quantum-classical hybrid technology. Could you ask me something specific about our architecture or capabilities?",
                name="technical_agent"
            )
            state["messages"].append(error_message)
            state["next"] = "END"
            return state
    
    def _business_agent_node(self, state: AgentState) -> AgentState:
        """Business Agent - Handles business cases, industry applications, and ROI discussions"""
        try:
            business_prompt = ChatPromptTemplate.from_messages([
                ("system",
                 "You are the Enterprise Strategy Advisor for iQore, specializing in quantum computing business transformation. "
                 "Your expertise lies in connecting iQore's quantum-classical hybrid architecture to real-world enterprise challenges and opportunities.\n\n"
                 "Core Industry Applications:\n"
                 "🏦 FINANCIAL SERVICES: Portfolio optimization, risk analysis, fraud detection, algorithmic trading\n"
                 "🧬 PHARMACEUTICALS: Drug discovery acceleration, molecular simulation, protein folding\n"
                 "🏭 MANUFACTURING: Supply chain optimization, quality control, predictive maintenance\n"
                 "🛡️ CYBERSECURITY: Advanced encryption, threat detection, security protocol development\n"
                 "⚡ ENERGY: Grid optimization, renewable integration, battery technology research\n"
                 "🚗 AUTOMOTIVE: Route optimization, autonomous systems, materials research\n\n"
                 "Enterprise Value Propositions:\n"
                 "📈 COMPETITIVE ADVANTAGE: First-mover advantage in quantum-enhanced operations\n"
                 "💰 ROI DRIVERS: Reduced computational costs, faster time-to-solution, improved accuracy\n"
                 "🔧 HYBRID FLEXIBILITY: Seamless integration with existing classical infrastructure\n"
                 "📊 SCALABLE DEPLOYMENT: From proof-of-concept to enterprise-wide implementation\n"
                 "🎯 MEASURABLE OUTCOMES: Clear performance benchmarks and business metrics\n\n"
                 "Your approach: Think like a C-suite consultant. Focus on strategic business impact, competitive positioning, and measurable outcomes. "
                 "Address how iQore's architecture solves specific enterprise pain points and creates new business opportunities. "
                 "Reference relevant industry trends and market dynamics. "
                 "Keep responses executive-level (2-3 sentences) and always tie back to business value and competitive advantage."),
                ("human", "{input}")
            ])
            
            latest_message = state["messages"][-1]
            formatted_prompt = business_prompt.format_messages(input=latest_message.content)
            response = self.llm.invoke(formatted_prompt)
            
            # Add agent response to messages
            agent_message = AIMessage(
                content=response.content,
                name="business_agent"
            )
            state["messages"].append(agent_message)
            state["next"] = "END"  # Business agent completes the interaction
            
            return state
            
        except Exception as e:
            logger.error(f"Error in business agent: {e}")
            error_message = AIMessage(
                content="I can help you understand how iQore delivers business value across industries. What specific use case or business challenge interests you?",
                name="business_agent"
            )
            state["messages"].append(error_message)
            state["next"] = "END"
            return state
    
    async def get_response(self, message: str, chat_history: List[Dict[str, str]]) -> Dict:
        """Get response from the multi-agent system"""
        try:
            # Use multi-agent system if available, fallback to traditional approach
            if self.multi_agent_graph:
                return await self._get_multi_agent_response(message, chat_history)
            else:
                # Fallback to single-agent RAG system
                return await self._get_traditional_response(message, chat_history)
                
        except Exception as e:
            logger.error(f"Error getting response: {e}")
            return {
                "response": "I'm sorry, but I encountered an error while processing your request. Please try again.",
                "chat_history": chat_history
            }
    
    async def _get_multi_agent_response(self, message: str, chat_history: List[Dict[str, str]]) -> Dict:
        """Get response using the multi-agent system"""
        try:
            # Convert chat history to LangGraph format
            messages = []
            for item in chat_history:
                if "user" in item and "assistant" in item:
                    messages.append(HumanMessage(content=item["user"]))
                    messages.append(AIMessage(content=item["assistant"]))
            
            # Add current user message
            messages.append(HumanMessage(content=message))
            
            # Create initial state
            initial_state = AgentState(
                messages=messages,
                next="",
                user_intent="",
                lead_info={},
                chat_history=chat_history
            )
            
            # Run the multi-agent workflow
            final_state = self.multi_agent_graph.invoke(initial_state)
            
            # Extract the final AI response
            ai_messages = [msg for msg in final_state["messages"] if isinstance(msg, AIMessage)]
            if ai_messages:
                response = ai_messages[-1].content
            else:
                response = "I'm here to help you learn about iQore. What would you like to know?"
            
            # Update chat history
            updated_history = chat_history.copy()
            updated_history.append({"user": message, "assistant": response})
            
            # Limit chat history to last 10 exchanges
            if len(updated_history) > 10:
                updated_history = updated_history[-10:]
            
            return {
                "response": response,
                "chat_history": updated_history
            }
            
        except Exception as e:
            logger.error(f"Error in multi-agent response: {e}")
            # Fallback to traditional response
            return await self._get_traditional_response(message, chat_history)
    
    async def _get_traditional_response(self, message: str, chat_history: List[Dict[str, str]]) -> Dict:
        """Fallback to traditional single-agent RAG response"""
        if not self.qa_chain:
            return {
                "response": "I'm sorry, but the chatbot service is not properly initialized. Please contact support.",
                "chat_history": chat_history
            }
        
        try:
            # Convert chat_history to the format expected by RAG chain
            langchain_history = []
            for item in chat_history:
                if "user" in item and "assistant" in item:
                    langchain_history.append(HumanMessage(content=item["user"]))
                    langchain_history.append(AIMessage(content=item["assistant"]))
            
            # Invoke the RAG chain
            result = self.qa_chain.invoke({
                "input": message, 
                "chat_history": langchain_history
            })
            
            response = result["answer"]
            
            # Update chat history
            updated_history = chat_history.copy()
            updated_history.append({"user": message, "assistant": response})
            
            # Limit chat history to last 10 exchanges
            if len(updated_history) > 10:
                updated_history = updated_history[-10:]
            
            return {
                "response": response,
                "chat_history": updated_history
            }
            
        except Exception as e:
            logger.error(f"Error getting traditional response: {e}")
            return {
                "response": "I'm sorry, but I encountered an error while processing your request. Please try again.",
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

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting iQore Multi-Agent Chatbot Backend...")
    try:
        success = await chatbot_service.initialize()
        if success:
            logger.info("✅ Multi-agent chatbot service initialized successfully")
        else:
            logger.warning("⚠️ Chatbot service initialized but with limited functionality")
    except Exception as e:
        logger.error(f"⚠️ Failed to initialize chatbot service: {e}")
        logger.info("🔄 Server will continue running, but chatbot functionality may be limited")
    
    yield
    
    # Shutdown
    chatbot_service.close_connections()
    logger.info("Multi-agent chatbot backend shutdown complete")

app = FastAPI(title="iQore Multi-Agent Chatbot Backend", version="2.0.0", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root() -> Dict[str, str]:
    """Root endpoint"""
    return {
        "message": "iQore Multi-Agent Chatbot Backend is running",
        "status": "healthy",
        "version": "2.0.0",
        "agents": ["supervisor", "demo_agent", "contact_agent", "technical_agent", "business_agent"],
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
        service="iqore-multi-agent-chatbot-backend",
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
        "message": "Multi-agent backend ready for chat interactions",
        "document_count": doc_count,
        "agents": ["supervisor", "demo_agent", "contact_agent", "technical_agent", "business_agent"]
    }

@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest) -> ChatResponse:
    """
    Main chat endpoint for processing user messages through multi-agent system
    """
    try:
        logger.info(f"Received chat request: {request.message[:50]}...")
        
        result = await chatbot_service.get_response(request.message, request.chat_history)
        
        response = ChatResponse(
            response=result["response"],
            chat_history=result["chat_history"]
        )
        
        logger.info(f"Multi-agent chat response sent successfully")
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
    logger.info(f"🚀 Starting iQore Multi-Agent Chatbot Backend on 0.0.0.0:{port}")
    logger.info(f"📊 Environment: {os.environ.get('GAE_ENV', 'local')}")
    logger.info(f"🔧 Port: {port}")
    logger.info(f"🤖 Agents: supervisor, demo_agent, contact_agent, technical_agent, business_agent")
    
    # Start the server with explicit configuration for Google Cloud Run
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port,
        log_level="info",
        access_log=True,
        timeout_keep_alive=30
    ) 