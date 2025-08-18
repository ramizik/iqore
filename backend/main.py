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
import uuid
import re

# LangChain imports for OpenAI integration
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Phase 2: LangChain Tools and Output Parsers
from langchain.tools import BaseTool
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

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

# Phase 1: Enhanced Pydantic Models for Demo Queue Management
from pydantic import EmailStr, validator
from enum import Enum
from datetime import datetime

class DemoStatus(str, Enum):
    WAITING = "waiting"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    NO_SHOW = "no_show"

class VisitorInfo(BaseModel):
    name: str
    email: EmailStr
    company: Optional[str] = None
    interest_areas: List[str] = []
    
    @validator('name')
    def validate_name(cls, v):
        if not v or len(v.strip()) < 2:
            raise ValueError('Name must be at least 2 characters')
        if len(v) > 50:
            raise ValueError('Name must be less than 50 characters')
        return v.strip().title()

class DemoRequest(BaseModel):
    visitor_info: VisitorInfo
    session_id: Optional[str] = None
    timestamp: Optional[datetime] = None
    status: DemoStatus = DemoStatus.WAITING
    notes: Optional[str] = None

class DemoQueueEntry(BaseModel):
    id: Optional[str] = None
    session_id: str
    name: str
    email: str
    company: Optional[str] = None
    interest_areas: List[str] = []
    status: DemoStatus
    timestamp: datetime
    queue_position: Optional[int] = None
    estimated_wait_time: Optional[int] = None
    notes: Optional[str] = None

class QueueStatusResponse(BaseModel):
    success: bool
    total_queue_length: int
    estimated_wait_time_minutes: int
    current_demo_in_progress: bool
    average_demo_duration_minutes: float = 17.5
    message: Optional[str] = None
    error: Optional[str] = None

class UserQueueStatusResponse(BaseModel):
    success: bool
    session_id: Optional[str] = None
    name: Optional[str] = None
    status: Optional[DemoStatus] = None
    queue_position: Optional[int] = None
    estimated_wait_time: Optional[int] = None
    total_in_queue: Optional[int] = None
    message: Optional[str] = None
    error: Optional[str] = None

class StaffQueueUpdateRequest(BaseModel):
    session_id: str
    new_status: DemoStatus
    notes: Optional[str] = None

class StaffQueueResponse(BaseModel):
    success: bool
    total_entries: int
    waiting_count: int
    in_progress_count: int
    completed_today: int
    queue_entries: List[DemoQueueEntry] = []
    message: Optional[str] = None
    error: Optional[str] = None

# Phase 2: Custom LangChain Tools for Demo Queue Management
class DemoQueueTool(BaseTool):
    """Tool for managing demo queue operations within LangChain agents"""
    name: str = "demo_queue_manager"
    description: str = (
        "Manage demo queue operations including adding visitors, checking status, and retrieving queue information. "
        "Use this tool when visitors want to sign up for demos or check their queue position. "
        "Input should be a JSON string with 'action' and relevant parameters."
    )
    chatbot_service: Optional['ChatbotService'] = None

    def __init__(self, chatbot_service: 'ChatbotService'):
        super().__init__()
        self.chatbot_service = chatbot_service

    async def _arun(self, action_input: str) -> str:
        """Asynchronous execution of demo queue operations"""
        try:
            import json
            action_data = json.loads(action_input)
            action = action_data.get("action")
            
            if action == "add_to_queue":
                result = await self.chatbot_service.add_to_demo_queue(
                    name=action_data["name"],
                    email=action_data["email"],
                    company=action_data.get("company"),
                    interest_areas=action_data.get("interest_areas", []),
                    session_id=action_data.get("session_id")
                )
                return json.dumps(result)
                
            elif action == "get_queue_status":
                if "session_id" in action_data:
                    result = await self.chatbot_service.get_queue_status(action_data["session_id"])
                else:
                    queue_length = await self.chatbot_service.get_current_queue_length()
                    wait_time = await self.chatbot_service.estimate_wait_time()
                    result = {
                        "success": True,
                        "queue_length": queue_length,
                        "estimated_wait_time": wait_time
                    }
                return json.dumps(result)
                
            elif action == "get_overall_status":
                queue_length = await self.chatbot_service.get_current_queue_length()
                wait_time = await self.chatbot_service.estimate_wait_time()
                in_progress = await self.chatbot_service.get_in_progress_count()
                result = {
                    "success": True,
                    "total_queue_length": queue_length,
                    "estimated_wait_time_minutes": wait_time,
                    "demos_in_progress": in_progress
                }
                return json.dumps(result)
                
            else:
                return json.dumps({"success": False, "error": f"Unknown action: {action}"})
                
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})

    def _run(self, action_input: str) -> str:
        """Synchronous fallback - not implemented for async operations"""
        return json.dumps({"success": False, "error": "This tool requires async execution"})

class UserInfoExtractionTool(BaseTool):
    """Tool for extracting structured visitor information from natural language"""
    name: str = "user_info_extractor"
    description: str = (
        "Extract structured visitor information (name, email, company) from natural language conversation. "
        "Use this when visitors provide their details for demo signup. "
        "Input should be the user's message text."
    )

    def _run(self, message_input: str) -> str:
        """Extract user information from natural language message"""
        try:
            import json
            # Basic extraction logic - this could be enhanced with more sophisticated NLP
            info = {}
            message_lower = message_input.lower().strip()
            
            # Extract email
            import re
            email_pattern = r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b'
            email_match = re.search(email_pattern, message_input)
            if email_match:
                info['email'] = email_match.group(0).lower()
            
            # Extract name patterns
            name_patterns = [
                r"my name is ([a-zA-Z\s]+)",
                r"i'm ([a-zA-Z\s]+)",
                r"name's ([a-zA-Z\s]+)",
                r"i am ([a-zA-Z\s]+)",
                r"call me ([a-zA-Z\s]+)"
            ]
            
            for pattern in name_patterns:
                name_match = re.search(pattern, message_lower)
                if name_match:
                    potential_name = name_match.group(1).strip().title()
                    if 2 <= len(potential_name) <= 50:
                        info['name'] = potential_name
                        break
            
            # If no name pattern found, check if message might be just a name
            if not info.get('name') and 2 <= len(message_input.strip()) <= 50 and '@' not in message_input:
                words = message_input.strip().split()
                if 1 <= len(words) <= 4 and re.match(r'^[a-zA-Z\s]+$', message_input.strip()):
                    info['name'] = message_input.strip().title()
            
            # Extract company
            company_patterns = [
                r"i work at ([a-zA-Z0-9\s&.-]+)",
                r"work for ([a-zA-Z0-9\s&.-]+)",
                r"from ([a-zA-Z0-9\s&.-]+)",
                r"company is ([a-zA-Z0-9\s&.-]+)"
            ]
            
            for pattern in company_patterns:
                company_match = re.search(pattern, message_lower)
                if company_match:
                    potential_company = company_match.group(1).strip().title()
                    if 2 <= len(potential_company) <= 100:
                        info['company'] = potential_company
                        break
            
            return json.dumps({
                "success": True,
                "extracted_info": info,
                "info_complete": bool(info.get('name') and info.get('email'))
            })
            
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})

# Phase 2: Pydantic Output Parsers for Structured Data
class ExtractedVisitorInfo(BaseModel):
    """Structured visitor information for demo signups"""
    name: Optional[str] = None
    email: Optional[str] = None
    company: Optional[str] = None
    interest_areas: List[str] = []
    info_complete: bool = False
    confidence_score: float = 0.0

class DemoSignupIntent(BaseModel):
    """Structured intent detection for demo signups"""
    wants_demo: bool = False
    has_questions: bool = False
    asking_about_queue: bool = False
    providing_info: bool = False
    intent_confidence: float = 0.0
    next_action: str = "continue_conversation"

# Phase 2: Output Parsers
visitor_info_parser = PydanticOutputParser(pydantic_object=ExtractedVisitorInfo)
demo_intent_parser = PydanticOutputParser(pydantic_object=DemoSignupIntent)

# LangGraph State for multi-agent system
class AgentState(TypedDict):
    """State shared between all agents in the system"""
    messages: Annotated[List[BaseMessage], operator.add]
    next: str  # Which agent to call next
    user_intent: str  # Detected user intent (demo, contact, technical, business)
    lead_info: Dict  # Information about potential leads
    chat_history: List[Dict[str, str]]  # Chat history for context
    # Phase 1: Demo-specific state management
    demo_state: str  # "initial", "collecting_info", "confirming", "queued"
    demo_user_info: Dict  # {"name": "", "email": "", "company": "", "interest_areas": []}
    demo_session_id: str  # Unique session identifier for queue tracking

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
        # Phase 2: Initialize demo queue collection reference
        self.demo_queue_collection = None
        self.multi_agent_graph = None
        self.llm = None
        # Phase 2: Initialize LangChain tools
        self.demo_queue_tool = None
        self.user_info_tool = None
        
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
            # Phase 2: Initialize demo queue collection
            self.demo_queue_collection = self.db['demo_queue']
            
            # Check if there are documents in the collection
            doc_count = self.pdf_chunks_collection.count_documents({})
            if doc_count == 0:
                logger.warning("No documents found in database. Technical agent will work with limited context.")
                
            logger.info(f"Found {doc_count} document chunks in database")
            
            # Phase 2: Check demo queue collection
            demo_queue_count = self.demo_queue_collection.count_documents({})
            logger.info(f"Demo queue initialized with {demo_queue_count} existing entries")
            
            # Initialize the traditional RAG chain for technical agent
            await self._initialize_rag_chain()
            
            # Phase 2: Initialize LangChain tools
            await self._initialize_langchain_tools()
            
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
    
    async def _initialize_langchain_tools(self):
        """Initialize custom LangChain tools for demo queue management"""
        try:
            # Initialize demo queue management tool
            self.demo_queue_tool = DemoQueueTool(chatbot_service=self)
            
            # Initialize user info extraction tool
            self.user_info_tool = UserInfoExtractionTool()
            
            logger.info("✅ LangChain tools initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing LangChain tools: {e}")
            self.demo_queue_tool = None
            self.user_info_tool = None
    
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
        Now with conversation context awareness for demo flows
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
            
            # Phase 3 Fix: Check if user is already in demo conversation flow
            demo_state = state.get("demo_state", "")
            previous_agent_messages = [msg for msg in state["messages"] if isinstance(msg, AIMessage)]
            
            # If user was recently talking to demo agent, check for demo-related responses
            if (demo_state or 
                (previous_agent_messages and 
                 getattr(previous_agent_messages[-1], 'name', '') == 'demo_agent')):
                
                user_message = latest_message.content.lower()
                
                # Check for demo signup responses (yes, sure, sign me up, etc.)
                if self._detect_signup_intent(user_message):
                    logger.info(f"Supervisor: User responding to demo signup - keeping in demo_agent")
                    state["user_intent"] = "demo_agent"
                    state["next"] = "demo_agent"
                    return state
                
                # Check for demo queue status requests
                if self._detect_queue_status_intent(user_message):
                    logger.info(f"Supervisor: User asking about queue status - routing to demo_agent")
                    state["user_intent"] = "demo_agent" 
                    state["next"] = "demo_agent"
                    return state
                
                # Check for demo-related follow-up questions
                demo_keywords = ['demo', 'demonstration', 'queue', 'wait', 'signup', 'reserve', 'spot']
                if any(keyword in user_message for keyword in demo_keywords):
                    logger.info(f"Supervisor: Demo context detected - continuing with demo_agent")
                    state["user_intent"] = "demo_agent"
                    state["next"] = "demo_agent"
                    return state
            
            # Default intent detection for new conversations
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
    
    async def _demo_agent_node(self, state: AgentState) -> AgentState:
        """Phase 3: Demo Agent with natural conversation flow and queue management"""
        try:
            latest_message = state["messages"][-1]
            demo_state = state.get("demo_state", "initial")
            
            logger.info(f"Demo agent: Current state='{demo_state}', Processing message: '{latest_message.content[:50]}...'")
            
            # Check if user is asking for queue status and has a session ID
            if (self._detect_queue_status_intent(latest_message.content) and 
                state.get("demo_session_id")):
                logger.info("Demo agent: User has session ID and asking for queue status")
                return await self._demo_queue_status_stage(state)
            
            # Route to appropriate stage based on current state
            if demo_state == "initial":
                return await self._demo_initial_stage(state)
            elif demo_state == "collecting_info":
                logger.info("Demo agent: In collecting_info state")
                return await self._demo_collect_info_stage(state)
            elif demo_state == "confirming":
                logger.info("Demo agent: In confirming state")
                return await self._demo_confirm_stage(state)
            elif demo_state == "queued":
                logger.info("Demo agent: In queued state")
                return await self._demo_queue_status_stage(state)
            else:
                # Fallback to initial stage
                logger.info(f"Demo agent: Unknown state '{demo_state}', falling back to initial")
                return await self._demo_initial_stage(state)
                
        except Exception as e:
            logger.error(f"Error in demo agent: {e}")
            error_message = AIMessage(
                content="I'd love to help you with our quantum computing demo! How can I assist you?",
                name="demo_agent"
            )
            state["messages"].append(error_message)
            state["next"] = "END"
            return state
    
    async def _demo_initial_stage(self, state: AgentState) -> AgentState:
        """Phase 3: Enhanced initial demo introduction with natural response handling"""
        try:
            latest_message = state["messages"][-1]
            user_message = latest_message.content
            
            # Check if this is the very first demo interaction or a follow-up response
            chat_history = state.get("chat_history", [])
            is_first_demo_message = len([msg for msg in state.get("messages", []) if getattr(msg, 'name', '') == 'demo_agent']) == 0
            
            if is_first_demo_message:
                # First time - show full demo description
                demo_prompt = ChatPromptTemplate.from_messages([
                    ("system", 
                     "You are the Demo Experience Coordinator for iQore's booth at this quantum computing convention. "
                     "Your mission is to get visitors excited about our live demonstration.\n\n"
                     "🎯 QUANTUM ALGORITHMS WE DEMONSTRATE:\n"
                     "• Variational Quantum Eigensolver (VQE) for molecular simulation\n"
                     "• Quantum Approximate Optimization Algorithm (QAOA) for logistics optimization\n"
                     "• Grover's Search Algorithm for database optimization\n"
                     "• Shor's Algorithm simulation for cryptography research\n"
                     "• Quantum Machine Learning algorithms for pattern recognition\n\n"
                     "🔧 WHAT YOU'LL SEE:\n"
                     "• iQD (quantum emulator) + iCD (classical compute) working together\n"
                     "• Real-time performance metrics and comparisons\n"
                     "• Interactive algorithm selection and execution\n"
                     "• Live Q&A with our quantum engineers (15-20 minutes)\n\n"
                     "Your goal: Get them excited and ask if they'd like to sign up for a demo slot! "
                     "Be enthusiastic but professional. Mention that slots are limited and popular. "
                     "End with: 'Would you like to reserve a spot in our demo queue?'"),
                    ("human", "{input}")
                ])
                
                formatted_prompt = demo_prompt.format_messages(input=user_message)
                response = self.llm.invoke(formatted_prompt)
                response_content = response.content
                
            else:
                # Follow-up response - check user intent
                signup_detected = self._detect_signup_intent(user_message)
                logger.info(f"Demo agent: Signup intent detected: {signup_detected} for message: '{user_message}'")
                
                if signup_detected:
                    # User wants to sign up - move to info collection
                    current_queue_length = await self.get_current_queue_length()
                    wait_time = await self.estimate_wait_time()
                    
                    response_content = (
                        f"Fantastic! Let me get you signed up. 🎉\n\n"
                        f"📊 **Current Queue Status:** {current_queue_length} people waiting (~{wait_time} minutes)\n\n"
                        f"To reserve your spot, I'll need your name and email address. What's your name?"
                    )
                    state["demo_state"] = "collecting_info"
                    
                    # Log the transition for debugging
                    logger.info(f"Demo agent: User confirmed signup, moving to collecting_info state")
                    
                elif self._detect_queue_status_intent(user_message):
                    # User asking about queue - provide general info
                    current_queue_length = await self.get_current_queue_length()
                    wait_time = await self.estimate_wait_time()
                    
                    response_content = (
                        f"📊 **Current Demo Queue Status:**\n"
                        f"• People waiting: {current_queue_length}\n"
                        f"• Estimated wait for new signups: {wait_time} minutes\n"
                        f"• Demo duration: 15-20 minutes each\n\n"
                        f"Would you like me to add you to the queue?"
                    )
                    
                else:
                    # General follow-up or questions about demo
                    demo_followup_prompt = ChatPromptTemplate.from_messages([
                        ("system",
                         "You are helping someone who's interested in our quantum computing demo. "
                         "Answer their questions about the demo content, what they'll see, or technical details. "
                         "Be helpful and encouraging. Always end by asking if they'd like to sign up for the queue."),
                        ("human", "{input}")
                    ])
                    
                    formatted_prompt = demo_followup_prompt.format_messages(input=user_message)
                    response = self.llm.invoke(formatted_prompt)
                    response_content = response.content
            
            # Add agent response
            agent_message = AIMessage(content=response_content, name="demo_agent")
            state["messages"].append(agent_message)
            
            # Stay in demo agent for continued interaction
            state["next"] = "demo_agent"
            
            return state
            
        except Exception as e:
            logger.error(f"Error in demo initial stage: {e}")
            error_message = AIMessage(
                content="I'd love to show you our quantum computing demo! Would you like to reserve a spot in our demo queue?",
                name="demo_agent"
            )
            state["messages"].append(error_message)
            state["next"] = "demo_agent"
            return state
    
    async def _demo_collect_info_stage(self, state: AgentState) -> AgentState:
        """Phase 3: Natural conversation for collecting user information"""
        try:
            latest_message = state["messages"][-1]
            current_info = state.get("demo_user_info", {})
            
            # Extract information from user's message
            extracted_info, info_complete = self._extract_user_info(latest_message.content, current_info)
            
            # Update demo_user_info with extracted information
            state["demo_user_info"] = extracted_info
            
            # Log what information was extracted for debugging
            logger.info(f"Extracted info: {extracted_info}, Complete: {info_complete}")
            
            # Generate natural response based on what information we have
            if info_complete:
                # We have name and email, move to confirmation
                response_content = (
                    f"Perfect! Let me confirm your details:\n"
                    f"📝 Name: {extracted_info['name']}\n"
                    f"📧 Email: {extracted_info['email']}\n"
                    f"🏢 Company: {extracted_info.get('company', 'Not specified')}\n\n"
                    f"I'll add you to our demo queue right away. Does this look correct?"
                )
                state["demo_state"] = "confirming"
                state["next"] = "demo_agent"
            else:
                # Generate natural follow-up question
                response_content = self._generate_info_request(extracted_info)
                state["demo_state"] = "collecting_info"
                state["next"] = "demo_agent"
            
            agent_message = AIMessage(content=response_content, name="demo_agent")
            state["messages"].append(agent_message)
            return state
            
        except Exception as e:
            logger.error(f"Error in demo collect info stage: {e}")
            agent_message = AIMessage(
                content="I'd love to get you signed up! Could you please share your name and email address?",
                name="demo_agent"
            )
            state["messages"].append(agent_message)
            state["next"] = "demo_agent"
            return state
    
    async def _demo_confirm_stage(self, state: AgentState) -> AgentState:
        """Phase 3: Handle confirmation and queue addition"""
        try:
            latest_message = state["messages"][-1]
            user_response = latest_message.content.lower().strip()
            
            # Check if user confirms (yes, correct, confirm, etc.)
            confirm_keywords = ['yes', 'correct', 'confirm', 'right', 'good', 'ok', 'okay', 'sure', 'yep', 'yeah']
            deny_keywords = ['no', 'wrong', 'incorrect', 'change', 'fix', 'edit']
            
            if any(keyword in user_response for keyword in confirm_keywords):
                # User confirmed - add to queue
                user_info = state.get("demo_user_info", {})
                
                if user_info.get("name") and user_info.get("email"):
                    # Add to demo queue
                    queue_result = await self.add_to_demo_queue(user_info)
                    
                    if queue_result.get("success"):
                        session_id = queue_result["session_id"]
                        position = queue_result["queue_position"]
                        wait_time = queue_result["estimated_wait_time"]
                        
                        # Store session ID for future reference
                        state["demo_session_id"] = session_id
                        state["demo_state"] = "queued"
                        
                        response_content = (
                            f"🎉 Excellent! You're now in our demo queue.\n\n"
                            f"📊 **Your Queue Status:**\n"
                            f"• Position: #{position}\n"
                            f"• Estimated wait: {wait_time} minutes\n"
                            f"• Demo duration: 15-20 minutes\n\n"
                            f"📍 **Next Steps:**\n"
                            f"• Find our demo station (look for the iQore quantum booth)\n"
                            f"• Our team will call your name when it's your turn\n"
                            f"• Feel free to explore other booths while you wait!\n\n"
                            f"💬 You can ask me for queue updates anytime by saying 'queue status' or 'how long is my wait?'\n"
                            f"🆔 Your session ID: `{session_id}`"
                        )
                        
                        logger.info(f"Successfully added {user_info['name']} to demo queue at position {position}")
                        
                    else:
                        response_content = (
                            "I apologize, but there was an issue adding you to the queue. "
                            "Please find one of our team members at the booth, and they'll help you sign up directly!"
                        )
                        state["demo_state"] = "initial"
                        
                else:
                    response_content = "I seem to be missing some information. Let me collect your details again."
                    state["demo_state"] = "collecting_info"
                    
            elif any(keyword in user_response for keyword in deny_keywords):
                # User wants to make changes
                response_content = (
                    "No problem! Let's update your information. "
                    "Please share your correct name and email address, and I'll update your details."
                )
                state["demo_state"] = "collecting_info"
                # Clear current info to restart collection
                state["demo_user_info"] = {}
                
            else:
                # Unclear response - ask for clarification
                response_content = (
                    "I want to make sure I have this right. "
                    "Should I add you to the demo queue with these details? "
                    "Please respond with 'yes' to confirm or 'no' to make changes."
                )
                # Stay in confirming state
                state["demo_state"] = "confirming"
            
            agent_message = AIMessage(content=response_content, name="demo_agent")
            state["messages"].append(agent_message)
            state["next"] = "demo_agent" if state["demo_state"] != "queued" else "END"
            
            return state
            
        except Exception as e:
            logger.error(f"Error in demo confirm stage: {e}")
            agent_message = AIMessage(
                content="There was an issue processing your signup. Please visit our booth directly for assistance!",
                name="demo_agent"
            )
            state["messages"].append(agent_message)
            state["next"] = "END"
            return state
    
    async def _demo_queue_status_stage(self, state: AgentState) -> AgentState:
        """Phase 3: Handle queue status requests and updates"""
        try:
            latest_message = state["messages"][-1]
            session_id = state.get("demo_session_id", "")
            
            if session_id:
                # Get current queue status
                status_result = await self.get_queue_status(session_id)
                
                if status_result.get("success"):
                    position = status_result["queue_position"]
                    wait_time = status_result["estimated_wait_time"]
                    name = status_result["name"]
                    total_queue = status_result["total_in_queue"]
                    
                    if position == 1:
                        response_content = (
                            f"🎉 Great news, {name}! You're **next in line**!\n\n"
                            f"📍 Please head to our demo station now - our team should be calling your name any moment. "
                            f"Look for the iQore quantum computing booth!\n\n"
                            f"⏱️ Your demo will last about 15-20 minutes with hands-on quantum algorithm exploration."
                        )
                    elif position <= 3:
                        response_content = (
                            f"📊 **Queue Update for {name}:**\n"
                            f"• Current position: #{position}\n"
                            f"• Estimated wait: {wait_time} minutes\n"
                            f"• Total people in queue: {total_queue}\n\n"
                            f"🔔 You're coming up soon! Stay nearby - we'll call your name when it's your turn."
                        )
                    else:
                        response_content = (
                            f"📊 **Queue Update for {name}:**\n"
                            f"• Current position: #{position}\n"
                            f"• Estimated wait: {wait_time} minutes\n"
                            f"• Total people in queue: {total_queue}\n\n"
                            f"⏰ You have some time to explore other booths! "
                            f"Come back in about {max(wait_time - 10, 5)} minutes, or ask me for another update anytime."
                        )
                else:
                    response_content = (
                        "I'm having trouble finding your queue entry. "
                        "Please visit our booth directly, and our team will help you!"
                    )
            else:
                # No session ID - user might be asking about general queue or wanting to sign up
                user_message = latest_message.content.lower()
                
                if any(word in user_message for word in ['status', 'wait', 'long', 'queue', 'position']):
                    # General queue inquiry
                    queue_length = await self.get_current_queue_length()
                    wait_time = await self.estimate_wait_time()
                    
                    response_content = (
                        f"📊 **Current Demo Queue Status:**\n"
                        f"• People waiting: {queue_length}\n"
                        f"• Estimated wait for new signups: {wait_time} minutes\n\n"
                        f"Would you like to join the queue? I can sign you up right now!"
                    )
                else:
                    # User might want to sign up
                    response_content = (
                        "I don't see you in our demo queue yet. "
                        "Would you like me to sign you up for a quantum computing demonstration? "
                        "It only takes a moment to get you added!"
                    )
                    state["demo_state"] = "initial"
            
            agent_message = AIMessage(content=response_content, name="demo_agent")
            state["messages"].append(agent_message)
            state["next"] = "END"
            
            return state
            
        except Exception as e:
            logger.error(f"Error in demo queue status stage: {e}")
            agent_message = AIMessage(
                content="I can help you check your demo queue status! Ask me about your queue position anytime.",
                name="demo_agent"
            )
            state["messages"].append(agent_message)
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
    
    # Phase 1: Enhanced Queue Management Methods
    async def get_in_progress_count(self) -> int:
        """Get count of demos currently in progress"""
        try:
            if self.demo_queue_collection is None:
                return 0
            return self.demo_queue_collection.count_documents({"status": "in_progress"})
        except Exception as e:
            logger.error(f"Error getting in-progress count: {e}")
            return 0

    async def get_full_queue_status(self) -> Dict:
        """Get comprehensive queue status for staff management"""
        try:
            if self.demo_queue_collection is None:
                return {
                    "success": False,
                    "error": "Demo queue collection not available",
                    "total_entries": 0,
                    "waiting_count": 0,
                    "in_progress_count": 0,
                    "completed_today": 0,
                    "queue_entries": []
                }
            
            # Get counts by status
            waiting_count = self.demo_queue_collection.count_documents({"status": "waiting"})
            in_progress_count = self.demo_queue_collection.count_documents({"status": "in_progress"})
            
            # Get today's completed demos
            today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
            completed_today = self.demo_queue_collection.count_documents({
                "status": "completed",
                "timestamp": {"$gte": today_start}
            })
            
            # Get all active queue entries (waiting + in_progress)
            queue_entries = []
            cursor = self.demo_queue_collection.find(
                {"status": {"$in": ["waiting", "in_progress"]}},
                {"_id": 0}  # Exclude MongoDB's _id field
            ).sort("timestamp", 1)
            
            position = 1
            for entry in cursor:
                queue_entry = DemoQueueEntry(
                    session_id=entry["session_id"],
                    name=entry["name"],
                    email=entry["email"],
                    company=entry.get("company"),
                    interest_areas=entry.get("interest_areas", []),
                    status=DemoStatus(entry["status"]),
                    timestamp=entry["timestamp"],
                    queue_position=position if entry["status"] == "waiting" else None,
                    estimated_wait_time=((position - 1) * 17.5) if entry["status"] == "waiting" else None,
                    notes=entry.get("notes")
                )
                queue_entries.append(queue_entry)
                if entry["status"] == "waiting":
                    position += 1
            
            total_entries = waiting_count + in_progress_count
            
            return {
                "success": True,
                "total_entries": total_entries,
                "waiting_count": waiting_count,
                "in_progress_count": in_progress_count,
                "completed_today": completed_today,
                "queue_entries": queue_entries,
                "message": f"Queue status: {waiting_count} waiting, {in_progress_count} in progress"
            }
            
        except Exception as e:
            logger.error(f"Error getting full queue status: {e}")
            return {
                "success": False,
                "error": str(e),
                "total_entries": 0,
                "waiting_count": 0,
                "in_progress_count": 0,
                "completed_today": 0,
                "queue_entries": []
            }

    async def update_demo_status(self, session_id: str, new_status: str, notes: Optional[str] = None) -> Dict:
        """Update demo status (for staff management)"""
        try:
            if self.demo_queue_collection is None:
                return {"success": False, "error": "Demo queue collection not available"}
            
            # Find the entry
            entry = self.demo_queue_collection.find_one({"session_id": session_id})
            if not entry:
                return {"success": False, "error": "Session not found"}
            
            # Prepare update data
            update_data = {
                "status": new_status,
                "updated_at": datetime.utcnow()
            }
            if notes:
                update_data["notes"] = notes
            
            # Update the entry
            result = self.demo_queue_collection.update_one(
                {"session_id": session_id},
                {"$set": update_data}
            )
            
            if result.modified_count > 0:
                return {
                    "success": True,
                    "message": f"Demo status updated to {new_status}",
                    "session_id": session_id
                }
            else:
                return {"success": False, "error": "Failed to update demo status"}
                
        except Exception as e:
            logger.error(f"Error updating demo status: {e}")
            return {"success": False, "error": str(e)}

    async def remove_from_queue(self, session_id: str) -> Dict:
        """Remove entry from demo queue"""
        try:
            if self.demo_queue_collection is None:
                return {"success": False, "error": "Demo queue collection not available"}
            
            # Remove the entry
            result = self.demo_queue_collection.delete_one({"session_id": session_id})
            
            if result.deleted_count > 0:
                return {
                    "success": True,
                    "message": "Entry removed from demo queue",
                    "session_id": session_id
                }
            else:
                return {"success": False, "error": "Session not found in queue"}
                
        except Exception as e:
            logger.error(f"Error removing from queue: {e}")
            return {"success": False, "error": str(e)}

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
                chat_history=chat_history,
                # Phase 1: Initialize demo state fields
                demo_state="initial",
                demo_user_info={},
                demo_session_id=""
            )
            
            # Run the multi-agent workflow (now supports async nodes)
            final_state = await self.multi_agent_graph.ainvoke(initial_state)
            
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
    
    # Phase 2: Demo Queue Management Methods
    async def add_to_demo_queue(self, user_info: Dict) -> Dict:
        """Add user to demo queue and return queue status"""
        try:
            if self.demo_queue_collection is None:
                raise Exception("Demo queue collection not initialized")
            
            # Generate unique session ID
            session_id = str(uuid.uuid4())
            
            # Get current queue position
            queue_length = await self.get_current_queue_length()
            queue_position = queue_length + 1
            
            # Estimate wait time (15-20 minutes per demo, assume 17.5 avg)
            estimated_wait_time = queue_position * 17.5 if queue_position > 1 else 0
            
            # Create queue entry
            queue_entry = {
                "session_id": session_id,
                "name": user_info.get("name", ""),
                "email": user_info.get("email", ""),
                "company": user_info.get("company", ""),
                "interest_areas": user_info.get("interest_areas", []),
                "timestamp": datetime.utcnow(),
                "status": "waiting",
                "queue_position": queue_position,
                "estimated_wait_time": int(estimated_wait_time),
                "notes": ""
            }
            
            # Insert into database
            result = self.demo_queue_collection.insert_one(queue_entry)
            
            logger.info(f"Added user {user_info.get('name')} to demo queue with session_id: {session_id}")
            
            return {
                "success": True,
                "session_id": session_id,
                "queue_position": queue_position,
                "estimated_wait_time": int(estimated_wait_time),
                "queue_length": queue_length + 1
            }
            
        except Exception as e:
            logger.error(f"Error adding to demo queue: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_queue_status(self, session_id: str) -> Dict:
        """Get current queue status for a specific user"""
        try:
            if self.demo_queue_collection is None:
                return {"success": False, "error": "Demo queue not available"}
            
            # Find user's queue entry
            user_entry = self.demo_queue_collection.find_one({"session_id": session_id})
            
            if not user_entry:
                return {"success": False, "error": "Session not found in queue"}
            
            # Calculate current position based on waiting entries before this user
            current_position = self.demo_queue_collection.count_documents({
                "status": "waiting",
                "timestamp": {"$lt": user_entry["timestamp"]}
            }) + 1
            
            # Recalculate estimated wait time
            estimated_wait_time = (current_position - 1) * 17.5 if current_position > 1 else 0
            
            return {
                "success": True,
                "session_id": session_id,
                "name": user_entry["name"],
                "status": user_entry["status"],
                "queue_position": current_position,
                "estimated_wait_time": int(estimated_wait_time),
                "total_in_queue": await self.get_current_queue_length()
            }
            
        except Exception as e:
            logger.error(f"Error getting queue status: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_current_queue_length(self) -> int:
        """Get current number of people waiting in demo queue"""
        try:
            if self.demo_queue_collection is None:
                return 0
            return self.demo_queue_collection.count_documents({"status": "waiting"})
        except Exception as e:
            logger.error(f"Error getting queue length: {e}")
            return 0
    
    async def estimate_wait_time(self) -> int:
        """Estimate wait time for new queue entry in minutes"""
        try:
            queue_length = await self.get_current_queue_length()
            # Average demo time is 17.5 minutes
            return int(queue_length * 17.5)
        except Exception as e:
            logger.error(f"Error estimating wait time: {e}")
            return 0
    
    def _validate_email(self, email: str) -> bool:
        """Validate email format"""
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(email_pattern, email) is not None
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID for queue tracking"""
        return str(uuid.uuid4())
    
    # Phase 3: Natural Language Processing for Demo Signups
    def _extract_user_info(self, message: str, current_info: Dict) -> tuple[Dict, bool]:
        """Extract user information from natural language message"""
        try:
            message_lower = message.lower().strip()
            
            # Initialize with current info
            info = current_info.copy()
            
            # Extract name (look for patterns like "my name is", "I'm", "name's")
            name_patterns = [
                r"my name is ([a-zA-Z\s]+)",
                r"i'm ([a-zA-Z\s]+)",
                r"name's ([a-zA-Z\s]+)",
                r"i am ([a-zA-Z\s]+)",
                r"call me ([a-zA-Z\s]+)",
                r"this is ([a-zA-Z\s]+)"
            ]
            
            for pattern in name_patterns:
                name_match = re.search(pattern, message_lower)
                if name_match:
                    potential_name = name_match.group(1).strip().title()
                    # Validate name (should be 2-50 chars, only letters and spaces)
                    if 2 <= len(potential_name) <= 50 and re.match(r'^[a-zA-Z\s]+$', potential_name):
                        info['name'] = potential_name
                        break
            
            # If no name pattern found, check if message might be just a name
            if not info.get('name') and not current_info.get('name'):
                # Check if message is likely just a name (no @ symbol, reasonable length)
                if 2 <= len(message.strip()) <= 50 and '@' not in message and re.match(r'^[a-zA-Z\s]+$', message.strip()):
                    # Could be a name response
                    words = message.strip().split()
                    if 1 <= len(words) <= 4:  # Reasonable name length
                        info['name'] = message.strip().title()
            
            # Handle conversational responses about having email but not wanting to share it yet
            if not info.get('email') and current_info.get('name') and not current_info.get('email'):
                # Check if they mentioned email but didn't provide it
                email_mention_patterns = [
                    r"email",
                    r"e-mail", 
                    r"contact",
                    r"reach me"
                ]
                if any(re.search(pattern, message_lower) for pattern in email_mention_patterns):
                    # They mentioned email - they're responding to email request
                    pass  # Will be handled by the email extraction above
            
            # Extract email
            email_pattern = r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b'
            email_match = re.search(email_pattern, message)
            if email_match:
                potential_email = email_match.group(0).lower()
                if self._validate_email(potential_email):
                    info['email'] = potential_email
            
            # Extract company (look for patterns like "I work at", "from", "company is")
            company_patterns = [
                r"i work at ([a-zA-Z0-9\s&.-]+)",
                r"work for ([a-zA-Z0-9\s&.-]+)",
                r"from ([a-zA-Z0-9\s&.-]+)",
                r"company is ([a-zA-Z0-9\s&.-]+)",
                r"at ([a-zA-Z0-9\s&.-]+)",
                r"with ([a-zA-Z0-9\s&.-]+)"
            ]
            
            for pattern in company_patterns:
                company_match = re.search(pattern, message_lower)
                if company_match:
                    potential_company = company_match.group(1).strip().title()
                    # Validate company name
                    if 2 <= len(potential_company) <= 100:
                        info['company'] = potential_company
                        break
            
            # Check if we have minimum required info (name and email)
            info_complete = bool(info.get('name') and info.get('email'))
            
            return info, info_complete
            
        except Exception as e:
            logger.error(f"Error extracting user info: {e}")
            return current_info, False
    
    def _generate_info_request(self, current_info: Dict) -> str:
        """Generate natural follow-up question based on what info we have"""
        try:
            has_name = bool(current_info.get('name'))
            has_email = bool(current_info.get('email'))
            
            if not has_name and not has_email:
                # Need both name and email
                return (
                    "Great! I'd love to get you signed up for our quantum computing demo. "
                    "To reserve your spot, I'll need your name and email address. What's your name?"
                )
            elif has_name and not has_email:
                # Have name, need email
                name = current_info['name']
                return (
                    f"Perfect, {name}! Now I'll need your email address to complete your demo signup. "
                    f"What email should I use for your reservation?"
                )
            elif not has_name and has_email:
                # Have email, need name  
                return (
                    "Thanks for the email! And what's your name so I can properly add you to our demo queue?"
                )
            else:
                # This shouldn't happen if info_complete logic is correct
                return (
                    "I think I have your information! Let me confirm the details with you."
                )
                
        except Exception as e:
            logger.error(f"Error generating info request: {e}")
            return "Could you please share your name and email address for the demo signup?"
    
    def _detect_signup_intent(self, message: str) -> bool:
        """Detect if user wants to sign up for demo"""
        message_lower = message.lower().strip()
        
        # Direct affirmative responses
        direct_yes = ['yes', 'yeah', 'yep', 'sure', 'ok', 'okay', 'alright', 'absolutely', 'definitely', 'of course']
        if any(message_lower == keyword or message_lower.startswith(keyword + ' ') for keyword in direct_yes):
            return True
            
        # Signup-specific phrases
        signup_keywords = [
            'sign me up', 'sign up', 'register', 'join', 'add me', 'reserve', 'book', 
            'interested', 'i want', "i'd like", 'count me in', 'put me', 'include me',
            'reserve a spot', 'get me signed up', 'add me to', 'put me in'
        ]
        return any(keyword in message_lower for keyword in signup_keywords)
    
    def _detect_queue_status_intent(self, message: str) -> bool:
        """Detect if user is asking about queue status"""
        message_lower = message.lower()
        status_keywords = [
            'queue', 'wait', 'status', 'position', 'how long', 'when', 'time',
            'check', 'update', 'where am i', 'my turn', 'next'
        ]
        return any(keyword in message_lower for keyword in status_keywords)
    
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
        "message": "iQore Multi-Agent ChatBot Backend with Phase 3 Demo Queue",
        "status": "healthy",
        "version": "3.0.0",
        "agents": ["supervisor", "demo_agent", "contact_agent", "technical_agent", "business_agent"],
        "features": ["natural_conversation", "demo_queue_management", "user_info_extraction", "vector_search"],
        "demo_capabilities": ["natural_signup", "queue_tracking", "status_updates", "multi_turn_conversations"],
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
    
    # Phase 2: Include demo queue status in health check
    try:
        queue_length = await chatbot_service.get_current_queue_length() if chatbot_service else 0
    except Exception as e:
        logger.warning(f"Could not get demo queue length: {e}")
        queue_length = 0
    
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
        queue_count = await chatbot_service.get_current_queue_length() if chatbot_service else 0
    except Exception as e:
        logger.warning(f"Could not get document/queue count for status: {e}")
        doc_count = 0
        queue_count = 0
    
    return {
        "api_version": "v1",
        "status": "active",
        "message": "Multi-agent backend with Phase 3 demo queue ready",
        "document_count": doc_count,
        "demo_queue_length": queue_count,
        "agents": ["supervisor", "demo_agent", "contact_agent", "technical_agent", "business_agent"],
        "features": ["natural_demo_signup", "queue_management", "user_info_extraction", "conversation_flow"]
    }

# Phase 1: Enhanced Demo Queue API Endpoints
@app.get("/api/v1/demo/queue-status", response_model=QueueStatusResponse)
async def get_demo_queue_status() -> QueueStatusResponse:
    """Get comprehensive demo queue status"""
    try:
        queue_length = await chatbot_service.get_current_queue_length()
        estimated_wait = await chatbot_service.estimate_wait_time()
        in_progress = await chatbot_service.get_in_progress_count()
        
        return QueueStatusResponse(
            success=True,
            total_queue_length=queue_length,
            estimated_wait_time_minutes=estimated_wait,
            current_demo_in_progress=in_progress > 0,
            message="Queue status retrieved successfully" if queue_length > 0 else "Queue is currently empty"
        )
    except Exception as e:
        logger.error(f"Error getting queue status: {e}")
        return QueueStatusResponse(
            success=False,
            total_queue_length=0,
            estimated_wait_time_minutes=0,
            current_demo_in_progress=False,
            error=str(e)
        )

@app.get("/api/v1/demo/queue/{session_id}", response_model=UserQueueStatusResponse)
async def get_user_queue_status(session_id: str) -> UserQueueStatusResponse:
    """Get queue status for specific user session"""
    try:
        status = await chatbot_service.get_queue_status(session_id)
        
        if status["success"]:
            return UserQueueStatusResponse(
                success=True,
                session_id=status["session_id"],
                name=status["name"],
                status=DemoStatus(status["status"]),
                queue_position=status["queue_position"],
                estimated_wait_time=status["estimated_wait_time"],
                total_in_queue=status["total_in_queue"],
                message=f"You are #{status['queue_position']} in queue"
            )
        else:
            return UserQueueStatusResponse(
                success=False,
                error=status["error"]
            )
    except Exception as e:
        logger.error(f"Error getting user queue status: {e}")
        return UserQueueStatusResponse(success=False, error=str(e))

@app.post("/api/v1/demo/request", response_model=UserQueueStatusResponse)
async def create_demo_request(request: DemoRequest) -> UserQueueStatusResponse:
    """Create a new demo request and add to queue"""
    try:
        user_info = {
            "name": request.visitor_info.name,
            "email": request.visitor_info.email,
            "company": request.visitor_info.company,
            "interest_areas": request.visitor_info.interest_areas
        }
        result = await chatbot_service.add_to_demo_queue(user_info)
        
        if result["success"]:
            return UserQueueStatusResponse(
                success=True,
                session_id=result["session_id"],
                name=user_info["name"],
                status=DemoStatus.WAITING,
                queue_position=result["queue_position"],
                estimated_wait_time=result["estimated_wait_time"],
                total_in_queue=result["queue_length"],
                message=f"Successfully added to demo queue. You are #{result['queue_position']} in line."
            )
        else:
            return UserQueueStatusResponse(
                success=False,
                error=result["error"]
            )
    except Exception as e:
        logger.error(f"Error creating demo request: {e}")
        return UserQueueStatusResponse(success=False, error=str(e))

# Staff Management Endpoints
@app.get("/api/v1/staff/demo/queue", response_model=StaffQueueResponse)
async def get_staff_queue_view() -> StaffQueueResponse:
    """Get comprehensive queue view for staff management"""
    try:
        queue_data = await chatbot_service.get_full_queue_status()
        return StaffQueueResponse(**queue_data)
    except Exception as e:
        logger.error(f"Error getting staff queue view: {e}")
        return StaffQueueResponse(
            success=False,
            total_entries=0,
            waiting_count=0,
            in_progress_count=0,
            completed_today=0,
            error=str(e)
        )

@app.put("/api/v1/staff/demo/queue/{session_id}", response_model=UserQueueStatusResponse)
async def update_demo_status(session_id: str, update_request: StaffQueueUpdateRequest) -> UserQueueStatusResponse:
    """Update demo status (staff only)"""
    try:
        result = await chatbot_service.update_demo_status(
            session_id=session_id,
            new_status=update_request.new_status.value,
            notes=update_request.notes
        )
        
        if result["success"]:
            return UserQueueStatusResponse(
                success=True,
                session_id=session_id,
                status=update_request.new_status,
                message=f"Demo status updated to {update_request.new_status.value}"
            )
        else:
            return UserQueueStatusResponse(
                success=False,
                error=result["error"]
            )
    except Exception as e:
        logger.error(f"Error updating demo status: {e}")
        return UserQueueStatusResponse(success=False, error=str(e))

@app.delete("/api/v1/staff/demo/queue/{session_id}")
async def remove_from_queue(session_id: str) -> Dict[str, any]:
    """Remove entry from demo queue (staff only)"""
    try:
        result = await chatbot_service.remove_from_queue(session_id)
        return result
    except Exception as e:
        logger.error(f"Error removing from queue: {e}")
        return {"success": False, "error": str(e)}

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