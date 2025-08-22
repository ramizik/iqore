from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import uvicorn
from typing import Dict, List, Optional, Literal, Annotated, Any
import logging
import asyncio
from datetime import datetime, timedelta
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
    phone: Optional[str] = None
    interest_areas: List[str] = []
    
    @validator('name')
    def validate_name(cls, v):
        if not v or len(v.strip()) < 2:
            raise ValueError('Name must be at least 2 characters')
        if len(v) > 50:
            raise ValueError('Name must be less than 50 characters')
        return v.strip().title()
    
    @validator('phone')
    def validate_phone(cls, v):
        if v and len(v.strip()) > 0:
            # Basic phone validation - remove non-digits and check length
            digits_only = ''.join(filter(str.isdigit, v))
            if len(digits_only) < 10:
                raise ValueError('Phone number must contain at least 10 digits')
        return v.strip() if v else None

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
    phone: Optional[str] = None
    interest_areas: List[str] = []
    status: DemoStatus
    timestamp: datetime
    queue_position: Optional[int] = None
    notes: Optional[str] = None

class QueueStatusResponse(BaseModel):
    success: bool
    total_queue_length: int
    current_demo_in_progress: bool
    message: Optional[str] = None
    error: Optional[str] = None

class UserQueueStatusResponse(BaseModel):
    success: bool
    session_id: Optional[str] = None
    name: Optional[str] = None
    status: Optional[DemoStatus] = None
    queue_position: Optional[int] = None
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
                    result = {
                        "success": True,
                        "queue_length": queue_length
                    }
                return json.dumps(result)
                
            elif action == "get_overall_status":
                queue_length = await self.chatbot_service.get_current_queue_length()
                in_progress = await self.chatbot_service.get_in_progress_count()
                result = {
                    "success": True,
                    "total_queue_length": queue_length,
                    "demos_in_progress": in_progress
                }
                return json.dumps(result)
                
            else:
                return json.dumps({"success": False, "error": f"Unknown action: {action}"})
                
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})

    def _run(self, action_input: str) -> str:
        """Synchronous fallback - not implemented for async operations"""
        import json
        return json.dumps({"success": False, "error": "This tool requires async execution"})





# LangGraph State for multi-agent system
class AgentState(TypedDict):
    """State shared between all agents in the system"""
    messages: Annotated[List[BaseMessage], operator.add]
    next: str  # Which agent to call next
    user_intent: str  # Detected user intent (demo, contact, technical, business)
    lead_info: Dict  # Information about potential leads
    chat_history: List[Dict[str, str]]  # Chat history for context
    # Phase 1: Demo-specific state management
    demo_state: str  # "initial", "queued"
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
        # Initialize demo_done collection reference
        self.demo_done_collection = None
        # Initialize chat history collection reference
        self.chat_history_collection = None
        self.multi_agent_graph = None
        self.llm = None
        # Phase 2: Initialize LangChain tools
        self.demo_queue_tool = None
        
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
            # Initialize demo_done collection for completed/removed demos
            self.demo_done_collection = self.db['demo_done']
            # Initialize chat history collection
            self.chat_history_collection = self.db['chat_history']
            
            # Check if there are documents in the collection
            doc_count = self.pdf_chunks_collection.count_documents({})
            if doc_count == 0:
                logger.warning("No documents found in database. Technical agent will work with limited context.")
                
            logger.info(f"Found {doc_count} document chunks in database")
            
            # Phase 2: Check demo queue collection
            demo_queue_count = self.demo_queue_collection.count_documents({})
            logger.info(f"Demo queue initialized with {demo_queue_count} existing entries")
            
            # Check demo_done collection
            demo_done_count = self.demo_done_collection.count_documents({})
            logger.info(f"Demo done collection initialized with {demo_done_count} completed entries")
            
            # Check chat history collection
            chat_history_count = self.chat_history_collection.count_documents({})
            logger.info(f"Chat history collection initialized with {chat_history_count} existing entries")
            
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
                "You are an experienced iQore sales professional and quantum computing expert having a natural conversation with booth visitors about our quantum computing innovations. Act professionally and proactively as a confident sales manager showcasing our quantum solutions.\n\n"
                "SALES APPROACH:\n"
                "Present iQore as the leading quantum-classical hybrid solution with proven competitive advantages. Emphasize measurable performance improvements and ROI benefits. Use conversational language while maintaining professional sales expertise. Focus on business impact and practical applications. Keep responses concise since visitors have limited time.\n\n"
                "CONVERSATION STYLE:\n"
                "Talk naturally like you're explaining to a colleague or business prospect in person. Use flowing sentences without any markdown formatting, bullet points, bold text, numbered lists, or special characters. Keep paragraphs short and conversational. Use natural transitions and ask follow-up questions naturally in conversation.\n\n"
                "KNOWLEDGE AREAS:\n"
                "I can discuss iQore Tech Stack including iQD technology, Common Misconceptions, Competitor Differentiators, FAQ, and our Patent Portfolio including SVE Core, Coherence Model, Dynamic Tensor Controller, Quantum Circuit Operations, and SVE Base technologies.\n\n"
                "ACCURACY REQUIREMENTS:\n"
                "For general quantum computing use your knowledge but keep brief and relevant. For iQore-specific questions only use verified information about our technology. When uncertain about iQore details suggest connecting with our technical team rather than speculating. Never mention backend systems or data sources.\n\n"
                "CONTEXT:\n{context}\n\n"
                "Guide conversations toward our solutions, competitive advantages, and demo opportunities. Position iQore as the optimal choice for quantum optimization challenges."
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
            
            logger.info("✅ LangChain tools initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing LangChain tools: {e}")
            self.demo_queue_tool = None
    
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
        
        # Demo intent detection - enhanced for on-site demo mentions
        demo_keywords = ['demo', 'demonstration', 'see the demo', 'live demo']
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
        Now with conversation context awareness for demo flows and welcome message
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
            
            # Check if this is the very first interaction (no previous AI messages)
            ai_messages = [msg for msg in state["messages"] if isinstance(msg, AIMessage)]
            if len(ai_messages) == 0:
                # First interaction - provide welcome message
                welcome_message = (
                    "Welcome to iQore! I'm your quantum computing specialist, and I'm here to show you how our breakthrough technology can transform your quantum applications.\n\n"
                    "As an experienced quantum solutions advisor, I can discuss our competitive advantages in quantum-classical hybrid computing, including our patent-pending iQD optimization technology that delivers measurable performance improvements over baseline quantum circuits.\n\n"
                    "I'm ready to cover our technology stack, industry applications, ROI benefits, and how we outperform competitors. Most importantly, I can arrange our quantum computing demonstration where you'll see real-time performance comparisons on both simulators and actual quantum hardware.\n\n"
                    "We have demo options available for you. You can join our live demo queue now if there's availability for immediate hands-on experience, or schedule the same demonstration for a time that works best for you.\n\n"
                    "What specific quantum computing challenges can I help you solve today?"
                )
                
                agent_message = AIMessage(content=welcome_message, name="supervisor")
                state["messages"].append(agent_message)
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
            elif demo_state == "queued":
                logger.info("Demo agent: In queued state")
                return await self._demo_queue_status_stage(state)
            else:
                # Fallback to initial stage for all other states since we're using UI for signup
                logger.info(f"Demo agent: State '{demo_state}', handling via initial stage with UI guidance")
                return await self._demo_initial_stage(state)
                
        except Exception as e:
            logger.error(f"Error in demo agent: {e}")
            error_message = AIMessage(
                content="I'd love to help you with our quantum computing demonstration! We offer the same hands-on experience either through our live demo queue (subject to availability) or by scheduling for a convenient time. How can I assist you?",
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
                     "You are an experienced iQore sales specialist focused on showcasing our quantum computing solutions through live demonstrations. Your goal is to professionally present our competitive advantages and guide users to the scheduling interface.\n\n"
                     "Demo Value Proposition:\n"
                     "Our demonstrations prove iQore's quantum optimization superiority with measurable performance improvements. Visitors see real-time comparisons between standard quantum circuits and our iQD-optimized versions, with concrete metrics on fidelity gains, gate count reductions, and execution time improvements.\n\n"
                     "Two Demo Scheduling Options (Same Experience):\n"
                     "Live Demo Queue: Join now for immediate 10-15 minute hands-on experience showing quantum algorithms like QAOA, VQE, Quantum Volume, and Grover's executed on both simulators and actual quantum computers with performance comparisons.\n"
                     "Schedule Demo Session: Book the exact same demonstration experience for a convenient time that works best for your schedule.\n\n"
                     "CRITICAL UI Guidance:\n"
                     "Always direct users to the scheduling window that appears next to this chat. Do NOT handle scheduling through text conversation. The user must use the browser interface to select their preferred option.\n\n"
                     "Communication Approach:\n"
                     "- Act as a confident sales professional highlighting our competitive advantages\n"
                     "- Keep responses concise and business-focused\n"
                     "- Emphasize measurable ROI and performance benefits\n"
                     "- Position iQore as the leading quantum optimization solution\n"
                     "- CRITICAL: Write in natural conversational style without any markdown formatting, bullet points, bold text, or special characters. Use flowing sentences like you're speaking to someone in person.\n"
                     "- Always end with: You should see our demo scheduling window appear next to this chat window where you can choose between joining the live queue for immediate experience or scheduling for your preferred time. If you're on mobile, you might need to scroll down to see the scheduling options. Both offer the identical hands-on quantum computing demonstration.\n\n"
                     "For hesitant prospects speak naturally: Our technical team will demonstrate the exact performance metrics and explain how these improvements translate to real-world applications in your industry. Use the scheduling window next to this chat to select your preferred demo timing."),
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
                    # User wants to sign up - direct them to the UI instead of text-based signup
                    current_queue_length = await self.get_current_queue_length()
                    wait_time = await self.estimate_wait_time()
                    
                    response_content = (
                        f"Excellent! I can see you're ready to experience our quantum computing demonstration.\n\n"
                        f"Please use the demo scheduling window that should appear next to this chat where you can choose your preferred option. If you're on mobile, you might need to scroll down to see the scheduling interface.\n\n"
                        f"Current live queue status: {current_queue_length} people waiting, approximately {wait_time} minutes wait time for immediate experience. The scheduled option offers the identical demonstration at your convenience.\n\n"
                        f"Both options provide the same 10-15 minute hands-on experience with our quantum optimization technology, showing real performance comparisons on simulators and actual quantum hardware."
                    )
                    # Don't change state - keep in initial for continued conversation
                    
                    # Log the interaction
                    logger.info(f"Demo agent: User wants signup, directing to UI interface")
                    
                elif self._detect_queue_status_intent(user_message):
                    # User asking about queue - provide general info and direct to UI
                    current_queue_length = await self.get_current_queue_length()
                    wait_time = await self.estimate_wait_time()
                    
                    response_content = (
                        f"Here's the current live demo queue status: {current_queue_length} people waiting with approximately {wait_time} minutes estimated wait time for immediate hands-on experience.\n\n"
                        f"You can use the demo scheduling window next to this chat to choose between joining the live queue or scheduling for your preferred time. If you're on mobile, scroll down to see the scheduling options.\n\n"
                        f"Both options offer the identical 10-15 minute demonstration featuring real-time quantum performance comparisons on simulators and actual quantum computers, showing exactly why iQore delivers superior optimization results."
                    )
                    
                else:
                    # General follow-up or questions about demo
                    demo_followup_prompt = ChatPromptTemplate.from_messages([
                        ("system",
                         "You are an experienced iQore sales specialist addressing questions about our quantum computing demonstrations. "
                         "Act professionally as a confident sales expert who understands our competitive advantages and ROI benefits. "
                         "Our demos prove iQore's quantum optimization superiority with measurable performance improvements - visitors see real-time comparisons with concrete fidelity, gate count, and execution time metrics. "
                         "Keep responses concise and business-focused, emphasizing competitive differentiation and business value. "
                         "CRITICAL: Write in natural conversational style without any markdown formatting, bullet points, bold text, or special characters. Speak like you're having a professional conversation in person. "
                         "CRITICAL UI Guidance: Always direct users to use the demo scheduling window that appears next to this chat for selecting their demo timing. Do NOT handle scheduling through text conversation. "
                         "Always end responses with: You can use the demo scheduling window next to this chat to select between immediate live queue or scheduled demo timing. If you're on mobile, you might need to scroll down to see the scheduling options. Both provide the identical hands-on quantum computing experience."),
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
                content="I'd love to show you our quantum computing demonstration! You can join our live demo queue (subject to availability) for immediate experience, or schedule the same demo for a convenient time later. Which timing option works better for you?",
                name="demo_agent"
            )
            state["messages"].append(error_message)
            state["next"] = "demo_agent"
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
                            f"• Total people in queue: {total_queue}\n\n"
                            f"🔔 You're coming up soon! Stay nearby - we'll call your name when it's your turn."
                        )
                    else:
                        response_content = (
                            f"📊 **Queue Update for {name}:**\n"
                            f"• Current position: #{position}\n"
                            f"• Total people in queue: {total_queue}\n\n"
                            f"⏰ You have some time to explore other booths! "
                            f"Feel free to ask me for another update anytime."
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
                    # General queue inquiry - direct to UI
                    queue_length = await self.get_current_queue_length()
                    wait_time = await self.estimate_wait_time()
                    
                    response_content = (
                        f"Current demo queue status: {queue_length} people waiting with approximately {wait_time} minutes estimated wait time.\n\n"
                        f"You can join using the demo scheduling window next to this chat where you can choose between the live queue for immediate experience or scheduling for your preferred time. If you're on mobile, you might need to scroll down to see the scheduling options."
                    )
                else:
                    # User might want to sign up - direct to UI
                    response_content = (
                        "I don't see you in our demo queue yet. You can sign up using the demo scheduling window that appears next to this chat. "
                        "Choose between joining the live queue for immediate experience or scheduling for a convenient time. Both offer the identical hands-on quantum computing demonstration!"
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
                 "You are an experienced iQore sales professional facilitating connections between qualified prospects and our solutions team. Act confidently as a senior sales manager focused on driving business outcomes.\n\n"
                 "Qualification Focus:\n"
                 "Identify decision-makers and technical evaluators who can benefit from our quantum optimization advantages. Prioritize prospects with quantum computing initiatives, optimization challenges, or performance requirements where iQore delivers competitive differentiation.\n\n"
                 "Key Information to Capture:\n"
                 "Name, company, and decision-making authority. Industry sector and specific quantum computing applications. Current quantum challenges or performance bottlenecks. Timeline for quantum computing implementation. Contact details for follow-up.\n\n"
                 "Communication Approach:\n"
                 "- Sound professional and consultative like an experienced sales executive\n"
                 "- Keep responses brief and business-focused\n"
                 "- Position follow-up as valuable consultation, not just information sharing\n"
                 "- Emphasize our competitive advantages and proven ROI\n"
                 "- CRITICAL: Write in natural conversational style without any markdown formatting, bullet points, bold text, or special characters. Speak like you're having a professional business conversation.\n"
                 "- Guide prospects naturally: I'll connect you with our solutions team for a strategic discussion about how iQore delivers quantum optimization advantages in your industry. Meanwhile, our quantum computing demonstration shows the performance improvements firsthand - you can access the demo scheduling window next to this chat to choose immediate queue or scheduled timing."),
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
                content="As your iQore sales professional, I'm ready to connect you with our solutions team for a strategic consultation about quantum optimization advantages in your industry. Please share your contact details and I'll arrange a valuable discussion with our specialists.",
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
                     "You are an experienced iQore sales professional and quantum computing expert engaging with booth visitors. Act professionally and proactively as a confident sales manager showcasing our quantum solutions.\n\n"
                     "Sales Approach:\n"
                     "Present iQore as the leading quantum-classical hybrid solution with proven competitive advantages. Emphasize measurable performance improvements and ROI benefits. Use conversational language while maintaining professional sales expertise. Focus on business impact and practical applications. Keep responses concise since visitors have limited time.\n\n"
                     "Accuracy Requirements:\n"
                     "For general quantum computing use your knowledge but keep brief and relevant. For iQore-specific questions only use verified information about our technology. When uncertain about iQore details suggest connecting with our technical team rather than speculating. Never mention backend systems or data sources.\n\n"
                     "CRITICAL: Write in natural conversational style without any markdown formatting, bullet points, bold text, numbered lists, or special characters. Speak like you're having a professional conversation in person.\n\n"
                     "Guide conversations toward our solutions, competitive advantages, and demo opportunities. Position iQore as the optimal choice for quantum optimization challenges. When appropriate, mention that visitors can see our demonstration using the scheduling window next to this chat for immediate or scheduled timing."),
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
                content="As your iQore quantum computing specialist, I'm ready to discuss our competitive advantages, performance benefits, and technical capabilities. What specific quantum computing challenges can I help you solve?",
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
                 "You are a senior iQore business development executive and quantum computing strategist engaging with enterprise decision-makers. Act as an experienced sales professional focused on demonstrating competitive business advantages and driving purchasing decisions.\n\n"
                 "Sales Positioning:\n"
                 "Position iQore as the premier quantum optimization solution delivering measurable ROI through superior performance. Emphasize our competitive differentiation: while others provide basic quantum computing, iQore delivers optimization that dramatically improves fidelity, reduces gate counts, and accelerates execution times.\n\n"
                 "Industry Value Propositions:\n"
                 "Financial Services gets portfolio optimization with provably better quantum advantage and risk analysis with higher accuracy. Pharmaceuticals benefits from accelerated drug discovery through optimized molecular simulation and competitive time-to-market advantages. Manufacturing achieves supply chain optimization with superior quantum performance and predictive maintenance with enhanced accuracy. Energy companies see grid optimization with measurable efficiency gains and renewable integration with proven quantum advantages. Automotive industry gains route optimization at scale and materials research with competitive quantum performance.\n\n"
                 "Communication Approach:\n"
                 "Act as a confident senior sales executive with deep industry expertise. Focus on competitive differentiation, ROI metrics, and implementation advantages. Keep responses concise and business-focused since executives have limited time. Emphasize proven results and customer success stories.\n\n"
                 "CRITICAL: Write in natural conversational style without any markdown formatting, bullet points, bold text, numbered lists, or special characters. Speak like you're having a strategic business conversation with an executive.\n\n"
                 "Guide prospects naturally: Our quantum computing demonstration proves these competitive advantages with real performance metrics. You'll see exactly why leading organizations choose iQore over alternatives. Use the demo scheduling window next to this chat to select immediate queue or scheduled timing - both offer identical hands-on experience with our optimization benefits.\n\n"
                 "For complex strategic discussions speak naturally: Our executive team is here to discuss enterprise implementation strategies and partnership opportunities tailored to your organization's quantum computing roadmap."),
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
                content="As your iQore business development specialist, I'm here to demonstrate how our quantum optimization solutions deliver competitive advantages and measurable ROI across industries. What quantum computing applications or challenges would you like to explore?",
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
                    phone=entry.get("phone"),
                    interest_areas=entry.get("interest_areas", []),
                    status=DemoStatus(entry["status"]),
                    timestamp=entry["timestamp"],
                    queue_position=position if entry["status"] == "waiting" else None,
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
        """Remove entry from demo queue and save to demo_done collection"""
        try:
            if self.demo_queue_collection is None:
                return {"success": False, "error": "Demo queue collection not available"}
            
            if self.demo_done_collection is None:
                return {"success": False, "error": "Demo done collection not available"}
            
            # First, get the entry to save its data
            entry = self.demo_queue_collection.find_one({"session_id": session_id})
            
            if not entry:
                return {"success": False, "error": "Session not found in queue"}
            
            # Prepare the entry for demo_done collection
            demo_done_entry = {
                "session_id": entry.get("session_id"),
                "name": entry.get("name"),
                "email": entry.get("email"),
                "company": entry.get("company", ""),
                "phone": entry.get("phone", ""),
                "interest_areas": entry.get("interest_areas", []),
                "original_timestamp": entry.get("timestamp"),  # When they originally joined queue
                "queue_position": entry.get("queue_position"),
                "status": entry.get("status"),  # Their status when removed
                "notes": entry.get("notes", ""),
                "created_via": entry.get("created_via", "unknown"),
                "completed_at": datetime.utcnow(),  # When they were removed/completed
                "removed_by": "admin"  # Indicate this was an admin action
            }
            
            # Save to demo_done collection
            save_result = self.demo_done_collection.insert_one(demo_done_entry)
            
            if not save_result.inserted_id:
                logger.error(f"Failed to save demo entry to demo_done collection for session {session_id}")
                return {"success": False, "error": "Failed to save demo completion data"}
            
            logger.info(f"Saved demo entry to demo_done collection: {session_id}, user: {entry.get('name')}")
            
            # Now remove the entry from the queue
            result = self.demo_queue_collection.delete_one({"session_id": session_id})
            
            if result.deleted_count > 0:
                return {
                    "success": True,
                    "message": f"Entry removed from demo queue and saved to completion records",
                    "session_id": session_id,
                    "user_name": entry.get("name"),
                    "saved_to_done": True
                }
            else:
                # If deletion failed, we should probably remove from demo_done as well
                # to maintain consistency
                self.demo_done_collection.delete_one({"session_id": session_id, "completed_at": demo_done_entry["completed_at"]})
                return {"success": False, "error": "Failed to remove from queue after saving completion data"}
                
        except Exception as e:
            logger.error(f"Error removing from queue: {e}")
            return {"success": False, "error": str(e)}

    async def get_demo_done_stats(self) -> Dict:
        """Get statistics from demo_done collection"""
        try:
            if self.demo_done_collection is None:
                return {"success": False, "error": "Demo done collection not available"}
            
            # Get total completed demos
            total_completed = self.demo_done_collection.count_documents({})
            
            # Get completed demos today
            today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
            today_completed = self.demo_done_collection.count_documents({
                "completed_at": {"$gte": today_start}
            })
            
            # Get completed demos this week
            week_start = today_start - timedelta(days=7)
            week_completed = self.demo_done_collection.count_documents({
                "completed_at": {"$gte": week_start}
            })
            
            return {
                "success": True,
                "total_completed": total_completed,
                "completed_today": today_completed,
                "completed_this_week": week_completed,
                "collection_available": True
            }
            
        except Exception as e:
            logger.error(f"Error getting demo done stats: {e}")
            return {"success": False, "error": str(e)}

    async def save_chat_history(self, user_message: str, assistant_response: str, session_id: Optional[str] = None) -> Dict:
        """Save chat interaction to MongoDB chat_history collection"""
        try:
            if self.chat_history_collection is None:
                return {"success": False, "error": "Chat history collection not available"}
            
            # Create chat history entry
            chat_entry = {
                "session_id": session_id or str(uuid.uuid4()),
                "user_message": user_message,
                "assistant_response": assistant_response,
                "timestamp": datetime.utcnow(),
                "message_length": len(user_message),
                "response_length": len(assistant_response)
            }
            
            # Insert into database
            result = self.chat_history_collection.insert_one(chat_entry)
            
            if result.inserted_id:
                logger.info(f"Chat history saved successfully with ID: {result.inserted_id}")
                return {
                    "success": True,
                    "message": "Chat history saved successfully",
                    "entry_id": str(result.inserted_id)
                }
            else:
                return {"success": False, "error": "Failed to save chat history"}
                
        except Exception as e:
            logger.error(f"Error saving chat history: {e}")
            return {"success": False, "error": str(e)}

    async def get_chat_history_stats(self) -> Dict:
        """Get statistics from chat_history collection"""
        try:
            if self.chat_history_collection is None:
                return {"success": False, "error": "Chat history collection not available"}
            
            # Get total chat interactions
            total_chats = self.chat_history_collection.count_documents({})
            
            # Get chats today
            today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
            today_chats = self.chat_history_collection.count_documents({
                "timestamp": {"$gte": today_start}
            })
            
            # Get chats this week
            week_start = today_start - timedelta(days=7)
            week_chats = self.chat_history_collection.count_documents({
                "timestamp": {"$gte": week_start}
            })
            
            return {
                "success": True,
                "total_chat_interactions": total_chats,
                "chats_today": today_chats,
                "chats_this_week": week_chats,
                "collection_available": True
            }
            
        except Exception as e:
            logger.error(f"Error getting chat history stats: {e}")
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
            logger.info(f"Adding user to demo queue: {user_info}")
            
            if self.demo_queue_collection is None:
                logger.error("Demo queue collection not initialized")
                raise Exception("Demo queue collection not initialized")
            
            # Validate required user info
            if not user_info.get("name") or not user_info.get("email"):
                logger.error(f"Missing required user info: {user_info}")
                raise Exception("Name and email are required")
            
            # Test MongoDB connection
            try:
                self.demo_queue_collection.find_one()
                logger.info("MongoDB connection test successful")
            except Exception as db_test_error:
                logger.error(f"MongoDB connection test failed: {db_test_error}")
                raise Exception(f"Database connection failed: {str(db_test_error)}")
            
            # Generate unique session ID
            session_id = str(uuid.uuid4())
            logger.info(f"Generated session ID: {session_id}")
            
            # Get current queue position
            queue_length = await self.get_current_queue_length()
            queue_position = queue_length + 1
            logger.info(f"Current queue length: {queue_length}, new position: {queue_position}")
            
            # Create queue entry
            queue_entry = {
                "session_id": session_id,
                "name": user_info.get("name", "").strip(),
                "email": user_info.get("email", "").strip().lower(),
                "company": user_info.get("company", "").strip() if user_info.get("company") else "",
                "phone": user_info.get("phone", "").strip() if user_info.get("phone") else "",
                "interest_areas": user_info.get("interest_areas", []),
                "timestamp": datetime.utcnow(),
                "status": "waiting",
                "queue_position": queue_position,
                "notes": "",
                "created_via": "direct_api"  # Track how the entry was created
            }
            
            logger.info(f"Queue entry to insert: {queue_entry}")
            
            # Insert into database
            result = self.demo_queue_collection.insert_one(queue_entry)
            
            if result.inserted_id:
                logger.info(f"Successfully inserted queue entry with MongoDB _id: {result.inserted_id}")
                
                # Verify the insertion
                verification = self.demo_queue_collection.find_one({"session_id": session_id})
                if verification:
                    logger.info(f"Verification successful: Queue entry exists in database")
                else:
                    logger.error("Verification failed: Queue entry not found after insertion")
                    raise Exception("Failed to verify queue entry insertion")
                
            else:
                logger.error("Insert operation did not return an inserted_id")
                raise Exception("Database insertion failed")
            
            logger.info(f"Successfully added {user_info.get('name')} to demo queue at position {queue_position}")
            
            return {
                "success": True,
                "session_id": session_id,
                "queue_position": queue_position,
                "queue_length": queue_length + 1,
                "name": user_info.get("name"),
                "message": f"Added to queue at position #{queue_position}"
            }
            
        except Exception as e:
            logger.error(f"Error adding to demo queue: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "details": f"Failed to add {user_info.get('name', 'user')} to queue"
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
            
            return {
                "success": True,
                "session_id": session_id,
                "name": user_entry["name"],
                "status": user_entry["status"],
                "queue_position": current_position,
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
        """Estimate wait time in minutes based on queue length"""
        try:
            queue_length = await self.get_current_queue_length()
            in_progress_count = await self.get_in_progress_count()
            
            # Base calculation: 15 minutes per demo (including transition time)
            minutes_per_demo = 15
            
            # If someone is currently in progress, they'll finish soon
            # If no one in progress, add base time to start first demo
            if in_progress_count > 0:
                estimated_minutes = queue_length * minutes_per_demo
            else:
                estimated_minutes = (queue_length * minutes_per_demo) + 5  # Add 5 min to start
            
            # Minimum wait time of 5 minutes, maximum of 60 minutes for display
            return max(5, min(estimated_minutes, 60))
            
        except Exception as e:
            logger.error(f"Error estimating wait time: {e}")
            return 10  # Default fallback

    
    def _validate_email(self, email: str) -> bool:
        """Validate email format"""
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(email_pattern, email) is not None
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID for queue tracking"""
        return str(uuid.uuid4())
    

    

    
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

@app.get("/api/v1/debug/db-test")
async def database_connection_test() -> Dict[str, Any]:
    """Test database connections for debugging"""
    try:
        results = {}
        
        if not chatbot_service:
            return {"success": False, "error": "Chatbot service not initialized"}
        
        # Test MongoDB connection
        try:
            if chatbot_service.mongo_client:
                # Test connection
                chatbot_service.mongo_client.admin.command('ping')
                results["mongodb_connection"] = "✅ Connected"
                
                # Test database access
                db_name = chatbot_service.db.name if chatbot_service.db else "Unknown"
                results["database_name"] = db_name
                
                # Test pdf_chunks collection
                if chatbot_service.pdf_chunks_collection:
                    pdf_count = chatbot_service.pdf_chunks_collection.count_documents({})
                    results["pdf_chunks_count"] = pdf_count
                else:
                    results["pdf_chunks_collection"] = "❌ Not initialized"
                
                # Test demo_queue collection
                if chatbot_service.demo_queue_collection:
                    queue_count = chatbot_service.demo_queue_collection.count_documents({})
                    results["demo_queue_count"] = queue_count
                    
                    # Test a simple insert and delete
                    test_doc = {
                        "test": True,
                        "timestamp": datetime.utcnow(),
                        "session_id": "test-connection"
                    }
                    insert_result = chatbot_service.demo_queue_collection.insert_one(test_doc)
                    if insert_result.inserted_id:
                        results["demo_queue_insert_test"] = "✅ Insert successful"
                        
                        # Clean up test document
                        delete_result = chatbot_service.demo_queue_collection.delete_one({"_id": insert_result.inserted_id})
                        if delete_result.deleted_count == 1:
                            results["demo_queue_delete_test"] = "✅ Delete successful"
                        else:
                            results["demo_queue_delete_test"] = "⚠️ Delete failed"
                    else:
                        results["demo_queue_insert_test"] = "❌ Insert failed"
                else:
                    results["demo_queue_collection"] = "❌ Not initialized"
                
                # Test chat_history collection
                if chatbot_service.chat_history_collection:
                    chat_history_count = chatbot_service.chat_history_collection.count_documents({})
                    results["chat_history_count"] = chat_history_count
                    
                    # Test a simple insert and delete for chat history
                    test_chat_doc = {
                        "test": True,
                        "user_message": "test message",
                        "assistant_response": "test response",
                        "timestamp": datetime.utcnow(),
                        "session_id": "test-chat-connection"
                    }
                    insert_result = chatbot_service.chat_history_collection.insert_one(test_chat_doc)
                    if insert_result.inserted_id:
                        results["chat_history_insert_test"] = "✅ Insert successful"
                        
                        # Clean up test document
                        delete_result = chatbot_service.chat_history_collection.delete_one({"_id": insert_result.inserted_id})
                        if delete_result.deleted_count == 1:
                            results["chat_history_delete_test"] = "✅ Delete successful"
                        else:
                            results["chat_history_delete_test"] = "⚠️ Delete failed"
                    else:
                        results["chat_history_insert_test"] = "❌ Insert failed"
                else:
                    results["chat_history_collection"] = "❌ Not initialized"
                    
            else:
                results["mongodb_connection"] = "❌ Client not initialized"
                
        except Exception as mongo_error:
            results["mongodb_error"] = str(mongo_error)
        
        # Test environment variables
        results["environment"] = {
            "MONGODB_URI": "✅ Set" if os.getenv('MONGODB_URI') else "❌ Missing",
            "MONGODB_DATABASE": os.getenv('MONGODB_DATABASE') or "❌ Missing",
            "OPENAI_API_KEY": "✅ Set" if os.getenv('OPENAI_API_KEY') else "❌ Missing"
        }
        
        results["success"] = True
        results["timestamp"] = datetime.utcnow().isoformat()
        
        return results
        
    except Exception as e:
        logger.error(f"Database test failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

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
        in_progress = await chatbot_service.get_in_progress_count()
        
        return QueueStatusResponse(
            success=True,
            total_queue_length=queue_length,
            current_demo_in_progress=in_progress > 0,
            message="Queue status retrieved successfully" if queue_length > 0 else "Queue is currently empty"
        )
    except Exception as e:
        logger.error(f"Error getting queue status: {e}")
        return QueueStatusResponse(
            success=False,
            total_queue_length=0,
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
            "phone": request.visitor_info.phone,
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
async def remove_from_queue(session_id: str) -> Dict[str, Any]:
    """Remove entry from demo queue and save to demo_done collection (staff only)"""
    try:
        result = await chatbot_service.remove_from_queue(session_id)
        return result
    except Exception as e:
        logger.error(f"Error removing from queue: {e}")
        return {"success": False, "error": str(e)}

@app.get("/api/v1/staff/demo/done-stats")
async def get_demo_done_statistics() -> Dict[str, Any]:
    """Get statistics from completed demos (staff only)"""
    try:
        result = await chatbot_service.get_demo_done_stats()
        return result
    except Exception as e:
        logger.error(f"Error getting demo done stats: {e}")
        return {"success": False, "error": str(e)}

@app.get("/api/v1/staff/chat/history-stats")
async def get_chat_history_statistics() -> Dict[str, Any]:
    """Get statistics from chat history (staff only)"""
    try:
        result = await chatbot_service.get_chat_history_stats()
        return result
    except Exception as e:
        logger.error(f"Error getting chat history stats: {e}")
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
        
        # Save chat history to database
        try:
            await chatbot_service.save_chat_history(
                user_message=request.message,
                assistant_response=result["response"]
            )
        except Exception as save_error:
            # Log the error but don't fail the chat response
            logger.error(f"Failed to save chat history: {save_error}")
        
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