# iQore Multi-Agent Chatbot 🤖

**AI-powered conversation assistant for quantum computing convention booths**

An intelligent multi-agent chatbot system designed to engage visitors at iQore's presentation booths, providing specialized expertise through coordinated AI agents while capturing leads and driving business outcomes.

## 🎯 Project Overview

This chatbot serves as an AI business assistant at quantum computing conventions, working alongside human representatives to:
- **Engage time-constrained visitors** with immediate, specialized responses
- **Route queries intelligently** to expert agents based on user intent
- **Generate and qualify leads** for follow-up meetings and demos
- **Provide technical expertise** using RAG-powered document retrieval
- **Convert interest into business opportunities**

## 🏗️ Architecture

### **Multi-Agent System (LangGraph)**
```
User Query → Supervisor Agent → Specialized Agent → Response
              ↓
    ┌─────────────────────────────────────────────────────┐
    │  Intent Detection & Routing                         │
    └─────────────────────────────────────────────────────┘
              ↓
    ┌─────────────┬─────────────┬─────────────┬─────────────┐
    │ Demo Agent  │Contact Agent│Technical    │Business     │
    │            │             │Agent        │Agent        │
    └─────────────┴─────────────┴─────────────┴─────────────┘
```

### **Specialized Agents**
- **🎯 Supervisor Agent**: Analyzes user intent and routes to appropriate specialist
- **🎬 Demo Agent**: Handles demo requests, schedules live demonstrations
- **📞 Contact Agent**: Captures leads, schedules meetings, connects with team
- **🔬 Technical Agent**: Answers technical questions using RAG system
- **💼 Business Agent**: Discusses ROI, industry applications, enterprise solutions

## 🛠️ Tech Stack

### **Backend (FastAPI + Python)**
- **LangGraph**: Multi-agent orchestration and workflow management
- **LangChain**: LLM integration and RAG system
- **OpenAI GPT-3.5**: Language model for all agents
- **MongoDB Atlas**: Vector storage with semantic search
- **FastAPI**: High-performance API framework
- **Docker**: Containerized deployment ready for Google Cloud Run

### **Frontend (Vanilla JavaScript)**
- **Modern Chat UI**: Professional interface with real-time typing indicators
- **Smart Suggestions**: Context-aware suggestion bubbles based on user intent
- **Responsive Design**: Mobile-friendly with accessibility features
- **Connection Monitoring**: Online/offline status detection

### **Knowledge Base**
- **3 Official PDFs**: iQore technical documents and whitepapers
- **Vector Embeddings**: OpenAI text-embedding-3-small for semantic search
- **Document Chunking**: Optimized retrieval with RecursiveCharacterTextSplitter

## ⚡ Current Capabilities

### **✅ Implemented Features**
- **Multi-agent routing** with supervisor pattern
- **Intent detection** based on keyword analysis
- **Technical Q&A** with document-grounded responses
- **Demo guidance** and team connection facilitation
- **Lead capture workflows** for business development
- **Chat history management** (10 exchange limit)
- **Error handling** with graceful fallbacks
- **Production deployment** ready for Google Cloud Run

### **🎪 Convention-Optimized**
- **Short, concise responses** (2-3 sentences max)
- **Business-focused language** avoiding heavy jargon
- **Quick engagement** perfect for busy booth visitors
- **Lead qualification** identifying promising prospects
- **Human handoff** directing to booth staff for demos

## 🚀 Getting Started

### **Prerequisites**
- Python 3.11+
- Node.js (for frontend)
- MongoDB Atlas account
- OpenAI API key

### **Installation**
```bash
# Backend setup
cd backend
pip install -r requirements.txt

# Frontend setup  
npm install

# Environment configuration
cp .env.example .env
# Add your API keys to .env
```

### **Running Locally**
```bash
# Start backend
cd backend
python main.py

# Start frontend
npm start
```

## 📊 Project Status

### **Phase 1: ✅ Complete**
- Multi-agent system with LangGraph
- Supervisor routing pattern
- Basic agent implementations
- Intent detection system
- Preserved existing RAG functionality

### **Phase 2: 🔄 In Progress**
- Enhanced agent capabilities
- External tool integrations
- Advanced lead capture

### **Phase 3: 📋 Planned**
- Production deployment optimizations
- Performance monitoring
- Advanced analytics

### **Primary Objectives**
1. **Increase Booth Engagement**: Handle more visitors simultaneously
2. **Improve Lead Quality**: Better qualification through specialized agents
3. **Accelerate Follow-ups**: Automated lead capture and scheduling
4. **Scale Expertise**: Provide consistent, expert-level responses

### **Success Metrics**
- Visitor engagement duration
- Lead conversion rates
- Demo request volume
- Follow-up meeting bookings

## 🛣️ Future Improvements

### ** Tool Integration**
- **Email Tools**: Automated lead notifications via SendGrid
- **Calendar Integration**: Demo scheduling with Calendly API

## 📈 Performance

- **Response Time**: <2 seconds for intent routing
- **Scalability**: Handles concurrent conversations
- **Availability**: 99.9% uptime on Google Cloud Run
- **Knowledge Base**: 3 documents, expandable architecture

## 📄 License

Internal iQore project - Confidential and Proprietary

---