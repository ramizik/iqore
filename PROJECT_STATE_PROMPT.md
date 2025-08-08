# iQore Event Chatbot - Project State Overview

## 🎯 Project Purpose
This is an **event-focused chatbot assistant** for iQore, designed to be deployed at presentation booths to help visitors learn about the company, products, and request demos or meetings. The chatbot provides intelligent responses using a RAG (Retrieval-Augmented Generation) system powered by iQore's internal documents.

## 🏗️ Architecture Overview

**Frontend**: Pure HTML/CSS/JavaScript chat interface
**Backend**: FastAPI (Python) with LangChain integration
**Database**: MongoDB Atlas with Vector Search capabilities
**Knowledge Base**: PDF documents (5 total) pre-ingested as embeddings
**LLM**: OpenAI GPT-3.5-turbo with text-embedding-3-small
**Deployment**: Dockerized backend on Google Cloud Run

## 📁 Current Project Structure

```
iqore/
├── backend/                           # FastAPI backend service
│   ├── main.py                       # ✅ COMPLETE - Main FastAPI app with modern LangChain LCEL
│   ├── requirements.txt              # ✅ COMPLETE - Python dependencies
│   ├── Dockerfile                    # ✅ COMPLETE - Container config for Cloud Run
│   ├── cloudbuild.yaml              # ✅ COMPLETE - Cloud Build configuration
│   ├── pdf/                         # ✅ COMPLETE - Knowledge base documents
│   │   ├── iQore - Execution Stack.pdf
│   │   ├── iQore - iQbit Paper.pdf
│   │   └── iQore - QPE - Proof of Function Brief - 2025.pdf
│   ├── cli_test.py                  # Testing utilities
│   └── start.sh / start-local.sh    # Startup scripts
├── index.html                        # ✅ COMPLETE - Modern chat UI
├── script.js                        # ✅ COMPLETE - Frontend JavaScript logic
├── styles.css                       # ✅ COMPLETE - Chat interface styling
├── server.js                        # ✅ COMPLETE - Express server for frontend
├── package.json                     # ✅ COMPLETE - Node.js dependencies
└── README.md                        # ⚠️ MINIMAL - Needs expansion
```

## 🚀 Current Implementation Status

### ✅ FULLY IMPLEMENTED & WORKING:

**Backend (FastAPI + LangChain)**:
- Modern LangChain LCEL implementation replacing deprecated ConversationalRetrievalChain
- History-aware retriever that contextualizes questions based on chat history
- MongoDB Atlas Vector Search integration for document retrieval
- OpenAI GPT-3.5-turbo with custom system prompt for iQore branding
- CORS middleware configured for cross-origin requests
- Health check endpoints for monitoring
- Proper error handling and logging
- Environment variable configuration
- Docker containerization ready for Google Cloud Run

**Frontend (Vanilla JS)**:
- Modern, responsive chat interface with professional styling
- Real-time typing indicators
- Chat history management (limited to last 10 exchanges)
- Proper error handling with user-friendly messages
- Connection status monitoring (online/offline detection)
- Enter key support for message sending
- Loading states and disabled inputs during processing
- Clean, accessible UI with proper avatars and message bubbles

**Knowledge Base**:
- 3 official iQore PDF documents ingested and ready
- Document chunking with RecursiveCharacterTextSplitter
- OpenAI embeddings (text-embedding-3-small) for semantic search
- MongoDB Atlas Vector Store with cosine similarity search

**Deployment**:
- Docker configuration optimized for Google Cloud Run
- Environment variables properly configured
- Health check endpoints implemented
- Production-ready logging and error handling

### ⚠️ AREAS THAT MAY NEED ATTENTION:

1. **Document Ingestion Pipeline**: 
   - No automated ingestion script visible in current structure
   - Documents appear to be manually ingested (need confirmation)
   - File hash checking mentioned in architecture but not clearly implemented

2. **Environment Configuration**:
   - Requires `.env` file with OPENAI_API_KEY, MONGODB_URI, MONGODB_DATABASE
   - Frontend currently hardcoded to specific Cloud Run URL

3. **Testing**:
   - Limited testing infrastructure
   - `cli_test.py` exists but implementation unclear

4. **Documentation**:
   - README.md is minimal and needs expansion
   - No API documentation or setup instructions

## 🔧 Tech Stack Details

**Backend Dependencies**:
```
fastapi==0.104.1          # Web framework
uvicorn[standard]==0.24.0  # ASGI server
langchain                  # LLM orchestration
langchain-mongodb         # MongoDB vector store
langchain-openai          # OpenAI integration
pymongo==4.6.1           # MongoDB driver
python-dotenv==1.0.0     # Environment management
```

**Frontend**:
- Pure vanilla JavaScript (no frameworks)
- Modern CSS with flexbox/grid layouts
- Google Fonts (Inter) integration
- Responsive design

## 🎯 AI Assistant Behavior

The chatbot is configured with a sophisticated system prompt that:
- Positions itself as a professional iQore virtual assistant
- Has deep knowledge of quantum-classical hybrid computing
- References official documents when answering questions
- Encourages demo requests and follow-up meetings
- Maintains helpful, confident, and persuasive tone
- Admits knowledge limitations honestly

## 🚀 Current Deployment Status

- **Backend**: Deployed on Google Cloud Run at `https://iqoregpt-529970623471.europe-west1.run.app`
- **Frontend**: Served via Express.js server (can be deployed separately)
- **Database**: MongoDB Atlas cluster configured with vector search
- **Environment**: Production-ready with proper logging and health checks

## 🛠️ Development Workflow

**To run locally**:
1. Backend: Set environment variables → `uvicorn main:app --reload`
2. Frontend: `npm start` (runs Express server on port 3000)
3. Database: Requires MongoDB Atlas connection

**For modifications**:
- Backend changes: Edit `backend/main.py`
- Frontend changes: Edit `index.html`, `script.js`, `styles.css`
- Styling: Modern CSS in `styles.css` with custom properties
- Deployment: Use `backend/Dockerfile` and Google Cloud Run

## 🎨 UI/UX Features

- Professional iQore-branded interface
- AI avatar with smooth animations
- Typing indicators for realistic conversation feel
- Error handling with contextual messages
- Connection status awareness
- Mobile-responsive design
- Accessible color scheme and typography

## 🔐 Security & Configuration

- CORS properly configured
- Environment variables for sensitive data
- Production-ready error handling
- Health check endpoints for monitoring
- Request/response logging for debugging

---

**Current Status**: The project is **production-ready** with a fully functional RAG chatbot system. The main components are implemented and working. Any modifications should focus on enhancing existing functionality, adding new features, or improving the document ingestion pipeline. 