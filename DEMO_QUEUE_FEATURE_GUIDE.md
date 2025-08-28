# 🎯 Demo Request Queue Feature - Implementation Guide

## Overview

The Demo Request Queue feature enables booth visitors to seamlessly request live quantum computing demos through natural conversation. This implementation uses **LangGraph for multi-agent orchestration**, **MongoDB Atlas for queue storage**, **FastAPI for backend APIs**, and **real-time frontend integration**.

## 🏗️ Architecture Components

### **Phase 1: Enhanced Data Models & API Endpoints** ✅ COMPLETED

#### **Pydantic Models** 
- `DemoStatus` - Enum for queue status tracking (waiting, in_progress, completed, cancelled, no_show)
- `VisitorInfo` - Structured visitor data with validation
- `DemoRequest` - Complete demo signup request format
- `QueueStatusResponse` - Comprehensive queue status response
- `UserQueueStatusResponse` - Individual user status response
- `StaffQueueUpdateRequest` - Staff management operations
- `StaffQueueResponse` - Staff queue overview

#### **Enhanced FastAPI Endpoints**
```
GET  /api/v1/demo/queue-status           - Overall queue status
GET  /api/v1/demo/queue/{session_id}     - Individual user status
POST /api/v1/demo/request                - Create demo request
GET  /api/v1/staff/demo/queue           - Staff management view
PUT  /api/v1/staff/demo/queue/{session_id} - Update demo status
DELETE /api/v1/staff/demo/queue/{session_id} - Remove from queue
```

### **Phase 2: LangChain Tools Integration** ✅ COMPLETED

#### **Custom LangChain Tools**
- `DemoQueueTool` - Async tool for queue operations within agents
- `UserInfoExtractionTool` - Natural language info extraction
- `PydanticOutputParser` integration for structured data

#### **Enhanced Agent Capabilities**
- Structured visitor info extraction from conversation
- Natural language intent detection for demo signups
- Queue status checking and management
- Multi-turn conversation state management

### **Phase 3: Frontend Integration** ✅ COMPLETED

#### **Real-time Queue Status Widget**
- Live queue metrics display (people waiting, estimated time, demo status)
- Auto-refresh every 30 seconds
- Personalized status updates for signed-up users
- Collapsible interface with smooth animations
- Mobile-responsive design

#### **Smart Context Detection**
- Automatic widget display when demo-related conversation detected
- Session ID extraction for personalized tracking
- Integration with existing chat interface

## 🚀 How It Works

### **1. Natural Conversation Flow**
```
User: "I'd like to see a demo"
AI: [Demo Agent] "Great! I'll show you our quantum algorithms..."
    → Queue widget automatically appears
    → Real-time queue status displayed

User: "Yes, sign me up"
AI: [Demo Agent] "What's your name?"
    → Natural info collection begins

User: "John Smith"
AI: [Demo Agent] "Thanks John! What's your email?"
    → Structured data extraction with validation

User: "john@company.com"
AI: [Demo Agent] "Perfect! You're added to the queue..."
    → Queue entry created, session ID assigned
```

### **2. LangGraph Agent Orchestration**
```
Supervisor Agent
    ├── Detects "demo" intent → routes to Demo Agent
    │
Demo Agent (Enhanced)
    ├── Initial Stage: Demo introduction & signup interest
    ├── Collection Stage: Natural info gathering with validation
    ├── Confirmation Stage: Review details & create queue entry
    └── Queue Status Stage: Ongoing status updates
```

### **3. Queue Management**
- **MongoDB Storage**: Persistent queue with visitor info, timestamps, status
- **Position Calculation**: Real-time queue position based on timestamps
- **Wait Time Estimation**: Dynamic calculation (17.5 min average per demo)
- **Status Tracking**: waiting → in_progress → completed/cancelled

### **4. Staff Management**
Staff can manage the queue through API endpoints:
```bash
# Get full queue overview
GET /api/v1/staff/demo/queue

# Start a demo (change status to in_progress)
PUT /api/v1/staff/demo/queue/{session_id}
{
  "new_status": "in_progress",
  "notes": "Demo started with John"
}

# Complete a demo
PUT /api/v1/staff/demo/queue/{session_id}
{
  "new_status": "completed",
  "notes": "Great demo session"
}
```

## 🎨 Frontend Features

### **Queue Status Widget**
- **Metrics Display**: Queue length, wait time, demo status
- **Action Buttons**: Join queue, refresh status
- **Status Messages**: Real-time updates with color coding
- **Auto-show Logic**: Appears when demos are requested or queue is active
- **Personalized Updates**: Shows user's position when signed up

### **Integration with Chat**
- **Context Detection**: Monitors chat for demo-related keywords
- **Session Tracking**: Extracts session IDs from AI responses
- **Seamless UX**: No page refresh required, all updates are live

## 💾 Data Models

### **Queue Entry Structure (MongoDB)**
```javascript
{
  "session_id": "uuid-v4-string",
  "name": "John Smith",
  "email": "john@company.com", 
  "company": "Tech Corp",
  "interest_areas": ["VQE", "QAOA"],
  "status": "waiting",
  "timestamp": "2024-01-15T10:30:00Z",
  "notes": "Interested in molecular simulation",
  "updated_at": "2024-01-15T10:30:00Z"
}
```

### **Queue Status Response**
```json
{
  "success": true,
  "total_queue_length": 3,
  "estimated_wait_time_minutes": 52,
  "current_demo_in_progress": true,
  "average_demo_duration_minutes": 17.5,
  "message": "Queue is active with demos in progress"
}
```

## 🔧 Configuration & Deployment

### **Environment Variables**
```
OPENAI_API_KEY=your_openai_key
MONGODB_URI=mongodb+srv://...
MONGODB_DATABASE=iqore_chatbot
```

### **MongoDB Collections**
- `demo_queue` - Queue entries with visitor information
- `pdf_chunks` - Existing RAG knowledge base

### **Frontend Configuration**
```javascript
// script.js
const API_BASE_URL = 'https://iqoregpt-529970623471.europe-west1.run.app';
const QUEUE_STATUS_ENDPOINT = `${API_BASE_URL}/api/v1/demo/queue-status`;
```

## 📱 User Experience Flow

1. **Discovery**: User asks about demos through natural conversation
2. **Interest**: Queue widget appears showing current demo activity
3. **Signup**: AI guides through natural info collection
4. **Confirmation**: Queue entry created, position & wait time shown
5. **Tracking**: Real-time updates on queue position
6. **Demo**: Staff manages demo execution through API
7. **Completion**: Status updated, user receives confirmation

## 🎯 Key Features Achieved

✅ **Natural Language Processing**: Info extraction from conversation  
✅ **Multi-Agent Orchestration**: LangGraph-powered agent routing  
✅ **Real-time Queue Management**: Live status updates  
✅ **Staff Management Interface**: Complete queue administration  
✅ **Seamless Frontend Integration**: Context-aware widget display  
✅ **Persistent Storage**: MongoDB-based queue persistence  
✅ **Structured Data Validation**: Pydantic model validation  
✅ **Session Tracking**: Unique session IDs for each user  
✅ **Responsive Design**: Mobile-optimized queue widget  
✅ **Error Handling**: Comprehensive error management throughout

## 🔮 Next Steps (Phase 4 - Future Enhancements)

- **Real-time WebSocket Updates**: Push notifications for queue changes
- **QR Code Generation**: Quick signup via QR codes at booth
- **Calendar Integration**: Schedule follow-up meetings
- **Analytics Dashboard**: Queue performance metrics
- **SMS Notifications**: Text updates for queue status
- **Priority Queue**: VIP or partner priority handling

## 🧪 Testing the Feature

### **Demo Conversation Examples**
```
"I want to see a demo"
"Show me the demo"
"Can I see a live demonstration?"
"I'd like to join the demo queue"
"What's the wait time for demos?"
```

### **API Testing**
```bash
# Check queue status
curl https://iqoregpt-529970623471.europe-west1.run.app/api/v1/demo/queue-status

# Create demo request
curl -X POST https://iqoregpt-529970623471.europe-west1.run.app/api/v1/demo/request \
  -H "Content-Type: application/json" \
  -d '{
    "visitor_info": {
      "name": "Test User",
      "email": "test@example.com",
      "company": "Test Corp"
    }
  }'
```

---

**Implementation Status**: ✅ **COMPLETE & PRODUCTION READY**

The Demo Request Queue feature is now fully integrated into the iQore chatbot system, providing a seamless experience for booth visitors to request and track live quantum computing demonstrations through natural conversation.
