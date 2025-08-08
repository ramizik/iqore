# Frontend Development Prompt for bolt.new - iQore Chatbot

## 🎯 Project Overview
You are developing the **frontend only** for iQore's event chatbot website. This is a **one-page website** built entirely around chatbot functionality - the chatbot IS the main feature, not a sidebar addition.

## 🚨 CRITICAL INSTRUCTIONS - READ CAREFULLY

### ✅ WHAT YOU SHOULD DO:
- **Focus on chatbot-first design** - everything revolves around the chat interface
- **Keep the page clean and uncluttered** - minimal distractions from the chat experience
- **Follow instructions strictly** - only add/modify/delete what is explicitly requested
- **Comment API integration points** for the FastAPI backend (more details below)
- **Create modern, professional UI** that reflects iQore's deep-tech quantum computing brand

### ❌ WHAT YOU SHOULD NOT DO:
- **DO NOT overcrowd the page** with unnecessary elements
- **DO NOT add features** unless specifically requested
- **DO NOT worry about backend development** - FastAPI backend already exists
- **DO NOT create complex navigation** - this is a single-page chat interface
- **DO NOT add social media widgets, footers, or marketing fluff** unless asked

## 🏗️ Backend Context (for API Integration)

**YOU DO NOT DEVELOP THE BACKEND** - but you need to know this for API calls:

- **Backend**: FastAPI already deployed and working
- **Main Chat Endpoint**: `POST /api/v1/chat`
- **Request Format**: `{ "message": "user input", "chat_history": [...] }`
- **Response Format**: `{ "response": "AI response", "chat_history": [...] }`
- **Health Check**: `GET /health` and `GET /api/v1/status`

**When you write frontend code that makes API calls, add comments explaining:**
```javascript
// API Integration Point: Chat endpoint
// Expected backend: FastAPI POST /api/v1/chat
// Request: { message: string, chat_history: array }
// Response: { response: string, chat_history: array }
```

## 🎨 Design Requirements

### Core Layout:
```
┌─────────────────────────────────────┐
│           Header (minimal)          │ ← Company branding, AI info
├─────────────────────────────────────┤
│                                     │
│                                     │
│         CHAT INTERFACE              │ ← Main focus area
│         (takes most space)          │
│                                     │
│                                     │
├─────────────────────────────────────┤
│      Input Area (bottom)            │ ← Message input + send button
└─────────────────────────────────────┘
```

### Visual Style:
- **Modern, clean aesthetic** suitable for tech company booth
- **Professional color scheme** (avoid bright/playful colors)
- **Readable typography** - good contrast and sizing
- **Mobile responsive** but primarily designed for desktop/tablet use at events
- **Subtle animations** for typing indicators, message appearance
- **Accessibility compliant** - proper ARIA labels, keyboard navigation

### Brand Context - iQore:
- **Industry**: Quantum-classical hybrid computing / Deep tech
- **Audience**: Enterprise clients, engineers, decision-makers at tech events
- **Tone**: Professional, innovative, trustworthy, cutting-edge

## 🔧 Technical Requirements

### Tech Stack:
- **Pure HTML/CSS/JavaScript** (no frameworks unless specifically requested)
- **Modern ES6+ JavaScript** with async/await for API calls
- **CSS Grid/Flexbox** for layouts
- **Responsive design** with mobile-first approach
- **Modern browser support** (Chrome, Firefox, Safari, Edge)

### Performance:
- **Fast loading** - minimal external dependencies
- **Smooth animations** - 60fps where possible
- **Efficient DOM manipulation** - batch updates where needed
- **Error handling** - graceful fallbacks for network issues

## 💬 Chatbot Interface Specifications

### Message Display:
- **Clear distinction** between user and AI messages
- **Avatar/icons** for visual message attribution
- **Timestamps** (optional, only if requested)
- **Typing indicators** when AI is responding
- **Auto-scroll** to latest messages
- **Message history preservation** during session

### Input Area:
- **Large, clear text input** - primary interaction element
- **Send button** with clear visual feedback
- **Enter key support** for message sending
- **Input validation** - prevent empty messages
- **Loading states** - disable input while processing
- **Character limit handling** (if needed)

### User Experience:
- **Immediate feedback** - show user message instantly
- **Clear loading states** - typing indicators, disabled states
- **Error handling** - network errors, API failures
- **Reconnection logic** - handle connection issues gracefully
- **Keyboard shortcuts** - Enter to send, basic navigation

## 🚀 Implementation Guidelines

### Code Structure:
```javascript
// Main application structure
const ChatApp = {
    // API integration points (comment these clearly)
    api: {
        baseUrl: 'YOUR_API_URL',
        endpoints: { /* comment each endpoint */ }
    },
    
    // UI management
    ui: {
        // DOM manipulation functions
    },
    
    // Message handling
    messages: {
        // Send/receive/display logic
    }
};
```

### API Integration Comments:
Always add comments like:
```javascript
// FastAPI Backend Integration
// Endpoint: POST /api/v1/chat
// This function sends user messages to the FastAPI backend
// and receives AI responses with updated chat history
async function sendMessage(message, chatHistory) {
    // Implementation here
}
```

## 📱 Responsive Behavior

### Desktop (Primary):
- **Full chat interface** taking most of viewport
- **Comfortable text sizes** for reading at booth displays
- **Clear touch targets** for booth interaction

### Tablet:
- **Optimized for touch** - larger buttons and inputs
- **Portrait/landscape** support
- **Comfortable viewing distance** for booth visitors

### Mobile (Secondary):
- **Stack layout** if needed
- **Touch-friendly** interface
- **Readable text** without zooming

## 🎯 Content Guidelines

### Initial Welcome Message:
Should reflect iQore's quantum computing focus:
```
"Hello! I'm the iQore AI Assistant, your guide to hybrid quantum-classical 
computing. I can help you understand iQore's technology, architecture, and 
applications. How can I assist you today?"
```

### Placeholder Text:
- Input: "Ask me about iQore's quantum computing solutions..."
- Loading: "iQore AI is thinking..."
- Error: "Connection issue - please try again"

## 🔍 Testing Considerations

### Manual Testing Points:
- **Message sending/receiving** flow
- **Error states** - network failures, API errors
- **Loading states** - typing indicators, disabled inputs
- **Responsive design** - different screen sizes
- **Keyboard navigation** - tab order, enter key
- **Long conversations** - scroll behavior, performance

### Edge Cases:
- **Very long messages** - handling and display
- **Network interruptions** - reconnection logic
- **Rapid message sending** - prevent spam/flooding
- **Empty responses** - graceful handling

## 📋 Deliverable Requirements

When creating the frontend:

1. **Single HTML file** with embedded CSS and JavaScript (unless separation requested)
2. **Clear API integration points** with detailed comments
3. **Professional, clean design** focused on chat functionality
4. **Responsive layout** that works on multiple screen sizes
5. **Error handling** for common failure scenarios
6. **Loading states** for better user experience

## 🚨 FINAL REMINDERS

- **CHATBOT IS THE STAR** - everything else is supporting
- **MINIMAL AND CLEAN** - don't add unnecessary elements
- **FOLLOW REQUESTS EXACTLY** - only build what's asked for
- **COMMENT API CALLS** - help with backend integration
- **PROFESSIONAL APPEARANCE** - this represents iQore at events

Remember: You're building a **chatbot-first website**, not a website with a chatbot feature. The entire page should be designed around facilitating great conversations with the AI assistant. 