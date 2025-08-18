// iQore Chatbot Frontend - script.js
// Configuration
const API_BASE_URL = 'https://iqoregpt-529970623471.europe-west1.run.app';  // Change this for production deployment
const CHAT_ENDPOINT = `${API_BASE_URL}/api/v1/chat`;

// Global state
let chatHistory = [];
let isLoading = false;
let currentSuggestions = [];

// DOM elements
const messagesContainer = document.getElementById('messagesContainer');
const messageInput = document.getElementById('messageInput');
const sendButton = document.getElementById('sendButton');
const suggestionsContainer = document.getElementById('suggestionsContainer');

// Initialize the chat interface
document.addEventListener('DOMContentLoaded', function() {
    console.log('iQore Chatbot Frontend initialized');
    messageInput.focus();
    showInitialSuggestions();
});

// Initial suggestions when chat starts
function showInitialSuggestions() {
    const initialSuggestions = [
        "What is iQore?",
        "What can I do here?",
        "I want see the demo"
    ];
    updateSuggestions(initialSuggestions);
}

// Generate contextual suggestions based on the user's message
function generateContextualSuggestions(userMessage) {
    // Simple keyword-based suggestion generation
    const message = userMessage.toLowerCase();
    
    if (message.includes('contact') || message.includes('meeting')) {
        return [
            "Schedule a call with iQore",
            "I have more questions about iQore",
            "I want to talk someone from iQore"
        ];
    } else if (message.includes('application') || message.includes('use case')) {
        return [
            "What industries benefit from iQore?",
            "Show me specific use cases",
            "How does this compare to classical computing?"
        ];
    } else if (message.includes('demo') || message.includes('demonstration')) {
        return [
            "What is demo about?",
            "Show me the demo!",
            "What quantum algorithms iQore works with?"
        ];
    } else {
        // Default contextual suggestions
        return [
            "Can I see the demo?",
            "Tell me about iQore",
            "What can you do?"
        ];
    }
}

// Update suggestion bubbles
function updateSuggestions(suggestions) {
    currentSuggestions = suggestions;
    
    // Clear existing suggestions
    suggestionsContainer.innerHTML = '';
    
    if (suggestions.length === 0) {
        suggestionsContainer.style.display = 'none';
        return;
    }
    
    suggestionsContainer.style.display = 'flex';
    
    suggestions.forEach((suggestion, index) => {
        const suggestionBubble = document.createElement('button');
        suggestionBubble.className = 'suggestion-bubble';
        suggestionBubble.textContent = suggestion;
        suggestionBubble.onclick = () => handleSuggestionClick(suggestion);
        
        // Add slight delay for animation
        setTimeout(() => {
            suggestionBubble.classList.add('visible');
        }, index * 100);
        
        suggestionsContainer.appendChild(suggestionBubble);
    });
}

// Handle suggestion bubble click
function handleSuggestionClick(suggestion) {
    // Set the input value and send the message
    messageInput.value = suggestion;
    sendMessage();
    
    // Hide suggestions temporarily while processing
    suggestionsContainer.style.display = 'none';
}
// Removed backend health check display as per user request

// Handle Enter key press in input field
function handleKeyDown(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
}

// Main function to send message
async function sendMessage() {
    const message = messageInput.value.trim();
    
    if (!message || isLoading) {
        return;
    }
    
    // Clear input and disable send button
    messageInput.value = '';
    setLoading(true);
    
    // Add user message to chat
    addMessage(message, 'user');
    
    // Show typing indicator
    const typingIndicator = showTypingIndicator();
    
    try {
        // Send request to backend
        const response = await fetch(CHAT_ENDPOINT, {
            method: 'POST',
            headers: {
                'Accept': 'application/json',
                'Content-Type': 'application/json',
            },
            mode: 'cors',  // Explicitly enable CORS
            body: JSON.stringify({
                message: message,
                chat_history: chatHistory
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        // Remove typing indicator
        removeTypingIndicator(typingIndicator);
        
        // Add AI response to chat
        addMessage(data.response, 'ai');
        
        // Generate and show new contextual suggestions based on user's message
        const newSuggestions = generateContextualSuggestions(message);
        setTimeout(() => {
            updateSuggestions(newSuggestions);
        }, 500); // Small delay to let user read the response first
        
        // Update chat history
        chatHistory = data.chat_history || [];
        
        console.log('Chat response received:', data);
        
    } catch (error) {
        console.error('Error sending message:', error);
        
        // Remove typing indicator
        removeTypingIndicator(typingIndicator);
        
        // Show appropriate error message based on error type
        let errorMessage = "I'm sorry, but I'm having trouble connecting right now. Please try again in a moment.";
        
        if (error.message.includes('Failed to fetch') || error.message.includes('NetworkError')) {
            errorMessage = "Unable to connect. Please check your internet connection and try again.";
        } else if (error.message.includes('500')) {
            errorMessage = "The service is experiencing issues. Please try again in a few moments.";
        } else if (error.message.includes('429')) {
            errorMessage = "Too many requests. Please wait a moment before trying again.";
        }
        
        addMessage(errorMessage, 'ai');
        
        // Show general suggestions on error
        setTimeout(() => {
            updateSuggestions([
                "Let me try a different question",
                "What is iQore's main technology?",
                "How can I learn more about iQore?"
            ]);
        }, 500);
    } finally {
        setLoading(false);
        messageInput.focus();
    }
}

// Add message to chat interface
function addMessage(text, sender) {
    const messageGroup = document.createElement('div');
    messageGroup.className = `message-group ${sender}-message`;
    
    // Create avatar
    const messageAvatar = document.createElement('div');
    messageAvatar.className = 'message-avatar';
    
    const avatar = document.createElement('div');
    avatar.className = `avatar ${sender}-avatar${sender === 'ai' ? '-small' : ''}`;
    messageAvatar.appendChild(avatar);
    
    // Create message bubble
    const messageBubble = document.createElement('div');
    messageBubble.className = 'message-bubble';
    
    // Add message text
    const messageText = document.createElement('p');
    messageText.textContent = text;
    messageBubble.appendChild(messageText);
    
    // Assemble message group
    messageGroup.appendChild(messageAvatar);
    messageGroup.appendChild(messageBubble);
    
    // Add to messages container
    messagesContainer.appendChild(messageGroup);
    
    // Scroll to bottom
    scrollToBottom();
}

// Show typing indicator
function showTypingIndicator() {
    const messageGroup = document.createElement('div');
    messageGroup.className = 'message-group ai-message typing-indicator-group';
    
    const messageAvatar = document.createElement('div');
    messageAvatar.className = 'message-avatar';
    
    const avatar = document.createElement('div');
    avatar.className = 'avatar ai-avatar-small';
    messageAvatar.appendChild(avatar);
    
    const typingBubble = document.createElement('div');
    typingBubble.className = 'message-bubble typing-indicator';
    
    const typingText = document.createElement('span');
    typingText.textContent = 'iQore AI is typing';
    
    const typingDots = document.createElement('div');
    typingDots.className = 'typing-dots';
    
    for (let i = 0; i < 3; i++) {
        const dot = document.createElement('div');
        dot.className = 'typing-dot';
        typingDots.appendChild(dot);
    }
    
    typingBubble.appendChild(typingText);
    typingBubble.appendChild(typingDots);
    
    messageGroup.appendChild(messageAvatar);
    messageGroup.appendChild(typingBubble);
    
    messagesContainer.appendChild(messageGroup);
    scrollToBottom();
    
    return messageGroup;
}

// Remove typing indicator
function removeTypingIndicator(indicator) {
    if (indicator && indicator.parentNode) {
        indicator.parentNode.removeChild(indicator);
    }
}

// Show system message (for errors, warnings, etc.)
function showSystemMessage(text, type = 'info') {
    const messageGroup = document.createElement('div');
    messageGroup.className = 'message-group system-message';
    messageGroup.style.justifyContent = 'center';
    messageGroup.style.margin = '16px 0';
    
    const messageBubble = document.createElement('div');
    messageBubble.style.background = type === 'warning' ? 
        'rgba(255, 193, 7, 0.2)' : 'rgba(139, 69, 255, 0.2)';
    messageBubble.style.border = type === 'warning' ? 
        '1px solid rgba(255, 193, 7, 0.4)' : '1px solid rgba(139, 69, 255, 0.3)';
    messageBubble.style.color = type === 'warning' ? '#ffc107' : '#8b45ff';
    messageBubble.style.padding = '12px 16px';
    messageBubble.style.borderRadius = '12px';
    messageBubble.style.fontSize = '14px';
    messageBubble.style.textAlign = 'center';
    messageBubble.style.maxWidth = '80%';
    
    messageBubble.textContent = text;
    messageGroup.appendChild(messageBubble);
    messagesContainer.appendChild(messageGroup);
    
    scrollToBottom();
}

// Set loading state
function setLoading(loading) {
    isLoading = loading;
    sendButton.disabled = loading;
    messageInput.disabled = loading;
    
    // Hide suggestions while loading
    if (loading) {
        suggestionsContainer.style.display = 'none';
    }
    
    if (loading) {
        sendButton.style.opacity = '0.5';
        sendButton.style.cursor = 'not-allowed';
        messageInput.placeholder = 'Please wait...';
    } else {
        sendButton.style.opacity = '1';
        sendButton.style.cursor = 'pointer';
        messageInput.placeholder = 'Type your message here...';
    }
}

// Scroll to bottom of chat
function scrollToBottom() {
    const chatMain = document.querySelector('.chat-main');
    setTimeout(() => {
        chatMain.scrollTop = chatMain.scrollHeight;
    }, 100);
}

// Handle window resize
window.addEventListener('resize', function() {
    scrollToBottom();
});

// Handle connection errors gracefully
window.addEventListener('online', function() {
    console.log('Connection restored');
    showSystemMessage('Connection restored! You can continue chatting.', 'info');
});

window.addEventListener('offline', function() {
    console.log('Connection lost');
    showSystemMessage('Connection lost. Please check your internet connection.', 'warning');
});

// Phase 3: Demo Queue Status Widget Functions
const QUEUE_STATUS_ENDPOINT = `${API_BASE_URL}/api/v1/demo/queue-status`;
const DEMO_REQUEST_ENDPOINT = `${API_BASE_URL}/api/v1/demo/request`;

// Queue widget state
let queueWidgetVisible = false;
let queueRefreshInterval = null;
let userSessionId = null;

// Initialize queue status checking
document.addEventListener('DOMContentLoaded', function() {
    // Start queue status monitoring
    startQueueMonitoring();
    
    // Check if demo-related conversation has started
    checkForDemoContext();
});

function startQueueMonitoring() {
    // Check queue status every 30 seconds
    refreshQueueStatus();
    queueRefreshInterval = setInterval(refreshQueueStatus, 30000);
}

function stopQueueMonitoring() {
    if (queueRefreshInterval) {
        clearInterval(queueRefreshInterval);
        queueRefreshInterval = null;
    }
}

async function refreshQueueStatus() {
    try {
        const response = await fetch(QUEUE_STATUS_ENDPOINT, {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json',
            },
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        updateQueueDisplay(data);
        
        // Show widget if there's queue activity or if user requested demo
        if (data.total_queue_length > 0 || data.current_demo_in_progress || queueWidgetVisible) {
            showQueueWidget();
        }

    } catch (error) {
        console.error('Error fetching queue status:', error);
        updateQueueMessage('Unable to fetch queue status', 'error');
    }
}

function updateQueueDisplay(queueData) {
    const queueLengthEl = document.getElementById('queueLength');
    const waitTimeEl = document.getElementById('waitTime');
    const demoStatusEl = document.getElementById('demoStatus');
    
    if (queueLengthEl) queueLengthEl.textContent = queueData.total_queue_length || 0;
    if (waitTimeEl) waitTimeEl.textContent = `${queueData.estimated_wait_time_minutes || 0} min`;
    if (demoStatusEl) demoStatusEl.textContent = queueData.current_demo_in_progress ? 'Yes' : 'No';
    
    // Update queue message
    if (queueData.message) {
        updateQueueMessage(queueData.message, queueData.success ? 'success' : 'info');
    }
}

function updateQueueMessage(message, type = 'info') {
    const messageEl = document.getElementById('queueMessage');
    if (messageEl) {
        messageEl.textContent = message;
        messageEl.className = `queue-message ${type}`;
    }
}

function showQueueWidget() {
    const widget = document.getElementById('queueStatusWidget');
    if (widget && !queueWidgetVisible) {
        widget.style.display = 'block';
        widget.classList.add('show');
        queueWidgetVisible = true;
    }
}

function hideQueueWidget() {
    const widget = document.getElementById('queueStatusWidget');
    if (widget) {
        widget.style.display = 'none';
        widget.classList.remove('show');
        queueWidgetVisible = false;
    }
}

function toggleQueueWidget() {
    const widget = document.getElementById('queueStatusWidget');
    const toggleBtn = document.getElementById('queueToggleBtn');
    
    if (widget.classList.contains('collapsed')) {
        widget.classList.remove('collapsed');
        if (toggleBtn) toggleBtn.textContent = '−';
    } else {
        widget.classList.add('collapsed');
        if (toggleBtn) toggleBtn.textContent = '+';
    }
}

function requestDemo() {
    // Send a demo request message through the chat
    const demoMessage = "I'd like to join the demo queue";
    sendMessage(demoMessage);
    
    // Show queue widget if not already visible
    showQueueWidget();
    updateQueueMessage('Demo request sent! The AI will help you sign up.', 'success');
}

function checkForDemoContext() {
    // Check if recent chat history contains demo-related keywords
    const demoKeywords = ['demo', 'demonstration', 'queue', 'signup', 'reserve'];
    const recentMessages = chatHistory.slice(-5); // Check last 5 messages
    
    const hasDemoContext = recentMessages.some(msg => 
        msg.user && demoKeywords.some(keyword => 
            msg.user.toLowerCase().includes(keyword)
        )
    );
    
    if (hasDemoContext) {
        showQueueWidget();
    }
}

// Monitor chat for demo-related conversations
function monitorChatForDemo(userMessage, aiResponse) {
    const demoKeywords = ['demo', 'demonstration', 'queue', 'signup', 'reserve', 'join'];
    const message = (userMessage + ' ' + aiResponse).toLowerCase();
    
    const isDemoRelated = demoKeywords.some(keyword => message.includes(keyword));
    
    if (isDemoRelated) {
        setTimeout(() => {
            showQueueWidget();
            refreshQueueStatus();
        }, 1000);
    }
    
    // Check for session ID in AI response
    const sessionIdMatch = aiResponse.match(/session[_\s]*id[:\s]*([a-f0-9\-]{36})/i);
    if (sessionIdMatch) {
        userSessionId = sessionIdMatch[1];
        console.log('User session ID detected:', userSessionId);
        
        // Start personalized status updates
        startPersonalizedStatusUpdates();
    }
}

function startPersonalizedStatusUpdates() {
    if (!userSessionId) return;
    
    // Check user's specific queue status every 15 seconds
    const personalizedInterval = setInterval(async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/api/v1/demo/queue/${userSessionId}`);
            if (response.ok) {
                const data = await response.json();
                if (data.success) {
                    updateQueueMessage(
                        `Your position: #${data.queue_position} | Wait time: ~${data.estimated_wait_time} min`,
                        'success'
                    );
                }
            }
        } catch (error) {
            console.error('Error fetching personal queue status:', error);
        }
    }, 15000);
    
    // Stop personal updates after 2 hours
    setTimeout(() => {
        clearInterval(personalizedInterval);
    }, 2 * 60 * 60 * 1000);
}

// Enhance the existing sendMessage function to monitor for demo conversations
const originalSendMessage = sendMessage;
window.sendMessage = async function(messageText = null) {
    const result = await originalSendMessage(messageText);
    
    // Monitor the conversation
    if (chatHistory.length >= 2) {
        const lastUserMsg = chatHistory[chatHistory.length - 2]?.user || '';
        const lastAiMsg = chatHistory[chatHistory.length - 1]?.assistant || '';
        monitorChatForDemo(lastUserMsg, lastAiMsg);
    }
    
    return result;
};

// Cleanup on page unload
window.addEventListener('beforeunload', function() {
    stopQueueMonitoring();
});

// Export new functions for global access
window.toggleQueueWidget = toggleQueueWidget;
window.refreshQueueStatus = refreshQueueStatus;
window.requestDemo = requestDemo;
window.handleKeyDown = handleKeyDown; 