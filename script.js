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

async function requestDemo() {
    // Try direct signup first, fall back to conversation if no user info
    try {
        // Check if we have user info from previous conversations
        const userInfo = extractUserInfoFromHistory();
        
        if (userInfo.name && userInfo.email) {
            // Direct API call to add to queue
            await addToQueueDirectly(userInfo);
        } else {
            // Try quick form signup first, then fall back to conversation
            const quickSignup = await tryQuickSignupForm();
            
            if (quickSignup.success) {
                await addToQueueDirectly(quickSignup.userInfo);
            } else {
                // Fall back to conversational signup
                const demoMessage = "I'd like to join the demo queue";
                await sendMessage(demoMessage);
                updateQueueMessage('The AI will help you sign up! Please provide your name and email.', 'info');
            }
        }
        
        // Show queue widget if not already visible
        showQueueWidget();
        
    } catch (error) {
        console.error('Error requesting demo:', error);
        updateQueueMessage('Error requesting demo. Please try through chat.', 'error');
        
        // Fall back to conversational method
        const demoMessage = "I'd like to join the demo queue";
        await sendMessage(demoMessage);
    }
}

async function addToQueueDirectly(userInfo) {
    try {
        updateQueueMessage('Adding you to the demo queue...', 'info');
        
        const response = await fetch(`${API_BASE_URL}/api/v1/demo/request`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                visitor_info: {
                    name: userInfo.name,
                    email: userInfo.email,
                    company: userInfo.company || null,
                    interest_areas: []
                }
            })
        });

        const result = await response.json();
        
        if (result.success) {
            userSessionId = result.session_id;
            updateQueueMessage(
                `✅ Success! You're #${result.queue_position} in queue. Wait time: ~${result.estimated_wait_time} min`,
                'success'
            );
            
            // Start personalized updates
            startPersonalizedStatusUpdates();
            
            // Add success message to chat
            addSystemMessageToChat(
                `🎉 Great! You've been added to our demo queue.\n\n` +
                `📊 Your Position: #${result.queue_position}\n` +
                `⏱️ Estimated Wait: ${result.estimated_wait_time} minutes\n` +
                `🆔 Session ID: ${result.session_id}`
            );
            
        } else {
            throw new Error(result.error || 'Failed to add to queue');
        }
        
    } catch (error) {
        console.error('Direct queue addition failed:', error);
        updateQueueMessage('Direct signup failed. Please use the chat to sign up.', 'error');
        throw error;
    }
}

function extractUserInfoFromHistory() {
    // Extract user info from recent chat history
    const userInfo = { name: null, email: null, company: null };
    
    // Look through recent chat history for user information
    const recentMessages = chatHistory.slice(-10); // Last 10 exchanges
    
    for (const exchange of recentMessages) {
        if (exchange.user && exchange.assistant) {
            const userMsg = exchange.user.toLowerCase();
            const aiMsg = exchange.assistant.toLowerCase();
            
            // Extract email
            const emailMatch = exchange.user.match(/\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b/);
            if (emailMatch) {
                userInfo.email = emailMatch[0];
            }
            
            // Extract name (look for patterns like "my name is", "I'm", etc.)
            const namePatterns = [
                /my name is ([a-zA-Z\s]{2,50})/i,
                /i'm ([a-zA-Z\s]{2,50})/i,
                /name's ([a-zA-Z\s]{2,50})/i,
                /i am ([a-zA-Z\s]{2,50})/i,
                /call me ([a-zA-Z\s]{2,50})/i
            ];
            
            for (const pattern of namePatterns) {
                const nameMatch = exchange.user.match(pattern);
                if (nameMatch) {
                    userInfo.name = nameMatch[1].trim();
                    break;
                }
            }
            
            // If no pattern found, check if user message might be just a name
            if (!userInfo.name && userMsg.length > 2 && userMsg.length < 50 && 
                !userMsg.includes('@') && /^[a-zA-Z\s]+$/.test(exchange.user.trim())) {
                const words = exchange.user.trim().split(/\s+/);
                if (words.length >= 1 && words.length <= 4) {
                    userInfo.name = exchange.user.trim();
                }
            }
            
            // Extract company
            const companyPatterns = [
                /work at ([a-zA-Z0-9\s&.-]{2,100})/i,
                /work for ([a-zA-Z0-9\s&.-]{2,100})/i,
                /from ([a-zA-Z0-9\s&.-]{2,100})/i,
                /company is ([a-zA-Z0-9\s&.-]{2,100})/i
            ];
            
            for (const pattern of companyPatterns) {
                const companyMatch = exchange.user.match(pattern);
                if (companyMatch) {
                    userInfo.company = companyMatch[1].trim();
                    break;
                }
            }
        }
    }
    
    return userInfo;
}

function addSystemMessageToChat(message) {
    // Add a system message to the chat interface
    const messagesContainer = document.getElementById('messagesContainer');
    
    if (messagesContainer) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message-group ai-message';
        messageDiv.innerHTML = `
            <div class="message-avatar">
                <div class="avatar ai-avatar-small"></div>
            </div>
            <div class="message-bubble">
                <p>${message.replace(/\n/g, '<br>')}</p>
            </div>
        `;
        messagesContainer.appendChild(messageDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }
}

async function tryQuickSignupForm() {
    return new Promise((resolve) => {
        // Create modal overlay
        const overlay = document.createElement('div');
        overlay.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.7);
            display: flex;
            justify-content: center;
            align-items: center;
            z-index: 10000;
        `;
        
        // Create form modal
        const modal = document.createElement('div');
        modal.style.cssText = `
            background: white;
            padding: 30px;
            border-radius: 12px;
            max-width: 400px;
            width: 90%;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        `;
        
        modal.innerHTML = `
            <h2 style="color: #2d1b69; margin-bottom: 20px; text-align: center;">🎯 Join Demo Queue</h2>
            <p style="color: #666; margin-bottom: 20px; text-align: center; font-size: 14px;">
                Quick signup for our live quantum computing demonstration
            </p>
            <form id="quickSignupForm">
                <div style="margin-bottom: 15px;">
                    <label style="display: block; margin-bottom: 5px; font-weight: 500; color: #333;">Name *</label>
                    <input type="text" id="quickName" required 
                           style="width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 6px; font-size: 14px;"
                           placeholder="Enter your full name">
                </div>
                <div style="margin-bottom: 15px;">
                    <label style="display: block; margin-bottom: 5px; font-weight: 500; color: #333;">Email *</label>
                    <input type="email" id="quickEmail" required
                           style="width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 6px; font-size: 14px;"
                           placeholder="Enter your email address">
                </div>
                <div style="margin-bottom: 20px;">
                    <label style="display: block; margin-bottom: 5px; font-weight: 500; color: #333;">Company (Optional)</label>
                    <input type="text" id="quickCompany"
                           style="width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 6px; font-size: 14px;"
                           placeholder="Enter your company name">
                </div>
                <div style="display: flex; gap: 10px; justify-content: center;">
                    <button type="submit" 
                            style="background: linear-gradient(135deg, #8b45ff 0%, #6b73ff 100%); color: white; border: none; padding: 12px 24px; border-radius: 6px; font-weight: 600; cursor: pointer;">
                        Join Queue
                    </button>
                    <button type="button" id="cancelSignup"
                            style="background: #6c757d; color: white; border: none; padding: 12px 24px; border-radius: 6px; font-weight: 600; cursor: pointer;">
                        Cancel
                    </button>
                </div>
            </form>
        `;
        
        overlay.appendChild(modal);
        document.body.appendChild(overlay);
        
        // Focus on name input
        setTimeout(() => {
            document.getElementById('quickName').focus();
        }, 100);
        
        // Handle form submission
        document.getElementById('quickSignupForm').addEventListener('submit', (e) => {
            e.preventDefault();
            const name = document.getElementById('quickName').value.trim();
            const email = document.getElementById('quickEmail').value.trim();
            const company = document.getElementById('quickCompany').value.trim();
            
            if (name && email) {
                document.body.removeChild(overlay);
                resolve({
                    success: true,
                    userInfo: { name, email, company: company || null }
                });
            }
        });
        
        // Handle cancel
        document.getElementById('cancelSignup').addEventListener('click', () => {
            document.body.removeChild(overlay);
            resolve({ success: false });
        });
        
        // Handle overlay click to close
        overlay.addEventListener('click', (e) => {
            if (e.target === overlay) {
                document.body.removeChild(overlay);
                resolve({ success: false });
            }
        });
        
        // Handle escape key
        const handleEscape = (e) => {
            if (e.key === 'Escape') {
                document.body.removeChild(overlay);
                document.removeEventListener('keydown', handleEscape);
                resolve({ success: false });
            }
        };
        document.addEventListener('keydown', handleEscape);
    });
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