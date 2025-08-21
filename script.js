// iQore Chatbot Frontend - script.js
// Configuration
const API_BASE_URL = 'https://iqoregpt-529970623471.europe-west1.run.app';  // Change this for production deployment
const CHAT_ENDPOINT = `${API_BASE_URL}/api/v1/chat`;

// Global state
let chatHistory = [];
let isLoading = false;
let currentSuggestions = [];
let hasUserSentMessage = false; // Track if user has sent their first message

// Session timer variables
let sessionTimer = null;
let sessionTimeLeft = 600; // 10 minutes in seconds
let oneMinuteWarningShown = false;

// DOM elements
const messagesContainer = document.getElementById('messagesContainer');
const messageInput = document.getElementById('messageInput');
const sendButton = document.getElementById('sendButton');
const suggestionsContainer = document.getElementById('suggestionsContainer');

// Initialize the chat interface
document.addEventListener('DOMContentLoaded', function() {
    console.log('iQore Chatbot Frontend initialized');
    // Only focus on desktop or when widgets are not visible
    if (window.innerWidth > 768) {
        messageInput.focus();
    }
    showInitialSuggestions();
    // Fetch initial welcome message from backend
    fetchWelcomeMessage();
    
    // Initialize mobile enhancements
    setupMobileInputHandling();
    addTouchFeedback();
    addSwipeSupport();
    initializeVirtualKeyboardHandling();
});

// Comprehensive list of questions about iQore
const ALL_IQORE_QUESTIONS = [
    "What is iQore?",
    "What is iQD?",
    "What does \"physics-augmented\" mean?",
    "Why are you at IEEE Quantum Week?",
    "Who are your target customers?",
    "How does iQD work?",
    "What hardware does iQD support?",
    "How is iQD different from other quantum optimizers?",
    "Can iQD work alongside other quantum toolchains?",
    "How do you prove your claims?",
    "How does iQD improve fidelity?",
    "What is \"coherence extension\" in iQD?",
    "What's the overhead of running iQD?",
    "How do I get iQD?",
    "Who uses iQD today?",
    "Will iQD work with future fault-tolerant QPUs?",
    "How will iQore impact the quantum industry?",
    "I want to see the demo",
    "Tell me about your technology stack",
    "What can I do here?"
];

// Function to get 3 random questions from the list
function getRandomQuestions(count = 5) {
    const shuffled = [...ALL_IQORE_QUESTIONS].sort(() => 0.5 - Math.random());
    return shuffled.slice(0, count);
}

// Initial suggestions when chat starts
function showInitialSuggestions() {
    // Only show suggestions if user hasn't sent a message yet
    if (hasUserSentMessage) {
        return;
    }
    
    const initialSuggestions = getRandomQuestions(5);
    updateSuggestions(initialSuggestions);
}

// Fetch welcome message from backend
async function fetchWelcomeMessage() {
    try {
        // Send an empty first message to trigger the welcome response
        const response = await fetch(CHAT_ENDPOINT, {
            method: 'POST',
            headers: {
                'Accept': 'application/json',
                'Content-Type': 'application/json',
            },
            mode: 'cors',
            body: JSON.stringify({
                message: "Hello",  // Simple greeting to trigger welcome
                chat_history: []
            })
        });
        
        if (response.ok) {
            const data = await response.json();
            // Add the welcome message to the chat
            addMessage(data.response, 'ai');
            // Update chat history
            chatHistory = data.chat_history || [];
        } else {
            // Fallback welcome message if backend is unavailable
            addMessage("Welcome to iQore! I'm here to help you learn about our quantum-classical hybrid computing innovations. What would you like to know?", 'ai');
        }
    } catch (error) {
        console.error('Error fetching welcome message:', error);
        // Fallback welcome message on error
        addMessage("Welcome to iQore! I'm here to help you learn about our quantum-classical hybrid computing innovations. What would you like to know?", 'ai');
    }
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
    } else if (message.includes('demo') || message.includes('demonstration')) {
        // Mix demo-specific suggestions with random iQore questions
        const demoSuggestions = [
            "What is demo about?",
            "Show me the demo!",
            "What quantum algorithms iQore works with?"
        ];
        const randomQuestions = getRandomQuestions(2);
        return [...demoSuggestions.slice(0, 1), ...randomQuestions];
    } else {
        // Always show 3 random questions from our comprehensive list
        return getRandomQuestions(3);
    }
}

// Update suggestion bubbles
function updateSuggestions(suggestions) {
    // Don't show any suggestions if user has already sent a message
    if (hasUserSentMessage) {
        suggestionsContainer.style.display = 'none';
        return;
    }
    
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
        suggestionBubble.setAttribute('aria-label', `Ask: ${suggestion}`);
        suggestionBubble.setAttribute('type', 'button');
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
    
    // Mark that user has sent their first message
    hasUserSentMessage = true;
    
    // Start session timer on first message
    startSessionTimer();
    
    sendMessage();
    
    // Hide suggestions permanently
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
    
    // Mark that user has sent their first message
    hasUserSentMessage = true;
    
    // Start session timer on first message
    startSessionTimer();
    
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
        
        // Monitor for demo-related conversation to show queue widget
        monitorChatForDemo(message, data.response);
        
        // Don't show suggestions anymore after first message
        suggestionsContainer.style.display = 'none';
        
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
        
        // Don't show suggestions after first message, even on error
        suggestionsContainer.style.display = 'none';
    } finally {
        setLoading(false);
        // Only focus on input if user is not scrolling or interacting with widgets
        if (window.innerWidth > 768 || !queueWidgetVisible) {
            messageInput.focus();
        }
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
    messageText.setAttribute('aria-label', `${sender === 'ai' ? 'AI' : 'User'} message: ${text}`);
    messageBubble.appendChild(messageText);
    
    // Add ARIA attributes for accessibility
    messageBubble.setAttribute('role', 'article');
    messageBubble.setAttribute('aria-label', `${sender === 'ai' ? 'AI' : 'User'} message`);
    
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

// Handle window resize (will be replaced by throttled version in mobile enhancements)
// window.addEventListener('resize', function() {
//     scrollToBottom();
// });

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
let calendlyWidgetVisible = false; // Add this line
let queueRefreshInterval = null;
let userSessionId = null;

// Initialize queue status checking
document.addEventListener('DOMContentLoaded', function() {
    // Don't start monitoring or show widget by default
    // Widget will be shown only when demo keywords are detected in conversation
    
    // Ensure widget is hidden on page load
    hideQueueWidget();
});

function startQueueMonitoring() {
    // Don't start if already monitoring
    if (queueRefreshInterval) {
        return;
    }
    
    // Check queue status every 30 seconds
    refreshQueueStatus();
    queueRefreshInterval = setInterval(refreshQueueStatus, 30000);
    
    console.log('Queue monitoring started');
}

function stopQueueMonitoring() {
    if (queueRefreshInterval) {
        clearInterval(queueRefreshInterval);
        queueRefreshInterval = null;
    }
}

async function refreshQueueStatus(showLoadingState = false) {
    const refreshBtn = document.getElementById('refreshQueueBtn');
    
    try {
        // Show loading state if requested (manual refresh)
        if (showLoadingState && refreshBtn) {
            refreshBtn.classList.add('loading');
            refreshBtn.disabled = true;
        }
        
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
        
        // Only update the display, don't auto-show widget unless it was already visible
        // Widget visibility is controlled by demo keyword detection only
        
        console.log('Queue status refreshed successfully');

    } catch (error) {
        console.error('Error fetching queue status:', error);
        // Queue message UI removed - errors logged to console
    } finally {
        // Remove loading state
        if (showLoadingState && refreshBtn) {
            setTimeout(() => {
                refreshBtn.classList.remove('loading');
                refreshBtn.disabled = false;
            }, 500); // Brief delay to show the animation completed
        }
    }
}

function updateQueueDisplay(queueData) {
    const queueLengthEl = document.getElementById('queueLength');
    
    if (queueLengthEl) {
        const newCount = queueData.total_queue_length || 0;
        const currentCount = parseInt(queueLengthEl.textContent) || 0;
        
        // Only animate if count changed
        if (newCount !== currentCount) {
            queueLengthEl.style.transform = 'scale(1.2)';
            queueLengthEl.style.color = '#8b45ff';
            
            setTimeout(() => {
                queueLengthEl.textContent = newCount;
                queueLengthEl.style.transform = 'scale(1)';
                queueLengthEl.style.color = '#8b45ff';
            }, 150);
        } else {
            queueLengthEl.textContent = newCount;
        }
    }
    
    // Update queue message
    if (queueData.message) {
        updateQueueMessage(queueData.message, queueData.success ? 'success' : 'info');
    }
}

async function handleManualRefresh() {
    // Handle manual refresh with loading state
    await refreshQueueStatus(true);
}

function updateQueueMessage(message, type = 'info') {
    // Queue message UI removed - no longer needed
}

function showQueueWidget() {
    const widget = document.getElementById('queueStatusWidget');
    if (widget && !queueWidgetVisible) {
        widget.style.display = 'block';
        widget.classList.add('show');
        queueWidgetVisible = true;
        console.log('Queue widget is now visible');
        
        // Also show Calendly widget
        setTimeout(() => {
            showCalendlyWidget();
        }, 300); // Small delay for smoother animation
    }
}

function hideQueueWidget() {
    const widget = document.getElementById('queueStatusWidget');
    if (widget) {
        widget.style.display = 'none';
        widget.classList.remove('show');
        queueWidgetVisible = false;
    }
    
    // Also hide Calendly widget
    hideCalendlyWidget();
    
    // Also stop monitoring when widget is hidden
    stopQueueMonitoring();
}

function toggleQueueWidget() {
    const widget = document.getElementById('queueStatusWidget');
    
    // Close/hide the queue card completely
    if (widget) {
        hideQueueWidget();
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
                // Queue message UI removed - no longer needed
            }
        }
    
        // Show queue widget if not already visible and start monitoring
    showQueueWidget();
        startQueueMonitoring();
        
    } catch (error) {
        console.error('Error requesting demo:', error);
        // Queue message UI removed - show error in chat instead
        
        // Fall back to conversational method
        const demoMessage = "I'd like to join the demo queue";
        await sendMessage(demoMessage);
    }
}

async function addToQueueDirectly(userInfo) {
    try {
        // Queue message UI removed - processing happens silently
        
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
                    phone: userInfo.phone || null,
                    interest_areas: []
                }
            })
        });

        const result = await response.json();
        
        if (result.success) {
            userSessionId = result.session_id;
            // Queue message UI removed - no longer needed
            
            // Start personalized updates
            startPersonalizedStatusUpdates();
            
            // Auto-refresh queue display to show updated status
            await refreshQueueStatus();
            
            // Add success message to chat
            addSystemMessageToChat(
                `🎉 Great! You've been added to our demo queue.\n` +
                `📊 Your Position: #${result.queue_position}\n`
            );
            
        } else {
            throw new Error(result.error || 'Failed to add to queue');
        }
        
    } catch (error) {
        console.error('Direct queue addition failed:', error);
        // Queue message UI removed - error handling through chat fallback
        throw error;
    }
}

function extractUserInfoFromHistory() {
    // Extract user info from recent chat history
    const userInfo = { name: null, email: null, company: null, phone: null };
    
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
            
            // Extract phone number
            const phonePatterns = [
                /my phone is ([+]?[\d\s\-\(\)]{10,})/i,
                /phone number is ([+]?[\d\s\-\(\)]{10,})/i,
                /call me at ([+]?[\d\s\-\(\)]{10,})/i,
                /number is ([+]?[\d\s\-\(\)]{10,})/i
            ];
            
            for (const pattern of phonePatterns) {
                const phoneMatch = exchange.user.match(pattern);
                if (phoneMatch) {
                    userInfo.phone = phoneMatch[1].trim();
                    break;
                }
            }
            
            // If no phone pattern found, check if message looks like a standalone phone number
            if (!userInfo.phone) {
                const digitsOnly = exchange.user.replace(/\D/g, '');
                if (digitsOnly.length >= 10 && exchange.user.length <= 20) {
                    userInfo.phone = exchange.user.trim();
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
                <div style="margin-bottom: 15px;">
                    <label style="display: block; margin-bottom: 5px; font-weight: 500; color: #333;">Company (Optional)</label>
                    <input type="text" id="quickCompany"
                           style="width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 6px; font-size: 14px;"
                           placeholder="Enter your company name">
                </div>
                <div style="margin-bottom: 20px;">
                    <label style="display: block; margin-bottom: 5px; font-weight: 500; color: #333;">Phone (Optional)</label>
                    <input type="tel" id="quickPhone"
                           style="width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 6px; font-size: 14px;"
                           placeholder="Enter your phone number">
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
            const phone = document.getElementById('quickPhone').value.trim();
            
            if (name && email) {
                document.body.removeChild(overlay);
                resolve({
                    success: true,
                    userInfo: { 
                        name, 
                        email, 
                        company: company || null,
                        phone: phone || null
                    }
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

// Removed checkForDemoContext function - widget now shows only on active demo keyword detection

// Monitor chat for demo-related conversations
function monitorChatForDemo(userMessage, aiResponse) {
    const demoKeywords = ['demo', 'demonstration', 'queue', 'signup', 'reserve', 'join', 'show me', 'live demo'];
    const message = (userMessage + ' ' + aiResponse).toLowerCase();
    
    const isDemoRelated = demoKeywords.some(keyword => message.includes(keyword));
    
    if (isDemoRelated) {
        console.log('Demo keywords detected, showing queue widget');
        
        setTimeout(() => {
            // Show the queue widget for the first time
            showQueueWidget();
            
            // Start queue monitoring now that demo conversation has begun
            startQueueMonitoring();
            
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
                    // Queue message UI removed - position updates handled silently
                    console.log(`Queue position: #${data.queue_position}, Wait time: ~${data.estimated_wait_time} min`);
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
window.toggleCalendlyWidget = toggleCalendlyWidget; // Add this line
window.requestDemo = requestDemo;
window.handleKeyDown = handleKeyDown; 
window.handleManualRefresh = handleManualRefresh;

// Note: refreshQueueStatus is kept for internal monitoring but not exposed globally 

// Session Timer Functions
function startSessionTimer() {
    // Don't start if timer is already running
    if (sessionTimer) {
        return;
    }
    
    // Show timer element
    const timerElement = document.getElementById('sessionTimer');
    if (timerElement) {
        timerElement.style.display = 'flex';
    }
    
    // Update timer display immediately
    updateTimerDisplay();
    
    // Start countdown
    sessionTimer = setInterval(() => {
        sessionTimeLeft--;
        updateTimerDisplay();
        
        // Show 1-minute warning
        if (sessionTimeLeft === 60 && !oneMinuteWarningShown) {
            showOneMinuteWarning();
            oneMinuteWarningShown = true;
        }
        
        // End session when timer reaches 0
        if (sessionTimeLeft <= 0) {
            endSession();
        }
    }, 1000);
    
    console.log('Session timer started - 10 minutes');
}

function updateTimerDisplay() {
    const timerText = document.getElementById('timerText');
    const timerElement = document.getElementById('sessionTimer');
    
    if (timerText) {
        const minutes = Math.floor(sessionTimeLeft / 60);
        const seconds = sessionTimeLeft % 60;
        timerText.textContent = `${minutes}:${seconds.toString().padStart(2, '0')}`;
        
        // Add warning style when 1 minute or less
        if (sessionTimeLeft <= 60 && timerElement) {
            timerElement.classList.add('warning');
        }
    }
}

function showOneMinuteWarning() {
    // Add warning message to chat
    addMessage(
        "⏰ Your session will end in 1 minute. We'd love to continue our conversation at our booth! Please visit us for more detailed discussions about iQore's quantum computing solutions.",
        'ai'
    );
    
    console.log('1-minute warning shown to user');
}

function endSession() {
    // Clear the timer
    if (sessionTimer) {
        clearInterval(sessionTimer);
        sessionTimer = null;
    }
    
    // Show final message
    addMessage(
        "🕐 Your session has ended. Thank you for your interest in iQore! Please visit our booth for continued assistance and live demonstrations of our quantum computing technology.",
        'ai'
    );
    
    console.log('Session ended - refreshing page in 3 seconds');
    
    // Refresh page after 3 seconds to allow user to read the message
    setTimeout(() => {
        window.location.reload();
    }, 3000);
}

// Cleanup timer on page unload
window.addEventListener('beforeunload', function() {
    if (sessionTimer) {
        clearInterval(sessionTimer);
    }
});

// Function to restart the session
function restartSession() {
    // Clear existing timer
    if (sessionTimer) {
        clearInterval(sessionTimer);
        sessionTimer = null;
    }
    
    // Reset timer variables
    sessionTimeLeft = 600; // 10 minutes
    oneMinuteWarningShown = false;
    
    // Reset user message flag
    hasUserSentMessage = false;
    
    // Hide timer
    const timerElement = document.getElementById('sessionTimer');
    if (timerElement) {
        timerElement.style.display = 'none';
        timerElement.classList.remove('warning');
    }
    
    // Clear chat history
    chatHistory = [];
    
    // Clear messages container
    const messagesContainer = document.getElementById('messagesContainer');
    if (messagesContainer) {
        messagesContainer.innerHTML = '';
    }
    
    // Show initial suggestions again
    showInitialSuggestions();
    
    // Fetch welcome message again
    fetchWelcomeMessage();
    
    // Focus on input only if not on mobile or widgets not visible
    if (window.innerWidth > 768 || !queueWidgetVisible) {
        messageInput.focus();
    }
    
    console.log('Session restarted by user');
}

// Add Calendly widget functions
function showCalendlyWidget() {
    const widget = document.getElementById('calendlyWidget');
    if (widget && !calendlyWidgetVisible) {
        widget.style.display = 'block';
        widget.classList.add('show');
        calendlyWidgetVisible = true;
        console.log('Calendly widget is now visible');
    }
}

function hideCalendlyWidget() {
    const widget = document.getElementById('calendlyWidget');
    if (widget) {
        widget.style.display = 'none';
        widget.classList.remove('show');
        calendlyWidgetVisible = false;
    }
}

function toggleCalendlyWidget() {
    const widget = document.getElementById('calendlyWidget');
    
    // Close/hide the Calendly widget completely
    if (widget) {
        hideCalendlyWidget();
    }
}

// Update existing showQueueWidget function
function showQueueWidget() {
    const widget = document.getElementById('queueStatusWidget');
    if (widget && !queueWidgetVisible) {
        widget.style.display = 'block';
        widget.classList.add('show');
        queueWidgetVisible = true;
        console.log('Queue widget is now visible');
        
        // Add widgets-visible class to chat container on mobile
        if (window.innerWidth <= 768) {
            const chatContainer = document.querySelector('.chat-container');
            if (chatContainer) {
                chatContainer.classList.add('widgets-visible');
            }
            
            // Scroll to widgets after a short delay to show them
            setTimeout(() => {
                scrollToWidgets();
            }, 800);
        }
        
        // Also show Calendly widget with a small delay for smoother animation
        setTimeout(() => {
            showCalendlyWidget();
        }, 300);
    }
}

// Update existing hideQueueWidget function
function hideQueueWidget() {
    const widget = document.getElementById('queueStatusWidget');
    if (widget) {
        widget.style.display = 'none';
        widget.classList.remove('show');
        queueWidgetVisible = false;
    }
    
    // Remove widgets-visible class from chat container on mobile
    if (window.innerWidth <= 768) {
        const chatContainer = document.querySelector('.chat-container');
        if (chatContainer) {
            chatContainer.classList.remove('widgets-visible');
        }
    }
    
    // Also hide Calendly widget
    hideCalendlyWidget();
    
    // Also stop monitoring when widget is hidden
    stopQueueMonitoring();
}

// Update existing toggleQueueWidget function
function toggleQueueWidget() {
    const widget = document.getElementById('queueStatusWidget');
    
    // Close/hide the queue card completely
    if (widget) {
        hideQueueWidget();
    }
}

// Mobile Enhancement Functions
let initialViewportHeight = window.innerHeight;

function setupMobileInputHandling() {
    // Prevent zoom on input focus (iOS Safari)
    messageInput.addEventListener('focus', function() {
        // Temporarily increase font size to prevent zoom
        if (window.innerWidth <= 768) {
            messageInput.style.fontSize = '16px';
        }
    });
    
    // Handle virtual keyboard appearance
    window.addEventListener('resize', throttle(function() {
        // Only scroll to input when keyboard appears if widgets are not visible
        // This prevents interrupting user's widget interaction
        if (document.activeElement === messageInput && !queueWidgetVisible) {
            setTimeout(() => {
                messageInput.scrollIntoView({ behavior: 'smooth', block: 'center' });
            }, 300);
        }
        // Only auto-scroll chat to bottom if user is actively in chat area
        if (!queueWidgetVisible) {
            scrollToBottom();
        }
    }, 100));
}

function addTouchFeedback() {
    // Add haptic feedback for touch interactions (if available)
    function addTouchListeners(element) {
        element.addEventListener('touchstart', function() {
            if (navigator.vibrate) {
                navigator.vibrate(10); // Subtle haptic feedback
            }
            // Add visual feedback
            element.style.transform = 'scale(0.95)';
        });
        
        element.addEventListener('touchend', function() {
            // Reset visual feedback
            setTimeout(() => {
                element.style.transform = '';
            }, 100);
        });
    }
    
    // Add to existing buttons
    const buttons = document.querySelectorAll('button, .suggestion-bubble');
    buttons.forEach(addTouchListeners);
    
    // Observer for dynamically added suggestion bubbles
    const observer = new MutationObserver(function(mutations) {
        mutations.forEach(function(mutation) {
            mutation.addedNodes.forEach(function(node) {
                if (node.nodeType === Node.ELEMENT_NODE) {
                    if (node.classList && node.classList.contains('suggestion-bubble')) {
                        addTouchListeners(node);
                    }
                    // Also check child elements
                    const newButtons = node.querySelectorAll && node.querySelectorAll('button, .suggestion-bubble');
                    if (newButtons) {
                        newButtons.forEach(addTouchListeners);
                    }
                }
            });
        });
    });
    
    // Start observing the suggestions container for changes
    const suggestionsContainer = document.getElementById('suggestionsContainer');
    if (suggestionsContainer) {
        observer.observe(suggestionsContainer, { childList: true, subtree: true });
    }
}

function initializeVirtualKeyboardHandling() {
    // Handle viewport height changes for virtual keyboard
    function handleVirtualKeyboard() {
        const currentHeight = window.visualViewport ? window.visualViewport.height : window.innerHeight;
        const heightDifference = initialViewportHeight - currentHeight;
        
        // If keyboard is likely open (height reduced significantly)
        if (heightDifference > 150) {
            document.body.classList.add('keyboard-open');
            
            // On mobile, adjust the chat container to work with scrollable layout
            if (window.innerWidth <= 768) {
                const chatContainer = document.querySelector('.chat-container');
                if (chatContainer) {
                    // Reduce chat height when keyboard is open to maintain usability
                    chatContainer.style.height = `${Math.min(currentHeight * 0.4, 300)}px`;
                    chatContainer.style.maxHeight = `${Math.min(currentHeight * 0.4, 300)}px`;
                }
            }
        } else {
            document.body.classList.remove('keyboard-open');
            
            // Reset heights on mobile
            if (window.innerWidth <= 768) {
                const chatContainer = document.querySelector('.chat-container');
                if (chatContainer) {
                    chatContainer.style.height = '';
                    chatContainer.style.maxHeight = '';
                }
            }
        }
    }

    // Listen for viewport changes (virtual keyboard)
    if (window.visualViewport) {
        window.visualViewport.addEventListener('resize', handleVirtualKeyboard);
    }
    
    // Fallback for older browsers
    window.addEventListener('resize', throttle(handleVirtualKeyboard, 100));
}

function addSwipeSupport() {
    let startX, startY, distX, distY;
    const threshold = 100; // minimum distance for swipe
    
    messagesContainer.addEventListener('touchstart', function(e) {
        const touch = e.touches[0];
        startX = touch.clientX;
        startY = touch.clientY;
    });
    
    messagesContainer.addEventListener('touchmove', function(e) {
        // Prevent default to avoid scrolling issues on horizontal swipes
        const touch = e.touches[0];
        const currentDistX = Math.abs(touch.clientX - startX);
        const currentDistY = Math.abs(touch.clientY - startY);
        
        if (currentDistX > currentDistY && currentDistX > 10) {
            e.preventDefault();
        }
    });
    
    messagesContainer.addEventListener('touchend', function(e) {
        const touch = e.changedTouches[0];
        distX = touch.clientX - startX;
        distY = touch.clientY - startY;
        
        // Check if it's a horizontal swipe
        if (Math.abs(distX) > Math.abs(distY) && Math.abs(distX) > threshold) {
            if (distX > 0) {
                // Swipe right - could show additional options
                console.log('Swiped right - future feature');
            } else {
                // Swipe left - could hide widgets or show menu
                console.log('Swiped left - future feature');
            }
        }
    });
}

// Throttle function for better performance
function throttle(func, delay) {
    let timeoutId;
    let lastExecTime = 0;
    return function (...args) {
        const currentTime = Date.now();
        
        if (currentTime - lastExecTime > delay) {
            func.apply(this, args);
            lastExecTime = currentTime;
        } else {
            clearTimeout(timeoutId);
            timeoutId = setTimeout(() => {
                func.apply(this, args);
                lastExecTime = Date.now();
            }, delay - (currentTime - lastExecTime));
        }
    };
}

// Scroll to widgets on mobile to help users find them
function scrollToWidgets() {
    if (window.innerWidth <= 768) {
        const widgetColumn = document.querySelector('.widget-column');
        if (widgetColumn) {
            widgetColumn.scrollIntoView({ 
                behavior: 'smooth', 
                block: 'start',
                inline: 'nearest'
            });
        }
    }
}

// Handle window resize to manage widgets-visible class
window.addEventListener('resize', throttle(function() {
    const chatContainer = document.querySelector('.chat-container');
    if (chatContainer && queueWidgetVisible) {
        if (window.innerWidth <= 768) {
            chatContainer.classList.add('widgets-visible');
        } else {
            chatContainer.classList.remove('widgets-visible');
        }
    }
}, 100));