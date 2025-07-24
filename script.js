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
        "What can you do?",
        "I want see the demo."
    ];
    updateSuggestions(initialSuggestions);
}

// Generate contextual suggestions based on the last AI response
function generateContextualSuggestions(lastAiMessage) {
    // Simple keyword-based suggestion generation
    const message = lastAiMessage.toLowerCase();
    
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
            "What quantum algorithms does iQore work with?"
        ];
    } else {
        // Default contextual suggestions
        return [
            "Can you give me a specific example?",
            "How does this benefit enterprises?",
            "What's the next step to learn more?"
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
        
        // Generate and show new contextual suggestions
        const newSuggestions = generateContextualSuggestions(data.response);
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

// Export functions for global access (if needed)
window.sendMessage = sendMessage;
window.handleKeyDown = handleKeyDown; 