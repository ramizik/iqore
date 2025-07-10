// Global variables
let isTyping = false;

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    const messageInput = document.getElementById('messageInput');
    const sendButton = document.getElementById('sendButton');
    
    // Enable/disable send button based on input
    messageInput.addEventListener('input', function() {
        const hasText = this.value.trim().length > 0;
        sendButton.disabled = !hasText || isTyping;
    });
    
    // Focus on input when page loads
    messageInput.focus();
});

// Handle keyboard shortcuts
function handleKeyDown(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        if (!isTyping && event.target.value.trim()) {
            sendMessage();
        }
    }
}

// Send message function
function sendMessage() {
    const messageInput = document.getElementById('messageInput');
    const message = messageInput.value.trim();
    
    if (!message || isTyping) return;
    
    // Add user message
    addMessage(message, 'user');
    
    // Clear input
    messageInput.value = '';
    document.getElementById('sendButton').disabled = true;
    
    // Show typing indicator and simulate AI response
    showTypingIndicator();
    
    // Call AWS Lambda function
    callOpenAIAgent(message).then(response => {
        hideTypingIndicator();
        addMessage(response, 'ai');
    }).catch(error => {
        hideTypingIndicator();
        addMessage("I'm sorry, I'm having trouble connecting right now. Please try again later.", 'ai');
        console.error('Error:', error);
    });
}

// Add message to chat
function addMessage(text, sender) {
    const messagesContainer = document.getElementById('messagesContainer');
    
    const messageGroup = document.createElement('div');
    messageGroup.className = `message-group ${sender}-message`;
    
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    
    const avatarDiv = document.createElement('div');
    avatarDiv.className = `avatar ${sender === 'user' ? 'user-avatar' : 'ai-avatar-small'}`;
    
    avatar.appendChild(avatarDiv);
    
    const messageBubble = document.createElement('div');
    messageBubble.className = 'message-bubble';
    messageBubble.innerHTML = `<p>${text}</p>`;
    
    messageGroup.appendChild(avatar);
    messageGroup.appendChild(messageBubble);
    
    messagesContainer.appendChild(messageGroup);
    
    // Scroll to bottom
    scrollToBottom();
}

// Show typing indicator
function showTypingIndicator() {
    isTyping = true;
    document.getElementById('sendButton').disabled = true;
    
    const messagesContainer = document.getElementById('messagesContainer');
    
    const typingGroup = document.createElement('div');
    typingGroup.className = 'message-group ai-message';
    typingGroup.id = 'typingIndicator';
    
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    
    const avatarDiv = document.createElement('div');
    avatarDiv.className = 'avatar ai-avatar-small';
    
    avatar.appendChild(avatarDiv);
    
    const messageBubble = document.createElement('div');
    messageBubble.className = 'message-bubble';
    
    const typingIndicator = document.createElement('div');
    typingIndicator.className = 'typing-indicator';
    typingIndicator.innerHTML = `
        <span>AI is thinking</span>
        <div class="typing-dots">
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
        </div>
    `;
    
    messageBubble.appendChild(typingIndicator);
    typingGroup.appendChild(avatar);
    typingGroup.appendChild(messageBubble);
    
    messagesContainer.appendChild(typingGroup);
    scrollToBottom();
}

// Hide typing indicator
function hideTypingIndicator() {
    isTyping = false;
    const typingIndicator = document.getElementById('typingIndicator');
    if (typingIndicator) {
        typingIndicator.remove();
    }
    
    // Re-enable send button if there's text
    const messageInput = document.getElementById('messageInput');
    const sendButton = document.getElementById('sendButton');
    sendButton.disabled = !messageInput.value.trim();
}

// Scroll to bottom of chat
function scrollToBottom() {
    const chatMain = document.querySelector('.chat-main');
    chatMain.scrollTop = chatMain.scrollHeight;
}

// Function to connect to your AWS Lambda function
async function callOpenAIAgent(message) {
    try {
        // Replace with your actual AWS Lambda API Gateway endpoint
        // Format: https://your-api-id.execute-api.YOUR-REGION.amazonaws.com/prod/chat
        const response = await fetch('https://fg9paa2iz5.execute-api.us-west-1.amazonaws.com/default/myChatbot', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                // Add any authentication headers if needed
                // 'Authorization': 'Bearer YOUR_TOKEN',
            },
            body: JSON.stringify({
                message: message,
                timestamp: new Date().toISOString(),
                // Add any other parameters your Lambda function expects:
                // userId: 'user123',
                // sessionId: 'session456',
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        // Handle the response from your Lambda function
        // Adjust based on your OpenAI Agentic SDK Lambda response structure
        return data.reply || data.response || data.message || "I received your message but couldn't generate a response.";
        
    } catch (error) {
        console.error('Error calling OpenAI agent:', error);
        throw error; // Re-throw to be handled in sendMessage()
    }
}