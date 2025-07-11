// Global variables
let isTyping = false;
const AWS_API_URL = 'https://yklxgcoavf.execute-api.us-west-1.amazonaws.com/dev'

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
    
    // Call AWS Lambda function with current agent
    callOpenAIAgent(message, currentAgent).then(response => {
        hideTypingIndicator();
        addMessage(response, 'ai');
    }).catch(error => {
        hideTypingIndicator();
        addMessage(error.message || "I'm sorry, I'm having trouble connecting right now. Please try again later.", 'ai');
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

// Configuration for different agents
const AGENTS = {
    'general': {
        name: 'iQore AI Assistant',
        endpoint: AWS_API_URL,
        description: 'General iQore assistant for technology and business questions'
    },
    // Future agents can be added here
    // 'technical': {
    //     name: 'iQore Technical Expert',
    //     endpoint: AWS_API_URL + '/technical',
    //     description: 'Deep technical guidance on iQD/iCD architecture'
    // },
    // 'business': {
    //     name: 'iQore Business Advisor',
    //     endpoint: AWS_API_URL + '/business',
    //     description: 'Investment and business strategy guidance'
    // }
};

// Current active agent
let currentAgent = 'general';

// Generate unique session ID
const sessionId = 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);

// Function to connect to your AWS Lambda function
async function callOpenAIAgent(message, agentType = 'general') {
    try {
        const agent = AGENTS[agentType];
        if (!agent) {
            throw new Error(`Unknown agent type: ${agentType}`);
        }

        const response = await fetch(agent.endpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                // Add any authentication headers if needed
                // 'Authorization': 'Bearer YOUR_TOKEN',
            },
            body: JSON.stringify({
                message: message,
                timestamp: new Date().toISOString(),
                sessionId: sessionId,
                agentType: agentType,
                userId: 'anonymous_user', // You can implement user identification later
            })
        });
        
        if (!response.ok) {
            // Get error details from response if available
            let errorMessage = `HTTP error! status: ${response.status}`;
            try {
                const errorData = await response.json();
                errorMessage = errorData.error || errorData.message || errorMessage;
            } catch (e) {
                // If we can't parse the error response, use the status message
            }
            throw new Error(errorMessage);
        }
        
        const data = await response.json();
        
        // Handle the response from your Lambda function
        const reply = data.reply || data.response || data.message;
        
        if (!reply) {
            throw new Error("No response content received from the agent");
        }
        
        // Log usage information if available (for debugging)
        if (data.usage) {
            console.log('OpenAI API Usage:', data.usage);
        }
        
        return reply;
        
    } catch (error) {
        console.error('Error calling OpenAI agent:', error);
        
        // Provide user-friendly error messages
        if (error.message.includes('HTTP error! status: 402')) {
            throw new Error("I'm temporarily unavailable due to API limits. Please try again later.");
        } else if (error.message.includes('HTTP error! status: 401')) {
            throw new Error("Authentication issue. Please contact support.");
        } else if (error.message.includes('HTTP error! status: 500')) {
            throw new Error("I'm experiencing technical difficulties. Please try again in a moment.");
        } else if (error.name === 'TypeError' && error.message.includes('fetch')) {
            throw new Error("Connection issue. Please check your internet connection and try again.");
        }
        
        throw error; // Re-throw to be handled in sendMessage()
    }
}

// Function to switch between agents (for future use)
function switchAgent(agentType) {
    if (AGENTS[agentType]) {
        currentAgent = agentType;
        console.log(`Switched to agent: ${AGENTS[agentType].name}`);
        
        // You could add UI updates here to show which agent is active
        // For example: updateAgentIndicator(agentType);
    } else {
        console.error(`Unknown agent type: ${agentType}`);
    }
}