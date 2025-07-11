const OpenAI = require('openai');

// In-memory conversation storage (for development)
// In production, use DynamoDB, Redis, or another persistent storage
const conversationHistory = new Map();

// iQore system prompt for the AI assistant
const SYSTEM_PROMPT = `You are the iQore AI Assistant, an expert guide for iQore's hybrid quantum-classical computing platform. 

About iQore:
- iQore is a software-native, physics-augmented execution framework
- Uses two modules: iQD (Intelligent Quantum Dynamics) and iCD (Intelligent Classical Dynamics)  
- Simulates quantum behavior on classical CPUs without requiring quantum hardware
- Focuses on optimization, cryptography, and AI acceleration
- Features entropy-driven collapse, fidelity tracking, and quantum state replayability
- Can simulate quantum algorithms like QPE (Quantum Phase Estimation) with 100% accuracy

Your role:
- Help users understand iQore's technology, architecture, and applications
- Explain quantum computing concepts in accessible terms
- Provide technical guidance on hybrid quantum-classical computing
- Answer questions about iQore's business vision and roadmap
- Be helpful, knowledgeable, and professional

Keep responses focused, informative, and tailored to the user's level of expertise.`;

// Get conversation history for a session
function getConversationHistory(sessionId) {
    if (!sessionId) {
        console.log('No session ID provided, returning empty history');
        return [];
    }
    
    const history = conversationHistory.get(sessionId) || [];
    console.log(`Retrieved ${history.length} messages for session ${sessionId}`);
    return history;
}

// Save conversation history for a session
function saveConversationHistory(sessionId, messages) {
    if (!sessionId) {
        console.log('No session ID provided, not saving history');
        return;
    }
    
    // Ensure messages is an array and all messages have proper format
    if (!Array.isArray(messages)) {
        console.error('Messages is not an array:', typeof messages);
        return;
    }
    
    // Validate each message
    const validMessages = messages.filter(msg => {
        if (!msg || typeof msg !== 'object' || !msg.role || !msg.content) {
            console.error('Invalid message in history:', msg);
            return false;
        }
        return true;
    });
    
    conversationHistory.set(sessionId, validMessages);
    console.log(`Saved ${validMessages.length} messages for session ${sessionId}`);
    
    // Clean up old conversations (keep only last 50 sessions)
    if (conversationHistory.size > 50) {
        const firstKey = conversationHistory.keys().next().value;
        conversationHistory.delete(firstKey);
        console.log(`Cleaned up old conversation: ${firstKey}`);
    }
}

// Estimate token count (rough approximation: 4 characters per token)
function estimateTokenCount(messages) {
    const totalChars = messages.reduce((sum, msg) => {
        return sum + (msg.content ? msg.content.length : 0);
    }, 0);
    return Math.ceil(totalChars / 4);
}

// Trim conversation history to fit within token limits
function trimConversationHistory(history, maxTokens = 3000) {
    const systemPromptTokens = Math.ceil(SYSTEM_PROMPT.length / 4);
    const availableTokens = maxTokens - systemPromptTokens - 200; // Reserve 200 tokens for response
    
    let trimmedHistory = [...history];
    let currentTokens = estimateTokenCount(trimmedHistory);
    
    // Remove older messages if we exceed token limit
    while (currentTokens > availableTokens && trimmedHistory.length > 2) {
        // Remove pairs (user + assistant) from the beginning
        trimmedHistory.splice(0, 2);
        currentTokens = estimateTokenCount(trimmedHistory);
    }
    
    return trimmedHistory;
}

exports.handler = async (event) => {
    // Add debug logging
    console.log('=== LAMBDA FUNCTION START ===');
    console.log('HTTP Method:', event.httpMethod);
    console.log('Headers:', JSON.stringify(event.headers, null, 2));
    console.log('Body:', event.body);
    console.log('Environment Check:', {
        hasOpenAIKey: !!process.env.OPENAI_API_KEY,
        openAIKeyLength: process.env.OPENAI_API_KEY ? process.env.OPENAI_API_KEY.length : 0,
        nodeEnv: process.env.NODE_ENV
    });
    
    // Set CORS headers
    const headers = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Content-Type',
        'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
        'Content-Type': 'application/json'
    };

    try {
        // Handle preflight OPTIONS request
        if (event.httpMethod === 'OPTIONS') {
            return {
                statusCode: 200,
                headers,
                body: JSON.stringify({ message: 'CORS preflight successful' })
            };
        }

        // Handle GET request (for testing)
        if (event.httpMethod === 'GET') {
            return {
                statusCode: 200,
                headers,
                body: JSON.stringify({ 
                    message: 'iQore AI Assistant Lambda is running',
                    status: 'ready',
                    timestamp: new Date().toISOString(),
                    activeConversations: conversationHistory.size
                })
            };
        }

        // Handle POST request (main chat functionality)
        if (event.httpMethod === 'POST') {
            // Parse the request body
            let requestBody;
            try {
                requestBody = JSON.parse(event.body);
            } catch (error) {
                console.error('JSON parsing error:', error);
                return {
                    statusCode: 400,
                    headers,
                    body: JSON.stringify({ 
                        error: 'Invalid JSON in request body',
                        details: error.message 
                    })
                };
            }

            const { message, sessionId, userId } = requestBody;

            console.log(`Processing request - SessionId: ${sessionId}, Message length: ${message?.length}`);

            // Validate required fields
            if (!message || typeof message !== 'string' || message.trim().length === 0) {
                console.log('Validation failed: Invalid message');
                return {
                    statusCode: 400,
                    headers,
                    body: JSON.stringify({ 
                        error: 'Message is required and must be a non-empty string' 
                    })
                };
            }

            // Get existing conversation history
            const history = getConversationHistory(sessionId);
            console.log(`Retrieved conversation history for session ${sessionId}: ${history.length} messages`);
            
            // Trim conversation history to prevent token limits
            const trimmedHistory = trimConversationHistory(history);
            console.log(`Trimmed history from ${history.length} to ${trimmedHistory.length} messages`);
            
            // Build messages array for OpenAI
            const messages = [
                {
                    role: 'system',
                    content: SYSTEM_PROMPT
                },
                ...trimmedHistory,  // Add trimmed conversation history
                {
                    role: 'user', 
                    content: message.trim()
                }
            ];

            const estimatedTokens = estimateTokenCount(messages);
            console.log(`Processing message for session ${sessionId}, messages: ${messages.length}, estimated tokens: ${estimatedTokens}`);

            // Validate message format for OpenAI
            for (let i = 0; i < messages.length; i++) {
                const msg = messages[i];
                if (!msg.role || !msg.content) {
                    console.error(`Invalid message format at index ${i}:`, msg);
                    return {
                        statusCode: 400,
                        headers,
                        body: JSON.stringify({ 
                            error: 'Invalid conversation history format',
                            message: 'Conversation history contains invalid messages'
                        })
                    };
                }
            }

            // Initialize OpenAI client inside handler for better error handling
            let openai;
            try {
                console.log('Initializing OpenAI client...');
                if (!process.env.OPENAI_API_KEY) {
                    console.error('CRITICAL: OPENAI_API_KEY environment variable is not set');
                    return {
                        statusCode: 500,
                        headers,
                        body: JSON.stringify({ 
                            error: 'Configuration error',
                            message: 'OpenAI API key is not configured'
                        })
                    };
                }
                
                openai = new OpenAI({
                    apiKey: process.env.OPENAI_API_KEY,
                    timeout: 25000, // 25 second timeout (Lambda has 30s max)
                });
                console.log('OpenAI client initialized successfully');
            } catch (initError) {
                console.error('Failed to initialize OpenAI client:', initError);
                return {
                    statusCode: 500,
                    headers,
                    body: JSON.stringify({ 
                        error: 'Configuration error',
                        message: 'Failed to initialize OpenAI service',
                        details: initError.message
                    })
                };
            }

            // Call OpenAI API with enhanced error handling
            let completion;
            try {
                console.log('Making OpenAI API call...');
                
                const startTime = Date.now();
                
                // Create a timeout promise
                const timeoutPromise = new Promise((_, reject) => {
                    setTimeout(() => reject(new Error('OpenAI API call timed out after 25 seconds')), 25000);
                });
                
                // Race between the API call and timeout
                completion = await Promise.race([
                    openai.chat.completions.create({
                        model: 'gpt-4o-mini',
                        messages: messages,
                        max_tokens: 500,
                        temperature: 0.7,
                        top_p: 1,
                        frequency_penalty: 0,
                        presence_penalty: 0
                    }),
                    timeoutPromise
                ]);
                const endTime = Date.now();
                
                console.log(`OpenAI API call successful in ${endTime - startTime}ms, tokens used: ${completion.usage?.total_tokens || 'unknown'}`);
                console.log('OpenAI response received:', completion.choices[0]?.message?.content?.substring(0, 100) + '...');
                
            } catch (openaiError) {
                console.error('OpenAI API error details:', {
                    name: openaiError.name,
                    message: openaiError.message,
                    status: openaiError.status,
                    code: openaiError.code,
                    type: openaiError.type,
                    stack: openaiError.stack
                });
                
                // Handle timeout specifically
                if (openaiError.message?.includes('timed out')) {
                    console.error('OpenAI API call timed out');
                    return {
                        statusCode: 408,
                        headers,
                        body: JSON.stringify({ 
                            error: 'Request timeout',
                            message: 'The AI service took too long to respond. Please try again.',
                            timeout: true
                        })
                    };
                }
                
                // Handle specific OpenAI errors
                if (openaiError.status === 429) {
                    return {
                        statusCode: 429,
                        headers,
                        body: JSON.stringify({ 
                            error: 'Rate limit exceeded',
                            message: 'Too many requests. Please wait a moment and try again.',
                            retryAfter: 30
                        })
                    };
                }
                
                if (openaiError.status === 400) {
                    return {
                        statusCode: 400,
                        headers,
                        body: JSON.stringify({ 
                            error: 'Bad request to OpenAI',
                            message: 'Invalid request format or parameters',
                            details: openaiError.message,
                            estimatedTokens: estimatedTokens
                        })
                    };
                }
                
                if (openaiError.status === 401) {
                    return {
                        statusCode: 401,
                        headers,
                        body: JSON.stringify({ 
                            error: 'Authentication failed',
                            message: 'Invalid API key or authentication error'
                        })
                    };
                }
                
                // Re-throw for general error handler
                throw openaiError;
            }

            const aiResponse = completion.choices[0]?.message?.content;
            
            if (!aiResponse) {
                console.error('No AI response received from OpenAI');
                return {
                    statusCode: 500,
                    headers,
                    body: JSON.stringify({ 
                        error: 'No response from AI',
                        message: 'The AI service did not return a response'
                    })
                };
            }

            // Update conversation history
            const updatedHistory = [
                ...trimmedHistory,
                { role: 'user', content: message.trim() },
                { role: 'assistant', content: aiResponse }
            ];
            
            console.log(`Saving updated conversation history: ${updatedHistory.length} messages`);
            saveConversationHistory(sessionId, updatedHistory);

            console.log(`Request completed successfully for session ${sessionId}`);
            
            const successResponse = {
                statusCode: 200,
                headers,
                body: JSON.stringify({
                    reply: aiResponse,
                    timestamp: new Date().toISOString(),
                    sessionId: sessionId || null,
                    userId: userId || null,
                    model: 'gpt-4o-mini',
                    usage: completion.usage,
                    conversationLength: updatedHistory.length,
                    estimatedTokens: estimatedTokens
                })
            };
            
            console.log('=== RETURNING SUCCESS RESPONSE ===');
            return successResponse;
        }

        // Handle unsupported HTTP methods
        return {
            statusCode: 405,
            headers,
            body: JSON.stringify({ 
                error: 'Method not allowed',
                allowedMethods: ['GET', 'POST', 'OPTIONS']
            })
        };

    } catch (error) {
        console.error('=== LAMBDA EXECUTION ERROR ===');
        console.error('Error details:', {
            name: error.name,
            message: error.message,
            stack: error.stack,
            code: error.code,
            status: error.status
        });
        
        // Handle OpenAI API errors specifically
        if (error.code === 'insufficient_quota') {
            return {
                statusCode: 402,
                headers,
                body: JSON.stringify({ 
                    error: 'OpenAI API quota exceeded',
                    message: 'Please check your OpenAI billing settings'
                })
            };
        }

        if (error.code === 'invalid_api_key') {
            return {
                statusCode: 401,
                headers,
                body: JSON.stringify({ 
                    error: 'Invalid OpenAI API key',
                    message: 'Please verify your API key configuration'
                })
            };
        }

        // Enhanced error response with more details
        return {
            statusCode: 500,
            headers,
            body: JSON.stringify({ 
                error: 'Internal server error',
                message: 'An unexpected error occurred while processing your request',
                timestamp: new Date().toISOString(),
                details: process.env.NODE_ENV === 'development' ? {
                    message: error.message,
                    stack: error.stack
                } : undefined
            })
        };
    }
}; 