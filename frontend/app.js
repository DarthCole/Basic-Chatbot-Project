// Ghana Chatbot Frontend
// File: frontend/app.js

// Configuration
const API_BASE_URL = '/api';

// DOM Elements - CORRECTED to match HTML IDs
const chatMessages = document.getElementById('messagesContainer');
const userInput = document.getElementById('messageInput');
const sendButton = document.getElementById('sendBtn');
const newChatButton = document.getElementById('newChatBtn');
const chatList = document.getElementById('chatList');
const voiceToggle = document.getElementById('voiceBtn');
const clearChatsButton = document.getElementById('clearChatBtn');
const uploadPdfButton = document.getElementById('uploadPdfBtn');
const pdfFileInput = document.getElementById('pdfInput');
const statusIndicator = document.getElementById('statusIndicator');

// State
let currentChatId = null;
let isVoiceEnabled = false;
let recognition = null;

// Initialize Speech Recognition
function initSpeechRecognition() {
    if ('webkitSpeechRecognition' in window) {
        recognition = new webkitSpeechRecognition();
        recognition.continuous = false;
        recognition.interimResults = false;
        recognition.lang = 'en-US';
        
        recognition.onstart = function() {
            console.log('Voice recognition started');
            voiceToggle.textContent = 'Listening...';
            voiceToggle.classList.add('listening');
        };
        
        recognition.onresult = function(event) {
            const transcript = event.results[0][0].transcript;
            userInput.value = transcript;
            sendMessage();
        };
        
        recognition.onerror = function(event) {
            console.error('Speech recognition error', event.error);
            voiceToggle.textContent = 'Voice';
            voiceToggle.classList.remove('listening');
            if (event.error === 'no-speech') {
                showNotification('No speech detected. Please try again.', 'warning');
            }
        };
        
        recognition.onend = function() {
            voiceToggle.textContent = 'Voice';
            voiceToggle.classList.remove('listening');
        };
    } else {
        console.warn('Web Speech API not supported');
        voiceToggle.style.display = 'none';
        showNotification('Voice input not supported in your browser', 'warning');
    }
}

// Initialize the application
async function init() {
    console.log('Initializing Ghana Chatbot...');
    
    // Initialize voice recognition
    initSpeechRecognition();
    
    // Load existing chats
    await loadChats();
    
    // If no chats exist, don't auto-create - just show welcome message
    if (!currentChatId) {
        // Just show welcome message, don't create chat automatically
        console.log('No existing chats found. Waiting for user to create one.');
    }
    
    // Event Listeners - FIXED: Check if elements exist before adding listeners
    if (sendButton) {
        sendButton.addEventListener('click', sendMessage);
    }
    
    if (userInput) {
        userInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });
    }
    
    if (newChatButton) {
        newChatButton.addEventListener('click', createNewChat);
    }
    
    if (voiceToggle) {
        voiceToggle.addEventListener('click', toggleVoiceInput);
    }
    
    if (clearChatsButton) {
        clearChatsButton.addEventListener('click', clearAllChats);
    }
    
    if (uploadPdfButton && pdfFileInput) {
        uploadPdfButton.addEventListener('click', () => pdfFileInput.click());
        pdfFileInput.addEventListener('change', handlePdfUpload);
    }
    
    // Add event listeners to example question chips
    document.querySelectorAll('.example-chip').forEach(chip => {
        chip.addEventListener('click', function() {
            const question = this.getAttribute('data-question');
            userInput.value = question;
            if (currentChatId) {
                sendMessage();
            } else {
                // If no chat exists, create one first
                createNewChatWithQuestion(question);
            }
        });
    });
    
    // Test connection
    await testConnection();
    
    console.log('Ghana Chatbot initialized');
}

// Test backend connection
async function testConnection() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        const data = await response.json();
        if (data.status === 'healthy') {
            showNotification('Connected to backend successfully', 'success');
            if (statusIndicator) {
                statusIndicator.querySelector('.status-text').textContent = 'Connected';
                statusIndicator.querySelector('.status-dot').style.backgroundColor = '#10B981';
            }
        }
    } catch (error) {
        console.error('Backend connection failed:', error);
        showNotification('Cannot connect to backend. Please ensure the server is running.', 'error');
        if (statusIndicator) {
            statusIndicator.querySelector('.status-text').textContent = 'Disconnected';
            statusIndicator.querySelector('.status-dot').style.backgroundColor = '#EF4444';
        }
    }
}

// Load all chat sessions
async function loadChats() {
    try {
        const response = await fetch(`${API_BASE_URL}/chats`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        
        const data = await response.json();
        
        if (chatList) {
            chatList.innerHTML = '';
            data.chats.forEach(chat => {
                const li = document.createElement('li');
                li.className = 'chat-item';
                li.innerHTML = `
                    <span class="chat-title">${chat.title}</span>
                    <span class="chat-meta">${chat.message_count} messages</span>
                    <button class="delete-chat" data-id="${chat.id}">×</button>
                `;
                
                li.addEventListener('click', (e) => {
                    if (!e.target.classList.contains('delete-chat')) {
                        switchChat(chat.id);
                    }
                });
                
                // Delete button
                const deleteBtn = li.querySelector('.delete-chat');
                deleteBtn.addEventListener('click', async (e) => {
                    e.stopPropagation();
                    if (confirm(`Delete chat "${chat.title}"?`)) {
                        await deleteChat(chat.id);
                    }
                });
                
                chatList.appendChild(li);
            });
            
            // Set first chat as current if available
            if (data.chats.length > 0) {
                currentChatId = data.chats[0].id;
                await loadChatMessages(currentChatId);
            }
        }
    } catch (error) {
        console.error('Failed to load chats:', error);
        // Don't show error on initial load - might just be empty
    }
}

// Create new chat session
async function createNewChat() {
    try {
        const title = prompt('Enter chat title:', `Chat ${new Date().toLocaleDateString()}`);
        if (!title) return;
        
        const response = await fetch(`${API_BASE_URL}/chats`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ title })
        });
        
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        
        const chat = await response.json();
        currentChatId = chat.id;
        await loadChats();
        
        // Clear welcome message and show chat interface
        const welcomeMessage = document.getElementById('welcomeMessage');
        if (welcomeMessage) {
            welcomeMessage.style.display = 'none';
        }
        
        // Add welcome message to chat
        addMessageToUI('assistant', 'Welcome to the Ghana Chatbot! Ask me anything about Ghana.', [], false);
        
        // Auto-focus input
        if (userInput) {
            userInput.focus();
        }
        
        showNotification(`New chat "${title}" created`, 'success');
    } catch (error) {
        console.error('Failed to create chat:', error);
        showError('Failed to create new chat');
    }
}

// Create new chat with a question
async function createNewChatWithQuestion(question) {
    try {
        const title = `Chat about ${question.substring(0, 30)}...`;
        
        const response = await fetch(`${API_BASE_URL}/chats`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ title })
        });
        
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        
        const chat = await response.json();
        currentChatId = chat.id;
        await loadChats();
        
        // Clear welcome message
        const welcomeMessage = document.getElementById('welcomeMessage');
        if (welcomeMessage) {
            welcomeMessage.style.display = 'none';
        }
        
        // Add the question and send it
        userInput.value = question;
        await sendMessage();
        
    } catch (error) {
        console.error('Failed to create chat with question:', error);
        showError('Failed to create chat');
    }
}

// Delete a chat
async function deleteChat(chatId) {
    try {
        const response = await fetch(`${API_BASE_URL}/chats/${chatId}`, {
            method: 'DELETE'
        });
        
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        
        // If we deleted the current chat, create a new one
        if (chatId === currentChatId) {
            await createNewChat();
        }
        
        await loadChats();
        showNotification('Chat deleted', 'success');
    } catch (error) {
        console.error('Failed to delete chat:', error);
        showError('Failed to delete chat');
    }
}

// Clear all chats
async function clearAllChats() {
    if (!confirm('Clear all chats? This action cannot be undone.')) return;
    
    try {
        const response = await fetch(`${API_BASE_URL}/chats`);
        const data = await response.json();
        
        // Delete each chat
        for (const chat of data.chats) {
            await fetch(`${API_BASE_URL}/chats/${chat.id}`, { method: 'DELETE' });
        }
        
        await createNewChat();
        await loadChats();
        showNotification('All chats cleared', 'success');
    } catch (error) {
        console.error('Failed to clear chats:', error);
        showError('Failed to clear chats');
    }
}

// Switch to a different chat
async function switchChat(chatId) {
    currentChatId = chatId;
    await loadChatMessages(chatId);
    
    // Highlight active chat
    document.querySelectorAll('.chat-item').forEach(item => {
        item.classList.remove('active');
        if (item.querySelector(`[data-id="${chatId}"]`)) {
            item.classList.add('active');
        }
    });
    
    if (userInput) {
        userInput.focus();
    }
}

// Load messages for a specific chat
async function loadChatMessages(chatId) {
    try {
        const response = await fetch(`${API_BASE_URL}/chats/${chatId}`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        
        const chat = await response.json();
        if (chatMessages) {
            chatMessages.innerHTML = '';
        }
        
        if (chat.messages && chat.messages.length > 0) {
            chat.messages.forEach(msg => {
                addMessageToUI(msg.role, msg.content, msg.sources, false);
            });
        } else {
            addMessageToUI('assistant', 'Welcome to the Ghana Chatbot! Ask me anything about Ghana.', [], false);
        }
    } catch (error) {
        console.error('Failed to load chat messages:', error);
        showError('Failed to load chat messages');
    }
}

// Send message to backend
async function sendMessage() {
    const message = userInput ? userInput.value.trim() : '';
    if (!message || !currentChatId) {
        // If no chat exists, create one first
        if (!currentChatId) {
            await createNewChat();
            // Try sending again after creating chat
            if (userInput && userInput.value.trim()) {
                await sendMessage();
            }
        }
        return;
    }
    
    // Add user message to UI immediately
    addMessageToUI('user', message, [], false);
    if (userInput) {
        userInput.value = '';
    }
    
    // Show loading indicator
    const loadingId = 'loading-' + Date.now();
    addMessageToUI('assistant', 'Thinking...', [], true, loadingId);
    
    try {
        const response = await fetch(`${API_BASE_URL}/direct-ask`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                question: message,
                chat_id: currentChatId
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${await response.text()}`);
        }
        
        const data = await response.json();
        
        // Update loading message with actual response
        updateMessage(loadingId, 'assistant', data.answer, data.sources);
        
        // Speak response if voice is enabled
        if (isVoiceEnabled && data.answer) {
            speakResponse(data.answer);
        }
        
    } catch (error) {
        console.error('Failed to send message:', error);
        updateMessage(loadingId, 'assistant', 
            `Error: ${error.message}. Please check if the backend server is running.`, []);
    }
}

// Handle PDF upload
async function handlePdfUpload(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    if (file.size > 200 * 1024 * 1024) {
        showNotification('File too large. Maximum size is 200MB.', 'error');
        return;
    }
    
    if (!file.name.toLowerCase().endsWith('.pdf')) {
        showNotification('Only PDF files are supported.', 'error');
        return;
    }
    
    const formData = new FormData();
    formData.append('file', file);
    
    const progressId = 'upload-' + Date.now();
    addMessageToUI('system', `Uploading PDF: ${file.name} (${formatFileSize(file.size)})...`, [], true, progressId);
    
    try {
        const response = await fetch(`${API_BASE_URL}/upload-pdf`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) throw new Error(`Upload failed: ${response.status}`);
        
        const result = await response.json();
        
        if (result.success) {
            updateMessage(progressId, 'system', 
                `PDF uploaded successfully! ${result.message}`, []);
            showNotification('PDF processed and added to knowledge base', 'success');
        } else {
            updateMessage(progressId, 'system', 
                `PDF upload failed: ${result.error || 'Unknown error'}`, []);
            showNotification('PDF upload failed', 'error');
        }
    } catch (error) {
        console.error('PDF upload error:', error);
        updateMessage(progressId, 'system', 
            `PDF upload failed: ${error.message}`, []);
        showNotification('PDF upload failed', 'error');
    }
    
    event.target.value = '';
}

// Voice functions
function toggleVoiceInput() {
    if (!recognition) {
        initSpeechRecognition();
    }
    
    if (isVoiceEnabled && recognition) {
        recognition.stop();
        isVoiceEnabled = false;
        voiceToggle.textContent = 'Voice';
        voiceToggle.classList.remove('active');
    } else {
        if (!recognition) {
            showNotification('Voice recognition not available', 'warning');
            return;
        }
        recognition.start();
        isVoiceEnabled = true;
        voiceToggle.classList.add('active');
    }
}

function speakResponse(text) {
    if (!('speechSynthesis' in window)) return;
    
    window.speechSynthesis.cancel();
    
    const utterance = new SpeechSynthesisUtterance(text.substring(0, 300));
    utterance.rate = 1.0;
    utterance.pitch = 1.0;
    utterance.volume = 1.0;
    
    utterance.onend = function() {
        console.log('Speech finished');
    };
    
    utterance.onerror = function(event) {
        console.error('Speech synthesis error', event);
    };
    
    window.speechSynthesis.speak(utterance);
}

// UI Helper Functions
function addMessageToUI(role, content, sources = [], isTemp = false, id = null) {
    if (!chatMessages) return null;
    
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}-message ${isTemp ? 'temp' : ''}`;
    if (id) {
        messageDiv.id = id;
    }
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.textContent = content;
    
    messageDiv.appendChild(contentDiv);
    
    if (sources && sources.length > 0) {
        const sourcesDiv = document.createElement('div');
        sourcesDiv.className = 'message-sources';
        sourcesDiv.innerHTML = `<strong>Sources:</strong> ` +
            sources.map((s, i) => 
                `<a href="${s.url}" target="_blank" title="${s.chunk_preview || ''}">Source ${i+1}</a>`
            ).join(', ');
        messageDiv.appendChild(sourcesDiv);
    }
    
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    return id;
}

function updateMessage(messageId, role, content, sources = []) {
    const messageDiv = document.getElementById(messageId);
    if (messageDiv) {
        messageDiv.className = `message ${role}-message`;
        
        const contentDiv = messageDiv.querySelector('.message-content');
        if (contentDiv) {
            contentDiv.textContent = content;
        }
        
        let sourcesDiv = messageDiv.querySelector('.message-sources');
        if (sources && sources.length > 0) {
            if (!sourcesDiv) {
                sourcesDiv = document.createElement('div');
                sourcesDiv.className = 'message-sources';
                messageDiv.appendChild(sourcesDiv);
            }
            sourcesDiv.innerHTML = `<strong>Sources:</strong> ` +
                sources.map((s, i) => 
                    `<a href="${s.url}" target="_blank" title="${s.chunk_preview || ''}">Source ${i+1}</a>`
                ).join(', ');
        } else if (sourcesDiv) {
            sourcesDiv.remove();
        }
    }
}

function showError(message) {
    if (!chatMessages) return;
    
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error-message';
    errorDiv.textContent = message;
    chatMessages.appendChild(errorDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.textContent = message;
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.classList.add('fade-out');
        setTimeout(() => notification.remove(), 300);
    }, 3000);
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// Initialize when page loads
document.addEventListener('DOMContentLoaded', init);