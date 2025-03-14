document.addEventListener('DOMContentLoaded', () => {
    const chatMessages = document.getElementById('chat-messages');
    const chatInput = document.getElementById('chat-input');
    const sendButton = document.getElementById('send-button');
    const sourcesContainer = document.getElementById('sources-container');
    const sourcesList = document.getElementById('sources-list');
    
    let chatHistory = [];
    
    // send message when send button is clicked
    sendButton.addEventListener('click', sendMessage);
    
    // send message when enter key is pressed (but allow shift+enter for new lines)
    chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
    
    function sendMessage() {
        const message = chatInput.value.trim();
        if (!message) return;
        
        // add user message to chat
        addMessage('user', message);
        chatInput.value = '';
        
        // create a placeholder for the assistant's response with loading indicator
        const assistantMessageDiv = document.createElement('div');
        assistantMessageDiv.className = 'message assistant loading';
        
        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        
        assistantMessageDiv.appendChild(messageContent);
        chatMessages.appendChild(assistantMessageDiv);
        
        // scroll to bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;
        
        // store in history
        chatHistory.push({ role: 'user', content: message });
        
        // fetch streaming response
        fetchStreamingResponse(message, messageContent, assistantMessageDiv);
        
        // fetch sources
        fetchSources(message);
    }
    
    function addMessage(role, content) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}`;
        
        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        messageContent.textContent = content;
        
        messageDiv.appendChild(messageContent);
        chatMessages.appendChild(messageDiv);
        
        // scroll to bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;
        
        return messageContent; // return the content element for updating later
    }
    
    async function fetchStreamingResponse(message, messageElement, messageDiv) {
        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    message: message,
                    history: chatHistory
                })
            });
            
            if (!response.ok) {
                throw new Error('network response was not ok');
            }
            
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let assistantResponse = '';
            
            // when we get the first chunk, remove loading indicator
            messageDiv.classList.remove('loading');
            
            while (true) {
                const { value, done } = await reader.read();
                if (done) break;
                
                // decode the chunk and append to assistant response
                const chunk = decoder.decode(value, { stream: true });
                assistantResponse += chunk;
                messageElement.textContent = assistantResponse;
                
                // scroll to bottom as new content arrives
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }
            
            // add the completed response to chat history
            chatHistory.push({ role: 'assistant', content: assistantResponse });
            
        } catch (error) {
            console.error('error fetching streaming response:', error);
            messageDiv.classList.remove('loading');
            messageElement.textContent = 'sorry, there was an error processing your request.';
        }
    }
    
    async function fetchSources(message) {
        try {
            const response = await fetch('/api/sources', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    message: message
                })
            });
            
            if (!response.ok) {
                throw new Error('network response was not ok');
            }
            
            const data = await response.json();
            
            // display sources
            if (data.sources && data.sources.length > 0) {
                // clear previous sources
                sourcesList.innerHTML = '';
                
                // add each source
                data.sources.forEach(source => {
                    const sourceItem = document.createElement('div');
                    sourceItem.className = 'source-item';
                    
                    const title = document.createElement('div');
                    title.className = 'source-title';
                    title.textContent = source.metadata.title || 'untitled';
                    
                    const score = document.createElement('div');
                    score.className = 'source-score';
                    score.textContent = `relevance: ${(source.relevance_score * 100).toFixed(1)}%`;
                    
                    sourceItem.appendChild(title);
                    sourceItem.appendChild(score);
                    sourcesList.appendChild(sourceItem);
                });
                
                // show sources container
                sourcesContainer.style.display = 'block';
            } else {
                sourcesContainer.style.display = 'none';
            }
            
        } catch (error) {
            console.error('error fetching sources:', error);
            sourcesContainer.style.display = 'none';
        }
    }
});
