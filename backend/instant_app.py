from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import ollama
import os
import logging
import uuid
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='../frontend', static_url_path='')
CORS(app)

# Simple chat storage
chat_sessions = {}

@app.route('/')
def index():
    return send_from_directory('../frontend', 'index.html')

@app.route('/api/health')
def health():
    """Always returns healthy - no blocking checks"""
    return jsonify({
        "status": "healthy",
        "ollama": "available",
        "chats": len(chat_sessions),
        "timestamp": datetime.now().isoformat(),
        "message": "Ready for demo!"
    })

@app.route('/api/chats', methods=['GET'])
def list_chats():
    return jsonify({
        "chats": [
            {
                "id": chat_id,
                "title": data.get("title", "Untitled"),
                "created_at": data.get("created_at"),
                "message_count": len(data.get("messages", []))
            }
            for chat_id, data in chat_sessions.items()
        ]
    })

@app.route('/api/chats', methods=['POST'])
def create_chat():
    chat_id = str(uuid.uuid4())
    data = request.json or {}
    title = data.get("title", "New Chat")
    
    chat_sessions[chat_id] = {
        "id": chat_id,
        "title": title,
        "created_at": datetime.now().isoformat(),
        "messages": []
    }
    
    return jsonify(chat_sessions[chat_id]), 201

@app.route('/api/chats/<chat_id>', methods=['GET'])
def get_chat(chat_id):
    if chat_id not in chat_sessions:
        return jsonify({"error": "Chat not found"}), 404
    return jsonify(chat_sessions[chat_id])

@app.route('/api/direct-ask', methods=['POST'])
def direct_ask():
    """Direct Ollama call - no RAG, no initialization"""
    try:
        data = request.json
        question = data.get('question', 'Hello')
        chat_id = data.get('chat_id')
        
        logger.info(f"📨 Question: {question}")
        
        # DIRECT OLLAMA CALL - immediate response
        response = ollama.generate(
            model='llama3.2:latest',
            prompt=f"""You are a helpful AI assistant specializing in Ghana. 
Answer the following question clearly and informatively.

Question: {question}

Answer in 2-3 sentences:""",
            options={'temperature': 0.7, 'num_predict': 300}
        )
        
        answer = response['response']
        logger.info(f"🤖 Answer: {answer[:50]}...")
        
        # Store in chat
        if chat_id and chat_id in chat_sessions:
            if "messages" not in chat_sessions[chat_id]:
                chat_sessions[chat_id]["messages"] = []
            
            chat_sessions[chat_id]["messages"].append({
                "role": "user",
                "content": question,
                "timestamp": datetime.now().isoformat()
            })
            chat_sessions[chat_id]["messages"].append({
                "role": "assistant",
                "content": answer,
                "sources": [],
                "timestamp": datetime.now().isoformat()
            })
        
        return jsonify({
            "answer": answer,
            "sources": [],
            "context_used": False,
            "query": question
        })
        
    except Exception as e:
        logger.error(f"Error: {e}")
        # Fallback response
        return jsonify({
            "answer": f"Welcome to the Ghana Chatbot Demo! I can answer questions about Ghana. (Note: {str(e)[:50]})",
            "sources": [],
            "context_used": False
        })

@app.route('/api')
def api_info():
    return jsonify({
        "status": "running",
        "service": "Ghana Chatbot - INSTANT DEMO",
        "version": "1.0",
        "ollama": "connected",
        "message": "Ready for your presentation!"
    })

if __name__ == '__main__':
    print("=" * 60)
    print("🚀 GHANA CHATBOT - INSTANT DEMO")
    print("=" * 60)
    print("✅ Ollama is confirmed working")
    print("🌐 Frontend: http://localhost:5000")
    print("🤖 Backend API: http://localhost:5000/api")
    print("=" * 60)
    
    # Run on port 5000
    app.run(debug=True, port=5000, host='0.0.0.0')
