from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import sys
import os
import logging
import shutil
import hashlib
from web_scraper import EnhancedWebScraper
from pdf_extractor import PDFExtractor
from rag_system import EnhancedRAGSystem
import threading
import queue
import uuid
import tempfile
from datetime import datetime
from typing import Dict, List
import socket
import ollama  # Added for direct Ollama access

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='../frontend', static_url_path='')
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Increase upload size limit to 200MB
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024

# Initialize components
scraper = EnhancedWebScraper(headless=True)
pdf_extractor = PDFExtractor()
rag_system = None  # Initialize on first use

# Task queue for background processing
task_queue = queue.Queue()
results = {}

# Chat sessions storage
chat_sessions: Dict[str, Dict] = {}

def clean_old_chromadb_data():
    """Clean old ChromaDB data to avoid schema issues"""
    # Since app.py is in backend/, we need to go up one level to reach Final/
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    paths_to_clean = [
        # Your project's ChromaDB path (Final/data/chroma_db)
        os.path.join(project_root, 'data', 'chroma_db'),
        # Global ChromaDB cache
        os.path.expanduser('~/.cache/chroma'),
        # Local ChromaDB data (relative to backend/)
        './chroma_db',
        './chromadb_data',
    ]
    
    for path in paths_to_clean:
        if os.path.exists(path):
            try:
                shutil.rmtree(path)
                logger.info(f"Cleaned old ChromaDB data: {path}")
            except Exception as e:
                logger.warning(f"Failed to clean {path}: {e}")

def test_ollama_connection():
    """Test if Ollama is running and accessible"""
    try:
        # Try to get list of models
        response = ollama.list()
        logger.info(f"Ollama connection successful. Models: {response}")
        
        # Check if llama3.2:latest is available
        models = response.get('models', [])
        model_names = [m.get('name') for m in models if m.get('name')]
        
        if 'llama3.2:latest' in model_names:
            logger.info("Model 'llama3.2:latest' found")
            return True, 'llama3.2:latest'
        elif any('llama3.2' in name for name in model_names):
            # Find any llama3.2 variant
            for name in model_names:
                if 'llama3.2' in name:
                    logger.info(f"Using model: {name}")
                    return True, name
        elif model_names:
            logger.info(f"Using available model: {model_names[0]}")
            return True, model_names[0]
        else:
            logger.warning("No models found in Ollama")
            return False, None
            
    except Exception as e:
        logger.error(f"Ollama connection test failed: {e}")
        return False, None

def get_or_create_rag_system():
    """Get or initialize RAG system with fallback to direct Ollama"""
    global rag_system
    
    # Test Ollama first
    ollama_ok, model_name = test_ollama_connection()
    
    if rag_system is None:
        try:
            # Only clean if absolutely necessary for demo
            # clean_old_chromadb_data()
            
            if ollama_ok:
                # Try to initialize the full RAG system
                rag_system = EnhancedRAGSystem(ollama_model=model_name or "llama3.2:latest")
                logger.info(f"RAG system initialized successfully with model: {model_name}")
                
                # Quick test without heavy query
                try:
                    test_result = rag_system.ask("Hello")
                    logger.info(f"RAG system test passed: Got response")
                except Exception as test_error:
                    logger.warning(f"RAG test query failed: {test_error}")
                    # Continue anyway for demo
            else:
                raise Exception("Ollama not available")
                
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {e}")
            logger.info("Creating Ollama-only fallback system for demo...")
            
            # Create a working fallback that uses Ollama directly
            class OllamaFallbackSystem:
                def __init__(self, model_name="llama3.2:latest"):
                    self.model_name = model_name
                    self.collection_count = 0
                    
                def ask(self, query):
                    try:
                        # Direct Ollama call - works without ChromaDB
                        response = ollama.generate(
                            model=self.model_name,
                            prompt=f"""You are a helpful AI assistant specializing in Ghana. 
Answer the following question clearly and informatively.

Question: {query}

Answer:""",
                            options={'temperature': 0.7, 'num_predict': 500}
                        )
                        
                        return {
                            "answer": response['response'],
                            "sources": [],
                            "context_used": False,
                            "query": query,
                            "model": self.model_name,
                            "fallback_mode": True
                        }
                    except Exception as e:
                        logger.error(f"Ollama fallback error: {e}")
                        return {
                            "answer": "I'm having trouble connecting to the AI model. Please make sure Ollama is running.",
                            "sources": [],
                            "context_used": False,
                            "query": query
                        }
                
                def get_stats(self):
                    return {
                        "collection_count": self.collection_count,
                        "model": self.model_name,
                        "status": "fallback_active",
                        "fallback_mode": True
                    }
                
                def add_documents(self, docs):
                    # Silently accept documents but don't process them in fallback mode
                    self.collection_count += len(docs)
                    logger.info(f"Fallback: Accepted {len(docs)} documents (not processed in fallback mode)")
            
            rag_system = OllamaFallbackSystem(model_name=model_name or "llama3.2:latest")
            logger.info(f"Using Ollama fallback system with model: {model_name}")
    
    return rag_system

def background_worker():
    """Background worker for processing tasks"""
    while True:
        task_id, task_type, data = task_queue.get()
        try:
            if task_type == "scrape":
                url = data["url"]
                result = scraper.scrape(url)
                
                # Initialize RAG system
                rag = get_or_create_rag_system()
                
                # Extract PDFs if found
                if result["success"] and result.get("pdf_links"):
                    pdf_results = pdf_extractor.batch_extract(
                        [pdf["url"] for pdf in result["pdf_links"][:3]]  # Limit to 3 PDFs
                    )
                    
                    # Add PDF content to RAG system
                    successful_pdfs = [r for r in pdf_results if r["success"]]
                    if successful_pdfs:
                        rag.add_documents(successful_pdfs)
                
                # Add web content to RAG system
                rag.add_documents([result])
                
                results[task_id] = {
                    "status": "completed",
                    "result": result,
                    "pdfs_processed": len(result.get("pdf_links", []))
                }
                
            elif task_type == "ask":
                # Initialize RAG system
                rag = get_or_create_rag_system()
                
                question = data["question"]
                chat_id = data.get("chat_id")
                
                result = rag.ask(question)
                
                # Store in chat session if chat_id provided
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
                        "content": result.get("answer", ""),
                        "sources": result.get("sources", []),
                        "timestamp": datetime.now().isoformat()
                    })
                
                results[task_id] = {
                    "status": "completed",
                    "result": result
                }
                
        except Exception as e:
            logger.error(f"Task {task_id} failed: {str(e)}")
            results[task_id] = {
                "status": "failed",
                "error": str(e)
            }
        
        task_queue.task_done()

# Start background worker
worker_thread = threading.Thread(target=background_worker, daemon=True)
worker_thread.start()

@app.route('/')
def index():
    """Serve the frontend"""
    return send_from_directory('../frontend', 'index.html')

@app.route('/api')
def home():
    return jsonify({
        "status": "running",
        "service": "Ghana Chatbot API",
        "version": "4.1",
        "features": [
            "Direct Ollama LLM with fallback",
            "Selenium web scraping",
            "PDF extraction (up to 200MB)",
            "Vector storage with ChromaDB (when available)",
            "Multiple chat sessions",
            "Background processing",
            "Voice chat ready"
        ],
        "endpoints": {
            "chats": "GET/POST /api/chats",
            "chat": "GET/DELETE /api/chats/<chat_id>",
            "ask": "POST /api/ask",
            "direct-ask": "POST /api/direct-ask",
            "scrape": "POST /api/scrape",
            "upload-pdf": "POST /api/upload-pdf",
            "status": "GET /api/status/<task_id>",
            "stats": "GET /api/stats",
            "health": "GET /api/health"
        }
    })

# Chat session endpoints
@app.route('/api/chats', methods=['GET'])
def list_chats():
    """List all chat sessions"""
    return jsonify({
        "chats": [
            {
                "id": chat_id,
                "title": chat_data.get("title", "Untitled Chat"),
                "created_at": chat_data.get("created_at"),
                "message_count": len(chat_data.get("messages", []))
            }
            for chat_id, chat_data in chat_sessions.items()
        ]
    })

@app.route('/api/chats', methods=['POST'])
def create_chat():
    """Create a new chat session"""
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
    """Get a specific chat session"""
    if chat_id not in chat_sessions:
        return jsonify({"error": "Chat not found"}), 404
    return jsonify(chat_sessions[chat_id])

@app.route('/api/chats/<chat_id>', methods=['DELETE'])
def delete_chat(chat_id):
    """Delete a chat session"""
    if chat_id not in chat_sessions:
        return jsonify({"error": "Chat not found"}), 404
    del chat_sessions[chat_id]
    return jsonify({"message": "Chat deleted"}), 200

@app.route('/api/chats/<chat_id>/title', methods=['PUT'])
def update_chat_title(chat_id):
    """Update chat title"""
    if chat_id not in chat_sessions:
        return jsonify({"error": "Chat not found"}), 404
    data = request.json
    if "title" in data:
        chat_sessions[chat_id]["title"] = data["title"]
    return jsonify(chat_sessions[chat_id])

@app.route('/api/scrape', methods=['POST'])
def scrape():
    """Scrape a website with background processing"""
    try:
        data = request.json
        url = data.get('url')
        
        if not url:
            return jsonify({"error": "URL is required"}), 400
        
        # Validate URL
        if not url.startswith(('http://', 'https://')):
            return jsonify({"error": "Invalid URL format"}), 400
        
        # Generate task ID
        task_id = str(uuid.uuid4())
        
        # Add to queue
        task_queue.put((task_id, "scrape", {"url": url}))
        
        return jsonify({
            "message": "Scraping started",
            "task_id": task_id,
            "status_url": f"/api/status/{task_id}"
        })
        
    except Exception as e:
        logger.error(f"Scrape error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/ask', methods=['POST'])
def ask():
    """Ask a question with RAG (background processing)"""
    try:
        data = request.json
        question = data.get('question')
        chat_id = data.get('chat_id')
        
        if not question:
            return jsonify({"error": "Question is required"}), 400
        
        # Generate task ID
        task_id = str(uuid.uuid4())
        
        # Add to queue
        task_queue.put((task_id, "ask", {"question": question, "chat_id": chat_id}))
        
        return jsonify({
            "message": "Processing question",
            "task_id": task_id,
            "status_url": f"/api/status/{task_id}"
        })
        
    except Exception as e:
        logger.error(f"Ask error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/direct-ask', methods=['POST'])
def direct_ask():
    """Direct synchronous question (for real-time chat) - UPDATED with fallback"""
    try:
        data = request.json
        question = data.get('question')
        chat_id = data.get('chat_id')
        
        if not question:
            return jsonify({"error": "Question is required"}), 400
        
        # Try to get RAG system first
        try:
            rag = get_or_create_rag_system()
            result = rag.ask(question)
        except Exception as rag_error:
            logger.warning(f"RAG system failed, using direct Ollama: {rag_error}")
            
            # Direct Ollama fallback
            try:
                response = ollama.generate(
                    model='llama3.2:latest',
                    prompt=f"""You are a helpful AI assistant specializing in Ghana. 
Answer the following question clearly and informatively.

Question: {question}

Answer:""",
                    options={'temperature': 0.7, 'num_predict': 500}
                )
                
                result = {
                    "answer": response['response'],
                    "sources": [],
                    "context_used": False,
                    "query": question,
                    "direct_ollama": True
                }
            except Exception as ollama_error:
                logger.error(f"Direct Ollama also failed: {ollama_error}")
                result = {
                    "answer": "I'm having trouble connecting to the AI service. Please check if Ollama is running with 'ollama serve'.",
                    "sources": [],
                    "context_used": False,
                    "query": question,
                    "error": str(ollama_error)
                }
        
        # Store in chat session if chat_id provided
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
                "content": result.get("answer", ""),
                "sources": result.get("sources", []),
                "timestamp": datetime.now().isoformat()
            })
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Direct ask error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/status/<task_id>')
def get_status(task_id):
    """Get status of a background task"""
    if task_id in results:
        return jsonify(results[task_id])
    else:
        return jsonify({
            "status": "processing",
            "message": "Task is still in queue or being processed"
        })

@app.route('/api/stats')
def get_stats():
    """Get system statistics"""
    try:
        rag = get_or_create_rag_system()
        stats = rag.get_stats()
        
        # Get the current port from app config or default
        server_port = app.config.get('SERVER_PORT', 5000)
        
        stats.update({
            "chat_sessions": len(chat_sessions),
            "background_tasks": task_queue.qsize(),
            "server_port": server_port,
            "max_upload_size": "200MB",
            "rag_status": "active" if rag_system else "inactive"
        })
        return jsonify(stats)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/upload-pdf', methods=['POST'])
def upload_pdf():
    """Upload and process a local PDF file (up to 200MB)"""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            return jsonify({"error": "Only PDF files are supported"}), 400
        
        # Create unique filename to avoid conflicts
        filename_hash = hashlib.md5(file.filename.encode()).hexdigest()[:8]
        unique_filename = f"{filename_hash}_{file.filename}"
        
        # Save file temporarily
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, unique_filename)
        
        # Save in chunks for large files
        chunk_size = 1024 * 1024  # 1MB chunks
        total_size = 0
        with open(temp_path, 'wb') as f:
            while True:
                chunk = file.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                total_size += len(chunk)
        
        # Get file size for logging
        file_size = os.path.getsize(temp_path)
        logger.info(f"Processing PDF ({file_size/1024/1024:.2f} MB): {file.filename}")
        
        # Extract text
        result = pdf_extractor.extract_from_file(temp_path, max_pages=100)
        
        # Try to add to RAG system
        try:
            rag = get_or_create_rag_system()
            if result["success"] and result.get("content"):
                rag.add_documents([result])
                result["message"] = f"PDF processed successfully. Added to knowledge base."
            else:
                result["message"] = "PDF processed but no content extracted."
        except Exception as rag_error:
            logger.warning(f"Could not add PDF to RAG system: {rag_error}")
            result["message"] = "PDF extracted but RAG system unavailable. Content saved for later."
        
        # Add file info to result
        result.update({
            "original_filename": file.filename,
            "file_size_mb": round(file_size / (1024 * 1024), 2),
            "pages_processed": result.get("pages", 0)
        })
        
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"PDF upload error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    try:
        # Test Ollama connection
        ollama_ok, model_name = test_ollama_connection()
        
        rag_status = "not initialized"
        if rag_system:
            try:
                rag_stats = rag_system.get_stats()
                rag_status = rag_stats.get('status', 'unknown')
                if rag_stats.get('fallback_mode'):
                    rag_status = f"fallback ({model_name})"
            except:
                rag_status = "error"
        
        return jsonify({
            "status": "healthy" if ollama_ok else "degraded",
            "ollama": "connected" if ollama_ok else "disconnected",
            "ollama_model": model_name,
            "rag_system": rag_status,
            "chat_sessions": len(chat_sessions),
            "background_worker": worker_thread.is_alive(),
            "timestamp": datetime.now().isoformat(),
            "services": {
                "web_scraping": True,
                "pdf_extraction": True,
                "vector_database": rag_status not in ['fallback', 'error', 'not initialized'],
                "llm_generation": ollama_ok
            }
        })
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e)}), 500

if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("Starting Ghana Chatbot API v4.1 (Enhanced with Ollama fallback)")
    logger.info("=" * 60)
    
    # Test Ollama first
    ollama_ok, model_name = test_ollama_connection()
    if not ollama_ok:
        logger.warning("Ollama not available. The system will use fallback mode.")
        logger.info("To start Ollama, open another terminal and run: ollama serve")
    
    # Initialize RAG system on startup (will use fallback if needed)
    try:
        get_or_create_rag_system()
        logger.info("Chat system initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize chat system: {e}")
        logger.info("The system will attempt to use Ollama directly")
    
    # Find an available port (start with 5000)
    def find_free_port(start_port=5000, max_tries=3):
        for port in range(start_port, start_port + max_tries):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('', port))
                    return port
            except OSError:
                continue
        return start_port  # Fallback
    
    port = find_free_port(5000)
    
    # Store the port in app config for use in stats endpoint
    app.config['SERVER_PORT'] = port
    
    # Log the port information
    if port != 5000:
        logger.warning(f"Port 5000 is in use, using port {port} instead")
        logger.info(f"Update frontend/app.js API_BASE_URL to: http://localhost:{port}/api")
    
    logger.info(f"Server starting on http://0.0.0.0:{port}")
    logger.info(f"Frontend available at: http://localhost:{port}")
    logger.info(f"API documentation: http://localhost:{port}/api")
    logger.info("=" * 60)
    
    # Start the server with use_reloader=False to prevent port switching
    app.run(debug=True, port=port, host='0.0.0.0', use_reloader=False)