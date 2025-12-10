# Ghana Chatbot - AI Assistant

An intelligent chatbot that provides answers about Ghana using Llama 3.2, RAG (Retrieval-Augmented Generation), web scraping, and PDF extraction.

## Features

- 🤖 **Llama 3.2 Integration**: Uses locally installed Llama 3.2 model via Ollama
- 📚 **RAG System**: Retrieval-Augmented Generation for accurate, context-aware responses
- 🌐 **Web Scraping**: Automatically scrapes websites for information about Ghana
- 📄 **PDF Extraction**: Extracts and processes PDF documents (including local files)
- 💬 **Multiple Chat Sessions**: Create and manage multiple conversation threads
- 🔊 **Voice-to-Speech**: Text-to-speech functionality for responses
- 🎤 **Voice Input**: Speech recognition for voice queries (browser-supported)
- 🎨 **Modern UI**: Beautiful, responsive HTML/CSS interface

## Prerequisites

1. **Python 3.8+** installed
2. **Ollama** installed and running
   ```bash
   # Install Ollama
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # Pull Llama 3.2 model
   ollama pull llama3.2
   ```
3. **Chrome/Chromium** browser (for Selenium web scraping)
4. **Node.js** (optional, for Playwright browsers)

## Installation

1. **Clone or navigate to the project directory**

2. **Create and activate virtual environment**:
   ```bash
   cd backend
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install Python dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r ../requirements.txt
   ```

4. **Install Playwright browsers**:
   ```bash
   playwright install chromium
   ```

5. **Ensure Ollama is running**:
   ```bash
   ollama serve
   ```

## Running the Application

### Option 1: Using the start scripts

1. **Start the backend**:
   ```bash
   chmod +x start_backend.sh
   ./start_backend.sh
   ```

2. **Open your browser** and navigate to:
   ```
   http://localhost:5000
   ```

### Option 2: Manual start

1. **Start the backend**:
   ```bash
   cd backend
   source .venv/bin/activate
   python app.py
   ```

2. **Open your browser** and navigate to:
   ```
   http://localhost:5000
   ```

## Usage

### Basic Chat

1. Click "New Chat" to start a conversation
2. Type your question about Ghana in the input box
3. Press Enter or click the send button
4. The chatbot will respond using RAG with information from scraped content and PDFs

### Voice Features

- **Voice Input**: Click the microphone button (🎤) to speak your question
- **Text-to-Speech**: Click the speaker button (🔊) on any assistant message to hear it read aloud

### PDF Upload

1. Click "📄 Upload PDF" in the input area
2. Select a PDF file from your computer
3. The PDF content will be extracted and added to the knowledge base

### Managing Chats

- **Create New Chat**: Click "New Chat" button in the sidebar
- **Switch Between Chats**: Click on any chat in the sidebar
- **Edit Chat Title**: Click the edit icon (✏️) in the chat header
- **Delete Chat**: Hover over a chat and click the delete button (🗑️)
- **Clear Chat**: Click the clear button (🗑️) in the chat header

## API Endpoints

The backend provides a RESTful API:

- `GET /api` - API information
- `GET /api/chats` - List all chat sessions
- `POST /api/chats` - Create a new chat session
- `GET /api/chats/<chat_id>` - Get a specific chat
- `DELETE /api/chats/<chat_id>` - Delete a chat
- `PUT /api/chats/<chat_id>/title` - Update chat title
- `POST /api/ask` - Ask a question (async)
- `POST /api/direct-ask` - Ask a question (sync)
- `POST /api/scrape` - Scrape a website
- `POST /api/upload-pdf` - Upload and process a PDF
- `GET /api/stats` - Get system statistics

## Project Structure

```
Final/
├── backend/
│   ├── app.py              # Flask API server
│   ├── web_scraper.py      # Web scraping functionality
│   ├── pdf_extractor.py    # PDF extraction
│   └── rag_system.py       # RAG system with Llama 3.2
├── frontend/
│   ├── index.html          # Main HTML file
│   ├── styles.css          # Styling
│   └── app.js              # Frontend JavaScript
├── data/
│   ├── chroma_db/          # Vector database (auto-created)
│   └── *.pdf               # PDF files (optional)
├── requirements.txt        # Python dependencies
├── start_backend.sh        # Backend startup script
└── README.md              # This file
```

## Configuration

### Changing the Model

To use a different Ollama model, edit `backend/rag_system.py`:

```python
rag_system = EnhancedRAGSystem(ollama_model="your-model-name")
```

### API Port

To change the backend port, edit `backend/app.py`:

```python
app.run(debug=True, port=YOUR_PORT, host='0.0.0.0')
```

## Troubleshooting

### Backend won't start

- Ensure Python 3.8+ is installed
- Check that all dependencies are installed: `pip install -r requirements.txt`
- Verify Ollama is running: `ollama list`

### Model not found

- Pull the model: `ollama pull llama3.2`
- Check available models: `ollama list`
- The system will attempt to use available Llama 3.2 variants automatically

### Web scraping fails

- Ensure Chrome/Chromium is installed
- Check internet connection
- Some websites may block automated scraping

### Voice features not working

- Voice input requires browser support (Chrome, Edge recommended)
- Text-to-speech requires browser support
- Check browser permissions for microphone access

### Frontend can't connect to backend

- Ensure backend is running on port 5000
- Check CORS settings if accessing from different origin
- Verify API_BASE_URL in `frontend/app.js`

## Development

### Adding New Features

1. **Backend**: Add new routes in `backend/app.py`
2. **Frontend**: Update `frontend/app.js` for new functionality
3. **Styling**: Modify `frontend/styles.css` for UI changes

### Testing

Test individual components:

```python
# Test PDF extraction
from pdf_extractor import PDFExtractor
extractor = PDFExtractor()
result = extractor.extract_from_file("path/to/file.pdf")

# Test web scraping
from web_scraper import EnhancedWebScraper
scraper = EnhancedWebScraper()
result = scraper.scrape("https://example.com")

# Test RAG system
from rag_system import EnhancedRAGSystem
rag = EnhancedRAGSystem()
result = rag.ask("What is the capital of Ghana?")
```

## License

This project is for educational purposes.

## Support

For issues or questions, please check:
- Ollama documentation: https://ollama.ai
- Flask documentation: https://flask.palletsprojects.com
- ChromaDB documentation: https://docs.trychroma.com

