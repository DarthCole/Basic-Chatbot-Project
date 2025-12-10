#!/bin/bash

# Ghana Chatbot - Backend Startup Script
# Run from Final/ directory

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo "Script directory: $SCRIPT_DIR"
cd "$SCRIPT_DIR/backend"
echo "Changed to: $(pwd)"

echo "Starting Ghana Chatbot Backend"
echo "========================================"

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Install/update dependencies
echo "Installing dependencies..."
pip install --upgrade pip

# Clean up old ChromaDB data
echo "Cleaning old ChromaDB data..."
rm -rf "$SCRIPT_DIR/data/chroma_db" 2>/dev/null || true
rm -rf "$HOME/.cache/chroma" 2>/dev/null || true

# Install ChromaDB 0.4.22
echo "Installing ChromaDB 0.4.22..."
pip install chromadb==0.4.22

# Install other dependencies
echo "Installing other dependencies..."
pip install flask flask-cors ollama numpy requests selenium webdriver-manager beautifulsoup4 pdfplumber pymupdf

# Verify installations
echo "Verifying installations..."
python -c "
import sys
print(f'Python: {sys.version}')

try:
    import chromadb
    print(f'ChromaDB {chromadb.__version__}')
except Exception as e:
    print(f'ChromaDB error: {e}')
"

echo "========================================"
echo "Setup complete! Starting server..."
echo ""

# Run the Flask app
echo "Starting backend server..."
"$SCRIPT_DIR/backend/.venv/bin/python" "$SCRIPT_DIR/backend/app.py"