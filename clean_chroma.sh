#!/bin/bash
echo "Cleaning ChromaDB data for fresh start..."
echo ""

# From the root Final/ directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "Cleaning project data..."
rm -rf "$SCRIPT_DIR/data/chroma_db" 2>/dev/null && echo "Removed: $SCRIPT_DIR/data/chroma_db" || echo "Not found: $SCRIPT_DIR/data/chroma_db"

echo "Cleaning global cache..."
rm -rf "$HOME/.cache/chroma" 2>/dev/null && echo "Removed: $HOME/.cache/chroma" || echo "Not found: $HOME/.cache/chroma"

echo "Cleaning local directories..."
rm -rf "$SCRIPT_DIR/backend/chroma_db" 2>/dev/null && echo "Removed: $SCRIPT_DIR/backend/chroma_db" || echo "Not found: $SCRIPT_DIR/backend/chroma_db"
rm -rf "$SCRIPT_DIR/backend/chromadb_data" 2>/dev/null && echo "Removed: $SCRIPT_DIR/backend/chromadb_data" || echo "Not found: $SCRIPT_DIR/backend/chromadb_data"

echo ""
echo "Cleanup complete! Now run: ./start_backend.sh"