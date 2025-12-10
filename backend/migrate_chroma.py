#!/usr/bin/env python3
"""
Migration script to fix ChromaDB schema issues
Run this once before starting the server
"""
import os
import shutil
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def migrate_chromadb():
    """Remove old ChromaDB data to force recreation with new schema"""
    
    paths_to_clean = [
        os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'chroma_db'),
        os.path.expanduser('~/.cache/chroma'),
        './chroma_db',
        './chromadb_data',
    ]
    
    cleaned_any = False
    for path in paths_to_clean:
        if os.path.exists(path):
            try:
                shutil.rmtree(path)
                logger.info(f"Removed: {path}")
                cleaned_any = True
            except Exception as e:
                logger.warning(f"Failed to remove {path}: {e}")
    
    if cleaned_any:
        logger.info("Migration complete. ChromaDB will create fresh database on next startup.")
    else:
        logger.info("No old ChromaDB data found. Ready to start.")

if __name__ == '__main__':
    print("ChromaDB Migration Tool")
    print("=" * 40)
    migrate_chromadb()