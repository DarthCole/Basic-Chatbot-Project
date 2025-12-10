import pdfplumber
import requests
import os
import tempfile
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

# Try to import PyMuPDF (optional)
try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False
    logger.warning("PyMuPDF not available, using pdfplumber only")

class PDFExtractor:
    def __init__(self):
        self.temp_dir = tempfile.gettempdir()
    
    def extract_from_url(self, pdf_url: str, max_pages: int = 20) -> Dict[str, Any]:
        """Extract text from a PDF URL"""
        try:
            logger.info(f"Extracting PDF from URL: {pdf_url}")
            
            # Download PDF
            response = requests.get(pdf_url, timeout=30)
            response.raise_for_status()
            
            # Save to temp file
            temp_path = os.path.join(self.temp_dir, "temp.pdf")
            with open(temp_path, 'wb') as f:
                f.write(response.content)
            
            # Extract text
            text = self._extract_text(temp_path, max_pages)
            
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            return {
                "url": pdf_url,
                "content": text,
                "success": True,
                "content_length": len(text),
                "source": "pdf"
            }
            
        except Exception as e:
            logger.error(f"PDF extraction error: {str(e)}")
            return {
                "url": pdf_url,
                "content": "",
                "success": False,
                "error": str(e)
            }
    
    def extract_from_file(self, pdf_path: str, max_pages: int = 20) -> Dict[str, Any]:
        """Extract text from a local PDF file"""
        try:
            logger.info(f"Extracting PDF from file: {pdf_path}")
            
            # Extract text
            text = self._extract_text(pdf_path, max_pages)
            
            return {
                "url": pdf_path,
                "content": text,
                "success": True,
                "content_length": len(text),
                "source": "pdf"
            }
            
        except Exception as e:
            logger.error(f"PDF extraction error: {str(e)}")
            return {
                "url": pdf_path,
                "content": "",
                "success": False,
                "error": str(e)
            }
    
    def _extract_text(self, pdf_path: str, max_pages: int) -> str:
        """Extract text from PDF using multiple methods for best results"""
        text_parts = []
        
        # Method 1: Use PyMuPDF (fitz) for fast extraction
        if HAS_PYMUPDF:
            try:
                with fitz.open(pdf_path) as doc:
                    for i, page in enumerate(doc):
                        if i >= max_pages:
                            break
                        text = page.get_text()
                        if text.strip():
                            text_parts.append(text)
            except Exception as e:
                logger.warning(f"PyMuPDF extraction failed: {e}")
        
        # Method 2: Use pdfplumber for better table extraction
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    if i >= max_pages:
                        break
                    text = page.extract_text()
                    if text and text.strip():
                        text_parts.append(text)
                        
                    # Extract tables
                    tables = page.extract_tables()
                    for table in tables:
                        if table:
                            table_text = "\n".join([" | ".join(filter(None, row)) for row in table if any(row)])
                            if table_text.strip():
                                text_parts.append(f"Table:\n{table_text}")
        except Exception as e:
            logger.warning(f"pdfplumber extraction failed: {e}")
        
        # Combine and clean text
        combined = "\n\n".join(text_parts)
        
        # Remove excessive whitespace
        lines = combined.split('\n')
        cleaned_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped:  # Keep non-empty lines
                cleaned_lines.append(stripped)
        
        return "\n".join(cleaned_lines)[:50000]  # Limit to 50k chars
    
    def batch_extract(self, pdf_urls: List[str], max_pages: int = 20) -> List[Dict[str, Any]]:
        """Extract text from multiple PDFs"""
        results = []
        for url in pdf_urls:
            result = self.extract_from_url(url, max_pages)
            results.append(result)
        return results

