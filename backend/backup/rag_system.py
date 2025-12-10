import ollama
# Patch NumPy compatibility for ChromaDB before importing
import numpy as np
# ChromaDB 0.4.22 uses np.float_, np.int_, np.uint which may cause issues
# Create aliases if they don't exist (for NumPy 2.0+ compatibility)
# Even in NumPy 1.x, these might trigger deprecation warnings
if not hasattr(np, 'float_'):
    np.float_ = np.float64
if not hasattr(np, 'int_'):
    np.int_ = np.int64  
if not hasattr(np, 'uint'):
    np.uint = np.uint64

import chromadb
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import os
import re
import time

logger = logging.getLogger(__name__)

class EnhancedRAGSystem:
    def __init__(self, 
                 ollama_model: str = 'llama3.2:latest',
                 chroma_path: str = None,
                 embedding_batch_size: int = 5):
        
        # Set default chroma path relative to project root
        if chroma_path is None:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            chroma_path = os.path.join(project_root, 'data', 'chroma_db')
        
        logger.info(f"Initializing Enhanced RAG System...")
        logger.info(f"Ollama model (for both embedding and generation): {ollama_model}")
        logger.info(f"ChromaDB path: {chroma_path}")
        
        # Ollama configuration
        self.ollama_model = ollama_model
        self.embedding_batch_size = embedding_batch_size
        
        # Check if Ollama is running and model is available
        self._check_ollama()
        
        # Initialize ChromaDB
        logger.info("Setting up vector database...")
        os.makedirs(chroma_path, exist_ok=True)
        
        # Create embedding function that uses Ollama
        def ollama_embedding_function(texts: List[str]) -> List[List[float]]:
            """Generate embeddings using Ollama"""
            embeddings = []
            
            # Process in batches to avoid overwhelming Ollama
            for i in range(0, len(texts), self.embedding_batch_size):
                batch = texts[i:i + self.embedding_batch_size]
                batch_embeddings = []
                
                for text in batch:
                    try:
                        # Get embedding from Ollama
                        response = ollama.embeddings(
                            model=self.ollama_model,
                            prompt=text
                        )
                        batch_embeddings.append(response["embedding"])
                    except Exception as e:
                        logger.warning(f"Failed to get embedding for text: {e}")
                        # Return a zero vector as fallback
                        batch_embeddings.append([0.0] * 4096)  # Llama 3.2 has 4096-dim embeddings
                
                embeddings.extend(batch_embeddings)
                # Small delay between batches
                if i + self.embedding_batch_size < len(texts):
                    time.sleep(0.1)
            
            return embeddings
        
        self.client = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.client.get_or_create_collection(
            name="ghana_rag",
            embedding_function=ollama_embedding_function,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Text splitting configuration
        self.chunk_size = 1000
        self.chunk_overlap = 200
        
        logger.info("Enhanced RAG System initialized successfully")
    
    def _check_ollama(self):
        """Verify Ollama is running and model is available"""
        try:
            # Check Ollama service
            models = ollama.list()
            
            # Handle different response formats
            if isinstance(models, dict):
                model_list = models.get('models', [])
            else:
                model_list = models if isinstance(models, list) else []
            
            # Extract model names safely
            model_names = []
            for model in model_list:
                if isinstance(model, dict):
                    name = model.get('name') or model.get('model', '')
                elif isinstance(model, str):
                    name = model
                else:
                    continue
                if name:
                    model_names.append(name)
            
            # Check for llama3.2 variants
            llama32_variants = [m for m in model_names if 'llama3.2' in m.lower() or 'llama3' in m.lower()]
            
            if not any(self.ollama_model in m for m in model_names) and not llama32_variants:
                logger.warning(f"Model {self.ollama_model} not found. Available models: {model_names}")
                logger.info(f"Attempting to pull {self.ollama_model}...")
                try:
                    ollama.pull(self.ollama_model)
                    logger.info(f"Model {self.ollama_model} pulled successfully")
                except Exception as e:
                    logger.warning(f"Could not pull {self.ollama_model}: {e}")
                    if llama32_variants:
                        self.ollama_model = llama32_variants[0]
                        logger.info(f"Using available model: {self.ollama_model}")
                    else:
                        # Try any available model
                        if model_names:
                            self.ollama_model = model_names[0]
                            logger.info(f"Using available model: {self.ollama_model}")
            elif llama32_variants and not any(self.ollama_model in m for m in model_names):
                # Use available llama3.2 variant
                self.ollama_model = llama32_variants[0]
                logger.info(f"Using available Llama 3.2 variant: {self.ollama_model}")
            
            # Test embedding capability
            test_response = ollama.embeddings(
                model=self.ollama_model,
                prompt="Test embedding"
            )
            logger.info(f"Ollama embedding test successful. Using model: {self.ollama_model}")
            
        except Exception as e:
            logger.error(f"Ollama connection failed: {str(e)}")
            logger.info("Please ensure Ollama is installed and running: curl -fsSL https://ollama.ai/install.sh | sh")
            logger.info("Then pull the model: ollama pull llama3.2")
            raise
    
    def _split_text(self, text: str) -> List[str]:
        """Split text into chunks with overlap"""
        if not text:
            return []
        
        # Try to split on paragraphs first
        paragraphs = re.split(r'\n\n+', text)
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # If adding this paragraph would exceed chunk size
            if current_chunk and len(current_chunk) + len(para) + 2 > self.chunk_size:
                # Save current chunk
                chunks.append(current_chunk)
                # Start new chunk with overlap
                overlap_text = current_chunk[-self.chunk_overlap:] if len(current_chunk) > self.chunk_overlap else current_chunk
                current_chunk = overlap_text + "\n\n" + para if overlap_text else para
            else:
                # Add to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
            
            # If current chunk is too large, split it further
            if len(current_chunk) > self.chunk_size:
                # Split by sentences
                sentences = re.split(r'(?<=[.!?])\s+', current_chunk)
                temp_chunk = ""
                for sentence in sentences:
                    if len(temp_chunk) + len(sentence) + 1 > self.chunk_size:
                        if temp_chunk:
                            chunks.append(temp_chunk)
                            # Start new with overlap
                            overlap = temp_chunk[-self.chunk_overlap:] if len(temp_chunk) > self.chunk_overlap else temp_chunk
                            temp_chunk = overlap + " " + sentence if overlap else sentence
                        else:
                            # Sentence itself is too long, split by words
                            words = sentence.split()
                            word_chunk = ""
                            for word in words:
                                if len(word_chunk) + len(word) + 1 > self.chunk_size:
                                    if word_chunk:
                                        chunks.append(word_chunk)
                                        overlap = word_chunk[-self.chunk_overlap:] if len(word_chunk) > self.chunk_overlap else word_chunk
                                        word_chunk = overlap + " " + word if overlap else word
                                    else:
                                        word_chunk = word
                                else:
                                    word_chunk = word_chunk + " " + word if word_chunk else word
                            if word_chunk:
                                temp_chunk = word_chunk
                    else:
                        temp_chunk = temp_chunk + " " + sentence if temp_chunk else sentence
                current_chunk = temp_chunk
        
        # Add remaining chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks if chunks else [text]
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add documents to vector database with enhanced processing"""
        if not documents:
            return
        
        all_chunks = []
        all_metadatas = []
        
        for doc_idx, doc in enumerate(documents):
            if not doc.get('success') or not doc.get('content'):
                continue
            
            content = doc['content']
            url = doc.get('url', 'unknown')
            doc_type = doc.get('source', 'web')
            
            # Split document into chunks
            chunks = self._split_text(content)
            
            for chunk_idx, chunk in enumerate(chunks):
                if len(chunk) < 50:  # Skip very short chunks
                    continue
                
                # Create metadata
                metadata = {
                    "url": url,
                    "doc_type": doc_type,
                    "chunk_index": chunk_idx,
                    "total_chunks": len(chunks),
                    "doc_index": doc_idx,
                    "timestamp": datetime.now().isoformat(),
                    "source": doc.get('method', 'unknown'),
                    "chunk_size": len(chunk)
                }
                
                all_chunks.append(chunk)
                all_metadatas.append(metadata)
        
        if all_chunks:
            # Add to ChromaDB in batches
            batch_size = 20  # Smaller batch size for Ollama embeddings
            for i in range(0, len(all_chunks), batch_size):
                end_idx = min(i + batch_size, len(all_chunks))
                
                batch_chunks = all_chunks[i:end_idx]
                batch_metadatas = all_metadatas[i:end_idx]
                batch_ids = [f"doc_{i+j}_{datetime.now().timestamp()}" for j in range(len(batch_chunks))]
                
                self.collection.add(
                    documents=batch_chunks,
                    metadatas=batch_metadatas,
                    ids=batch_ids
                )
                
                logger.info(f"Added batch {i//batch_size + 1}/{(len(all_chunks)-1)//batch_size + 1}: {len(batch_chunks)} chunks")
            
            logger.info(f"Added {len(all_chunks)} chunks from {len(documents)} documents to vector store")
    
    def retrieve_relevant_chunks(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant chunks with enhanced scoring"""
        # Query the collection (embeddings are handled by ChromaDB using Ollama)
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results * 2,  # Get more for filtering
            include=['documents', 'metadatas', 'distances']
        )
        
        if not results['documents']:
            return []
        
        # Process and score results
        scored_chunks = []
        for i, (chunk, metadata, distance) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        )):
            # Calculate relevance score (inverse of distance with some enhancements)
            relevance_score = 1 / (1 + distance)
            
            # Boost score for certain conditions
            if metadata.get('doc_type') == 'pdf':
                relevance_score *= 1.1  # Slight boost for PDF content
            
            # Penalize very short chunks
            if len(chunk) < 100:
                relevance_score *= 0.8
            
            # Boost for recent documents
            try:
                timestamp = metadata.get('timestamp', '')
                if timestamp:
                    doc_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    age_days = (datetime.now() - doc_time).days
                    if age_days < 7:  # Recent within a week
                        relevance_score *= 1.05
            except:
                pass
            
            scored_chunks.append({
                "chunk": chunk,
                "metadata": metadata,
                "relevance_score": relevance_score,
                "distance": distance
            })
        
        # Sort by relevance score and take top n
        scored_chunks.sort(key=lambda x: x['relevance_score'], reverse=True)
        return scored_chunks[:n_results]
    
    def generate_answer_with_llama(self, query: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate answer using Llama 3.2 with RAG"""
        
        if not context_chunks:
            # Fallback to direct generation if no context
            response = ollama.generate(
                model=self.ollama_model,
                prompt=f"""You are a helpful AI assistant specializing in information about Ghana. 
Answer the following question about Ghana to the best of your knowledge.

Question: {query}

Answer:""",
                options={
                    'temperature': 0.3,
                    'top_p': 0.9,
                    'num_predict': 500
                }
            )
            
            return {
                "answer": response['response'],
                "sources": [],
                "context_used": False
            }
        
        # Prepare context
        context_text = ""
        for i, chunk_info in enumerate(context_chunks):
            context_text += f"[Source {i+1}]\n{chunk_info['chunk'][:800]}\n\n"
        
        # Construct prompt for Llama 3.2
        prompt = f"""You are a helpful AI assistant specializing in information about Ghana. Use the provided context to answer the question accurately and informatively.

Context:
{context_text}

Question: {query}

Instructions:
1. Answer based primarily on the provided context
2. If the context doesn't contain sufficient information, you may supplement with general knowledge about Ghana
3. Keep the answer clear, concise, and informative
4. Cite sources when appropriate using [Source X] notation
5. Focus on accuracy and relevance to Ghana

Answer:"""
        
        # Generate answer with Ollama
        response = ollama.generate(
            model=self.ollama_model,
            prompt=prompt,
            options={
                'temperature': 0.2,  # Low temperature for factual accuracy
                'top_p': 0.9,
                'num_predict': 800,  # Response length
                'repeat_penalty': 1.1
            }
        )
        
        # Extract sources
        sources = []
        for chunk_info in context_chunks:
            sources.append({
                "url": chunk_info['metadata'].get('url', 'Unknown'),
                "relevance_score": chunk_info['relevance_score'],
                "source_type": chunk_info['metadata'].get('doc_type', 'web'),
                "chunk_preview": chunk_info['chunk'][:200] + "..."
            })
        
        return {
            "answer": response['response'],
            "sources": sources,
            "context_used": True,
            "model": self.ollama_model,
            "prompt_length": len(prompt),
            "response_length": len(response['response'])
        }
    
    def ask(self, query: str, n_context: int = 4) -> Dict[str, Any]:
        """Main method to answer a question"""
        logger.info(f"Processing question: {query}")
        
        # Retrieve relevant chunks
        context_chunks = self.retrieve_relevant_chunks(query, n_results=n_context)
        
        # Generate answer with Llama
        answer_result = self.generate_answer_with_llama(query, context_chunks)
        
        # Add query and metadata
        answer_result.update({
            "query": query,
            "context_chunks_retrieved": len(context_chunks),
            "timestamp": datetime.now().isoformat()
        })
        
        return answer_result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        try:
            count = self.collection.count()
            return {
                "collection_count": count,
                "embedding_model": self.ollama_model,
                "llm_model": self.ollama_model,
                "embedding_batch_size": self.embedding_batch_size,
                "status": "active"
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}