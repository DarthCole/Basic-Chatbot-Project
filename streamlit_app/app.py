import streamlit as st
import requests
import json
import time
import pandas as pd
from datetime import datetime

st.set_page_config(
    page_title="SafoRAG Pro",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #4B5563;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-box {
        padding: 1rem;
        background-color: #D1FAE5;
        border-radius: 0.5rem;
        border-left: 4px solid #10B981;
    }
    .info-box {
        padding: 1rem;
        background-color: #DBEAFE;
        border-radius: 0.5rem;
        border-left: 4px solid #3B82F6;
    }
    .source-card {
        padding: 0.75rem;
        background-color: #F3F4F6;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
        border-left: 3px solid #8B5CF6;
    }
</style>
""", unsafe_allow_html=True)

# App header
st.markdown('<h1 class="main-header">🤖 SafoRAG Pro</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Advanced Web Scraping & RAG System with Llama 2</p>', unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    api_url = st.text_input("Backend API URL", "http://localhost:5000")
    
    st.markdown("---")
    st.header("📊 System Status")
    
    if st.button("Check System Health"):
        try:
            response = requests.get(f"{api_url}/stats")
            if response.status_code == 200:
                stats = response.json()
                st.success("✅ System is healthy")
                st.json(stats)
            else:
                st.error("❌ System not responding")
        except:
            st.error("❌ Cannot connect to backend")
    
    st.markdown("---")
    st.header("📖 Instructions")
    st.info("""
    1. Enter a URL to scrape websites and extract PDFs
    2. Ask questions about the scraped content
    3. View detailed sources and relevance scores
    4. System uses Llama 2 via Ollama for intelligent answers
    """)

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs(["🌐 Web Scraping", "❓ Ask Questions", "📊 Analytics", "⚡ Direct Query"])

with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Web Scraping")
        url = st.text_input("Enter URL to scrape:", placeholder="https://example.com")
        
        col1a, col1b = st.columns(2)
        with col1a:
            scrape_method = st.selectbox(
                "Scraping Method",
                ["auto", "selenium", "playwright", "static"],
                help="Auto tries all methods, Selenium for JS-heavy sites"
            )
        with col1b:
            extract_pdfs = st.checkbox("Extract PDFs", value=True)
        
        if st.button("🚀 Start Scraping", type="primary", use_container_width=True):
            if url:
                with st.spinner("Scraping website and extracting content..."):
                    try:
                        response = requests.post(
                            f"{api_url}/scrape",
                            json={"url": url, "method": scrape_method},
                            timeout=60
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            st.session_state.last_scrape = result
                            
                            st.success("✅ Scraping started!")
                            st.info(f"Task ID: {result['task_id']}")
                            st.info(f"Check status: {result['status_url']}")
                            
                            # Poll for completion
                            task_id = result['task_id']
                            for i in range(10):
                                time.sleep(2)
                                status_response = requests.get(f"{api_url}/status/{task_id}")
                                if status_response.status_code == 200:
                                    status_data = status_response.json()
                                    if status_data['status'] == 'completed':
                                        st.balloons()
                                        st.success("✅ Scraping completed!")
                                        
                                        scrape_result = status_data['result']
                                        
                                        with st.expander("📄 Scraped Content Preview", expanded=False):
                                            st.text_area(
                                                "Content",
                                                scrape_result.get('content', '')[:3000],
                                                height=200
                                            )
                                        
                                        if scrape_result.get('pdf_links'):
                                            st.subheader("📎 PDFs Found")
                                            for pdf in scrape_result['pdf_links'][:5]:
                                                st.write(f"• {pdf['text']}")
                                                st.code(pdf['url'])
                                        
                                        break
                                    elif status_data['status'] == 'failed':
                                        st.error(f"Scraping failed: {status_data.get('error', 'Unknown error')}")
                                        break
                                else:
                                    st.warning(f"Still processing... ({i+1}/10)")
                        else:
                            st.error(f"Scraping failed: {response.text}")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            else:
                st.warning("Please enter a URL")
    
    with col2:
        st.header("Recent Scrapes")
        if 'last_scrape' in st.session_state:
            last = st.session_state.last_scrape
            st.metric("Last URL", last.get('url', 'N/A')[:50] + "...")
            st.metric("Method", last.get('method', 'unknown'))
            st.metric("Content Length", f"{last.get('content_length', 0):,} chars")
            
            if last.get('pdf_links'):
                st.metric("PDFs Found", len(last['pdf_links']))

with tab2:
    st.header("Ask Questions")
    
    question = st.text_area(
        "Enter your question:",
        placeholder="What information did you find about...",
        height=100
    )
    
    col2a, col2b = st.columns([1, 3])
    with col2a:
        context_chunks = st.slider("Context chunks", 2, 10, 4)
    
    if st.button("🔍 Get Answer", type="primary", use_container_width=True):
        if question:
            with st.spinner("🧠 Thinking... (Using Llama 2 via Ollama)"):
                try:
                    # Use direct endpoint for synchronous response
                    response = requests.post(
                        f"{api_url}/direct-ask",
                        json={"question": question},
                        timeout=120
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Display answer
                        st.markdown("### 🤖 Answer")
                        st.markdown("---")
                        st.markdown(result['answer'])
                        
                        # Display sources
                        if result.get('sources'):
                            st.markdown("### 📚 Sources")
                            for i, source in enumerate(result['sources'], 1):
                                with st.expander(f"Source {i}: {source.get('url', 'Unknown')[:80]}...", expanded=False):
                                    col_source = st.columns([3, 1])
                                    with col_source[0]:
                                        st.write(f"**URL:** {source.get('url', 'N/A')}")
                                        st.write(f"**Type:** {source.get('source_type', 'web')}")
                                        st.write(f"**Preview:** {source.get('chunk_preview', 'N/A')}")
                                    with col_source[1]:
                                        st.metric("Relevance", f"{source.get('relevance_score', 0):.2%}")
                        
                        # Display metadata
                        with st.expander("📊 Metadata", expanded=False):
                            st.json({
                                "query": result.get('query'),
                                "model": result.get('model'),
                                "context_used": result.get('context_used'),
                                "context_chunks": result.get('context_chunks_retrieved'),
                                "timestamp": result.get('timestamp')
                            })
                    
                    else:
                        st.error(f"Failed to get answer: {response.text}")
                        
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.info("Make sure Ollama is running: `ollama serve`")
        else:
            st.warning("Please enter a question")

with tab3:
    st.header("System Analytics")
    
    try:
        # Get system stats
        stats_response = requests.get(f"{api_url}/stats")
        if stats_response.status_code == 200:
            stats = stats_response.json()
            
            col3a, col3b, col3c = st.columns(3)
            
            with col3a:
                st.metric("Documents in DB", stats.get('collection_count', 0))
            
            with col3b:
                st.metric("Embedding Model", stats.get('embedding_model', 'N/A'))
            
            with col3c:
                st.metric("LLM Model", stats.get('llm_model', 'N/A'))
            
            # Performance metrics
            st.subheader("Performance")
            
            # Example queries
            example_queries = [
                "What is the main topic?",
                "Summarize the key points",
                "What are the important findings?"
            ]
            
            query_to_test = st.selectbox("Test query", example_queries)
            
            if st.button("Test Performance"):
                with st.spinner("Testing..."):
                    start_time = time.time()
                    response = requests.post(
                        f"{api_url}/direct-ask",
                        json={"question": query_to_test},
                        timeout=60
                    )
                    end_time = time.time()
                    
                    if response.status_code == 200:
                        st.success(f"Response time: {(end_time - start_time):.2f} seconds")
                        result = response.json()
                        st.write(f"Answer length: {len(result.get('answer', ''))} characters")
                        st.write(f"Chunks used: {result.get('context_chunks_retrieved', 0)}")
        
        else:
            st.error("Could not retrieve system stats")
            
    except:
        st.error("Cannot connect to backend API")

with tab4:
    st.header("⚡ Direct Ollama Query")
    
    st.info("This bypasses RAG and directly queries Llama 2")
    
    direct_prompt = st.text_area(
        "Direct prompt to Llama 2:",
        placeholder="Explain quantum computing in simple terms...",
        height=150
    )
    
    col4a, col4b, col4c = st.columns(3)
    with col4a:
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
    with col4b:
        max_tokens = st.slider("Max Tokens", 100, 2000, 500)
    with col4c:
        top_p = st.slider("Top P", 0.1, 1.0, 0.9, 0.1)
    
    if st.button("Generate Direct Response", type="secondary"):
        if direct_prompt:
            with st.spinner("Generating with Llama 2..."):
                try:
                    import subprocess
                    result = subprocess.run(
                        ['ollama', 'run', 'llama2:7b', direct_prompt],
                        capture_output=True,
                        text=True,
                        timeout=30
                    )
                    
                    if result.returncode == 0:
                        st.markdown("### 📝 Llama 2 Response")
                        st.markdown("---")
                        st.write(result.stdout)
                    else:
                        st.error(f"Error: {result.stderr}")
                        
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.info("Make sure Ollama is installed and running")
        else:
            st.warning("Please enter a prompt")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6B7280;">
    <p>🤖 SafoRAG Pro • Powered by Llama 2, Selenium, and ChromaDB</p>
    <p>Backend: Flask • Frontend: Streamlit • Vector DB: ChromaDB • LLM: Ollama</p>
</div>
""", unsafe_allow_html=True)