"""
Streamlit Interface for Domain Expert System - Task 4
Provides a web interface for scientific paper search, summarization, and concept explanation.
"""

import os
import sys
import streamlit as st
import pandas as pd
import json
from datetime import datetime
from typing import List, Dict
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add shared directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from shared.vector_db_manager import VectorDatabaseManager
from shared.llm_integration import LLMIntegration
from shared.embedding_service import EmbeddingService
from task4_domain_expert.domain_expert import DomainExpertSystem
from task4_domain_expert.data_loader import ArxivDataLoader
from task4_domain_expert.dataset_cache import DatasetCache


# Page configuration
st.set_page_config(
    page_title="Domain Expert System - arXiv Papers",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for improved contrast and modern design - consistent with other task apps
st.markdown("""
<style>
    /* Main theme colors - consistent with Task 1 Knowledge Updater */
    :root {
        --primary-color: #2E86AB;
        --secondary-color: #A23B72;
        --accent-color: #F18F01;
        --success-color: #28A745;
        --warning-color: #FFC107;
        --error-color: #DC3545;
        --dark-bg: #1E1E1E;
        --light-bg: #FFFFFF;
        --card-bg: #F8F9FA;
        --border-color: #DEE2E6;
        --text-primary: #FFFFFF;
        --text-secondary: #E0E0E0;
    }
    
    /* Main header styling - consistent with other apps */
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        color: var(--primary-color);
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 8px rgba(46, 134, 171, 0.3);
        /* Enhanced shadow for better visibility */
        filter: drop-shadow(0 2px 4px rgba(0,0,0,0.1));
    }
    
    /* Enhanced metric cards - consistent styling */
    .metric-card {
        background: linear-gradient(135deg, var(--card-bg) 0%, #FFFFFF 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 2px solid var(--border-color);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        border-color: var(--primary-color);
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 4px;
        background: linear-gradient(90deg, var(--primary-color), var(--accent-color));
    }
    
    .metric-card h3 {
        color: var(--text-primary);
        font-size: 1rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .metric-card h2 {
        color: var(--primary-color);
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        line-height: 1;
    }
    
    /* Paper card styling with consistent theme */
    .paper-card {
        background: linear-gradient(135deg, var(--card-bg) 0%, #FFFFFF 100%);
        border: 2px solid var(--border-color);
        border-radius: 12px;
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .paper-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        border-color: var(--primary-color);
    }
    
    .paper-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 4px;
        background: linear-gradient(90deg, var(--primary-color), var(--accent-color));
    }
    
    .paper-title {
        font-size: 1.4rem;
        font-weight: bold;
        color: var(--primary-color) !important;
        margin-bottom: 1rem;
        line-height: 1.4;
    }
    
    .paper-meta {
        font-size: 1.1rem;
        color: white !important;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    
    .similarity-score {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        color: white;
        padding: 0.4rem 0.8rem;
        border-radius: 6px;
        font-weight: bold;
        border: none;
        box-shadow: 0 2px 8px rgba(46, 134, 171, 0.3);
    }
    
    /* Chat messages with consistent theme */
    .conversation-message {
        background: var(--card-bg);
        padding: 2rem;
        margin: 1.5rem 0;
        border-radius: 15px;
        border: 2px solid var(--border-color);
        font-size: 1.1rem;
        line-height: 1.7;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    
    .user-message {
        background: linear-gradient(135deg, rgba(46, 134, 171, 0.2) 0%, rgba(46, 134, 171, 0.3) 100%);
        border-left: 4px solid var(--primary-color);
        color: white !important;
    }
    
    .user-message * {
        color: white !important;
    }
    
    .assistant-message {
        background: linear-gradient(135deg, rgba(40, 167, 69, 0.2) 0%, rgba(40, 167, 69, 0.3) 100%);
        border-left: 4px solid var(--success-color);
        color: white !important;
    }
    
    .assistant-message * {
        color: white !important;
    }
    
    /* Concept explanation with consistent theme */
    .concept-explanation {
        background: linear-gradient(135deg, #FFF8E1 0%, #FFECB3 100%);
        border: 2px solid var(--accent-color);
        border-radius: 15px;
        padding: 2.5rem;
        margin: 2rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        position: relative;
        overflow: hidden;
    }
    
    .concept-explanation::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 4px;
        background: linear-gradient(90deg, var(--secondary-color), var(--accent-color));
    }
    
    .explanation-content {
        font-size: 1.2rem;
        line-height: 1.8;
        color: white !important;
        font-weight: 500;
    }
    
    .explanation-content * {
        color: white !important;
    }
    
    /* Enhanced buttons - consistent with other apps */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(46, 134, 171, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(46, 134, 171, 0.4);
        color: white;
    }
    
    /* Status messages - consistent styling */
    .success-message {
        background: linear-gradient(135deg, #D4EDDA 0%, #C3E6CB 100%);
        color: var(--success-color);
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid var(--success-color);
        font-weight: 500;
        box-shadow: 0 2px 8px rgba(40, 167, 69, 0.2);
    }
    
    .error-message {
        background: linear-gradient(135deg, #F8D7DA 0%, #F5C6CB 100%);
        color: var(--error-color);
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid var(--error-color);
        font-weight: 500;
        box-shadow: 0 2px 8px rgba(220, 53, 69, 0.2);
    }
    
    .warning-message {
        background: linear-gradient(135deg, #FFF3CD 0%, #FFEAA7 100%);
        color: #856404;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid var(--warning-color);
        font-weight: 500;
        box-shadow: 0 2px 8px rgba(255, 193, 7, 0.2);
    }
    
    /* Sidebar styling - consistent with other apps */
    .css-1d391kg {
        background: linear-gradient(180deg, var(--dark-bg) 0%, #2C2C2C 100%);
    }
    
    .css-1d391kg .css-1v0mbdj {
        color: white !important;
    }
    
    /* Comprehensive text visibility fixes */
    
    /* Sidebar text visibility - force white text on dark background */
    .css-1d391kg, .css-1d391kg * {
        color: white !important;
    }
    
    .css-1d391kg .stMarkdown, .css-1d391kg .stMarkdown * {
        color: white !important;
    }
    
    .css-1d391kg .stSelectbox label, .css-1d391kg .stSelectbox div,
    .css-1d391kg .stMetric label, .css-1d391kg .stMetric div,
    .css-1d391kg .stInfo, .css-1d391kg .stInfo * {
        color: white !important;
    }
    
    /* Main content text visibility - ensure white text on dark backgrounds */
    .stMarkdown, .stMarkdown p, .stMarkdown div, .stMarkdown span,
    .stText, .stText p, .stText div, .stText span {
        color: white !important;
    }
    
    /* Form labels and inputs - high contrast white text */
    .stSelectbox label, .stTextInput label, .stTextArea label,
    .stButton label, .stMetric label, .stMetric div,
    .stSlider label, .stCheckbox label, .stRadio label {
        color: white !important;
        font-weight: 600 !important;
    }
    
    /* Tab text visibility */
    .stTabs [data-baseweb="tab"] {
        color: white !important;
        font-weight: 600 !important;
    }
    
    .stTabs [aria-selected="true"] {
        color: white !important;
        font-weight: 700 !important;
    }
    
    /* Headers and subheaders */
    h1, h2, h3, h4, h5, h6 {
        color: white !important;
        font-weight: 600 !important;
    }
    
    /* Expander headers */
    .streamlit-expanderHeader {
        color: white !important;
        font-weight: 600 !important;
    }
    
    /* Metric values */
    .metric-value {
        color: var(--primary-color) !important;
        font-weight: 700 !important;
    }
    
    /* Info, warning, error, success message text */
    .stInfo, .stInfo * {
        color: var(--primary-color) !important;
        font-weight: 500 !important;
    }
    
    .stWarning, .stWarning * {
        color: #856404 !important;
        font-weight: 500 !important;
    }
    
    .stError, .stError * {
        color: var(--error-color) !important;
        font-weight: 500 !important;
    }
    
    .stSuccess, .stSuccess * {
        color: var(--success-color) !important;
        font-weight: 500 !important;
    }
    
    /* Input field styling */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stSelectbox > div > div > select {
        background-color: var(--light-bg);
        color: var(--text-primary);
        border: 2px solid var(--border-color);
        border-radius: 8px;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: var(--light-bg);
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: var(--card-bg);
        color: var(--text-secondary);
        border: 2px solid var(--border-color);
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        color: white;
        border-color: var(--primary-color);
        box-shadow: 0 4px 12px rgba(46, 134, 171, 0.3);
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, var(--primary-color), var(--accent-color));
    }
    
    /* Info boxes */
    .stInfo {
        background: linear-gradient(135deg, #D1ECF1 0%, #BEE5EB 100%);
        border-left: 4px solid var(--primary-color);
    }
    
    /* Section headers */
    h2, h3 {
        color: var(--text-primary);
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    /* Additional text visibility fixes */
    
    /* Ensure all text in containers is visible */
    .stContainer, .stContainer * {
        color: white !important;
    }
    
    /* Override Streamlit's default text colors */
    .stApp {
        color: white !important;
    }
    
    /* Main app background and text */
    .main .block-container {
        color: white !important;
    }
    
    /* All text elements */
    * {
        color: white !important;
    }
    
    /* Placeholder text visibility */
    input::placeholder, textarea::placeholder {
        color: var(--text-secondary) !important;
        opacity: 0.7;
    }
    
    /* Dropdown and select text */
    .stSelectbox > div > div > div {
        color: white !important;
    }
    
    /* Input field text */
    .stTextInput > div > div > input {
        color: white !important;
        background-color: rgba(255, 255, 255, 0.1) !important;
    }
    
    /* Text area text */
    .stTextArea > div > div > textarea {
        color: white !important;
        background-color: rgba(255, 255, 255, 0.1) !important;
    }
    
    /* Button text - ensure it stays white on colored backgrounds */
    .stButton > button {
        color: white !important;
    }
    
    .stButton > button:hover {
        color: white !important;
    }
    
    /* Dataframe text */
    .stDataFrame, .stDataFrame * {
        color: white !important;
    }
    
    /* JSON display text */
    .stJson, .stJson * {
        color: white !important;
    }
    
    /* Code block text */
    .stCode, .stCode * {
        color: white !important;
    }
    
    /* Ensure spinner text is visible */
    .stSpinner > div {
        color: var(--primary-color) !important;
    }
    
    /* Progress bar text */
    .stProgress > div > div > div > div {
        color: white !important;
        font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_system():
    """Initialize the domain expert system with caching."""
    try:
        # Initialize components
        vector_db = VectorDatabaseManager()
        llm = LLMIntegration()
        embedding_service = EmbeddingService()
        
        # Initialize domain expert system
        expert_system = DomainExpertSystem(
            domain="computer_science",
            vector_db=vector_db,
            llm=llm,
            embedding_service=embedding_service
        )
        
        return expert_system, True
    except Exception as e:
        st.error(f"Failed to initialize system: {str(e)}")
        return None, False


def load_dataset_cached(expert_system):
    """Load dataset with error handling for Streamlit."""
    try:
        success = expert_system.load_arxiv_dataset()
        if success:
            stats = expert_system.get_domain_statistics()
            return True, stats
        else:
            return False, {}
    except Exception as e:
        st.error(f"Error in dataset loading: {str(e)}")
        return False, {}


def load_dataset_if_needed(expert_system):
    """Load dataset into vector database if not already loaded."""
    try:
        # Check if dataset exists first
        data_loader = ArxivDataLoader()
        if not data_loader.processed_data_file.exists():
            st.warning("⚠️ No dataset found. Creating sample dataset for demonstration...")
            
            # Create a minimal sample dataset
            sample_papers = [
                {
                    "id": "sample_001",
                    "title": "Introduction to Machine Learning",
                    "abstract": "This paper provides an introduction to machine learning concepts, algorithms, and applications in various domains.",
                    "authors": ["Sample Author"],
                    "categories": ["cs.LG"],
                    "primary_category": "cs.LG",
                    "category_name": "Machine Learning",
                    "published_date": "2023-01-01",
                    "pdf_link": "https://arxiv.org/pdf/sample_001.pdf",
                    "text_for_embedding": "Introduction to Machine Learning This paper provides an introduction to machine learning concepts, algorithms, and applications in various domains.",
                    "processed_date": datetime.now().isoformat()
                }
            ]
            
            # Save sample dataset
            import json
            data_loader.data_dir.mkdir(parents=True, exist_ok=True)
            with open(data_loader.processed_data_file, 'w', encoding='utf-8') as f:
                json.dump(sample_papers, f, indent=2, ensure_ascii=False)
            
            st.info("📝 Sample dataset created. For full functionality, run: `python task4_domain_expert/data_loader.py`")
        
        # Try to load the dataset
        with st.spinner("Loading dataset..."):
            success, stats = load_dataset_cached(expert_system)
        
        if success:
            papers_count = stats.get('papers_loaded', 0)
            if papers_count > 0:
                st.success(f"✅ Dataset ready with {papers_count} papers")
            else:
                st.success("✅ Dataset loaded successfully!")
            return True
        else:
            st.warning("⚠️ Dataset loading failed, but system can still work with limited functionality")
            return True  # Allow system to continue with limited functionality
            
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        st.info("💡 The system can still work for concept explanations without the dataset")
        return True  # Allow system to continue


def display_paper_card(paper: Dict, show_abstract: bool = True):
    """Display a paper in a card format."""
    with st.container():
        st.markdown(f"""
        <div class="paper-card">
            <div class="paper-title">{paper['title']}</div>
            <div class="paper-meta">
                <strong>Category:</strong> {paper.get('category_name', 'Unknown')} ({paper.get('primary_category', 'N/A')}) |
                <strong>Authors:</strong> {', '.join(paper.get('authors', [])[:3])}{'...' if len(paper.get('authors', [])) > 3 else ''} |
                <strong>Similarity:</strong> <span class="similarity-score">{paper.get('similarity_score', 0):.3f}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if show_abstract and paper.get('abstract'):
            with st.expander("📄 Abstract"):
                st.write(paper['abstract'])
        
        # Action buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Use a more unique key combining paper ID and timestamp
            paper_id = paper.get('id', f"paper_{hash(paper.get('title', 'unknown'))}")
            if st.button(f"📄 Summarize", key=f"summarize_{paper_id}_{hash(str(paper))}"):
                return "summarize"
        
        with col2:
            if paper.get('pdf_link'):
                st.markdown(f"[📎 PDF Link]({paper['pdf_link']})")
        
        with col3:
            if st.button(f"💬 Discuss", key=f"discuss_{paper_id}_{hash(str(paper))}"):
                return "discuss"
    
    return None


def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🤖 Domain Expert System</h1>', unsafe_allow_html=True)
    st.markdown("**Task 4**: Scientific Paper Search, Summarization & Expert Explanations")
    
    # Check API key first
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        st.error("❌ Google API key not found!")
        st.info("Please set the GOOGLE_API_KEY environment variable in your .env file")
        st.stop()
    
    # Initialize system with error handling
    try:
        expert_system, init_success = initialize_system()
        
        if not init_success or expert_system is None:
            st.error("❌ Failed to initialize the domain expert system.")
            st.info("Please check your configuration and try refreshing the page.")
            st.stop()
        
        st.success("✅ System initialized successfully")
        
    except Exception as e:
        st.error(f"❌ System initialization error: {str(e)}")
        st.info("Please check your configuration and try refreshing the page.")
        st.stop()
    
    # Load dataset with graceful fallback
    dataset_loaded = load_dataset_if_needed(expert_system)
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 System Status")
        
        # Get system statistics
        stats = expert_system.get_domain_statistics()
        st.metric("Papers Loaded", f"{stats.get('papers_loaded', 0):,}")
        st.metric("Active Conversations", stats.get('active_conversations', 0))
        st.metric("System Status", stats.get('system_status', 'Unknown'))
        
        st.header("📚 Categories")
        st.info("""
        **Available Categories:**
        - 🤖 Artificial Intelligence (cs.AI)
        - 🧠 Machine Learning (cs.LG)
        - 💬 Natural Language Processing (cs.CL)
        - 👁️ Computer Vision (cs.CV)
        - 🧬 Neural Networks (cs.NE)
        - 🤖 Robotics (cs.RO)
        - 🔍 Information Retrieval (cs.IR)
        - 👥 Human-Computer Interaction (cs.HC)
        - 📊 Data Structures & Algorithms (cs.DS)
        - 🗄️ Databases (cs.DB)
        """)
    
    # Main interface tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Paper Search", "💡 Concept Explanation", "💬 Expert Chat", "📊 System Info"])
    
    # Tab 1: Paper Search
    with tab1:
        st.header("🔍 Scientific Paper Search")
        
        # Search interface
        col1, col2 = st.columns([3, 1])
        
        with col1:
            search_query = st.text_input(
                "Enter your search query:",
                placeholder="e.g., machine learning neural networks, computer vision CNN, natural language processing...",
                help="Search for papers using keywords related to your research interest"
            )
        
        with col2:
            num_results = st.selectbox("Results", [5, 10, 15, 20], index=1)
        
        if st.button("🔍 Search Papers", type="primary"):
            if search_query.strip():
                with st.spinner("Searching for relevant papers..."):
                    papers = expert_system.search_papers(search_query, top_k=num_results)
                
                if papers:
                    st.success(f"Found {len(papers)} relevant papers")
                    
                    # Display results
                    for i, paper in enumerate(papers, 1):
                        st.subheader(f"📄 Result {i}")
                        action = display_paper_card(paper)
                        
                        if action == "summarize":
                            with st.spinner("Generating summary..."):
                                summary = expert_system.summarize_paper(paper_data=paper)
                                
                                if summary and not summary.startswith("Error:"):
                                    st.success("📄 Paper Summary Generated")
                                    st.write(summary)
                                else:
                                    st.warning("⚠️ Summary generation requires API key configuration")
                                    st.info("The system can search and retrieve papers, but summarization requires LLM access.")
                        
                        elif action == "discuss":
                            st.session_state['discuss_paper'] = paper
                            st.session_state['active_tab'] = 'expert_chat'
                            st.rerun()
                        
                        st.divider()
                else:
                    st.warning("No papers found for your query. Try different keywords or broader terms.")
            else:
                st.warning("Please enter a search query.")
    
    # Tab 2: Concept Explanation
    with tab2:
        st.header("💡 Concept Explanation")
        
        # Concept explanation interface
        concept_query = st.text_input(
            "Enter a concept to explain:",
            placeholder="e.g., transformer architecture, convolutional neural networks, reinforcement learning...",
            help="Ask for explanations of scientific concepts and methodologies"
        )
        
        context_input = st.text_area(
            "Additional context (optional):",
            placeholder="Provide any additional context or specific aspects you'd like to focus on...",
            height=100
        )
        
        if st.button("💡 Explain Concept", type="primary"):
            if concept_query.strip():
                with st.spinner("Generating explanation..."):
                    explanation = expert_system.explain_concept(concept_query, context_input)
                
                if explanation and not explanation.startswith("Error:"):
                    st.success("💡 Concept Explanation Generated")
                    
                    # Display explanation with better formatting
                    st.markdown(f"""
                    <div class="concept-explanation">
                        <div class="explanation-content">
                            {explanation.replace('\n', '<br>')}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show related papers
                    with st.expander("📚 Related Papers"):
                        related_papers = expert_system.search_papers(concept_query, top_k=3)
                        if related_papers:
                            for paper in related_papers:
                                st.write(f"**{paper['title']}**")
                                st.write(f"*Category: {paper['category_name']} | Similarity: {paper['similarity_score']:.3f}*")
                                st.write(f"{paper['abstract'][:200]}...")
                                st.divider()
                else:
                    st.warning("⚠️ Concept explanation requires API key configuration")
                    st.info("The system can find related papers, but detailed explanations require LLM access.")
                    
                    # Show related papers as fallback
                    st.subheader("📚 Related Papers Found:")
                    related_papers = expert_system.search_papers(concept_query, top_k=5)
                    if related_papers:
                        for paper in related_papers:
                            display_paper_card(paper, show_abstract=False)
            else:
                st.warning("Please enter a concept to explain.")
    
    # Tab 3: Expert Chat
    with tab3:
        st.header("💬 Expert Chat")
        
        # Initialize conversation
        if 'conversation_id' not in st.session_state:
            st.session_state['conversation_id'] = f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        if 'chat_history' not in st.session_state:
            st.session_state['chat_history'] = []
        
        # Display chat history with improved visibility
        if st.session_state['chat_history']:
            st.subheader("💬 Conversation History")
            for message in st.session_state['chat_history']:
                if message['role'] == 'user':
                    st.markdown(f"""
                    <div class="conversation-message user-message">
                        <strong style="color: #0d6efd; font-size: 1.1rem;">👤 You:</strong><br>
                        <span style="color: #212529; font-size: 1rem; font-weight: 400;">{message['content']}</span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="conversation-message assistant-message">
                        <strong style="color: #198754; font-size: 1.1rem;">🤖 Expert:</strong><br>
                        <span style="color: #212529; font-size: 1rem; font-weight: 400;">{message['content']}</span>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Chat input
        st.subheader("💭 Ask the Expert")
        
        # Check if we're discussing a specific paper
        if 'discuss_paper' in st.session_state:
            paper = st.session_state['discuss_paper']
            st.info(f"💬 Discussing: **{paper['title'][:80]}...**")
            
            if st.button("🔄 Start New Conversation"):
                if 'discuss_paper' in st.session_state:
                    del st.session_state['discuss_paper']
                st.session_state['chat_history'] = []
                st.session_state['conversation_id'] = f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                st.rerun()
        
        user_question = st.text_area(
            "Your question:",
            placeholder="Ask follow-up questions, request clarifications, or explore related topics...",
            height=100
        )
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("💬 Send Message", type="primary"):
                if user_question.strip():
                    # Add user message to history
                    st.session_state['chat_history'].append({
                        'role': 'user',
                        'content': user_question,
                        'timestamp': datetime.now().isoformat()
                    })
                    
                    with st.spinner("Expert is thinking..."):
                        response = expert_system.handle_followup(
                            user_question, 
                            st.session_state['conversation_id']
                        )
                    
                    if response and not response.startswith("Error:"):
                        # Add assistant response to history
                        st.session_state['chat_history'].append({
                            'role': 'assistant',
                            'content': response,
                            'timestamp': datetime.now().isoformat()
                        })
                        st.rerun()
                    else:
                        st.warning("⚠️ Expert chat requires API key configuration")
                        st.info("The system can search for relevant papers, but conversational responses require LLM access.")
                else:
                    st.warning("Please enter a question.")
        
        with col2:
            if st.button("🔄 Clear Conversation"):
                st.session_state['chat_history'] = []
                st.session_state['conversation_id'] = f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                if 'discuss_paper' in st.session_state:
                    del st.session_state['discuss_paper']
                st.rerun()
    
    # Tab 4: System Information
    with tab4:
        st.header("📊 System Information")
        
        # System statistics
        stats = expert_system.get_domain_statistics()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 System Statistics")
            st.json(stats)
        
        with col2:
            st.subheader("🔧 Configuration")
            
            # Check API key status
            api_key_status = "✅ Configured" if os.getenv('GOOGLE_API_KEY') else "❌ Not Set"
            st.write(f"**Google API Key:** {api_key_status}")
            
            if not os.getenv('GOOGLE_API_KEY'):
                st.warning("⚠️ LLM features (summarization, explanation, chat) require API key configuration")
                st.info("Set the GOOGLE_API_KEY environment variable to enable all features.")
            
            st.write(f"**Domain:** {stats.get('domain', 'Unknown')}")
            st.write(f"**Collection:** {stats.get('collection_name', 'Unknown')}")
        
        # Sample queries
        st.subheader("💡 Sample Queries")
        
        sample_queries = [
            "machine learning neural networks deep learning",
            "computer vision image recognition CNN",
            "natural language processing transformers BERT",
            "reinforcement learning policy gradient",
            "robotics autonomous navigation",
            "artificial intelligence reasoning knowledge"
        ]
        
        st.write("Try these sample queries to explore the system:")
        for i, query in enumerate(sample_queries):
            if st.button(f"🔍 {query}", key=f"sample_query_{i}_{hash(query)}"):
                st.session_state['sample_query'] = query
                # Switch to search tab
                st.rerun()
        
        # Dataset information
        st.subheader("📚 Dataset Information")
        
        try:
            data_loader = ArxivDataLoader()
            papers = data_loader.load_processed_papers()
            
            if papers:
                df = pd.DataFrame(papers)
                
                st.write(f"**Total Papers:** {len(df):,}")
                
                # Category distribution
                category_counts = df['primary_category'].value_counts()
                st.write("**Category Distribution:**")
                
                chart_data = pd.DataFrame({
                    'Category': category_counts.index,
                    'Count': category_counts.values
                })
                
                st.bar_chart(chart_data.set_index('Category'))
                
            else:
                st.warning("No dataset information available")
                
        except Exception as e:
            st.error(f"Error loading dataset information: {str(e)}")


if __name__ == "__main__":
    main()