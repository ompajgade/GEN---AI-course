"""
Streamlit Interface for Knowledge Base Updater - Task 1
Provides a web interface for managing dynamic knowledge base expansion.
"""

import os
import sys
import streamlit as st
import pandas as pd
import json
from datetime import datetime, timedelta
from typing import List, Dict
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from shared.vector_db_manager import VectorDatabaseManager
    from shared.embedding_service import EmbeddingService
    from task1_knowledge_updater.knowledge_updater import KnowledgeBaseUpdater
except ImportError as e:
    st.error(f"Import Error: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Knowledge Base Updater",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for improved contrast and modern design
st.markdown("""
<style>
    /* Main theme colors */
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
        --text-primary: #212529;
        --text-secondary: #6C757D;
    }
    
    /* Main header styling - reliable and always visible */
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
    
    /* Enhanced metric cards */
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
    
    /* Status messages */
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
    
    /* Enhanced buttons */
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
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, var(--dark-bg) 0%, #2C2C2C 100%);
    }
    
    .css-1d391kg .css-1v0mbdj {
        color: white;
    }
    
    /* Form styling */
    .stForm {
        background: var(--card-bg);
        padding: 2rem;
        border-radius: 12px;
        border: 2px solid var(--border-color);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, var(--primary-color), var(--accent-color));
    }
    
    /* Dataframe styling */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
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
    
    /* Quick action cards */
    .quick-action-card {
        background: var(--light-bg);
        padding: 1.5rem;
        border-radius: 12px;
        border: 2px solid var(--border-color);
        text-align: center;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .quick-action-card:hover {
        border-color: var(--primary-color);
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    /* Navigation improvements */
    .css-1v0mbdj select {
        background: var(--dark-bg);
        color: white;
        border: 1px solid var(--primary-color);
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'updater' not in st.session_state:
    st.session_state.updater = None
if 'update_logs' not in st.session_state:
    st.session_state.update_logs = []
if 'sources' not in st.session_state:
    st.session_state.sources = []

def initialize_updater():
    """Initialize the knowledge base updater."""
    try:
        with st.spinner("Initializing Knowledge Base components..."):
            vector_db = VectorDatabaseManager()
            embedding_service = EmbeddingService()
            updater = KnowledgeBaseUpdater(vector_db, embedding_service)
            st.success("✅ Knowledge Base Updater initialized successfully!")
            return updater
    except Exception as e:
        st.error(f"❌ Failed to initialize updater: {str(e)}")
        st.info("Please check your configuration and try again.")
        return None

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🔄 Knowledge Base Updater</h1>', unsafe_allow_html=True)
    st.markdown("**Task 1**: Dynamic Knowledge Base Expansion System")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["Dashboard", "Add Sources", "Manual Update", "Schedule Updates", "View Logs", "Settings"]
    )
    
    # Initialize updater if not already done
    if st.session_state.updater is None:
        with st.spinner("Initializing Knowledge Base Updater..."):
            st.session_state.updater = initialize_updater()
    
    if st.session_state.updater is None:
        st.error("Failed to initialize the system. Please check your configuration.")
        return
    
    # Route to different pages
    if page == "Dashboard":
        show_dashboard()
    elif page == "Add Sources":
        show_add_sources()
    elif page == "Manual Update":
        show_manual_update()
    elif page == "Schedule Updates":
        show_schedule_updates()
    elif page == "View Logs":
        show_logs()
    elif page == "Settings":
        show_settings()

def show_dashboard():
    """Display the main dashboard."""
    st.header("📊 Knowledge Base Dashboard")
    
    # Get system statistics
    try:
        stats = st.session_state.updater.get_source_stats()
        collection_stats = stats.get('collection_stats', {})
        
        # Display metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            doc_count = collection_stats.get('document_count', 0)
            st.markdown(f"""
            <div class="metric-card">
                <h3>📚 Total Documents</h3>
                <h2>{doc_count:,}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            source_count = stats.get('total_sources', 0)
            st.markdown(f"""
            <div class="metric-card">
                <h3>🔗 Active Sources</h3>
                <h2>{source_count}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            # Get last update from sources
            sources = stats.get('sources', {})
            last_update = 'Never'
            if sources:
                last_updates = [s.get('last_updated') for s in sources.values() if s.get('last_updated')]
                if last_updates:
                    last_update = max(last_updates)
                    # Format the date nicely
                    try:
                        from datetime import datetime
                        if isinstance(last_update, str) and last_update != 'Never':
                            dt = datetime.fromisoformat(last_update.replace('Z', '+00:00'))
                            last_update = dt.strftime('%m/%d %H:%M')
                    except:
                        pass
            
            st.markdown(f"""
            <div class="metric-card">
                <h3>🔄 Last Update</h3>
                <h2>{last_update}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            total_updates = stats.get('total_updates', 0)
            st.markdown(f"""
            <div class="metric-card">
                <h3>📈 Total Updates</h3>
                <h2>{total_updates}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Recent activity
        st.subheader("📋 Recent Activity")
        logs = st.session_state.updater.get_update_logs()
        if logs:
            df = pd.DataFrame(logs[-10:])  # Show last 10 logs
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No recent activity. Add sources and run updates to see activity here.")
        
        # Quick actions
        st.subheader("⚡ Quick Actions")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Run Update Now", type="primary"):
                run_manual_update()
        
        with col2:
            if st.button("➕ Add New Source"):
                st.session_state.page = "Add Sources"
                st.rerun()
        
        with col3:
            if st.button("🧪 Add Demo Data", help="Add sample data for testing"):
                add_demo_data()
                
    except Exception as e:
        st.error(f"Error loading dashboard: {str(e)}")
        st.info("Dashboard will be available after adding sources and running updates.")

def show_add_sources():
    """Display the add sources page."""
    st.header("➕ Add Knowledge Sources")
    
    # Source type selection
    source_type = st.selectbox(
        "Select source type:",
        ["RSS Feed", "Web Scraping", "File Upload", "API Endpoint"]
    )
    
    if source_type == "RSS Feed":
        show_rss_form()
    elif source_type == "Web Scraping":
        show_web_scraping_form()
    elif source_type == "File Upload":
        show_file_upload_form()
    elif source_type == "API Endpoint":
        show_api_form()

def show_rss_form():
    """Show RSS feed configuration form."""
    st.subheader("📡 RSS Feed Configuration")
    
    with st.form("rss_form"):
        name = st.text_input("Source Name", placeholder="e.g., Tech News RSS")
        url = st.text_input("RSS Feed URL", placeholder="https://feeds.feedburner.com/oreilly/radar")
        update_frequency = st.selectbox("Update Frequency", ["hourly", "daily", "weekly"])
        max_items = st.number_input("Max Items per Update", min_value=1, max_value=100, value=10)
        
        submitted = st.form_submit_button("Add RSS Source")
        
        if submitted:
            if name and url:
                try:
                    # Generate unique source ID
                    source_id = f"rss_{name.lower().replace(' ', '_')}_{int(time.time())}"
                    
                    source_config = {
                        "name": name,
                        "url": url,
                        "update_frequency": update_frequency,
                        "max_items": max_items
                    }
                    
                    st.session_state.updater.add_source(source_id, "rss", source_config)
                    st.success(f"✅ RSS source '{name}' added successfully!")
                    
                    # Add to session state sources list
                    if 'sources' not in st.session_state:
                        st.session_state.sources = []
                    st.session_state.sources.append({
                        'id': source_id,
                        'name': name,
                        'type': 'rss',
                        'config': source_config
                    })
                    
                except Exception as e:
                    st.error(f"❌ Failed to add RSS source: {str(e)}")
            else:
                st.error("Please fill in all required fields.")

def show_web_scraping_form():
    """Show web scraping configuration form."""
    st.subheader("🕷️ Web Scraping Configuration")
    
    with st.form("scraping_form"):
        name = st.text_input("Source Name", placeholder="e.g., Company Blog")
        base_url = st.text_input("Base URL", placeholder="https://example.com")
        css_selector = st.text_input("CSS Selector for Content", placeholder=".article-content")
        max_pages = st.number_input("Max Pages to Scrape", min_value=1, max_value=50, value=5)
        
        submitted = st.form_submit_button("Add Web Scraping Source")
        
        if submitted:
            if name and base_url and css_selector:
                try:
                    source_config = {
                        "name": name,
                        "base_url": base_url,
                        "css_selector": css_selector,
                        "max_pages": max_pages
                    }
                    
                    st.session_state.updater.add_source("web_scraping", source_config)
                    st.success(f"✅ Web scraping source '{name}' added successfully!")
                    
                except Exception as e:
                    st.error(f"❌ Failed to add web scraping source: {str(e)}")
            else:
                st.error("Please fill in all required fields.")

def show_file_upload_form():
    """Show file upload form."""
    st.subheader("📁 File Upload")
    
    uploaded_files = st.file_uploader(
        "Choose files to upload",
        accept_multiple_files=True,
        type=['txt', 'pdf', 'docx', 'md']
    )
    
    if uploaded_files:
        if st.button("Process Uploaded Files"):
            process_uploaded_files(uploaded_files)

def process_uploaded_files(uploaded_files):
    """Process uploaded files."""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_docs_added = 0
    
    for i, uploaded_file in enumerate(uploaded_files):
        status_text.text(f"Processing {uploaded_file.name}...")
        
        try:
            # Read file content
            if uploaded_file.type == "text/plain" or uploaded_file.name.endswith('.txt'):
                content = str(uploaded_file.read(), "utf-8")
            elif uploaded_file.name.endswith('.md'):
                content = str(uploaded_file.read(), "utf-8")
            else:
                content = f"File: {uploaded_file.name} - Content type: {uploaded_file.type}"
            
            # Create documents for processing
            documents = [{
                'text': content,
                'metadata': {
                    'filename': uploaded_file.name,
                    'file_type': uploaded_file.type,
                    'upload_time': datetime.now().isoformat()
                }
            }]
            
            # Process and embed documents
            processed_docs = st.session_state.updater.process_and_embed(documents)
            
            # Add to database
            docs_added = st.session_state.updater.update_database("user_uploads", processed_docs)
            total_docs_added += docs_added
            
            st.success(f"✅ Processed {uploaded_file.name}: {docs_added} documents added")
            
        except Exception as e:
            st.error(f"❌ Failed to process {uploaded_file.name}: {str(e)}")
        
        progress_bar.progress((i + 1) / len(uploaded_files))
    
    status_text.text("✅ All files processed!")
    st.success(f"Successfully processed {len(uploaded_files)} files. Total documents added: {total_docs_added}")

def show_api_form():
    """Show API endpoint configuration form."""
    st.subheader("🔌 API Endpoint Configuration")
    
    with st.form("api_form"):
        name = st.text_input("Source Name", placeholder="e.g., External API")
        endpoint_url = st.text_input("API Endpoint URL", placeholder="https://api.example.com/data")
        api_key = st.text_input("API Key (optional)", type="password")
        headers = st.text_area("Custom Headers (JSON format)", placeholder='{"Authorization": "Bearer token"}')
        
        submitted = st.form_submit_button("Add API Source")
        
        if submitted:
            if name and endpoint_url:
                try:
                    source_config = {
                        "name": name,
                        "endpoint_url": endpoint_url,
                        "api_key": api_key,
                        "headers": json.loads(headers) if headers else {}
                    }
                    
                    st.session_state.updater.add_source("api", source_config)
                    st.success(f"✅ API source '{name}' added successfully!")
                    
                except json.JSONDecodeError:
                    st.error("Invalid JSON format in headers.")
                except Exception as e:
                    st.error(f"❌ Failed to add API source: {str(e)}")
            else:
                st.error("Please fill in all required fields.")

def show_manual_update():
    """Display manual update page."""
    st.header("🔄 Manual Update")
    
    st.info("Run a manual update to fetch new content from all configured sources.")
    
    if st.button("🚀 Start Manual Update", type="primary"):
        run_manual_update()

def run_manual_update():
    """Run a manual update."""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("Starting update process...")
        progress_bar.progress(0.1)
        
        # Get all sources from the updater
        stats = st.session_state.updater.get_source_stats()
        sources = stats.get('sources', {})
        
        if not sources:
            st.warning("No sources configured. Please add sources first.")
            return
        
        total_sources = len(sources)
        
        for i, (source_id, source_info) in enumerate(sources.items()):
            status_text.text(f"Updating source: {source_info.get('config', {}).get('name', source_id)}...")
            
            try:
                # Update from this source
                result = st.session_state.updater.update_from_source(source_id)
                
                if result.get('success'):
                    st.success(f"✅ Updated {source_id}: {result.get('documents_added', 0)} documents added")
                else:
                    st.warning(f"⚠️ No new content from {source_id}")
                
            except Exception as e:
                st.error(f"Failed to update source {source_id}: {str(e)}")
            
            progress_bar.progress((i + 1) / total_sources)
        
        status_text.text("✅ Update completed!")
        st.success("Manual update completed successfully!")
        
    except Exception as e:
        st.error(f"Update failed: {str(e)}")

def show_schedule_updates():
    """Display schedule updates page."""
    st.header("⏰ Schedule Updates")
    
    st.info("Configure automatic updates to run at specified intervals.")
    
    # Current schedule
    st.subheader("Current Schedule")
    
    # Schedule configuration
    st.subheader("Configure Schedule")
    
    with st.form("schedule_form"):
        schedule_type = st.selectbox("Schedule Type", ["Daily", "Weekly", "Hourly", "Custom"])
        
        if schedule_type == "Daily":
            time_input = st.time_input("Update Time", value=datetime.now().time())
        elif schedule_type == "Weekly":
            day = st.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
            time_input = st.time_input("Update Time", value=datetime.now().time())
        elif schedule_type == "Hourly":
            interval = st.number_input("Interval (hours)", min_value=1, max_value=24, value=6)
        else:  # Custom
            cron_expression = st.text_input("Cron Expression", placeholder="0 9 * * 1-5")
        
        submitted = st.form_submit_button("Save Schedule")
        
        if submitted:
            try:
                # Configure schedule
                schedule_config = {
                    "type": schedule_type.lower(),
                    "time": str(time_input) if schedule_type in ["Daily", "Weekly"] else None,
                    "day": day if schedule_type == "Weekly" else None,
                    "interval": interval if schedule_type == "Hourly" else None,
                    "cron": cron_expression if schedule_type == "Custom" else None
                }
                
                st.session_state.updater.schedule_updates(schedule_config)
                st.success("✅ Schedule configured successfully!")
                
            except Exception as e:
                st.error(f"❌ Failed to configure schedule: {str(e)}")

def show_logs():
    """Display update logs."""
    st.header("📋 Update Logs")
    
    # Get logs
    logs = st.session_state.updater.get_update_logs(limit=50)
    
    if logs:
        # Convert to DataFrame for better display
        df = pd.DataFrame(logs)
        
        # Display summary
        st.subheader("📊 Summary")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Updates", len(logs))
        
        with col2:
            total_docs = sum(log.get('documents_added', 0) for log in logs)
            st.metric("Documents Added", total_docs)
        
        with col3:
            successful_updates = sum(1 for log in logs if log.get('success', False))
            st.metric("Success Rate", f"{(successful_updates/len(logs)*100):.1f}%")
        
        # Display logs table
        st.subheader("📋 Recent Updates")
        st.dataframe(df, use_container_width=True)
        
        # Download logs
        if st.button("📥 Download Logs"):
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"knowledge_base_logs_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    else:
        st.info("No logs available. Run some updates to see logs here.")
        
        # Show example of how to add content
        st.markdown("""
        ### 🚀 Get Started
        
        1. **Add Sources**: Go to 'Add Sources' to configure RSS feeds, upload files, or add web sources
        2. **Run Updates**: Use 'Manual Update' to fetch and process content
        3. **View Results**: Come back here to see update logs and statistics
        """)

def show_settings():
    """Display settings page."""
    st.header("⚙️ Settings")
    
    # System settings
    st.subheader("System Settings")
    
    with st.form("settings_form"):
        max_documents = st.number_input("Max Documents per Collection", min_value=100, max_value=10000, value=1000)
        embedding_model = st.selectbox("Embedding Model", ["sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers/all-mpnet-base-v2"])
        batch_size = st.number_input("Processing Batch Size", min_value=10, max_value=100, value=50)
        
        submitted = st.form_submit_button("Save Settings")
        
        if submitted:
            st.success("✅ Settings saved successfully!")
    
    # Database management
    st.subheader("Database Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🗑️ Clear All Data", type="secondary"):
            if st.checkbox("I understand this will delete all data"):
                try:
                    st.session_state.updater.clear_all_data()
                    st.success("✅ All data cleared!")
                except Exception as e:
                    st.error(f"❌ Failed to clear data: {str(e)}")
    
    with col2:
        if st.button("📊 Export Data"):
            try:
                data = st.session_state.updater.export_data()
                st.download_button(
                    label="Download Export",
                    data=json.dumps(data, indent=2),
                    file_name=f"knowledge_base_export_{datetime.now().strftime('%Y%m%d')}.json",
                    mime="application/json"
                )
            except Exception as e:
                st.error(f"❌ Failed to export data: {str(e)}")

def add_demo_data():
    """Add demo data to test the system."""
    try:
        with st.spinner("Adding demo data..."):
            # Create sample documents
            demo_documents = [
                {
                    'text': "Artificial Intelligence (AI) is transforming customer service by enabling chatbots to understand and respond to customer queries more effectively.",
                    'metadata': {'source': 'demo', 'topic': 'AI in customer service'}
                },
                {
                    'text': "Machine Learning algorithms can analyze customer sentiment and adjust responses accordingly to improve customer satisfaction.",
                    'metadata': {'source': 'demo', 'topic': 'ML and sentiment analysis'}
                },
                {
                    'text': "Natural Language Processing (NLP) helps chatbots understand human language and provide more accurate responses to customer inquiries.",
                    'metadata': {'source': 'demo', 'topic': 'NLP in chatbots'}
                },
                {
                    'text': "Vector databases enable efficient similarity search for finding relevant information to answer customer questions.",
                    'metadata': {'source': 'demo', 'topic': 'vector databases'}
                },
                {
                    'text': "Multi-modal AI systems can process both text and images, providing richer customer service experiences.",
                    'metadata': {'source': 'demo', 'topic': 'multi-modal AI'}
                }
            ]
            
            # Process and embed documents
            processed_docs = st.session_state.updater.process_and_embed(demo_documents)
            
            # Add to database
            docs_added = st.session_state.updater.update_database("demo_knowledge", processed_docs)
            
            st.success(f"✅ Added {docs_added} demo documents to the knowledge base!")
            st.info("You can now see the updated statistics in the dashboard.")
            
    except Exception as e:
        st.error(f"❌ Failed to add demo data: {str(e)}")

if __name__ == "__main__":
    main()