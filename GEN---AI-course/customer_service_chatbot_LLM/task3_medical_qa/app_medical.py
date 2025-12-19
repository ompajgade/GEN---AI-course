"""
Medical Q&A Streamlit Application - Task 3
Interactive web interface for medical question answering using MedQuAD dataset.
"""

import sys
from pathlib import Path
import streamlit as st

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from task3_medical_qa.medical_qa import MedicalQASystem
from shared.vector_db_manager import VectorDatabaseManager
from shared.embedding_service import EmbeddingService
try:
    from task3_medical_qa.entity_recognizer import MedicalEntityRecognizer
except ImportError:
    # Fallback entity recognizer
    class MedicalEntityRecognizer:
        def extract_medical_entities(self, text):
            return {}


# Page configuration
st.set_page_config(
    page_title="Medical Q&A System",
    page_icon="🏥",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .entity-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        margin: 0.25rem;
        border-radius: 1rem;
        font-size: 0.875rem;
        font-weight: 500;
    }
    .symptom { background-color: #ffebee; color: #c62828; }
    .disease { background-color: #e3f2fd; color: #1565c0; }
    .treatment { background-color: #e8f5e9; color: #2e7d32; }
    .medication { background-color: #fff3e0; color: #e65100; }
    .body_part { background-color: #f3e5f5; color: #6a1b9a; }
    .procedure { background-color: #e0f2f1; color: #00695c; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_system():
    """Initialize the Medical Q&A System (cached)"""
    try:
        with st.spinner("Initializing Medical Q&A components..."):
            # Initialize components with error handling
            try:
                vector_db = VectorDatabaseManager()
            except Exception as e:
                st.warning(f"Vector database initialization failed: {e}")
                vector_db = None
            
            try:
                embedding_service = EmbeddingService()
            except Exception as e:
                st.warning(f"Embedding service initialization failed: {e}")
                embedding_service = None
            
            try:
                entity_recognizer = MedicalEntityRecognizer()
            except Exception as e:
                st.warning(f"Entity recognizer initialization failed: {e}")
                entity_recognizer = None
            
            # Initialize QA system with available components
            qa_system = MedicalQASystem(
                vector_db=vector_db,
                embedding_service=embedding_service,
                enable_sentiment=False,
                enable_multilingual=False
            )
            
            st.success("✅ Medical Q&A System initialized successfully!")
            return qa_system, entity_recognizer
    except Exception as e:
        st.error(f"Failed to initialize system: {str(e)}")
        # Try to return a minimal system
        try:
            qa_system = MedicalQASystem(
                vector_db=None,
                embedding_service=None,
                enable_sentiment=False,
                enable_multilingual=False
            )
            return qa_system, None
        except:
            return None, None


def display_entities(entities):
    """Display extracted medical entities with color-coded badges"""
    if not entities:
        return
        
    st.markdown("### 🔍 Detected Medical Entities")
    
    has_entities = False
    try:
        for entity_type, entity_list in entities.items():
            if entity_list:
                has_entities = True
                st.markdown(f"**{entity_type.replace('_', ' ').title()}:**")
                
                html = ""
                for entity in entity_list:
                    try:
                        # Handle both dict and object formats
                        if hasattr(entity, 'text'):
                            text = entity.text
                            confidence = entity.confidence
                        elif isinstance(entity, dict):
                            text = entity.get("text", str(entity))
                            confidence = entity.get("confidence", 0.7)
                        else:
                            text = str(entity)
                            confidence = 0.7
                        
                        html += f'<span class="entity-badge {entity_type}">{text} ({confidence:.0%})</span>'
                    except Exception as e:
                        # Skip problematic entities
                        continue
                
                if html:  # Only display if we have valid HTML
                    st.markdown(html, unsafe_allow_html=True)
        
        if not has_entities:
            st.info("No medical entities detected in the query.")
            
    except Exception as e:
        st.warning(f"Could not display entities: {e}")
        st.info("Entity recognition encountered an issue, but search will continue.")


def display_answer(answer, index):
    """Display a single answer with metadata"""
    with st.expander(f"📄 Answer {index + 1} - {answer.get('source', 'Medical Source')}", expanded=(index == 0)):
        if 'confidence' in answer:
            st.markdown(f"**Confidence:** {answer['confidence']:.0%}")
        if 'similarity_score' in answer:
            st.markdown(f"**Similarity Score:** {answer['similarity_score']:.3f}")
        
        st.markdown("**Answer:**")
        st.write(answer.get('answer', answer.get('text', 'No answer available')))
        
        if answer.get('url'):
            st.markdown(f"[🔗 Source Link]({answer['url']})")

@st.cache_data
def get_demo_medical_data():
    """Get demo medical data (cached)."""
    return [
        {
            "question": "What are the symptoms of diabetes?",
            "answer": "Common symptoms of diabetes include frequent urination, excessive thirst, unexplained weight loss, extreme fatigue, blurred vision, slow-healing cuts and bruises, and frequent infections.",
            "source": "Medical Demo Data",
            "category": "diabetes"
        },
        {
            "question": "How is hypertension treated?",
            "answer": "Hypertension is typically treated through lifestyle changes (diet, exercise, weight management) and medications such as ACE inhibitors, diuretics, beta-blockers, or calcium channel blockers.",
            "source": "Medical Demo Data", 
            "category": "hypertension"
        },
        {
            "question": "What causes pneumonia?",
            "answer": "Pneumonia can be caused by bacteria (most common), viruses, fungi, or other microorganisms. Bacterial pneumonia is often caused by Streptococcus pneumoniae.",
            "source": "Medical Demo Data",
            "category": "pneumonia"
        },
        {
            "question": "What are the risk factors for heart disease?",
            "answer": "Risk factors for heart disease include high blood pressure, high cholesterol, smoking, diabetes, obesity, physical inactivity, family history, age, and stress.",
            "source": "Medical Demo Data",
            "category": "heart_disease"
        },
        {
            "question": "What medications are used for asthma?",
            "answer": "Asthma medications include quick-relief inhalers (bronchodilators) and long-term control medications (corticosteroids, leukotriene modifiers).",
            "source": "Medical Demo Data",
            "category": "asthma"
        },
        {
            "question": "What are the side effects of antibiotics?",
            "answer": "Common side effects of antibiotics include nausea, diarrhea, stomach upset, allergic reactions, and disruption of normal gut bacteria. Some antibiotics may cause more specific side effects.",
            "source": "Medical Demo Data",
            "category": "medications"
        },
        {
            "question": "How is depression diagnosed?",
            "answer": "Depression is diagnosed through clinical evaluation including assessment of symptoms, duration, severity, and impact on daily functioning. Healthcare providers use standardized criteria and may conduct psychological assessments.",
            "source": "Medical Demo Data",
            "category": "mental_health"
        },
        {
            "question": "What is the difference between Type 1 and Type 2 diabetes?",
            "answer": "Type 1 diabetes is an autoimmune condition where the body doesn't produce insulin, typically diagnosed in childhood. Type 2 diabetes occurs when the body becomes resistant to insulin or doesn't produce enough, usually developing in adulthood.",
            "source": "Medical Demo Data",
            "category": "diabetes"
        }
    ]

def load_demo_medical_data(qa_system):
    """Load demo medical data for testing."""
    try:
        demo_qa_pairs = get_demo_medical_data()
        
        # Check if demo data is already loaded
        try:
            stats = qa_system.vector_db.get_collection_stats("medical_demo")
            if stats.get('document_count', 0) > 0:
                qa_system.collection_name = "medical_demo"
                qa_system.is_loaded = True
                st.success(f"✅ Demo data already loaded! ({stats['document_count']} documents)")
                return True
        except:
            pass  # Collection doesn't exist yet
        
        # Simple fallback - store data directly in session state
        if not hasattr(qa_system, 'vector_db') or qa_system.vector_db is None:
            st.warning("Vector database not available. Using simple in-memory storage.")
            st.session_state.demo_qa_data = demo_qa_pairs
            qa_system.is_loaded = True
            qa_system.collection_name = "simple_demo"
            st.success("✅ Demo data loaded in memory!")
            return True
        
        # Process data for vector database
        documents = []
        for qa in demo_qa_pairs:
            documents.append({
                'text': f"Question: {qa['question']} Answer: {qa['answer']}",
                'metadata': {
                    'question': qa['question'],
                    'answer': qa['answer'],
                    'source': qa['source'],
                    'category': qa['category']
                }
            })
        
        # Try to generate embeddings
        with st.spinner("Generating embeddings for demo data..."):
            try:
                # Try batch processing first
                processed_docs = qa_system.embedding_service.generate_embeddings_batch(
                    [doc['text'] for doc in documents]
                )
            except Exception as e:
                st.warning(f"Embedding generation failed: {e}. Using simple text matching...")
                # Fallback to simple storage
                st.session_state.demo_qa_data = demo_qa_pairs
                qa_system.is_loaded = True
                qa_system.collection_name = "simple_demo"
                st.success("✅ Demo data loaded with simple text matching!")
                return True
        
        # Try to create vector database collection
        try:
            qa_system.collection_name = "medical_demo"
            qa_system.vector_db.create_collection("medical_demo")
            qa_system.vector_db.add_documents(
                collection_name="medical_demo",
                documents=[doc['text'] for doc in documents],
                embeddings=processed_docs,
                metadata=[doc['metadata'] for doc in documents]
            )
            qa_system.is_loaded = True
            st.success("✅ Demo data loaded with vector search!")
            return True
        except Exception as e:
            st.warning(f"Vector database failed: {e}. Using simple text matching...")
            # Final fallback
            st.session_state.demo_qa_data = demo_qa_pairs
            qa_system.is_loaded = True
            qa_system.collection_name = "simple_demo"
            st.success("✅ Demo data loaded with simple text matching!")
            return True
        
    except Exception as e:
        st.error(f"Failed to load demo data: {str(e)}")
        # Last resort fallback
        try:
            demo_qa_pairs = get_demo_medical_data()
            st.session_state.demo_qa_data = demo_qa_pairs
            qa_system.is_loaded = True
            qa_system.collection_name = "simple_demo"
            st.warning("✅ Demo data loaded with basic functionality!")
            return True
        except:
            return False       


def main():
    # Header
    st.markdown('<h1 class="main-header">🏥 Medical Q&A System</h1>', unsafe_allow_html=True)
    st.markdown("Ask medical questions and get answers from the MedQuAD dataset")
    
    # Initialize system
    qa_system, entity_recognizer = initialize_system()
    
    if qa_system is None:
        st.error("❌ Failed to initialize the Medical Q&A System")
        st.info("Please check your configuration and try again.")
        return
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Dataset loading
        st.subheader("📊 Dataset Status")
        
        # Check if dataset is loaded
        try:
            is_loaded = hasattr(qa_system, 'is_loaded') and qa_system.is_loaded
        except:
            is_loaded = False
        
        if not is_loaded:
            st.warning("⚠️ No dataset loaded")
            st.info("Choose one of the options below to get started:")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📥 Load MedQuAD Dataset", help="Load the full medical Q&A dataset (may take several minutes)", use_container_width=True):
                    with st.spinner("Loading MedQuAD dataset... This may take several minutes on first load."):
                        try:
                            # Check if we have the dataset files
                            from pathlib import Path
                            data_path = Path("data/medquad")
                            
                            if not data_path.exists():
                                st.info("📥 Dataset not found locally. Downloading from GitHub...")
                            
                            success = qa_system.load_medquad_dataset()
                            if success:
                                st.success("✅ MedQuAD dataset loaded successfully!")
                                st.balloons()
                                st.rerun()
                            else:
                                st.error("❌ Failed to load MedQuAD dataset")
                                st.info("💡 Try the demo data option for quick testing")
                        except Exception as e:
                            st.error(f"❌ Error loading MedQuAD dataset: {str(e)}")
                            st.info("💡 Try the demo data option for quick testing")
            
            with col2:
                if st.button("🧪 Load Demo Data", help="Load sample medical data for quick testing", use_container_width=True):
                    with st.spinner("Loading demo medical data..."):
                        try:
                            success = load_demo_medical_data(qa_system)
                            if success:
                                st.success("✅ Demo data loaded!")
                                st.balloons()
                                st.rerun()
                            else:
                                st.error("❌ Failed to load demo data")
                        except Exception as e:
                            st.error(f"❌ Error loading demo data: {str(e)}")
            
            # Show loading tips
            with st.expander("💡 Loading Tips"):
                st.markdown("""
                **MedQuAD Dataset:**
                - Full medical Q&A dataset from trusted sources
                - ~16,000+ question-answer pairs
                - First load downloads ~50MB and may take 5-10 minutes
                - Subsequent loads are much faster (cached)
                
                **Demo Data:**
                - Quick sample dataset for testing
                - 8 common medical Q&A pairs
                - Loads in seconds
                - Perfect for trying out the system
                """)
        else:
            # Dataset is loaded - show status and stats
            collection_name = getattr(qa_system, 'collection_name', 'Unknown')
            
            if collection_name == "medical_demo":
                st.success("✅ Demo dataset loaded")
                st.info("🧪 Using sample medical data for demonstration")
            else:
                st.success("✅ MedQuAD dataset loaded")
                st.info("📚 Using full medical Q&A dataset")
            
            # Show dataset stats
            try:
                stats = qa_system.get_dataset_stats()
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Q&A Pairs", f"{stats.get('total_qa_pairs', 0):,}")
                with col2:
                    st.metric("Sources", stats.get('num_sources', 0))
                with col3:
                    st.metric("Collection", collection_name.replace('_', ' ').title())
                
                # Show sources breakdown if available
                sources = stats.get('sources', {})
                if sources and len(sources) > 1:
                    with st.expander("📋 Sources Breakdown"):
                        for source, count in sources.items():
                            st.write(f"• **{source}**: {count:,} pairs")
                            
            except Exception as e:
                st.warning(f"Could not load dataset statistics: {str(e)}")
            
            # Option to reload or switch datasets
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Reload Current Dataset", help="Reload the current dataset"):
                    qa_system.is_loaded = False
                    st.rerun()
            with col2:
                if st.button("🔀 Switch Dataset", help="Switch to a different dataset"):
                    qa_system.is_loaded = False
                    if hasattr(qa_system, 'collection_name'):
                        delattr(qa_system, 'collection_name')
                    st.rerun()
        
        # Query settings
        st.subheader("Query Settings")
        num_results = st.slider("Number of results", 1, 10, 5)
        
        # About
        st.subheader("About")
        st.info("""
        This Medical Q&A system uses:
        - **MedQuAD Dataset**: Medical Q&A from trusted sources
        - **Entity Recognition**: Identifies symptoms, diseases, treatments
        - **Vector Search**: Finds relevant answers using embeddings
        """)
    
    # Main content
    if not hasattr(qa_system, 'is_loaded') or not qa_system.is_loaded:
        # Welcome section with better guidance
        st.markdown("## 🏥 Welcome to Medical Q&A System")
        st.markdown("Get reliable medical information from trusted sources using AI-powered search.")
        
        st.info("👈 **Get Started:** Load a dataset from the sidebar to begin asking medical questions.")
        
        # Show example queries in a more attractive format
        st.markdown("### 💡 Example Medical Questions")
        st.markdown("Once you load a dataset, you can ask questions like these:")
        
        examples = [
            ("🩺 Symptoms", "What are the symptoms of diabetes?"),
            ("💊 Treatment", "How is hypertension treated?"),
            ("🦠 Causes", "What causes pneumonia?"),
            ("💉 Medications", "What medications are used for asthma?"),
            ("⚠️ Risk Factors", "What are the risk factors for heart disease?"),
            ("🧬 Conditions", "What is the difference between Type 1 and Type 2 diabetes?"),
            ("🩹 Side Effects", "What are the side effects of antibiotics?"),
            ("🧠 Mental Health", "How is depression diagnosed?")
        ]
        
        col1, col2 = st.columns(2)
        for i, (category, example) in enumerate(examples):
            with col1 if i % 2 == 0 else col2:
                st.markdown(f"**{category}**")
                st.markdown(f"*{example}*")
                st.markdown("")
        
        # Quick start guide
        with st.expander("🚀 Quick Start Guide"):
            st.markdown("""
            1. **Load Dataset**: Choose either MedQuAD (full) or Demo Data (quick) from the sidebar
            2. **Ask Questions**: Type your medical question in natural language
            3. **Get Answers**: Receive evidence-based responses with source attribution
            4. **Explore Entities**: See detected medical terms highlighted in your query
            
            **Tips for Better Results:**
            - Be specific about symptoms, conditions, or treatments
            - Use medical terminology when known
            - Ask one question at a time for focused answers
            """)
    
    else:
        # Query input
        query = st.text_input(
            "🔍 Enter your medical question:",
            placeholder="e.g., What are the symptoms of diabetes?",
            help="Ask any medical question and get answers from trusted medical sources"
        )
        
        col1, col2 = st.columns([1, 5])
        with col1:
            search_button = st.button("🔎 Search", type="primary", use_container_width=True)
        with col2:
            clear_button = st.button("🗑️ Clear", use_container_width=True)
        
        if clear_button:
            st.rerun()
        
        if search_button and query:
            with st.spinner("🔍 Searching for medical answers..."):
                try:
                    # Extract entities first
                    entities = {}
                    if entity_recognizer is not None:
                        with st.spinner("Analyzing medical terms..."):
                            try:
                                entities = entity_recognizer.extract_medical_entities(query)
                            except Exception as e:
                                st.warning(f"Entity recognition failed: {e}")
                                entities = {}  # Empty entities dict as fallback
                    else:
                        st.info("Entity recognition not available - continuing with search...")
                    
                    # Search for answers
                    with st.spinner("Finding relevant information..."):
                        answers = []
                        
                        # Check if we have simple demo data
                        if (hasattr(qa_system, 'collection_name') and 
                            qa_system.collection_name == "simple_demo" and 
                            'demo_qa_data' in st.session_state):
                            
                            # Simple text matching search
                            demo_data = st.session_state.demo_qa_data
                            query_lower = query.lower()
                            
                            for qa in demo_data:
                                # Simple keyword matching
                                question_lower = qa['question'].lower()
                                answer_lower = qa['answer'].lower()
                                
                                # Calculate simple similarity score
                                query_words = set(query_lower.split())
                                qa_words = set(question_lower.split() + answer_lower.split())
                                
                                # Find common words
                                common_words = query_words.intersection(qa_words)
                                if common_words:
                                    similarity = len(common_words) / len(query_words) if query_words else 0
                                    
                                    answers.append({
                                        'question': qa['question'],
                                        'answer': qa['answer'],
                                        'source': qa['source'],
                                        'confidence': similarity,
                                        'similarity_score': similarity
                                    })
                            
                            # Sort by similarity and take top results
                            answers.sort(key=lambda x: x['similarity_score'], reverse=True)
                            answers = answers[:num_results]
                            
                        elif hasattr(qa_system, 'process_medical_query'):
                            # Try advanced processing
                            try:
                                result = qa_system.process_medical_query(query, top_k=num_results)
                                
                                # Check for errors in the result
                                if 'error' in result:
                                    st.warning(f"⚠️ {result['error']}")
                                    # Fallback to simple search if available
                                    if 'demo_qa_data' in st.session_state:
                                        st.info("Using simple text matching instead...")
                                        demo_data = st.session_state.demo_qa_data
                                        for qa in demo_data[:3]:  # Show first 3
                                            answers.append({
                                                'question': qa['question'],
                                                'answer': qa['answer'],
                                                'source': qa['source'],
                                                'confidence': 0.7,
                                                'similarity_score': 0.7
                                            })
                                else:
                                    answers = result.get('answers', [])
                                    
                                    # Show language detection if multilingual
                                    if result.get('original_language') and result.get('original_language') != 'en':
                                        st.info(f"🌍 Detected language: {result['original_language']}")
                                        if result.get('translated_query'):
                                            st.info(f"🔄 Translated query: {result['translated_query']}")
                                    
                                    # Show sentiment if available
                                    if result.get('sentiment'):
                                        sentiment_info = result['sentiment']
                                        sentiment_emoji = {"positive": "😊", "negative": "😔", "neutral": "😐"}
                                        st.info(f"😊 Query sentiment: {sentiment_emoji.get(sentiment_info['label'], '🤔')} {sentiment_info['label'].title()}")
                            except Exception as e:
                                st.warning(f"Advanced search failed: {e}")
                                # Fallback to simple demo data
                                if 'demo_qa_data' in st.session_state:
                                    st.info("Using simple text matching...")
                                    demo_data = st.session_state.demo_qa_data
                                    for qa in demo_data[:3]:
                                        answers.append({
                                            'question': qa['question'],
                                            'answer': qa['answer'],
                                            'source': qa['source'],
                                            'confidence': 0.7,
                                            'similarity_score': 0.7
                                        })
                        else:
                            # Final fallback - show some demo answers
                            if 'demo_qa_data' in st.session_state:
                                demo_data = st.session_state.demo_qa_data
                                for qa in demo_data[:num_results]:
                                    answers.append({
                                        'question': qa['question'],
                                        'answer': qa['answer'],
                                        'source': qa['source'],
                                        'confidence': 0.6,
                                        'similarity_score': 0.6
                                    })
                    
                    # Display results
                    st.markdown("---")
                    
                    # Display entities
                    if entities and any(entities.values()):
                        display_entities(entities)
                        st.markdown("---")
                    
                    # Display answers
                    if answers:
                        st.markdown(f"### 📚 Found {len(answers)} Relevant Medical Answers")
                        
                        # Show confidence indicator
                        avg_confidence = sum(answer.get('confidence', 0) for answer in answers) / len(answers)
                        if avg_confidence > 0.8:
                            st.success(f"🎯 High confidence results (avg: {avg_confidence:.1%})")
                        elif avg_confidence > 0.6:
                            st.info(f"✅ Good confidence results (avg: {avg_confidence:.1%})")
                        else:
                            st.warning(f"⚠️ Moderate confidence results (avg: {avg_confidence:.1%})")
                        
                        for i, answer in enumerate(answers):
                            display_answer(answer, i)
                    else:
                        st.warning("🔍 No relevant answers found for your question.")
                        
                        # Provide helpful suggestions
                        st.markdown("### 💡 Try These Tips:")
                        st.markdown("""
                        - **Rephrase your question** using different medical terms
                        - **Be more specific** about symptoms, conditions, or treatments  
                        - **Check spelling** of medical terms
                        - **Try simpler language** if using complex terminology
                        """)
                        
                        # Show similar example questions
                        st.markdown("### 🔄 Example Questions:")
                        example_questions = [
                            "What are the symptoms of diabetes?",
                            "How is high blood pressure treated?", 
                            "What causes chest pain?",
                            "What medications help with asthma?"
                        ]
                        for example in example_questions:
                            if st.button(f"Try: {example}", key=f"example_{example}"):
                                st.session_state.example_query = example
                                st.rerun()
                        
                except Exception as e:
                    st.error(f"❌ Error processing your medical query")
                    
                    # Show detailed error in expander for debugging
                    with st.expander("🔧 Technical Details"):
                        st.code(f"Error: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
                    
                    st.info("💡 Please try a different question or reload the dataset from the sidebar.")
        
        elif search_button:
            st.warning("⚠️ Please enter a medical question to search for answers.")
        
        # Handle example query selection
        if 'example_query' in st.session_state:
            st.session_state.query_input = st.session_state.example_query
            del st.session_state.example_query
            st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "Medical Q&A System | Task 3 - GenAI Customer Service Bot"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
