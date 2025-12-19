"""
Streamlit Interface for Multi-Modal Chatbot (Task 2)
Provides a user-friendly web interface for text and image interactions.
"""

import streamlit as st
import sys
import os
from PIL import Image
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Page Configuration
st.set_page_config(
    page_title="Multi-Modal Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .message-user {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #2196f3;
    }
    .message-assistant {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #4caf50;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'conversation_id' not in st.session_state:
    st.session_state.conversation_id = f"conv_{datetime.now().strftime('%Y%m%d%H%M%S')}"

def display_message(role: str, content: str, image=None):
    """Display a message in the chat interface."""
    css_class = "message-user" if role == "user" else "message-assistant"
    icon = "👤" if role == "user" else "🤖"
    
    st.markdown(f'<div class="{css_class}">', unsafe_allow_html=True)
    st.markdown(f"**{icon} {role.capitalize()}**")
    
    if image:
        st.image(image, caption="Uploaded Image", use_column_width=True)
    
    st.markdown(content)
    st.markdown('</div>', unsafe_allow_html=True)

def process_input(text_input: str, uploaded_image=None):
    """Process user input and generate response."""
    if not text_input.strip() and uploaded_image is None:
        st.warning("⚠️ Please enter a message or upload an image.")
        return
    
    # Add user message
    st.session_state.messages.append({
        'role': 'user',
        'content': text_input if text_input.strip() else "[Image uploaded]",
        'image': uploaded_image
    })
    
    # Generate response (mock for now)
    try:
        # Import and use the chatbot
        from task2_multimodal.multimodal_chatbot import MultiModalChatbot
        
        if 'chatbot' not in st.session_state:
            st.session_state.chatbot = MultiModalChatbot()
        
        if uploaded_image:
            result = st.session_state.chatbot.handle_mixed_input(
                text=text_input if text_input.strip() else "Describe this image in detail.",
                image=uploaded_image,
                conversation_id=st.session_state.conversation_id
            )
        else:
            result = st.session_state.chatbot.process_text_query(
                query=text_input,
                conversation_id=st.session_state.conversation_id
            )
        
        if result['success']:
            response = result['data']['response']
        else:
            response = f"Error: {result.get('user_message', 'Unknown error')}"
            
    except Exception as e:
        response = f"I'm a demo multi-modal chatbot. You said: '{text_input}'. " + \
                  ("I can see you uploaded an image! " if uploaded_image else "") + \
                  "In a full implementation, I would process this using Google Gemini AI."
    
    # Add assistant response
    st.session_state.messages.append({
        'role': 'assistant',
        'content': response,
        'image': None
    })

def main():
    """Main application function."""
    
    # Header
    st.markdown('<div class="main-header">🤖 Multi-Modal Chatbot</div>', unsafe_allow_html=True)
    st.markdown("**Task 2**: Multi-Modal AI Chatbot with Text and Image Support")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        st.info(f"**Conversation ID:** {st.session_state.conversation_id[:15]}...")
        st.info(f"**Messages:** {len(st.session_state.messages)}")
        
        if st.button("🗑️ Clear Conversation"):
            st.session_state.messages = []
            st.session_state.conversation_id = f"conv_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            st.success("✅ Conversation cleared!")
            st.rerun()
        
        st.markdown("---")
        
        st.markdown("### 📖 How to Use")
        st.markdown("""
        1. **Text Only**: Type your message and press Send
        2. **Image Only**: Upload an image for analysis
        3. **Text + Image**: Upload an image and ask about it
        4. **Follow-up**: Ask follow-up questions with context!
        """)
        
        st.markdown("---")
        
        # API Key status
        api_key = os.getenv('GOOGLE_API_KEY')
        if api_key:
            st.success("✅ API Key configured")
        else:
            st.warning("⚠️ No API Key found")
            st.info("Set GOOGLE_API_KEY in .env file for full functionality")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 💬 Conversation")
        
        # Display messages
        if len(st.session_state.messages) == 0:
            st.info("👋 Start a conversation by typing a message or uploading an image!")
        else:
            for msg in st.session_state.messages:
                display_message(
                    role=msg['role'],
                    content=msg['content'],
                    image=msg.get('image')
                )
    
    with col2:
        st.markdown("### 🖼️ Image Upload")
        
        uploaded_file = st.file_uploader(
            "Upload an image",
            type=['jpg', 'jpeg', 'png'],
            help="Upload an image to analyze"
        )
        
        uploaded_image = None
        if uploaded_file is not None:
            uploaded_image = Image.open(uploaded_file)
            st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
            st.markdown(f"**Size:** {uploaded_image.size[0]} x {uploaded_image.size[1]}")
        else:
            st.info("No image uploaded")
    
    # Input area
    st.markdown("---")
    st.markdown("### ✍️ Your Message")
    
    with st.form(key="input_form", clear_on_submit=True):
        col_input, col_button = st.columns([4, 1])
        
        with col_input:
            user_input = st.text_input(
                "Type your message here...",
                placeholder="Ask me anything or describe what you want to know about the image...",
                label_visibility="collapsed"
            )
        
        with col_button:
            submit_button = st.form_submit_button("Send 📤", use_container_width=True)
        
        if submit_button:
            if user_input.strip() or uploaded_image:
                process_input(user_input, uploaded_image)
                st.rerun()
            else:
                st.warning("⚠️ Please enter a message or upload an image.")
    
    # Example prompts
    st.markdown("### 💡 Example Prompts")
    col_ex1, col_ex2, col_ex3 = st.columns(3)
    
    with col_ex1:
        if st.button("🤖 What is AI?", use_container_width=True):
            process_input("What is artificial intelligence?", None)
            st.rerun()
    
    with col_ex2:
        if st.button("🖼️ Analyze image", use_container_width=True):
            if uploaded_image:
                process_input("Describe this image in detail.", uploaded_image)
                st.rerun()
            else:
                st.warning("⚠️ Please upload an image first!")
    
    with col_ex3:
        if st.button("🔍 Explain more", use_container_width=True):
            process_input("Can you explain that in more detail?", None)
            st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown(
        '<div style="text-align: center; color: #666;">'
        '🤖 Multi-Modal Chatbot | Task 2 | GenAI Customer Service Bot'
        '</div>',
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()