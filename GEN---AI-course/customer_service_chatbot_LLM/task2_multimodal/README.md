# Task 2: Multi-Modal Chatbot

## Overview

The Multi-Modal Chatbot enables users to interact with the AI using both text and images. It leverages Google's Gemini AI for vision capabilities and maintains conversation context across multi-turn interactions.

## Features Implemented

### Core Functionality
- ✅ **Text Query Processing**: Handle text-only conversations with context awareness
- ✅ **Image Query Processing**: Analyze and understand images using Gemini AI vision
- ✅ **Mixed Input Handling**: Process inputs containing both text and images
- ✅ **Conversation Context Management**: Maintain conversation history across exchanges
- ✅ **Multi-turn Conversations**: Support follow-up questions with context
- ✅ **Sentiment Analysis Integration**: Emotion-aware responses based on user sentiment
- ✅ **Multi-lingual Support**: Automatic language detection and culturally appropriate responses

### Key Components

#### MultiModalChatbot Class
The main class that orchestrates all multi-modal interactions.

**Methods:**
- `process_text_query()`: Process text-only queries with conversation context
- `process_image_query()`: Analyze images and generate descriptions
- `handle_mixed_input()`: Unified interface for text and/or image inputs
- `get_conversation_history()`: Retrieve conversation history
- `clear_conversation()`: Clear a conversation's history
- `list_conversations()`: List all active conversations

#### Data Models
- `Message`: Represents a single message (text and/or image)
- `Conversation`: Represents a conversation with message history

## Requirements Addressed

- **Requirement 2.1**: Process and understand image content using Gemini AI ✅
- **Requirement 2.3**: Generate responses for text queries requesting visual content ✅
- **Requirement 2.4**: Maintain conversation context across text and image exchanges ✅

## Usage Example

```python
from task2_multimodal.multimodal_chatbot import MultiModalChatbot
from PIL import Image

# Initialize chatbot
chatbot = MultiModalChatbot()

# Text-only conversation
result = chatbot.process_text_query(
    query="What is machine learning?",
    conversation_id="conv_1"
)
print(result['data']['response'])

# Follow-up question (uses context)
result = chatbot.process_text_query(
    query="Can you give me an example?",
    conversation_id="conv_1"
)
print(result['data']['response'])

# Image analysis
image = Image.open("example.jpg")
result = chatbot.process_image_query(
    image=image,
    query="What's in this image?",
    conversation_id="conv_1"
)
print(result['data']['response'])

# Mixed input (text + image)
result = chatbot.handle_mixed_input(
    text="Is this a cat or a dog?",
    image=image,
    conversation_id="conv_1"
)
print(result['data']['response'])
```

## Architecture

```
MultiModalChatbot
├── LLM Integration (Gemini AI)
│   ├── Text generation
│   ├── Image analysis
│   └── Multi-modal processing
├── Vector Database (ChromaDB)
│   └── Conversation history storage
└── Conversation Management
    ├── Message tracking
    ├── Context building
    └── History management
```

## Testing

### Basic Tests (No API Key Required)
```bash
python task2_multimodal/test_multimodal_basic.py
```

Tests covered:
- ✅ Chatbot initialization
- ✅ Conversation creation
- ✅ Message addition
- ✅ Context building
- ✅ Conversation history retrieval
- ✅ List conversations
- ✅ Clear conversation
- ✅ Max context messages limit

### Integration Tests (Requires API Key)
Set your Google API key in `.env`:
```
GOOGLE_API_KEY=your_api_key_here
```

Then run:
```bash
python task2_multimodal/multimodal_chatbot.py
```

## Dependencies

- `google-generativeai`: For Gemini AI integration
- `chromadb`: For vector database storage
- `Pillow`: For image processing
- `python-dotenv`: For environment variable management
- `transformers`: For sentiment analysis
- `langdetect`: For language detection
- `torch`: For ML model inference

## File Structure

```
task2_multimodal/
├── multimodal_chatbot.py      # Main implementation
├── app_multimodal.py          # Streamlit web interface
├── demo_multimodal.ipynb      # Jupyter demonstration notebook
├── test_multimodal_basic.py   # Basic unit tests
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

## Streamlit Interface

### Running the App

```bash
streamlit run task2_multimodal/app_multimodal.py
```

### Features

- **Text Chat**: Natural conversation with context awareness
- **Image Upload**: Upload and analyze images (JPG, PNG, JPEG)
- **Mixed Input**: Ask questions about uploaded images
- **Conversation History**: View all messages in the current conversation
- **Context Preservation**: Bot remembers previous messages
- **Clear Conversation**: Start fresh anytime
- **Example Prompts**: Quick-start buttons for common queries

### Interface Components

1. **Main Chat Area**: Displays conversation history with user and assistant messages
2. **Image Upload Panel**: Upload and preview images
3. **Input Form**: Type messages and send
4. **Sidebar**: Settings, conversation info, and instructions
5. **Example Prompts**: Quick access to common queries

## Demonstration

### Jupyter Notebook Demo

Run the comprehensive demonstration notebook:
```bash
jupyter notebook task2_multimodal/demo_multimodal.ipynb
```

The notebook demonstrates:
- **Text-only conversations** with context awareness
- **Image upload and analysis** using Gemini AI vision
- **Mixed text+image interactions** in the same conversation
- **Conversation context preservation** across multi-modal exchanges
- **Context window management** and limits
- **Error handling** for various edge cases

### Interactive Demos Included

1. **Demo 1**: Text-only conversations showing context awareness
2. **Demo 2**: Image upload and analysis with sample images
3. **Demo 3**: Mixed text+image interactions in one conversation
4. **Demo 4**: Conversation history and management features
5. **Demo 5**: Context window limit testing
6. **Demo 6**: Error handling validation

## Next Steps

1. **Task 12.1**: Write property test for image processing (optional) ✅
2. **Task 12.2**: Write property test for context preservation (optional) ✅
3. **Task 13**: Create Streamlit interface for Task 2 ✅
4. **Task 14**: Create demonstration notebook and documentation ✅

## Notes

- The chatbot maintains a maximum of 10 messages in context by default (configurable)
- Conversation history is stored in memory (not persisted to disk)
- Image analysis requires a valid Google API key with Gemini AI access
- All responses follow a standardized format with success/error indicators

## Error Handling

The implementation includes comprehensive error handling:
- Invalid image inputs
- API failures (with retry logic in LLM integration)
- Missing conversations
- Context overflow protection

All errors return user-friendly messages with suggested actions.
