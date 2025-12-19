"""
Multi-Modal Chatbot (Task 2) with Multi-lingual Support
Handles text and image inputs/outputs using Gemini AI capabilities.
Supports conversation context management across text and image exchanges.
Integrated with sentiment analysis and multi-lingual support.
"""

import sys
import os
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, field
import logging
from PIL import Image
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.llm_integration import LLMIntegration
from shared.vector_db_manager import VectorDatabaseManager
from shared.utils import create_error_response, create_success_response, get_timestamp

# Import sentiment analysis
try:
    sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'task5_sentiment'))
    from sentiment_analysis import SentimentAnalysisEngine
    SENTIMENT_AVAILABLE = True
except ImportError:
    SENTIMENT_AVAILABLE = False

# Import multilingual support
try:
    sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'task6_multilingual'))
    from multilingual_system import MultiLingualSystem
    MULTILINGUAL_AVAILABLE = True
except ImportError:
    MULTILINGUAL_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Message:
    """Represents a single message in a conversation."""
    message_id: str
    role: str  # 'user' or 'assistant'
    content: str
    image: Optional[Image.Image] = None
    sentiment: Optional[str] = None
    sentiment_score: Optional[float] = None
    language: str = "en"
    timestamp: str = field(default_factory=get_timestamp)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Conversation:
    """Represents a conversation with message history."""
    conversation_id: str
    user_id: str
    messages: List[Message] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    language: str = "en"
    sentiment_history: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=get_timestamp)
    updated_at: str = field(default_factory=get_timestamp)


class MultiModalChatbot:
    """
    Multi-modal chatbot with multi-lingual and sentiment support.
    
    Key Features:
    - Process text queries with context
    - Analyze and understand images
    - Handle mixed text + image inputs
    - Maintain conversation history
    - Support multiple languages
    - Sentiment-aware responses
    """
    
    def __init__(
        self,
        llm: Optional[LLMIntegration] = None,
        vector_db: Optional[VectorDatabaseManager] = None,
        max_context_messages: int = 10,
        enable_sentiment: bool = True,
        enable_multilingual: bool = True
    ):
        """Initialize the multi-modal chatbot with all integrations."""
        self.llm = llm or LLMIntegration(primary_model="gemini-pro")
        self.vector_db = vector_db or VectorDatabaseManager()
        self.max_context_messages = max_context_messages
        
        # Initialize sentiment analysis
        self.sentiment_engine = None
        if enable_sentiment and SENTIMENT_AVAILABLE:
            try:
                self.sentiment_engine = SentimentAnalysisEngine()
                logger.info("✅ Sentiment analysis enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize sentiment analysis: {e}")
        
        # Initialize multilingual support
        self.multilingual_system = None
        if enable_multilingual and MULTILINGUAL_AVAILABLE:
            try:
                self.multilingual_system = MultiLingualSystem()
                logger.info("✅ Multi-lingual support enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize multilingual system: {e}")
        
        # Store active conversations
        self.conversations: Dict[str, Conversation] = {}
        
        # Initialize conversation history collection
        try:
            self.vector_db.create_collection(
                name="conversation_history",
                metadata={"purpose": "Store conversation context for retrieval"}
            )
            logger.info("✅ Multi-modal chatbot initialized")
        except Exception as e:
            logger.warning(f"⚠️ Could not create conversation_history collection: {e}")

    def _detect_and_process_language(self, text: str) -> Dict[str, Any]:
        """Detect language and prepare for processing."""
        if not self.multilingual_system:
            return {"language": "en", "original_text": text, "translated_text": text}
        
        try:
            result = self.multilingual_system.process_multilingual_query(text)
            return {
                "language": result["detected_language"],
                "original_text": text,
                "translated_text": result.get("translation", text)
            }
        except Exception as e:
            logger.warning(f"Language processing failed: {e}")
            return {"language": "en", "original_text": text, "translated_text": text}

    def _generate_culturally_appropriate_response(self, response: str, target_language: str) -> str:
        """Generate culturally appropriate response in target language."""
        if not self.multilingual_system or target_language == "en":
            return response
        
        try:
            return self.multilingual_system.generate_culturally_appropriate_response(
                response, target_language
            )
        except Exception as e:
            logger.warning(f"Cultural adaptation failed: {e}")
            return response

    def _add_message_to_conversation(
        self,
        conversation: Conversation,
        role: str,
        content: str,
        image: Optional[Image.Image] = None,
        language: str = "en",
        metadata: Optional[Dict] = None
    ) -> Message:
        """Add a message to conversation with sentiment and language analysis."""
        # Analyze sentiment for user messages
        sentiment = None
        sentiment_score = None
        
        if role == "user" and self.sentiment_engine and content:
            try:
                sentiment_result = self.sentiment_engine.analyze_sentiment(content)
                sentiment = sentiment_result.label
                sentiment_score = sentiment_result.score
                conversation.sentiment_history.append(sentiment)
                
                if len(conversation.sentiment_history) > self.max_context_messages:
                    conversation.sentiment_history = conversation.sentiment_history[-self.max_context_messages:]
                    
            except Exception as e:
                logger.warning(f"Sentiment analysis failed: {e}")
        
        message = Message(
            message_id=f"{conversation.conversation_id}_msg_{datetime.now().strftime('%Y%m%d%H%M%S%f')}",
            role=role,
            content=content,
            image=image,
            sentiment=sentiment,
            sentiment_score=sentiment_score,
            language=language,
            metadata=metadata or {}
        )
        
        conversation.messages.append(message)
        conversation.updated_at = get_timestamp()
        
        # Update conversation language if user message
        if role == "user":
            conversation.language = language
        
        # Keep only recent messages
        if len(conversation.messages) > self.max_context_messages:
            conversation.messages = conversation.messages[-self.max_context_messages:]
        
        return message

    def process_text_query(
        self,
        query: str,
        conversation_id: Optional[str] = None,
        context: Optional[Dict] = None,
        use_context: bool = True
    ) -> Dict[str, Any]:
        """Process a text query with multi-lingual and sentiment support."""
        try:
            if conversation_id is None:
                conversation_id = f"conv_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            # Get or create conversation
            if conversation_id not in self.conversations:
                self.conversations[conversation_id] = Conversation(
                    conversation_id=conversation_id,
                    user_id="default_user"
                )
            
            conversation = self.conversations[conversation_id]
            
            # Detect and process language
            lang_info = self._detect_and_process_language(query)
            detected_language = lang_info["language"]
            translated_query = lang_info["translated_text"]
            
            # Add user message to history
            self._add_message_to_conversation(
                conversation=conversation,
                role="user",
                content=query,
                language=detected_language
            )
            
            # Build context from conversation history
            context_string = ""
            if use_context and conversation.messages:
                context_parts = ["Previous conversation:"]
                for msg in conversation.messages[-self.max_context_messages:]:
                    role_label = "User" if msg.role == "user" else "Assistant"
                    image_note = " [with image]" if msg.image else ""
                    context_parts.append(f"{role_label}{image_note}: {msg.content}")
                context_string = "\n".join(context_parts)
            
            if context:
                context_string += f"\n\nAdditional context: {context}"
            
            # Generate response using LLM (use translated query for English processing)
            response = self.llm.generate_text(
                prompt=translated_query,
                context=context_string,
                max_tokens=1024
            )
            
            # Apply sentiment-aware response adjustment
            user_sentiment = None
            if conversation.messages and conversation.messages[-1].sentiment:
                user_sentiment = conversation.messages[-1].sentiment
                
                if self.sentiment_engine:
                    try:
                        response = self.sentiment_engine.adjust_response_tone(response, user_sentiment)
                    except Exception as e:
                        logger.warning(f"Failed to adjust response tone: {e}")
            
            # Generate culturally appropriate response in target language
            final_response = self._generate_culturally_appropriate_response(response, detected_language)
            
            # Add assistant response to history
            self._add_message_to_conversation(
                conversation=conversation,
                role="assistant",
                content=final_response,
                language=detected_language
            )
            
            logger.info(f"✅ Processed text query in {detected_language} for conversation {conversation_id}")
            
            return create_success_response(
                data={
                    'response': final_response,
                    'conversation_id': conversation_id,
                    'detected_language': detected_language,
                    'original_query': query,
                    'translated_query': translated_query if detected_language != "en" else None,
                    'user_sentiment': user_sentiment,
                    'message_count': len(conversation.messages),
                    'has_context': use_context and len(conversation.messages) > 2
                },
                message="Text query processed successfully"
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to process text query: {e}")
            return create_error_response(
                error_code="TEXT_QUERY_ERROR",
                error_message=str(e),
                user_message="Failed to process your message. Please try again.",
                suggested_action="Check your input and try again"
            )

    def process_image_query(
        self,
        image: Image.Image,
        query: str = "Describe this image in detail.",
        conversation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process an image with multi-lingual support."""
        try:
            if image is None:
                raise ValueError("Image cannot be None")
            
            if conversation_id is None:
                conversation_id = f"conv_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            # Get or create conversation
            if conversation_id not in self.conversations:
                self.conversations[conversation_id] = Conversation(
                    conversation_id=conversation_id,
                    user_id="default_user"
                )
            
            conversation = self.conversations[conversation_id]
            
            # Detect and process language
            lang_info = self._detect_and_process_language(query)
            detected_language = lang_info["language"]
            translated_query = lang_info["translated_text"]
            
            # Add user message with image to history
            self._add_message_to_conversation(
                conversation=conversation,
                role="user",
                content=query,
                image=image,
                language=detected_language,
                metadata={
                    'image_size': image.size,
                    'image_mode': image.mode
                }
            )
            
            # Analyze image using Gemini AI (use translated query)
            analysis = self.llm.analyze_image(image, question=translated_query)
            
            if not analysis.get('success', False):
                raise Exception(analysis.get('error', 'Image analysis failed'))
            
            response = analysis['description']
            
            # Generate culturally appropriate response
            final_response = self._generate_culturally_appropriate_response(response, detected_language)
            
            # Add assistant response to history
            self._add_message_to_conversation(
                conversation=conversation,
                role="assistant",
                content=final_response,
                language=detected_language,
                metadata={
                    'image_analyzed': True,
                    'image_size': image.size
                }
            )
            
            logger.info(f"✅ Processed image query in {detected_language} for conversation {conversation_id}")
            
            return create_success_response(
                data={
                    'response': final_response,
                    'conversation_id': conversation_id,
                    'detected_language': detected_language,
                    'original_query': query,
                    'image_info': {
                        'size': image.size,
                        'mode': image.mode
                    },
                    'message_count': len(conversation.messages)
                },
                message="Image query processed successfully"
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to process image query: {e}")
            return create_error_response(
                error_code="IMAGE_QUERY_ERROR",
                error_message=str(e),
                user_message="Failed to analyze the image. Please try again.",
                suggested_action="Ensure the image is valid and try again"
            )

    def handle_mixed_input(
        self,
        text: str,
        image: Optional[Image.Image] = None,
        conversation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Handle mixed input with full multi-lingual support."""
        try:
            if conversation_id is None:
                conversation_id = f"conv_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            if image is not None:
                # Multi-modal: text + image
                return self.process_image_query(image, text, conversation_id)
            else:
                # Text only
                return self.process_text_query(text, conversation_id, use_context=True)
            
        except Exception as e:
            logger.error(f"❌ Failed to handle mixed input: {e}")
            return create_error_response(
                error_code="MIXED_INPUT_ERROR",
                error_message=str(e),
                user_message="Failed to process your input. Please try again.",
                suggested_action="Check your input and try again"
            )

    def get_conversation_history(self, conversation_id: str) -> Dict[str, Any]:
        """Get conversation history with language information."""
        try:
            if conversation_id not in self.conversations:
                return create_error_response(
                    error_code="CONVERSATION_NOT_FOUND",
                    error_message=f"Conversation {conversation_id} not found",
                    user_message="Conversation not found",
                    retry_possible=False
                )
            
            conversation = self.conversations[conversation_id]
            
            messages = []
            for msg in conversation.messages:
                messages.append({
                    'message_id': msg.message_id,
                    'role': msg.role,
                    'content': msg.content,
                    'language': msg.language,
                    'sentiment': msg.sentiment,
                    'has_image': msg.image is not None,
                    'timestamp': msg.timestamp,
                    'metadata': msg.metadata
                })
            
            return create_success_response(
                data={
                    'conversation_id': conversation.conversation_id,
                    'user_id': conversation.user_id,
                    'primary_language': conversation.language,
                    'messages': messages,
                    'message_count': len(messages),
                    'sentiment_history': conversation.sentiment_history,
                    'created_at': conversation.created_at,
                    'updated_at': conversation.updated_at
                },
                message="Conversation history retrieved"
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to get conversation history: {e}")
            return create_error_response(
                error_code="HISTORY_ERROR",
                error_message=str(e),
                user_message="Failed to retrieve conversation history"
            )

    def get_supported_features(self) -> Dict[str, Any]:
        """Get information about supported features."""
        return {
            'multimodal': True,
            'sentiment_analysis': self.sentiment_engine is not None,
            'multilingual': self.multilingual_system is not None,
            'supported_languages': self.multilingual_system.supported_languages if self.multilingual_system else ['en'],
            'context_management': True,
            'conversation_history': True
        }


# Example usage
if __name__ == "__main__":
    print("🧪 Testing Multi-Modal Chatbot with Multi-lingual Support\n")
    
    # Initialize chatbot
    chatbot = MultiModalChatbot()
    
    # Test multilingual text queries
    test_queries = [
        "What is artificial intelligence?",  # English
        "¿Qué es la inteligencia artificial?",  # Spanish
        "कृत्रिम बुद्धिमत्ता क्या है?",  # Hindi
        "Qu'est-ce que l'intelligence artificielle?"  # French
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"{i}️⃣ Testing query: {query}")
        result = chatbot.process_text_query(query, f"test_conv_{i}")
        
        if result['success']:
            data = result['data']
            print(f"   Detected Language: {data['detected_language']}")
            print(f"   Response: {data['response'][:100]}...")
        else:
            print(f"   Error: {result['user_message']}")
        print()
    
    # Show supported features
    features = chatbot.get_supported_features()
    print("🔧 Supported Features:")
    for feature, enabled in features.items():
        status = "✅" if enabled else "❌"
        print(f"   {status} {feature}: {enabled}")
    
    print("\n✅ Multi-lingual multi-modal chatbot test completed!")