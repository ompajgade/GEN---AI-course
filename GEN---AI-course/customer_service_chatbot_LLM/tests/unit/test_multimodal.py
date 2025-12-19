"""
Unit tests for Multi-Modal Chatbot (Task 2)
"""

import unittest
import sys
import os
from unittest.mock import Mock, patch, MagicMock
from PIL import Image
import numpy as np

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from task2_multimodal.multimodal_chatbot import MultiModalChatbot, Message, Conversation


class TestMultiModalChatbot(unittest.TestCase):
    """Test cases for MultiModalChatbot class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock dependencies to avoid actual API calls
        self.mock_llm = Mock()
        self.mock_vector_db = Mock()
        
        # Initialize chatbot with mocked dependencies
        self.chatbot = MultiModalChatbot(
            llm=self.mock_llm,
            vector_db=self.mock_vector_db,
            enable_sentiment=False,
            enable_multilingual=False
        )
    
    def test_initialization(self):
        """Test chatbot initialization."""
        self.assertIsNotNone(self.chatbot)
        self.assertEqual(self.chatbot.max_context_messages, 10)
        self.assertIsInstance(self.chatbot.conversations, dict)
        self.assertEqual(len(self.chatbot.conversations), 0)
    
    def test_conversation_creation(self):
        """Test conversation creation."""
        conv_id = "test_conv_1"
        conversation = self.chatbot._get_or_create_conversation(conv_id)
        
        self.assertIsInstance(conversation, Conversation)
        self.assertEqual(conversation.conversation_id, conv_id)
        self.assertEqual(conversation.user_id, "default_user")
        self.assertEqual(len(conversation.messages), 0)
        
        # Test that same conversation is returned
        same_conversation = self.chatbot._get_or_create_conversation(conv_id)
        self.assertEqual(conversation, same_conversation)
    
    def test_message_addition(self):
        """Test adding messages to conversation."""
        conv_id = "test_conv_2"
        conversation = self.chatbot._get_or_create_conversation(conv_id)
        
        # Add user message
        message = self.chatbot._add_message_to_conversation(
            conversation=conversation,
            role="user",
            content="Hello, how are you?",
            language="en"
        )
        
        self.assertIsInstance(message, Message)
        self.assertEqual(message.role, "user")
        self.assertEqual(message.content, "Hello, how are you?")
        self.assertEqual(message.language, "en")
        self.assertEqual(len(conversation.messages), 1)
    
    def test_context_limit(self):
        """Test conversation context limit."""
        conv_id = "test_conv_3"
        conversation = self.chatbot._get_or_create_conversation(conv_id)
        
        # Add more messages than the limit
        for i in range(15):
            self.chatbot._add_message_to_conversation(
                conversation=conversation,
                role="user",
                content=f"Message {i}",
                language="en"
            )
        
        # Should only keep the last 10 messages
        self.assertEqual(len(conversation.messages), 10)
        self.assertEqual(conversation.messages[0].content, "Message 5")
        self.assertEqual(conversation.messages[-1].content, "Message 14")
    
    @patch('task2_multimodal.multimodal_chatbot.logger')
    def test_process_text_query_success(self, mock_logger):
        """Test successful text query processing."""
        # Mock LLM response
        self.mock_llm.generate_text.return_value = "This is a test response."
        
        result = self.chatbot.process_text_query(
            query="What is AI?",
            conversation_id="test_conv_4"
        )
        
        self.assertTrue(result['success'])
        self.assertEqual(result['data']['response'], "This is a test response.")
        self.assertEqual(result['data']['conversation_id'], "test_conv_4")
        self.assertIn('message_count', result['data'])
        
        # Verify LLM was called
        self.mock_llm.generate_text.assert_called_once()
    
    @patch('task2_multimodal.multimodal_chatbot.logger')
    def test_process_text_query_error(self, mock_logger):
        """Test text query processing with error."""
        # Mock LLM to raise exception
        self.mock_llm.generate_text.side_effect = Exception("API Error")
        
        result = self.chatbot.process_text_query(
            query="What is AI?",
            conversation_id="test_conv_5"
        )
        
        self.assertFalse(result['success'])
        self.assertEqual(result['error_code'], "TEXT_QUERY_ERROR")
        self.assertIn("API Error", result['error_message'])
    
    def test_process_image_query_invalid_image(self):
        """Test image query processing with invalid image."""
        result = self.chatbot.process_image_query(
            image=None,
            query="What's in this image?",
            conversation_id="test_conv_6"
        )
        
        self.assertFalse(result['success'])
        self.assertEqual(result['error_code'], "IMAGE_QUERY_ERROR")
        self.assertIn("Image cannot be None", result['error_message'])
    
    @patch('task2_multimodal.multimodal_chatbot.logger')
    def test_process_image_query_success(self, mock_logger):
        """Test successful image query processing."""
        # Create a mock image
        mock_image = Mock(spec=Image.Image)
        mock_image.size = (100, 100)
        mock_image.mode = "RGB"
        
        # Mock LLM response
        self.mock_llm.analyze_image.return_value = {
            'success': True,
            'description': "This is a test image description."
        }
        
        result = self.chatbot.process_image_query(
            image=mock_image,
            query="What's in this image?",
            conversation_id="test_conv_7"
        )
        
        self.assertTrue(result['success'])
        self.assertEqual(result['data']['response'], "This is a test image description.")
        self.assertEqual(result['data']['image_info']['size'], (100, 100))
        
        # Verify LLM was called
        self.mock_llm.analyze_image.assert_called_once()
    
    def test_handle_mixed_input_text_only(self):
        """Test mixed input handling with text only."""
        # Mock LLM response
        self.mock_llm.generate_text.return_value = "Text response."
        
        result = self.chatbot.handle_mixed_input(
            text="Hello",
            image=None,
            conversation_id="test_conv_8"
        )
        
        self.assertTrue(result['success'])
        self.assertEqual(result['data']['response'], "Text response.")
    
    def test_handle_mixed_input_with_image(self):
        """Test mixed input handling with image."""
        # Create a mock image
        mock_image = Mock(spec=Image.Image)
        mock_image.size = (100, 100)
        mock_image.mode = "RGB"
        
        # Mock LLM response
        self.mock_llm.analyze_image.return_value = {
            'success': True,
            'description': "Mixed input response."
        }
        
        result = self.chatbot.handle_mixed_input(
            text="What's this?",
            image=mock_image,
            conversation_id="test_conv_9"
        )
        
        self.assertTrue(result['success'])
        self.assertEqual(result['data']['response'], "Mixed input response.")
    
    def test_get_conversation_history_success(self):
        """Test getting conversation history."""
        conv_id = "test_conv_10"
        conversation = self.chatbot._get_or_create_conversation(conv_id)
        
        # Add some messages
        self.chatbot._add_message_to_conversation(
            conversation=conversation,
            role="user",
            content="Hello",
            language="en"
        )
        self.chatbot._add_message_to_conversation(
            conversation=conversation,
            role="assistant",
            content="Hi there!",
            language="en"
        )
        
        result = self.chatbot.get_conversation_history(conv_id)
        
        self.assertTrue(result['success'])
        self.assertEqual(result['data']['conversation_id'], conv_id)
        self.assertEqual(result['data']['message_count'], 2)
        self.assertEqual(len(result['data']['messages']), 2)
    
    def test_get_conversation_history_not_found(self):
        """Test getting history for non-existent conversation."""
        result = self.chatbot.get_conversation_history("nonexistent_conv")
        
        self.assertFalse(result['success'])
        self.assertEqual(result['error_code'], "CONVERSATION_NOT_FOUND")
    
    def test_clear_conversation_success(self):
        """Test clearing conversation."""
        conv_id = "test_conv_11"
        conversation = self.chatbot._get_or_create_conversation(conv_id)
        
        # Add a message
        self.chatbot._add_message_to_conversation(
            conversation=conversation,
            role="user",
            content="Hello",
            language="en"
        )
        
        # Clear conversation
        result = self.chatbot.clear_conversation(conv_id)
        
        self.assertTrue(result['success'])
        self.assertNotIn(conv_id, self.chatbot.conversations)
    
    def test_clear_conversation_not_found(self):
        """Test clearing non-existent conversation."""
        result = self.chatbot.clear_conversation("nonexistent_conv")
        
        self.assertFalse(result['success'])
        self.assertEqual(result['error_code'], "CONVERSATION_NOT_FOUND")
    
    def test_list_conversations(self):
        """Test listing conversations."""
        # Create some conversations
        self.chatbot._get_or_create_conversation("conv_1")
        self.chatbot._get_or_create_conversation("conv_2")
        
        result = self.chatbot.list_conversations()
        
        self.assertTrue(result['success'])
        self.assertEqual(len(result['data']['conversations']), 2)
    
    def test_get_supported_features(self):
        """Test getting supported features."""
        features = self.chatbot.get_supported_features()
        
        self.assertIsInstance(features, dict)
        self.assertTrue(features['multimodal'])
        self.assertTrue(features['context_management'])
        self.assertTrue(features['conversation_history'])
        self.assertIn('sentiment_analysis', features)
        self.assertIn('multilingual', features)


class TestMessage(unittest.TestCase):
    """Test cases for Message class."""
    
    def test_message_creation(self):
        """Test message creation."""
        message = Message(
            message_id="msg_1",
            role="user",
            content="Hello",
            language="en"
        )
        
        self.assertEqual(message.message_id, "msg_1")
        self.assertEqual(message.role, "user")
        self.assertEqual(message.content, "Hello")
        self.assertEqual(message.language, "en")
        self.assertIsNone(message.image)
        self.assertIsNone(message.sentiment)


class TestConversation(unittest.TestCase):
    """Test cases for Conversation class."""
    
    def test_conversation_creation(self):
        """Test conversation creation."""
        conversation = Conversation(
            conversation_id="conv_1",
            user_id="user_1"
        )
        
        self.assertEqual(conversation.conversation_id, "conv_1")
        self.assertEqual(conversation.user_id, "user_1")
        self.assertEqual(len(conversation.messages), 0)
        self.assertEqual(conversation.language, "en")
        self.assertEqual(len(conversation.sentiment_history), 0)


if __name__ == '__main__':
    unittest.main()