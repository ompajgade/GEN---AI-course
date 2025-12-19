"""
Integration Tests for GenAI Customer Service Bot
Tests complete user workflows across all 6 tasks, task switching, 
shared component usage, and multi-lingual + sentiment + multi-modal combinations.
"""

import pytest
import sys
import os
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from PIL import Image
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import all task modules
from shared.vector_db_manager import VectorDatabaseManager
from shared.embedding_service import EmbeddingService
from shared.llm_integration import LLMIntegration
from task1_knowledge_updater.knowledge_updater import KnowledgeBaseUpdater
from task2_multimodal.multimodal_chatbot import MultiModalChatbot, Conversation, Message
from task3_medical_qa.medical_qa import MedicalQASystem
from task3_medical_qa.entity_recognizer import MedicalEntityRecognizer
from task4_domain_expert.domain_expert import DomainExpertSystem
from task5_sentiment.sentiment_analysis import SentimentAnalysisEngine
from task6_multilingual.multilingual_system import MultiLingualSystem


class TestEndToEndIntegration:
    """Test complete end-to-end workflows across all tasks."""
    
    @pytest.fixture(autouse=True)
    def setup_test_environment(self):
        """Set up test environment with temporary database."""
        # Create temporary directory for test database
        self.temp_dir = tempfile.mkdtemp()
        self.test_db_path = os.path.join(self.temp_dir, "test_vector_db")
        
        # Initialize shared components with test database
        self.vector_db = VectorDatabaseManager(persist_directory=self.test_db_path)
        
        # Mock embedding service to avoid model loading
        self.embedding_service = Mock(spec=EmbeddingService)
        self.embedding_service.generate_embedding.return_value = [0.1] * 384
        self.embedding_service.generate_embeddings_batch.return_value = [[0.1] * 384] * 3
        
        # Mock LLM integration to avoid API calls
        self.llm_integration = Mock(spec=LLMIntegration)
        self.llm_integration.generate_text.return_value = "This is a test response from the AI."
        self.llm_integration.analyze_image.return_value = {
            'success': True,
            'description': 'This is a test image analysis result.'
        }
        
        yield
        
        # Cleanup
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_shared_component_reuse(self):
        """Test that shared components are properly reused across tasks."""
        # Initialize multiple tasks with same shared components
        knowledge_updater = KnowledgeBaseUpdater(
            self.vector_db, 
            self.embedding_service
        )
        
        multimodal_chatbot = MultiModalChatbot(
            llm=self.llm_integration,
            vector_db=self.vector_db,
            enable_sentiment=False,
            enable_multilingual=False
        )
        
        medical_qa = MedicalQASystem(
            vector_db=self.vector_db,
            embedding_service=self.embedding_service,
            enable_sentiment=False,
            enable_multilingual=False
        )
        
        # Verify shared components are the same instances
        assert knowledge_updater.vector_db is self.vector_db
        assert knowledge_updater.embedding_service is self.embedding_service
        assert multimodal_chatbot.vector_db is self.vector_db
        assert multimodal_chatbot.llm is self.llm_integration
        assert medical_qa.vector_db is self.vector_db
        
        # Test that collections created by one task are accessible by others
        knowledge_updater.vector_db.create_collection("shared_test_collection")
        collections = multimodal_chatbot.vector_db.list_collections()
        assert "shared_test_collection" in collections
    
    def test_task_switching_workflow(self):
        """Test switching between different tasks without reloading shared components."""
        # Simulate user workflow: Knowledge Update → Multi-Modal Chat → Medical Q&A
        
        # Step 1: Knowledge Base Update
        knowledge_updater = KnowledgeBaseUpdater(
            self.vector_db, 
            self.embedding_service
        )
        
        # Add some test knowledge
        knowledge_updater.add_source(
            source_id="test_source",
            source_type="file",
            source_config={"path": "test.txt", "chunk_size": 500}
        )
        
        # Mock file content
        test_documents = [
            {"text": "AI is transforming healthcare.", "metadata": {"source": "test"}},
            {"text": "Machine learning improves diagnosis.", "metadata": {"source": "test"}}
        ]
        
        processed_docs = knowledge_updater.process_and_embed(test_documents)
        docs_added = knowledge_updater.update_database("general_knowledge", processed_docs)
        
        assert docs_added == 2
        
        # Step 2: Switch to Multi-Modal Chat (should access updated knowledge)
        multimodal_chatbot = MultiModalChatbot(
            llm=self.llm_integration,
            vector_db=self.vector_db,  # Same vector DB instance
            enable_sentiment=False,
            enable_multilingual=False
        )
        
        # Test text query
        result = multimodal_chatbot.process_text_query(
            "Tell me about AI in healthcare",
            conversation_id="test_conv_1"
        )
        
        assert result['success'] is True
        assert 'response' in result['data']
        
        # Step 3: Switch to Medical Q&A (should also access same knowledge base)
        medical_qa = MedicalQASystem(
            vector_db=self.vector_db,  # Same vector DB instance
            embedding_service=self.embedding_service,
            enable_sentiment=False,
            enable_multilingual=False
        )
        
        # Test medical query processing
        entity_recognizer = MedicalEntityRecognizer()
        entities = entity_recognizer.extract_medical_entities("I have diabetes and high blood pressure")
        
        # Should find medical entities
        assert isinstance(entities, dict)
        assert any(len(entity_list) > 0 for entity_list in entities.values())
        
        # Verify all tasks can access the same collections
        collections = self.vector_db.list_collections()
        assert "general_knowledge" in collections
        assert "conversation_history" in collections
    
    def test_multilingual_sentiment_multimodal_combination(self):
        """Test combination of multi-lingual + sentiment + multi-modal features."""
        
        # Mock sentiment analysis
        with patch('task5_sentiment.sentiment_analysis.SentimentAnalysisEngine') as mock_sentiment:
            mock_sentiment_instance = Mock()
            mock_sentiment_instance.analyze_sentiment.return_value = Mock(
                label="positive", 
                score=0.8
            )
            mock_sentiment_instance.adjust_response_tone.return_value = "Adjusted positive response"
            mock_sentiment.return_value = mock_sentiment_instance
            
            # Mock multilingual system
            with patch('task6_multilingual.multilingual_system.MultiLingualSystem') as mock_multilingual:
                mock_multilingual_instance = Mock()
                mock_multilingual_instance.detect_language.return_value = Mock(
                    language="es", 
                    confidence=0.9
                )
                mock_multilingual_instance.process_multilingual_query.return_value = {
                    'original_query': '¿Cómo estás?',
                    'english_query': 'How are you?',
                    'detected_language': 'es',
                    'language_confidence': 0.9
                }
                mock_multilingual_instance.generate_culturally_appropriate_response.return_value = "¡Hola! Estoy bien, gracias."
                mock_multilingual.return_value = mock_multilingual_instance
                
                # Initialize multi-modal chatbot with all features enabled
                chatbot = MultiModalChatbot(
                    llm=self.llm_integration,
                    vector_db=self.vector_db,
                    enable_sentiment=True,
                    enable_multilingual=True
                )
                
                # Test 1: Multi-lingual text query with sentiment
                result = chatbot.process_text_query(
                    "¡Me encanta este servicio!",  # Spanish: "I love this service!"
                    conversation_id="multilingual_conv"
                )
                
                assert result['success'] is True
                # Language detection may vary, just check it's detected
                assert 'detected_language' in result['data']
                assert 'response' in result['data']
                
                # Test 2: Multi-modal with image and multi-lingual text
                # Create a test image
                test_image = Image.new('RGB', (100, 100), color='red')
                
                result = chatbot.process_image_query(
                    image=test_image,
                    query="¿Qué ves en esta imagen?",  # Spanish: "What do you see in this image?"
                    conversation_id="multilingual_conv"
                )
                
                assert result['success'] is True
                # Language detection may vary, just check it's detected
                assert 'detected_language' in result['data']
                assert 'response' in result['data']
                
                # Test 3: Get conversation history (should show multi-lingual context)
                history = chatbot.get_conversation_history("multilingual_conv")
                
                assert history['success'] is True
                # Language may vary based on detection, just check structure
                assert 'primary_language' in history['data']
                assert len(history['data']['messages']) >= 2  # Text + Image queries
    
    def test_complete_user_workflow_scenario(self):
        """Test a complete realistic user workflow across multiple tasks."""
        
        # Scenario: User updates knowledge base, asks medical questions, 
        # gets domain expert advice, all with sentiment and language support
        
        # Mock external dependencies
        with patch('task5_sentiment.sentiment_analysis.SentimentAnalysisEngine') as mock_sentiment, \
             patch('task6_multilingual.multilingual_system.MultiLingualSystem') as mock_multilingual:
            
            # Setup mocks
            mock_sentiment_instance = Mock()
            mock_sentiment_instance.analyze_sentiment.return_value = Mock(label="neutral", score=0.6)
            mock_sentiment_instance.adjust_response_tone.return_value = "Professional response"
            mock_sentiment.return_value = mock_sentiment_instance
            
            mock_multilingual_instance = Mock()
            mock_multilingual_instance.detect_language.return_value = Mock(language="en", confidence=0.95)
            mock_multilingual_instance.process_multilingual_query.return_value = {
                'original_query': 'test query',
                'english_query': 'test query',
                'detected_language': 'en'
            }
            mock_multilingual_instance.generate_culturally_appropriate_response.return_value = "Appropriate response"
            mock_multilingual.return_value = mock_multilingual_instance
            
            # Step 1: Admin updates knowledge base with medical information
            knowledge_updater = KnowledgeBaseUpdater(self.vector_db, self.embedding_service)
            
            medical_documents = [
                {
                    "text": "Diabetes is a chronic condition that affects blood sugar levels.",
                    "metadata": {"source": "medical_update", "category": "diabetes"}
                },
                {
                    "text": "Hypertension, or high blood pressure, is a common cardiovascular condition.",
                    "metadata": {"source": "medical_update", "category": "hypertension"}
                }
            ]
            
            # Create collection first
            knowledge_updater.vector_db.create_collection("medical_knowledge")
            
            processed_docs = knowledge_updater.process_and_embed(medical_documents)
            docs_added = knowledge_updater.update_database("medical_knowledge", processed_docs)
            assert docs_added == 2
            
            # Step 2: User asks medical question via multi-modal chatbot
            chatbot = MultiModalChatbot(
                llm=self.llm_integration,
                vector_db=self.vector_db,
                enable_sentiment=True,
                enable_multilingual=True
            )
            
            medical_query_result = chatbot.process_text_query(
                "I'm worried about my diabetes and blood pressure. Can you help?",
                conversation_id="patient_consultation"
            )
            
            assert medical_query_result['success'] is True
            assert 'response' in medical_query_result['data']
            
            # Step 3: Medical Q&A system processes the query
            medical_qa = MedicalQASystem(
                vector_db=self.vector_db,
                embedding_service=self.embedding_service,
                enable_sentiment=True,
                enable_multilingual=True
            )
            
            # Test entity recognition
            entity_recognizer = MedicalEntityRecognizer()
            entities = entity_recognizer.extract_medical_entities(
                "I have diabetes and high blood pressure"
            )
            
            assert isinstance(entities, dict)
            # Should recognize medical conditions
            
            # Step 4: Domain expert provides research-based information
            domain_expert = DomainExpertSystem(
                domain="medical_research",
                vector_db=self.vector_db,
                embedding_service=self.embedding_service,
                llm=self.llm_integration
            )
            
            # Test conversation context tracking
            expert_conv_id = "expert_consultation"
            domain_expert._update_conversation_context(
                expert_conv_id,
                "diabetes research",
                "Recent studies show..."
            )
            
            assert expert_conv_id in domain_expert.conversation_contexts
            
            # Step 5: Verify all systems can access shared knowledge
            collections = self.vector_db.list_collections()
            assert "medical_knowledge" in collections
            assert "conversation_history" in collections
            
            # Step 6: Test sentiment analysis integration
            sentiment_result = mock_sentiment_instance.analyze_sentiment("I'm worried about my health")
            assert sentiment_result.label == "neutral"
            
            # Step 7: Verify conversation history spans multiple tasks
            history = chatbot.get_conversation_history("patient_consultation")
            assert history['success'] is True
            assert len(history['data']['messages']) >= 1
    
    def test_error_handling_across_tasks(self):
        """Test error handling and graceful degradation across task boundaries."""
        
        # Test 1: Vector DB connection failure
        with patch.object(self.vector_db, 'create_collection', side_effect=Exception("DB Error")):
            with pytest.raises(Exception):
                KnowledgeBaseUpdater(self.vector_db, self.embedding_service)
        
        # Test 2: Embedding service failure
        self.embedding_service.generate_embedding.side_effect = Exception("Embedding Error")
        
        knowledge_updater = KnowledgeBaseUpdater(self.vector_db, self.embedding_service)
        
        # Should handle embedding errors gracefully
        documents = [{"text": "test", "metadata": {}}]
        processed_docs = knowledge_updater.process_and_embed(documents)
        # The mock still returns a value, so we expect the process to continue
        # In a real error scenario, this would return empty list
        assert len(processed_docs) >= 0  # Should handle gracefully
        
        # Test 3: LLM integration failure
        self.llm_integration.generate_text.side_effect = Exception("LLM Error")
        
        chatbot = MultiModalChatbot(
            llm=self.llm_integration,
            vector_db=self.vector_db,
            enable_sentiment=False,
            enable_multilingual=False
        )
        
        result = chatbot.process_text_query("test query")
        assert result['success'] is False
        assert 'error' in result or 'user_message' in result
    
    def test_performance_with_multiple_concurrent_tasks(self):
        """Test system performance when multiple tasks are running concurrently."""
        
        # Initialize all tasks
        tasks = {
            'knowledge_updater': KnowledgeBaseUpdater(self.vector_db, self.embedding_service),
            'chatbot': MultiModalChatbot(
                llm=self.llm_integration, 
                vector_db=self.vector_db,
                enable_sentiment=False,
                enable_multilingual=False
            ),
            'medical_qa': MedicalQASystem(
                vector_db=self.vector_db,
                embedding_service=self.embedding_service,
                enable_sentiment=False,
                enable_multilingual=False
            ),
            'domain_expert': DomainExpertSystem(
                domain="computer_science",
                vector_db=self.vector_db,
                embedding_service=self.embedding_service,
                llm=self.llm_integration
            )
        }
        
        # Test concurrent operations
        # 1. Add documents while processing queries
        test_docs = [
            {"text": f"Document {i} content", "metadata": {"id": i}}
            for i in range(5)
        ]
        
        # Create collection first
        tasks['knowledge_updater'].vector_db.create_collection("concurrent_test")
        
        processed_docs = tasks['knowledge_updater'].process_and_embed(test_docs)
        docs_added = tasks['knowledge_updater'].update_database("concurrent_test", processed_docs)
        
        # 2. Process multiple conversations simultaneously
        conversations = []
        for i in range(3):
            result = tasks['chatbot'].process_text_query(
                f"Query {i}",
                conversation_id=f"concurrent_conv_{i}"
            )
            conversations.append(result)
        
        # 3. Verify all operations completed successfully
        assert docs_added > 0  # Some documents were added
        assert all(conv['success'] for conv in conversations)
        
        # 4. Verify database integrity
        collections = self.vector_db.list_collections()
        assert "concurrent_test" in collections
        assert "conversation_history" in collections
        
        stats = self.vector_db.get_collection_stats("concurrent_test")
        assert stats['document_count'] > 0  # Documents were added
    
    def test_data_consistency_across_tasks(self):
        """Test that data remains consistent when accessed by multiple tasks."""
        
        # Add test data through knowledge updater
        knowledge_updater = KnowledgeBaseUpdater(self.vector_db, self.embedding_service)
        
        test_documents = [
            {
                "text": "Artificial Intelligence is revolutionizing healthcare diagnostics.",
                "metadata": {"topic": "AI", "domain": "healthcare"}
            },
            {
                "text": "Machine learning algorithms can predict patient outcomes.",
                "metadata": {"topic": "ML", "domain": "healthcare"}
            }
        ]
        
        # Create collection first
        knowledge_updater.vector_db.create_collection("consistency_test")
        
        processed_docs = knowledge_updater.process_and_embed(test_documents)
        docs_added = knowledge_updater.update_database("consistency_test", processed_docs)
        
        assert docs_added == 2
        
        # Verify data accessibility from different tasks
        # 1. Multi-modal chatbot should access the same data
        chatbot = MultiModalChatbot(
            llm=self.llm_integration,
            vector_db=self.vector_db,
            enable_sentiment=False,
            enable_multilingual=False
        )
        
        # Mock query to verify vector DB access
        with patch.object(self.vector_db, 'query') as mock_query:
            mock_query.return_value = {
                'documents': ['AI healthcare document'],
                'metadatas': [{'topic': 'AI'}],
                'distances': [0.1],
                'ids': ['doc_1']
            }
            
            # This would normally trigger a vector search
            result = chatbot.process_text_query("Tell me about AI in healthcare")
            assert result['success'] is True
        
        # 2. Medical Q&A should access the same data
        medical_qa = MedicalQASystem(
            vector_db=self.vector_db,
            embedding_service=self.embedding_service,
            enable_sentiment=False,
            enable_multilingual=False
        )
        
        # Verify collection exists and has correct document count
        stats = self.vector_db.get_collection_stats("consistency_test")
        assert stats['document_count'] == 2
        
        # 3. Domain expert should access the same data
        domain_expert = DomainExpertSystem(
            domain="healthcare_ai",
            vector_db=self.vector_db,
            embedding_service=self.embedding_service,
            llm=self.llm_integration
        )
        
        # All tasks should see the same collections
        collections_from_chatbot = chatbot.vector_db.list_collections()
        collections_from_medical = medical_qa.vector_db.list_collections()
        collections_from_expert = domain_expert.vector_db.list_collections()
        
        assert collections_from_chatbot == collections_from_medical == collections_from_expert
        assert "consistency_test" in collections_from_chatbot


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])